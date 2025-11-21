# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimator framework with automatic type normalization decorators.

This module provides the :class:`Estimator` base class and decorators that
automatically normalize return types to :class:`Mapping` format, enabling
uniform handling of both single-key and multi-key estimators.

Decorator Type Normalization
-----------------------------

The :func:`@shape <shape>`, :func:`@headers <headers>`, and :func:`@estimate <estimate>`
decorators automatically convert return types to :class:`Mapping` format:

**@shape decorator**:
    - Input: ``Mapping[K, tuple[int, int]] | tuple[int, int]``
    - Output: **Always** ``Mapping[K, tuple[int, int]]``
    - Single value ``(V, C)`` → ``{key: (V, C)}``

**@headers decorator**:
    - Input: ``Mapping[K, NDArrayStr] | NDArrayStr``
    - Output: **Always** ``Mapping[K, NDArrayStr]``
    - Single array → ``{key: array}``

**@estimate decorator**:
    - Input: ``Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None``
    - Output: **Always** ``Mapping[K, NDArrayFloat]``
    - Single value → ``{key: value}``
    - ``None`` values → replaced with ``configure_missing_array(key)``

This means that even if an estimator's implementation returns a single value,
the decorated property/method will always return a :class:`Mapping`, allowing
code to uniformly access results via ``.keys()``, ``.values()``, and ``.items()``.

Single-Key Estimator Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    class PoseEstimator(Estimator["mediapipe.pose"]):
        @property
        @shape
        def shape(self) -> tuple[int, int]:
            return (33, 4)  # Returns tuple

        # After decoration, shape property returns:
        # {"mediapipe.pose": (33, 4)}  # Mapping!

Multi-Key Estimator Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    class HolisticEstimator(
        Estimator[Literal["pose", "left_hand", "right_hand", "face"]]
    ):
        @property
        @shape
        def shape(self) -> Mapping[str, tuple[int, int]]:
            return {
                "pose": (33, 4),
                "left_hand": (21, 4),
                "right_hand": (21, 4),
                "face": (468, 4),
            }

        # After decoration, shape property returns the same Mapping
        # No conversion needed, but type is guaranteed

Key Benefits
~~~~~~~~~~~~

1. **Uniform Access**: All estimators expose :class:`Mapping` interface
2. **Simplified Logic**: Framework code can use ``.keys()`` without type checking
3. **Automatic ``None`` Handling**: Missing landmarks replaced with configured arrays
4. **Type Safety**: Decorators ensure consistent return types

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

When implementing estimators:
    - Return natural types (single values or Mappings) from implementations
    - Decorators handle normalization automatically
    - Access ``self.shape[key]``, ``self.headers[key]`` safely in all methods
    - ``configure_missing_array(key)`` works correctly for both single and multi-key

See Also
--------
:class:`Estimator` : Base class for landmark estimators
:func:`shape` : Shape decorator with type normalization
:func:`headers` : Headers decorator with type normalization
:func:`estimate` : Estimate decorator with type normalization and None handling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Mapping,
    Callable,
    Concatenate,
    TypedDict,
    overload,
    is_typeddict,
    get_origin,
)
from typing_extensions import TypeIs
from dataclasses import dataclass
from functools import update_wrapper, cache
from itertools import product

import numpy as np

from ..typings import ArrayLikeFloat, NDArrayFloat, ArrayLikeStr, NDArrayStr, MatLike
from .options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS


@dataclass
class ProcessResult[K: str]:
    """Result of a single frame processing operation.

    Contains the frame identifier, landmark headers, detected landmarks,
    and the annotated frame with visualization overlays.

    Attributes:
        frame_id (:obj:`int`): Sequential frame identifier.
        headers (
            :class:`~typing.Mapping`\\[
                :obj:`K`, :class:`numpy.typing.NDArray`\\[:obj:`str`\\]
            \\]
        ):
            Mapping from landmark keys to their string headers.
        landmarks (
            :class:`~typing.Mapping`\\[
                :obj:`K`, :class:`numpy.typing.NDArray`\\[:obj:`float`\\]
            \\]
        ):
            Mapping from landmark keys to their numeric coordinate arrays.
        annotated_frame (:class:`MatLike`): Frame with visualization
            annotations.
    """

    frame_id: int
    headers: Mapping[K, NDArrayStr]
    landmarks: Mapping[K, NDArrayFloat]
    annotated_frame: MatLike


# ============================================================
# Estimator Base Class
# ============================================================

type EstimatorWithKey = Estimator[Any]


class _DecoratorWithOptions[
    E: EstimatorWithKey,
    M: Mapping[str, object],
    **Pi,
    Ri,
    **Po,
    Ro,
]:
    def __init__(
        self,
        decorator: Callable[
            [Callable[Concatenate[E, Pi], Ri], M], Callable[Concatenate[E, Po], Ro]
        ],
        options: M,
    ):
        self._decorator = decorator
        self._options = options

    def __call__(
        self, func: Callable[Concatenate[E, Pi], Ri]
    ) -> Callable[Concatenate[E, Po], Ro]:
        return self._decorator(func, self._options)


class KeyOptions[K: str](TypedDict, total=False):
    key: K


def _typeddict_typeis[T: Mapping[str, object]](
    obj: object, type_: type[T]
) -> TypeIs[T]:
    origin = type_
    while (new_origin := get_origin(origin)) is not None:
        origin = new_origin

    assert is_typeddict(origin), f"`{origin}` is not a TypedDict"
    return isinstance(obj, Mapping)


def _get_key_from_options_or_estimator[K: str](
    estimater: Estimator[K],
    options: KeyOptions[K] | None,
) -> K:
    key_from_options: K | None = None if options is None else options.get("key")

    try:
        key_from_estimator: K = estimater.configure_estimator_name()
    except (TypeError, ValueError, AttributeError) as e:
        # Runtime error detecting implementation mistakes in subclasses
        estimator_class = estimater.__class__.__name__
        raise TypeError(
            f"Error calling configure_estimator_name() on "
            f"{estimator_class}: {e}\n\n"
            f"IMPLEMENTATION CHECK REQUIRED:\n"
            f"  1. Verify {estimator_class}.configure_estimator_name() "
            f"returns a SINGLE key value\n"
            f"  2. Do NOT return tuple, list, or collection - "
            f"only one key of type K\n"
            f"  3. For multi-key estimators, return a representative "
            f"primary name\n"
            f"  4. Check the method signature matches: "
            f"def configure_estimator_name(self) -> K\n\n"
            f"Common mistakes:\n"
            f"  ❌ return (Key.A, Key.B)  # Wrong: tuple of keys\n"
            f"  ❌ return [Key.A, Key.B]  # Wrong: list of keys\n"
            f"  ✅ return Key.PRIMARY     # Correct: single key value\n\n"
            f"See {estimator_class}.configure_estimator_name() implementation."
        ) from e

    return key_from_options or key_from_estimator


# ============================================================
# shape decorator
# ============================================================

# Decorator Information
# External Function
# (E) -> Mapping[K, tuple[int, int]]  # ALWAYS returns Mapping
# Internal Function
# (E) -> Mapping[K, tuple[int, int]] | tuple[int, int]  # Can return either
#
# TYPE NORMALIZATION:
# - If internal returns tuple[int, int] → wrapped as {key: tuple}
# - If internal returns Mapping → returned as-is
# - Result is ALWAYS Mapping, so .keys() is always available

# override1: with key options
# override2: without key options


@overload
def shape[E: EstimatorWithKey, K: str](
    func: Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]],
    options: KeyOptions[K] | None = None,
    /,
) -> Callable[[E], Mapping[K, tuple[int, int]]]:
    """Decorator overload definition for shape without key options.

    Wraps the :meth:`shape` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@shape, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        func (`Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]]`):
            Method that returns shape information.
        options (`KeyOptions[K] | None`, optional):
            TypedDict containing key options. Defaults to None.

    Returns:
        :class:`Callable[[E], Mapping[K, tuple[int, int]]]`:
            Decorated method.
    """


@overload
def shape[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
) -> Callable[
    [Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]]],
    Callable[[E], Mapping[K, tuple[int, int]]],
]:
    """Decorator overload definition for shape with key options.

    Wraps the :meth:`shape` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@shape, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options.

    Returns:
        :class:`Callable[[Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]]],
        Callable[[E], Mapping[K, tuple[int, int]]]]`:
            Decorated method.
    """


# implementation
def shape[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]] | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None,
    /,
) -> (
    Callable[[E], Mapping[K, tuple[int, int]]]
    | Callable[
        [Callable[[E], Mapping[K, tuple[int, int]] | tuple[int, int]]],
        Callable[[E], Mapping[K, tuple[int, int]]],
    ]
):
    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E,
            KeyOptions[K],
            [],
            Mapping[K, tuple[int, int]] | tuple[int, int],
            [],
            Mapping[K, tuple[int, int]],
        ](shape, arg1)

    if not callable(arg1):
        raise TypeError("First argument must be a callable or KeyOptions TypedDict.")

    def wrapper(self: E, /) -> Mapping[K, tuple[int, int]]:
        result = arg1(self)

        converted: Mapping[K, tuple[int, int]]
        if isinstance(result, Mapping):
            converted = {k: v for k, v in result.items()}
        else:
            key = _get_key_from_options_or_estimator(self, arg2)
            converted = {key: result}

        return converted

    update_wrapper(wrapper, arg1)
    return wrapper


# ============================================================
# headers decorator
# ============================================================

# Decorator Information

# External Function
# (E) -> Mapping[K, NDArrayStr]  # ALWAYS returns Mapping
# Internal Function
# (E) -> Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None  # Can return either
#
# TYPE NORMALIZATION:
# - If internal returns ArrayLikeStr → wrapped as {key: np.array(value)}
# - If internal returns Mapping → each value converted to NDArrayStr
# - None values → replaced with empty array
# - Result is ALWAYS Mapping, so .keys() is always available

# override1: with key options
# override2: without key options


@overload
def headers[E: EstimatorWithKey, K: str](
    func: Callable[[E], Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None],
    options: KeyOptions[K] | None = None,
    /,
) -> Callable[[E], Mapping[K, NDArrayStr]]:
    """Decorator overload definition for headers without key options.

    Wraps the :meth:`headers` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@headers, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        func (`Callable[[E], Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None]`):
            Method that returns header array.
        options (`KeyOptions[K] | None`, optional):
            TypedDict containing key options. Defaults to None.

    Returns:
        :class:`Callable[[E], Mapping[K, NDArrayStr]]`:
            Decorated method.
    """


@overload
def headers[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
) -> Callable[
    [Callable[[E], Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None]],
    Callable[[E], Mapping[K, NDArrayStr]],
]:
    """Decorator overload definition for headers with key options.

    Wraps the :meth:`headers` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@headers, <item type>]`
    item type: :class:`ArrayLikeStr` -> :class:`NDArrayStr`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options.

    Returns:
        :class:`Callable[[Callable[[E], Mapping[K, ArrayLikeStr | None]
        | ArrayLikeStr | None]], Callable[[E], Mapping[K, NDArrayStr]]]`:
            Decorator that converts the return type of the decorated
            method.
    """


def headers[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[[E], Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None,
    /,
) -> (
    Callable[[E], Mapping[K, NDArrayStr]]
    | Callable[
        [Callable[[E], Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None]],
        Callable[[E], Mapping[K, NDArrayStr]],
    ]
):
    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E,
            KeyOptions[K],
            [],
            Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None,
            [],
            Mapping[K, NDArrayStr],
        ](headers, arg1)

    if not callable(arg1):
        raise TypeError("First argument must be a callable or KeyOptions TypedDict.")

    def wrapper(self: E, /) -> Mapping[K, NDArrayStr]:
        result = arg1(self)

        mapping: Mapping[K, ArrayLikeStr | None]
        if isinstance(result, Mapping):
            mapping = result
        else:
            key = _get_key_from_options_or_estimator(self, arg2)
            mapping = {key: result}

        return {
            k: (
                np.asarray(
                    [
                        str(coord)
                        for coord in product(
                            range(self.shape[k][0]), range(self.shape[k][1])
                        )
                    ]
                ).reshape(self.shape[k])
                if v is None
                else np.asarray(v)
            )
            for k, v in mapping.items()
        }

    update_wrapper(wrapper, arg1)
    return wrapper


# ============================================================
# estimate decorator
# ============================================================

# Decorator Information

# External Function
# (E, MatLike | None, int) -> Mapping[K, NDArrayFloat]  # ALWAYS returns Mapping
# Internal Function
# (E, MatLike, int) -> Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None
#
# TYPE NORMALIZATION AND NONE HANDLING:
# - If internal returns ArrayLikeFloat → wrapped as {key: np.array(value)}
# - If internal returns Mapping → each value processed individually:
#   * None values → replaced with configure_missing_array(key)
#   * Non-None values → converted to NDArrayFloat
# - If frame_src is None → returns missing arrays for all keys in shape
# - Result is ALWAYS Mapping with NO None values
#
# CRITICAL: configure_missing_array(key) is called with individual keys,
# so it works correctly for both single-key and multi-key estimators.
# For multi-key: self.shape[key] returns the specific shape for that key.

# override1: with key options
# override2: without key options


@overload
def estimate[E: EstimatorWithKey, K: str](
    func: Callable[
        [E, MatLike, int], Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None
    ],
    options: KeyOptions[K] | None = None,
    /,
) -> Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]:
    """Decorator overload definition for estimate without key options.

    Wraps the :meth:`estimate` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@estimate, <item type>]`
    item type: :class:`ArrayLikeFloat` -> :class:`NDArrayFloat`

    Args:
        func (
            `Callable[[E, MatLike, int], Mapping[K, ArrayLikeFloat | None]
            | ArrayLikeFloat | None]`
        ):
            Method that performs estimation processing.
        options (`KeyOptions[K] | None`, optional):
            TypedDict containing key options. Defaults to None.

    Returns:
        :class:`Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]`:
            Decorated method.

    """


@overload
def estimate[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
) -> Callable[
    [
        Callable[
            [E, MatLike, int], Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None
        ]
    ],
    Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]],
]:
    """Decorator overload definition for estimate with key options.

    Wraps the :meth:`estimate` method of :class:`Estimator` class
    to converted mapping

    mapping type: wrapping to :class:`Mapping[K@estimate, <item type>]`
    item type: :class:`ArrayLikeFloat` -> :class:`NDArrayFloat`

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options.

    Returns:
        :class:`Callable[[Callable[[E, MatLike, int],
        Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None]],
        Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]]`:
            Decorated method.
    """


# implementation
def estimate[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[
            [E, MatLike, int], Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None
        ]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None,
    /,
) -> (
    Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]]
    | Callable[
        [
            Callable[
                [E, MatLike, int],
                Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None,
            ]
        ],
        Callable[[E, MatLike | None, int], Mapping[K, NDArrayFloat]],
    ]
):
    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E,
            KeyOptions[K],
            [MatLike, int],
            Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None,
            [MatLike | None, int],
            Mapping[K, NDArrayFloat],
        ](estimate, arg1)

    if not callable(arg1):
        raise TypeError("First argument must be a callable or KeyOptions TypedDict.")

    def wrapper(
        self: E, frame_src: MatLike | None, frame_idx: int
    ) -> Mapping[K, NDArrayFloat]:
        if frame_src is None:
            return {
                klm: np.asarray(self.configure_missing_array(klm))
                for klm in self.shape.keys()
            }

        result = arg1(self, frame_src, frame_idx)

        mapping: Mapping[K, ArrayLikeFloat | None]
        if isinstance(result, Mapping):
            mapping = result
        else:
            key = _get_key_from_options_or_estimator(self, arg2)
            mapping = {key: result}

        return {
            k: (
                np.asarray(self.configure_missing_array(k))
                if v is None
                else np.asarray(v)
            )
            for k, v in mapping.items()
        }

    update_wrapper(wrapper, arg1)
    return wrapper


# ============================================================
# annotate decorator
# ============================================================

# Decorator Information

# External Function
# (E, MatLike, int, Mapping[K, NDArrayFloat])
#     -> MatLike
# Internal Function
# (E, MatLike, int, Mapping[K, NDArrayFloat])
#     -> MatLike | None

# override1: without key options
# override2: with key options


@overload
def annotate[E: EstimatorWithKey, K: str](
    func: Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike | None],
    options: KeyOptions[K] | None = None,
    /,
) -> Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike]:
    """Decorator overload definition for annotate without key options.

    Wraps the :meth:`annotate` method of :class:`Estimator` class.
    If the internal method returns None, the original frame is used.

    Types:
        InternalMethod:
            (:class:`E@annotate`, :class:`MatLike`, :class:`int`,
            :class:`Mapping[K, NDArrayFloat]`)
            -> :class:`MatLike` | :code:`None`

    Args:
        func (`InternalMethod`):
            Method that annotates landmarks on frames
        options (`KeyOptions[K] | None, optional`):
            TypedDict containing key options

    Returns:
        :code:`((self, frame_src: MatLike, frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]) -> MatLike)`:
            Decorated method

    """


@overload
def annotate[E: EstimatorWithKey, K: str](
    options: KeyOptions[K], /
) -> Callable[
    [Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike | None]],
    Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike],
]:
    """Decorator overload definition for annotate with key options.

    Wraps the :meth:`annotate` method of :class:`Estimator` class.
    If the internal method returns None, the original frame is used.

    Args:
        options (`KeyOptions[K]`):
            TypedDict containing key options
    Returns:
        :code:`((self, frame_src: MatLike, frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]) -> MatLike)`:
            Decorated method
    """


def annotate[E: EstimatorWithKey, K: str](
    arg1: (
        Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike | None]
        | KeyOptions[K]
    ),
    arg2: KeyOptions[K] | None = None,
    /,
) -> (
    Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike]
    | Callable[
        [Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike | None]],
        Callable[[E, MatLike, int, Mapping[K, NDArrayFloat]], MatLike],
    ]
):
    if _typeddict_typeis(arg1, KeyOptions[K]):
        return _DecoratorWithOptions[
            E,
            KeyOptions[K],
            [MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike | None,
            [MatLike, int, Mapping[K, NDArrayFloat]],
            MatLike,
        ](annotate, arg1)

    if not callable(arg1):
        raise TypeError("First argument must be a callable or KeyOptions TypedDict.")

    def wrapper(
        self: E, frame_src: MatLike, frame_idx: int, landmarks: Mapping[K, NDArrayFloat]
    ) -> MatLike:
        result = arg1(self, frame_src, frame_idx, landmarks)

        if result is None:
            return frame_src
        else:
            return result

    update_wrapper(wrapper, arg1)
    return wrapper


# ============================================================
# Estimator Base Class
# ============================================================

class Estimator[K: str](ABC):
    """Abstract base class for landmark estimation models.

    Defines the interface for all landmark estimators in the LMPipe pipeline.
    Subclasses must implement the core methods to define estimation behavior,
    output shape, and optional header information.

    Type Parameters:
        K: String type for landmark keys identifying different body parts or outputs.

    Attributes:
        missing_value (:obj:`float`): Default value for missing/invalid landmarks.
            Defaults to :obj:`numpy.nan`.

    Note:
        Subclasses must implement :attr:`shape` and :meth:`estimate` methods.
        The :attr:`headers` and :meth:`annotate` methods have default implementations
        but can be overridden for custom behavior.
    """

    missing_value: float = np.nan
    "Default missing value used in missing arrays."

    # ===========================================================
    # Core (Abstract) Methods
    # ===========================================================

    @property
    @abstractmethod
    @shape
    def shape(self) -> Mapping[K, tuple[int, int]] | tuple[int, int]:
        """Abstract property that returns shape of the estimation result array.

        The shape is defined as a mapping from key :obj:`K` to a tuple of ``(V, C)``.

        Where ``V`` is the number of landmark points,
        and ``C`` is the coordinate dimension (e.g., ``2`` for ``(x, y)`` coordinates).

        Returns:
            :class:`~typing.Mapping`\\[
                :obj:`K`, :obj:`tuple`\\[:obj:`int`, :obj:`int`\\]
            \\]:
                Shape of the estimation result array mapped by key :obj:`K`.

        Note:
            Override Guidelines:

            - If the estimator has a single output, return a tuple of ``(V, C)``.
            - If the estimator has multiple outputs, return a mapping from each
              key :obj:`K` to its corresponding ``(V, C)`` tuple.
        """

    @abstractmethod
    @estimate
    def estimate(
        self, frame_src: MatLike, frame_idx: int
    ) -> Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None:
        """Abstract method that estimates landmarks from frames.

        Args:
            frame_src (:class:`MatLike`): Source frame for estimation.
            frame_idx (:obj:`int`): Index of the current frame.

        Returns:
            :class:`~typing.Mapping`\\[
                :obj:`K`, :class:`ArrayLikeFloat` | :obj:`None`
            \\]:
                Estimated landmarks mapped by key :obj:`K`.

        Note:
            Override Guidelines:

            - If the estimator has no output or wants to use a dummy array,
              return :obj:`None`.
            - If the estimator has a single output, return an
              :class:`ArrayLikeFloat`.
            - If the estimator has multiple outputs, return a mapping from
              each
              key :obj:`K` to its corresponding :class:`ArrayLikeFloat`.
        """

    # ===========================================================
    # Core (Non-Abstract) Methods
    # ===========================================================

    @property
    @headers
    @cache
    # overrideable method
    def headers(self) -> Mapping[K, ArrayLikeStr] | ArrayLikeStr | None:
        """Returns array of header names corresponding to each element of
        estimation result.

        Returns:
            :class:`Mapping[K, ArrayLikeStr]`:
                Array of header names matching the shape of estimation results

        Override Guidelines:
            - If the estimator has no single output or want to use default headers,
              return `None`.
            - If the estimator has a single output, return an :class:`ArrayLikeStr`.
            - If the estimator has multiple outputs, return a mapping from each key K
              to its corresponding :class:`ArrayLikeStr`.
        """
        return None

    @annotate
    # overrideable method
    def annotate(
        self, frame_src: MatLike, frame_idx: int, landmarks: Mapping[K, NDArrayFloat]
    ) -> MatLike | None:
        """Returns annotated frame with landmarks drawn on it.

        Args:
            frame_src (`MatLike`):
                Source frame for annotation
            frame_idx (`int`):
                Index of the current frame
            landmarks (`Mapping[K, NDArrayFloat]`):
                Landmarks to draw on the frame

        Returns:
            :class:`MatLike` | :code:`None`:
                Annotated frame, or None to use the original frame

        Override Guidelines:
            - If nothing to do, return `None` to use the original frame.
            - Otherwise, return an annotated :class:`MatLike`.
        """

        return None

    # ===========================================================
    # Configuration Methods
    # ===========================================================

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    def setup(self):
        """Setup method called before processing begins.

        Override Guidelines:
            - Override this method to perform any necessary setup or
              initialization before processing starts.
        """
        pass

    @abstractmethod
    def configure_estimator_name(self) -> K:
        # This method must be abstract
        # because the key K is not determined at Estimator definition time.
        """Configures and returns the estimator name used as key K.

        CRITICAL: This method ALWAYS returns a SINGLE key value, never a collection.

        For single-key estimators (where @estimate returns a single value),
        this name identifies that one output.

        For multi-key estimators (where @estimate returns Mapping[K, ...]),
        this name serves as the estimator's identifier or primary name, but
        does NOT represent all output keys. The actual output keys come from
        the Mapping returned by @estimate.

        Common Mistake:
            # WRONG: Trying to return multiple keys
            def configure_estimator_name(self) -> MyEnum:
                return (MyEnum.POSE, MyEnum.HAND)  # ❌ Type error!

            # CORRECT: Return a single representative key
            def configure_estimator_name(self) -> MyEnum:
                return MyEnum.HOLISTIC  # ✅ Single identifier

        Example - Single Key Estimator:
            class PoseEstimator(Estimator[PoseKey]):
                def configure_estimator_name(self) -> PoseKey:
                    return PoseKey.POSE  # Only one output

                @estimate
                def estimate_landmark(self, frame):
                    # Returns single array, auto-wrapped to {PoseKey.POSE: array}
                    return pose_array

        Example - Multi-Key Estimator:
            class HolisticEstimator(Estimator[HolisticKey]):
                def configure_estimator_name(self) -> HolisticKey:
                    # Return primary identifier, not all keys
                    return HolisticKey.HOLISTIC

                @estimate
                def estimate_landmark(self, frame):
                    # Returns multiple keys explicitly
                    return {
                        HolisticKey.POSE: pose_array,
                        HolisticKey.LEFT_HAND: left_hand_array,
                        HolisticKey.RIGHT_HAND: right_hand_array,
                    }

        Returns:
            key (`K@Estimator`):
                A single key value identifying this estimator. For multi-key
                estimators, this is typically the estimator's primary name
                or category, not the collection of all output keys.

        Override Guidelines:
            - MUST be overridden in subclasses to provide the appropriate key K.
            - MUST return exactly one key value (type K), never a tuple or list.
            - For multi-key estimators, choose a representative primary name.
            - The return type matches the K type parameter of Estimator[K].
        """
        ...

    # overrideable method
    def configure_missing_array(self, key: K) -> ArrayLikeFloat:
        """Configures and returns a missing value array for the given key K.

        Args:
            key (`K@Estimator`):
                The key for which to configure the missing array.

        Returns:
            :class:`ArrayLikeFloat`:
                The configured missing value array.
        """
        return np.full(self.shape[key], self.missing_value, dtype=float)
