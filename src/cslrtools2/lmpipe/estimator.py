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


############################# Estimator Decorators #############################

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
    key_from_estimator: K = estimater.configure_estimator_name()
    return key_from_options or key_from_estimator


### shape decorator ###

# Decorator Information
# External Function
# (E) -> Mapping[K, NDArrayStr]
# Internal Function
# (E) -> Mapping[K, ArrayLikeStr] | ArrayLikeStr | None

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


### headers decorator ###

# Decorator Information

# External Function
# (E) -> Mapping[K, NDArrayStr]
# Internal Function
# (E) -> Mapping[K, ArrayLikeStr | None] | ArrayLikeStr | None

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


### estimate decorator ###

# Decorator Information

# External Function
# (E, MatLike | None, int) -> Mapping[K, NDArrayFloat]
# Internal Function
# (E, MatLike, int) -> Mapping[K, ArrayLikeFloat | None] | ArrayLikeFloat | None

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


### annotate decorator ###

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


########################## Estimator Class Definition ##########################


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

    ### Core (Abstract) Methods ###

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

    ### Core (Non-Abstract) Methods ###

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

    ### Configuration Methods ###

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

        Returns:
            key (`K@Estimator`):
                The key used to identify the estimator.

        Override Guidelines:
            - This method must be overridden in subclasses to provide the
              appropriate key K.
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
