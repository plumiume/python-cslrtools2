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

from typing import TYPE_CHECKING, Protocol
from os import PathLike as _PathLike

type PathLike = _PathLike[str] | str
"Type alias for path-like objects (str or PathLike)."

if TYPE_CHECKING:
    # Lazy import for type checking
    import numpy as np
    from numpy._typing import (
        _ArrayLikeInt_co,  # pyright: ignore[reportPrivateUsage]
        _ArrayLikeFloat_co,  # pyright: ignore[reportPrivateUsage]
        _ArrayLikeStr_co,  # pyright: ignore[reportPrivateUsage]
    )
    from numpy.typing import (
        NDArray as _NDArray,
        ArrayLike as _ArrayLike,
        DTypeLike as _DTypeLike,
    )
    from cv2.typing import MatLike as _MatLike
else:
    _ArrayLike = object
    _ArrayLikeFloat_co = object
    _ArrayLikeStr_co = object
    _NDArray = object
    _DTypeLike = object
    _MatLike = object


class SupportsArray[R: _ArrayLike](Protocol):
    """Protocol for objects that support conversion to arrays.

    This protocol defines objects that implement the :meth:`__array__` method,
    allowing them to be converted to NumPy arrays or similar array types.

    Type Parameters:
        R: The return type of :meth:`__array__`, must be array-like.

    Example:
        Custom class implementing the array protocol::

            >>> import numpy as np
            >>> class MyArray:
            ...     def __init__(self, data):
            ...         self.data = data
            ...     def __array__(self, dtype=None, copy=None):
            ...         return np.array(self.data, dtype=dtype)
            >>> obj = MyArray([1, 2, 3])
            >>> np.asarray(obj)
            array([1, 2, 3])
    """

    def __array__(
        self, dtype: _DTypeLike | None = None, copy: bool | None = None
    ) -> R: ...


# re-declare types for sphinx autodoc

type ArrayLike = _ArrayLike | SupportsArray[_ArrayLike]
"Type alias for array-like objects."

type ArrayLikeInt = _ArrayLikeInt_co | SupportsArray[_ArrayLikeInt_co]
"Type alias for array-like objects of integers."
type ArrayLikeFloat = _ArrayLikeFloat_co | SupportsArray[_ArrayLikeFloat_co]
"Type alias for array-like objects of floats."
type ArrayLikeStr = _ArrayLikeStr_co | SupportsArray[_ArrayLikeStr_co]
"Type alias for array-like objects of strings."

type MatLike = _MatLike
"Type alias for OpenCV matrix-like objects."

if TYPE_CHECKING:
    type NDArrayInt = _NDArray[np.integer]
    "Type alias for NumPy arrays of integers."
    type NDArrayFloat = _NDArray[np.floating]
    "Type alias for NumPy arrays of floats."
    type NDArrayStr = _NDArray[np.str_]
    "Type alias for NumPy arrays of strings."
else:
    NDArrayInt = object
    NDArrayFloat = object
    NDArrayStr = object
