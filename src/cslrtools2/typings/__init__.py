from typing import TYPE_CHECKING, Protocol
from os import PathLike as _PathLike

type PathLike = _PathLike[str] | str
"Type alias for path-like objects (str or PathLike)."

if TYPE_CHECKING:
    # Lazy import for type checking
    import numpy as np
    from numpy._typing import (
        _ArrayLikeInt_co, _ArrayLikeFloat_co, _ArrayLikeStr_co # pyright: ignore[reportPrivateUsage]
    )
    from numpy.typing import (
        NDArray as _NDArray,
        ArrayLike as _ArrayLike,
        DTypeLike as _DTypeLike 
    )
    from cv2.typing import MatLike as _MatLike
else:
    _ArrayLikeFloat_co = object
    _ArrayLikeStr_co = object
    _NDArray = object
    _MatLike = object

class SupportsArray[R: _ArrayLike](Protocol):
    def __array__(
        self,
        dtype: _DTypeLike | None = None,
        copy: bool | None = None
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


