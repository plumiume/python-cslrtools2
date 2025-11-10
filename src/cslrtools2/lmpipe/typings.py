from typing import TYPE_CHECKING, Literal
from os import PathLike as _PathLike

type PathLike = _PathLike[str] | str
"Type alias for path-like objects (str or PathLike)."

if TYPE_CHECKING:
    # Lazy import for type checking
    import numpy as np
    from numpy._typing import (
        _ArrayLikeFloat_co, _ArrayLikeStr_co # pyright: ignore[reportPrivateUsage]
    )
    from numpy.typing import NDArray as _NDArray
    from cv2.typing import MatLike as _MatLike
else:
    _ArrayLikeFloat_co = object
    _ArrayLikeStr_co = object
    _NDArray = object
    _MatLike = object


# re-declare types for sphinx autodoc

type ArrayLikeFloat = _ArrayLikeFloat_co
"Type alias for array-like objects of floats."
type ArrayLikeStr = _ArrayLikeStr_co
"Type alias for array-like objects of strings."
type MatLike = _MatLike
"Type alias for OpenCV matrix-like objects."

if TYPE_CHECKING:
    type NDArrayFloat = _NDArray[np.floating]
    "Type alias for NumPy arrays of floats."
    type NDArrayStr = _NDArray[np.str_]
    "Type alias for NumPy arrays of strings."
else:
    NDArrayFloat = object
    NDArrayStr = object

type ExecutorMode = Literal["batch", "frames"] | None
"Type alias for executor modes."
type ExecutorType = Literal["thread", "process"] | None
"Type alias for executor types."

type ExistRule = Literal["skip", "overwrite", "suffix", "error"]
"Type alias for file existence handling rules."
