import numpy as np
from os import PathLike as _PathLike

type PathLike = _PathLike[str]

def save(
    tensor_dict: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
    ) -> bytes: ...

def save_file(
    tensor_dict: dict[str, np.ndarray],
    filename: str | PathLike,
    metadata: dict[str, str] | None = None,
    ) -> None: ...

def load(data: bytes) -> dict[str, np.ndarray]: ...

def load_file(filename: str | PathLike) -> dict[str, np.ndarray]: ...