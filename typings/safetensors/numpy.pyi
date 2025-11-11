from typing import Any
import os
import numpy as np

from safetensors import safe_open


__all__ = [
    "desirialize",
    "safe_open",
    "serialize",
    "serialize_file",
    "save",
    "save_file",
    "load",
    "load_file",
]


### Rust implemented low-level functions ###

def desirialize(data: bytes) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    """Opens a safetensors lazily and returns tensors as asked
    
    Args:
        data (`bytes`): The byte content of a file

    Returns:
        :code:`list[tuple[str, dict[str, dict[str, Any]]]]`:
            The deserialized content is like:
                :code:`[("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]`
    """


def serialize(
    tensor_dict: dict[str, tuple[str, Any]],
    metadata: dict[str, str] | None = None,
    ) -> bytes:
    """Serializes raw data.
    
    Args:
        tensor_dict (`dict[str, tuple[str, Any]]`):
            The tensor dict is like:
                :code:`{"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}`
        metadata (`dict[str, str]`, *optional*):
            The optional purely text annotations.

    Returns:
        :code:`bytes`: The serialized content.
    """


def serialize_file(
    tensor_dict: dict[str, tuple[str, Any]],
    filename: str | os.PathLike[Any],
    metadata: dict[str, str] | None = None,
    ) -> None:
    """Serializes raw data into file.
    
    Args:
        tensor_dict (`dict[str, tuple[str, Any]]`):
            The tensor dict is like:
                :code:`{"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}`
        filename (`str | os.PathLike`):
            The name of the file to write into.
        metadata (`dict[str, str]`, *optional*):
            The optional purely text annotations
    """


### High-level convenience functions ###

def save(
    tensor_dict: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> bytes: ...



def save_file(
    tensor_dict: dict[str, np.ndarray],
    filename: str | os.PathLike[Any],
    metadata: dict[str, str] | None = None,
) -> None: ...


def load(data: bytes) -> dict[str, np.ndarray]: ...


def load_file(filename: str | os.PathLike[Any]) -> dict[str, np.ndarray]: ...
