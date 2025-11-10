from os import PathLike
from typing import AnyStr
import torch
from torch import Tensor



def storage_ptr(tensor: Tensor) -> int: ...


def storage_size(tensor: Tensor) -> int: ...


def save_model(
    model: torch.nn.Module,
    filename: str,
    metadata: dict[str, str] | None = None,
    fource_contiguous: bool = True,
) -> None: ...


def load_model(
    model: torch.nn.Module,
    filename: str | PathLike[AnyStr],
    strict: bool = True,
    device: str | int = "cpu",
) -> tuple[list[str], list[str]]: ...


def save(
    tensors: dict[str, Tensor],
    metadata: dict[str, str] | None,
) -> bytes: ...


def save_file(
    tensors: dict[str, Tensor],
    filename: str | PathLike[AnyStr],
    metadata: dict[str, str] | None = None,
) -> None: ...


def load_file(
    filename: str | PathLike[AnyStr],
    device: str | int = "cpu",
) -> dict[str, Tensor]: ...


def load(data: bytes) -> dict[str, Tensor]: ...
