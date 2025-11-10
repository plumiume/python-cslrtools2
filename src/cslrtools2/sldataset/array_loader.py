# CsvLoader(PreKeyLoader)
# JsonLoader(PreKeyLoader)
# NpyLoader(PreKeyLoader)
# NpzLoader(ContainerLoader)
# TorchLoader(PreKeyLoader, ContainerLoader)
# SafetensorsLoader(ContainerLoader)
# ZarrLoader(ContainerLoader)

from abc import ABC, abstractmethod
from typing import Any, Mapping, Callable, TypeGuard, get_origin

import csv
import json
import numpy as np
import torch
import safetensors.torch
import zarr
from torch import Tensor

from .typings import PathLike

def _is_mapping[T: Mapping[Any, Any]](obj: object, type_: type[T]) -> TypeGuard[T]:
    origin = type_
    while (new_origin := get_origin(origin)) is not None:
        origin = new_origin
    return isinstance(obj, origin)

class PreKeyLoader(ABC):

    @abstractmethod
    def load_array(self, path: PathLike) -> Tensor: ...

class ContainerLoader(ABC):

    @abstractmethod
    def load_mapping(self, path: PathLike) -> Mapping[str, Tensor]: ...


class CsvLoader(PreKeyLoader):

    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter

    def load_array(self, path: PathLike) -> Tensor:
        return torch.tensor([
            [float(value) for value in row]
            for row in csv.reader(
                open(path, "r", newline=""),
                delimiter=self.delimiter
            )
        ])

class JsonLoader(PreKeyLoader):

    def load_array(self, path: PathLike) -> Tensor:
        return torch.tensor(
            json.load(
                open(path, "r")
            )
        )

class NpyLoader(PreKeyLoader):

    def load_array(self, path: PathLike) -> Tensor:
        return torch.tensor(
            np.load(path)
        )

class NpzLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, Tensor]:
        npz: np.lib.npyio.NpzFile[Any] = np.load(path)
        return {
            key: torch.tensor(npz[key])
            for key in npz.files
        }

class TorchLoader(PreKeyLoader, ContainerLoader):

    def load_array(self, path: PathLike) -> Tensor:
        ret = torch.load(path)
        if not isinstance(ret, Tensor):
            raise ValueError(f"Expected a Tensor in Torch file at {path}, got {type(ret)}")
        return ret

    def load_mapping(self, path: PathLike) -> Mapping[str, Tensor]:
        data = torch.load(path)
        if not _is_mapping(data, Mapping[str, Tensor]):
            raise ValueError(f"Expected a dict of str to Tensor in Torch file at {path}")
        return data

class SafetensorsLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, Tensor]:
        data = safetensors.torch.load_file(path)
        if not _is_mapping(data, Mapping[str, Tensor]):
            raise ValueError(f"Expected a dict of str to Tensor in Safetensors file at {path}")
        return data

class ZarrLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, Tensor]:
        group = zarr.open_group(path)
        return {
            key: torch.tensor(array[:])
            for key, array in group.arrays()
        }

prekey_loaders: Mapping[str, Callable[[PathLike], Tensor]] = {
    ".csv": CsvLoader().load_array,
    ".tsv": CsvLoader(delimiter="\t").load_array,
    ".ssv": CsvLoader(delimiter=" ").load_array,
    ".json": JsonLoader().load_array,
    ".npy": NpyLoader().load_array,
    ".pt": TorchLoader().load_array,
    ".pth": TorchLoader().load_array,
}

container_loaders: Mapping[str, Callable[[PathLike], Mapping[str, Tensor]]] = {
    ".npz": NpzLoader().load_mapping,
    ".pt": TorchLoader().load_mapping,
    ".pth": TorchLoader().load_mapping,
    ".safetensors": SafetensorsLoader().load_mapping,
    ".zarr": ZarrLoader().load_mapping,
}
