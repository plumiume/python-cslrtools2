# CsvLoader(PreKeyLoader)
# JsonLoader(PreKeyLoader)
# NpyLoader(PreKeyLoader)
# NpzLoader(ContainerLoader)
# TorchLoader(PreKeyLoader, ContainerLoader)
# SafetensorsLoader(ContainerLoader)
# ZarrLoader(ContainerLoader)

from abc import ABC, abstractmethod
from typing import Any, Mapping, Callable, TypeGuard, get_origin
from pathlib import Path

import csv
import json
import numpy as np
import torch
import safetensors.torch
import zarr

from ..typings import PathLike, ArrayLike

def _is_mapping[T: Mapping[Any, Any]](obj: object, type_: type[T]) -> TypeGuard[T]:
    origin = type_
    while (new_origin := get_origin(origin)) is not None:
        origin = new_origin
    return isinstance(obj, origin)

class PreKeyLoader(ABC):

    @abstractmethod
    def load_array(self, path: PathLike) -> ArrayLike: ...

class ContainerLoader(ABC):

    @abstractmethod
    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]: ...


class CsvLoader(PreKeyLoader):

    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter

    def load_array(self, path: PathLike) -> ArrayLike:
        return [
            [float(value) for value in row]
            for row in csv.reader(
                open(path, "r", newline=""),
                delimiter=self.delimiter
            )
        ]

class JsonLoader(PreKeyLoader):

    def load_array(self, path: PathLike) -> ArrayLike:
        return json.load(
            open(path, "r")
        )

class NpyLoader(PreKeyLoader):

    def load_array(self, path: PathLike) -> ArrayLike:
        result = np.load(path)
        if isinstance(result, Mapping):
            raise ValueError(f"Expected a single array in NPY file at {path}, got a mapping")
        return result

class NpzLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        npz: np.lib.npyio.NpzFile[Any] | Any = np.load(path)
        if not _is_mapping(npz, np.lib.npyio.NpzFile[Any]):
            raise ValueError(f"Expected a NPZ file at {path}, got {type(npz)}")
        return npz

class TorchLoader(PreKeyLoader, ContainerLoader):

    def load_array(self, path: PathLike) -> ArrayLike:
        result = torch.load(path)
        if not torch.is_tensor(result):
            raise ValueError(f"Expected a Tensor in Torch file at {path}, got {type(result)}")
        return result

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        data = torch.load(path)
        if not _is_mapping(data, Mapping[str, ArrayLike]):
            raise ValueError(f"Expected a dict of str to ArrayLike in Torch file at {path}")
        return data

class SafetensorsLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        data = safetensors.torch.load_file(path)
        if not _is_mapping(data, Mapping[str, ArrayLike]):
            raise ValueError(f"Expected a dict of str to ArrayLike in Safetensors file at {path}")
        return data

class ZarrLoader(ContainerLoader):

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        group = zarr.open_group(Path(path))
        return {
            key: array
            for key, array in group.arrays()
        }

type PrekeyLoadFunc = Callable[[PathLike], ArrayLike]
type ContainerLoadFunc = Callable[[PathLike], Mapping[str, ArrayLike]]

prekey_loaders: Mapping[str, PrekeyLoadFunc] = {
    ".csv": CsvLoader().load_array,
    ".tsv": CsvLoader(delimiter="\t").load_array,
    ".ssv": CsvLoader(delimiter=" ").load_array,
    ".json": JsonLoader().load_array,
    ".npy": NpyLoader().load_array,
    ".pt": TorchLoader().load_array,
    ".pth": TorchLoader().load_array,
}

container_loaders: Mapping[str, ContainerLoadFunc] = {
    ".npz": NpzLoader().load_mapping,
    ".pt": TorchLoader().load_mapping,
    ".pth": TorchLoader().load_mapping,
    ".safetensors": SafetensorsLoader().load_mapping,
    ".zarr": ZarrLoader().load_mapping,
}
