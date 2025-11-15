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
from ..exceptions import DataFormatError

def _is_mapping[T: Mapping[Any, Any]](obj: object, type_: type[T]) -> TypeGuard[T]:
    """Check if object is an instance of a mapping type.

    Args:
        obj (:obj:`object`): Object to check.
        type_ (:obj:`type`\\[:obj:`T`\\]): Target mapping type.

    Returns:
        :obj:`bool`: :obj:`True` if obj is an instance of the mapping type.
    """
    origin = type_
    while (new_origin := get_origin(origin)) is not None:
        origin = new_origin
    return isinstance(obj, origin)

class PreKeyLoader(ABC):
    """Abstract base class for loaders that load single arrays from files.

    Pre-key loaders handle file formats where each file contains a single array
    associated with a key (derived from the filename).

    Examples:
        Formats: CSV, JSON (single array), NPY
    """

    @abstractmethod
    def load_array(self, path: PathLike) -> ArrayLike:
        """Load a single array from the specified file path.

        Args:
            path (:class:`PathLike`): Path to the file to load.

        Returns:
            :class:`ArrayLike`: The loaded array.
        """
        ...

class ContainerLoader(ABC):
    """Abstract base class for loaders that load multiple keyed arrays from files.

    Container loaders handle file formats where a single file contains multiple
    arrays, each associated with a string key.

    Examples:
        Formats: NPZ, PyTorch (pt/pth), SafeTensors, Zarr
    """

    @abstractmethod
    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        """Load a mapping of keyed arrays from the specified file path.

        Args:
            path (:class:`PathLike`): Path to the container file to load.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`str`, :class:`ArrayLike`\\]:
                Mapping from keys to arrays.
        """
        ...


class CsvLoader(PreKeyLoader):
    """CSV file loader for array data.

    Loads numeric data from CSV files, converting each row to a list of floats.

    Attributes:
        delimiter (:obj:`str`): Column delimiter character. Defaults to ``','``.
    """

    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter

    def load_array(self, path: PathLike) -> ArrayLike:
        """Load CSV file as a 2D array.

        Args:
            path (:class:`PathLike`): Path to the CSV file.

        Returns:
            :class:`ArrayLike`: 2D list of float values.
        """
        return [
            [float(value) for value in row]
            for row in csv.reader(
                open(path, "r", newline=""),
                delimiter=self.delimiter
            )
        ]

class JsonLoader(PreKeyLoader):
    """JSON file loader for array data.

    Loads array data from JSON files. The JSON file should contain a single
    array or nested array structure.
    """

    def load_array(self, path: PathLike) -> ArrayLike:
        """Load JSON file as an array.

        Args:
            path (:class:`PathLike`): Path to the JSON file.

        Returns:
            :class:`ArrayLike`: The parsed JSON array.
        """
        return json.load(
            open(path, "r")
        )

class NpyLoader(PreKeyLoader):
    """NumPy NPY file loader for array data.

    Loads NumPy arrays from NPY files. Only supports single arrays,
    not NPZ archives.
    """

    def load_array(self, path: PathLike) -> ArrayLike:
        """Load NPY file as a NumPy array.

        Args:
            path (:class:`PathLike`): Path to the NPY file.

        Returns:
            :class:`ArrayLike`: The loaded NumPy array.

        Raises:
            :exc:`DataFormatError`: If the file contains a mapping instead of a single array.
        """
        result = np.load(path)
        if isinstance(result, Mapping):
            raise DataFormatError(
                f"Expected a single array in NPY file at {path}, got a mapping. "
                f"Use load_npy_multi() for multi-array files."
            )
        return result

class NpzLoader(ContainerLoader):
    """NumPy NPZ archive loader for multiple keyed arrays.

    Loads multiple NumPy arrays from NPZ archive files, where each array
    is associated with a string key.
    """

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        """Load NPZ archive as a mapping of arrays.

        Args:
            path (:class:`PathLike`): Path to the NPZ file.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`str`, :class:`ArrayLike`\\]:
                Mapping from array names to NumPy arrays.

        Raises:
            :exc:`DataFormatError`: If the file is not a valid NPZ archive.
        """
        npz: np.lib.npyio.NpzFile[Any] | Any = np.load(path)
        if not _is_mapping(npz, np.lib.npyio.NpzFile[Any]):
            raise DataFormatError(
                f"Expected a NPZ file at {path}, got {type(npz)}. "
                f"Ensure the file is a valid NumPy .npz archive."
            )
        return npz

class TorchLoader(PreKeyLoader, ContainerLoader):
    """PyTorch file loader for tensors and tensor dictionaries.

    Supports loading both single tensors (PreKeyLoader) and dictionaries
    of tensors (ContainerLoader) from PyTorch .pt/.pth files.
    """

    def load_array(self, path: PathLike) -> ArrayLike:
        """Load PyTorch file as a single tensor.

        Args:
            path (:class:`PathLike`): Path to the PyTorch file.

        Returns:
            :class:`ArrayLike`: The loaded :class:`torch.Tensor`.

        Raises:
            :exc:`DataFormatError`: If the file does not contain a single tensor.
        """
        result = torch.load(path)
        if not torch.is_tensor(result):
            raise DataFormatError(
                f"Expected a Tensor in Torch file at {path}, got {type(result)}. "
                f"Use load_array_dict() for dictionary-based Torch files."
            )
        return result

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        """Load PyTorch file as a mapping of tensors.

        Args:
            path (:class:`PathLike`): Path to the PyTorch file.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`str`, :class:`ArrayLike`\\]:
                Mapping from keys to tensors.

        Raises:
            :exc:`DataFormatError`: If the file does not contain a valid mapping.
        """
        data = torch.load(path)
        if not _is_mapping(data, Mapping[str, ArrayLike]):
            raise DataFormatError(
                f"Expected a dict of str to ArrayLike in Torch file at {path}, got {type(data)}. "
                f"Use load_array() for single tensor files."
            )
        return data

class SafetensorsLoader(ContainerLoader):
    """SafeTensors file loader for secure tensor storage.

    Loads tensors from SafeTensors format files, which provide safe
    and efficient tensor serialization.
    """

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        """Load SafeTensors file as a mapping of tensors.

        Args:
            path (:class:`PathLike`): Path to the SafeTensors file.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`str`, :class:`ArrayLike`\\]:
                Mapping from keys to tensors.

        Raises:
            :exc:`DataFormatError`: If the file does not contain a valid mapping.
        """
        data = safetensors.torch.load_file(path)
        if not _is_mapping(data, Mapping[str, ArrayLike]):
            raise DataFormatError(
                f"Expected a dict of str to ArrayLike in Safetensors file at {path}, got {type(data)}. "
                f"Ensure the file contains a valid safetensors mapping."
            )

        return data

class ZarrLoader(ContainerLoader):
    """Zarr group loader for chunked array storage.

    Loads arrays from Zarr groups, which support efficient chunked
    storage and compression for large datasets.
    """

    def load_mapping(self, path: PathLike) -> Mapping[str, ArrayLike]:
        """Load Zarr group as a mapping of arrays.

        Args:
            path (:class:`PathLike`): Path to the Zarr group directory.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`str`, :class:`ArrayLike`\\]:
                Mapping from array names to :class:`zarr.Array` objects.
        """
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
