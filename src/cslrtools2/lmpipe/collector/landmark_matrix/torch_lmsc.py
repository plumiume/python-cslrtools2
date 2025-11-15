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

from pathlib import Path
from typing import Mapping

import torch

from ....typings import NDArrayFloat
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class TorchLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into PyTorch tensor files.

    Supports two modes:
    - Container mode (per_key=False): Single ``landmarks.pt`` (or ``.pth``) file with all keys
    - Per-key mode (per_key=True): Multiple ``{key}.pt`` (or ``.pth``) files in ``landmarks/`` directory
    """

    def __init__(self, *, per_key: bool = False, extension: str = ".pt") -> None:
        """Initialize the PyTorch landmark matrix save collector.

        Args:
            per_key (:class:`bool`, optional): If True, save each key to a separate file.
                If False, save all keys to a single landmarks file. Defaults to False.
            extension (:class:`str`, optional): File extension. Either ".pt" or ".pth". Defaults to ".pt".
        """
        if extension not in (".pt", ".pth"):
            raise ValueError(f"Invalid extension: {extension}. Must be '.pt' or '.pth'.")
        self.per_key = per_key
        self.extension = extension
        self.USE_LANDMARK_DIR = per_key  # Use directory for per-key mode
        self._base_dir: Path | None = None
        self._path: Path | None = None
        self._buffer: dict[str, list[torch.Tensor]] = {}

    @property
    def is_perkey(self) -> bool:
        """Per-key mode when per_key=True."""
        return self.per_key

    @property
    def is_container(self) -> bool:
        """Container mode when per_key=False."""
        return not self.per_key

    @property
    def file_ext(self) -> str:
        """File extension for PyTorch tensor files (.pt or .pth)."""
        return self.extension

    def _open_file(self, path: Path):
        if self.per_key:
            self._base_dir = self._prepare_landmark_dir(path)
            self._path = None
        else:
            self._path = self._get_landmark_file_path(path, f"landmarks{self.extension}")
            self._base_dir = None
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            bucket = self._buffer.setdefault(str(key), [])
            bucket.append(torch.tensor(value))

    def _close_file(self):
        if self.per_key:
            # Per-key mode: save each key to a separate file
            if self._base_dir is None:
                return
            for key, tensors in self._buffer.items():
                file_path = self._base_dir / f"{key}{self.extension}"
                if tensors:
                    torch.save(torch.stack(tensors), file_path)
                else:
                    torch.save(torch.empty(0), file_path)
            self._base_dir = None
        else:
            # Container mode: save all keys to a single file
            if self._path is None:
                return
            if not self._buffer:
                torch.save({}, self._path)
            else:
                tensors = {key: torch.stack(values) for key, values in self._buffer.items()}
                torch.save(tensors, self._path)
            self._path = None
        self._buffer = {}


def pt_lmsc_creator[K: str](key_type: type[K]) -> TorchLandmarkMatrixSaveCollector[K]:
    """Create a PyTorch .pt landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`TorchLandmarkMatrixSaveCollector[K]`: PyTorch landmark matrix saver with .pt extension.
    """
    return TorchLandmarkMatrixSaveCollector[K]()

def pth_lmsc_creator[K: str](key_type: type[K]) -> TorchLandmarkMatrixSaveCollector[K]:
    """Create a PyTorch .pth landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`TorchLandmarkMatrixSaveCollector[K]`: PyTorch landmark matrix saver with .pth extension.
    """
    return TorchLandmarkMatrixSaveCollector[K](extension=".pth")

lmsc_aliases.update({
    ".pt": pt_lmsc_creator,
    ".pth": pth_lmsc_creator,
})
