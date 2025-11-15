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

from pathlib import Path
from typing import Mapping

import numpy as np

from ....typings import NDArrayFloat
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class NpyLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist landmarks into per-key ``*.npy`` files."""

    def __init__(self) -> None:
        """Initialize the NumPy .npy landmark matrix save collector."""
        self._base_dir: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

    @property
    def is_perkey(self) -> bool:
        """Per-key mode: saves individual .npy files."""
        return True

    @property
    def is_container(self) -> bool:
        """Not a container collector."""
        return False

    @property
    def file_ext(self) -> str:
        """File extension for NumPy binary files."""
        return ".npy"

    def _open_file(self, path: Path):
        self._base_dir = self._prepare_landmark_dir(path)
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, value in result.items():
            key = str(raw_key)
            bucket = self._buffer.setdefault(key, [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._base_dir is None:
            return
        for key, arrays in self._buffer.items():
            file_path = self._base_dir / f"{key}.npy"
            if arrays:
                try:
                    np.save(file_path, np.stack(arrays))
                except ValueError as e:
                    shapes = dict[tuple[int, int], int]()
                    for a in arrays:
                        s = shapes.get(a.shape, 0)
                        shapes[a.shape] = s + 1
                    raise ValueError(
                        f"with shape {shapes} at key '{key}'"
                    ) from e
            else:
                np.save(file_path, np.empty((0,), dtype=float))
        self._base_dir = None
        self._buffer = {}


def npy_lmsc_creator[K: str](key_type: type[K]) -> NpyLandmarkMatrixSaveCollector[K]:
    """Create a NumPy .npy landmark matrix save collector.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`NpyLandmarkMatrixSaveCollector[K]`: Per-key NumPy .npy landmark matrix saver.
    """
    return NpyLandmarkMatrixSaveCollector[K]()

lmsc_aliases.update({
    ".npy": npy_lmsc_creator,
})
