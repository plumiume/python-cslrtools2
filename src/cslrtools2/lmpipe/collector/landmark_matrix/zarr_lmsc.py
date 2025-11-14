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
import zarr

from ....typings import NDArrayFloat
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class ZarrLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into Zarr array storage.
    
    Supports two modes:
    - Container mode (per_key=False): Single ``landmarks.zarr/`` directory with all keys as datasets
    - Per-key mode (per_key=True): Multiple ``{key}.zarr/`` directories in ``landmarks/`` directory
    """

    def __init__(self, *, per_key: bool = False) -> None:
        """Initialize the Zarr landmark matrix save collector.
        
        Args:
            per_key (:class:`bool`, optional): If True, save each key to a separate .zarr directory.
                If False, save all keys to a single landmarks.zarr directory. Defaults to False.
        """
        self.per_key = per_key
        self.USE_LANDMARK_DIR = per_key  # Use directory for per-key mode
        self._base_dir: Path | None = None
        self._path: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

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
        """File extension for Zarr array storage."""
        return ".zarr"

    def _open_file(self, path: Path):
        if self.per_key:
            self._base_dir = self._prepare_landmark_dir(path)
            self._path = None
        else:
            self._path = self._get_landmark_file_path(path, "landmarks.zarr")
            self._base_dir = None
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            bucket = self._buffer.setdefault(str(key), [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self.per_key:
            # Per-key mode: save each key to a separate zarr directory
            if self._base_dir is None:
                return
            for key, arrays in self._buffer.items():
                zarr_path = self._base_dir / f"{key}.zarr"
                store = zarr.open_group(zarr_path, mode="w")
                if arrays:
                    store["data"] = np.stack(arrays)
                else:
                    store["data"] = np.empty((0,), dtype=float)
            self._base_dir = None
        else:
            # Container mode: save all keys to a single zarr directory
            if self._path is None:
                return
            store = zarr.open_group(self._path, mode="w")
            if self._buffer:
                for key, values in self._buffer.items():
                    store[key] = np.stack(values)
            self._path = None
        self._buffer = {}


def zarr_lmsc_creator[K: str](key_type: type[K]) -> ZarrLandmarkMatrixSaveCollector[K]:
    """Create a Zarr landmark matrix save collector.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`ZarrLandmarkMatrixSaveCollector[K]`: Zarr landmark matrix saver (container mode by default).
    """
    return ZarrLandmarkMatrixSaveCollector[K]()

lmsc_aliases.update({
    ".zarr": zarr_lmsc_creator,
})
