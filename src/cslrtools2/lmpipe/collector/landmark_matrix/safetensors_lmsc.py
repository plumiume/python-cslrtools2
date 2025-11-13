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
from safetensors.numpy import save_file

from ....typings import NDArrayFloat
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class SafetensorsLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into a ``landmarks.safetensors`` file."""

    USE_LANDMARK_DIR = False  # Container file at dst/landmarks.safetensors

    def __init__(self) -> None:
        """Initialize the Safetensors landmark matrix save collector."""
        self._path: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

    @property
    def is_perkey(self) -> bool:
        """Not per-key mode: saves a container file."""
        return False

    @property
    def is_container(self) -> bool:
        """Container mode: saves all keys in landmarks.safetensors."""
        return True

    @property
    def file_ext(self) -> str:
        """File extension for Safetensors files."""
        return ".safetensors"

    def _open_file(self, path: Path):
        self._path = self._get_landmark_file_path(path, "landmarks.safetensors")
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            bucket = self._buffer.setdefault(str(key), [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._path is None:
            return
        if not self._buffer:
            save_file({}, self._path)
        else:
            arrays = {key: np.stack(values) for key, values in self._buffer.items()}
            save_file(arrays, self._path)
        self._path = None
        self._buffer = {}


def safetensors_lmsc_creator[K: str](key_type: type[K]) -> SafetensorsLandmarkMatrixSaveCollector[K]:
    """Create a Safetensors landmark matrix save collector.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`SafetensorsLandmarkMatrixSaveCollector[K]`: Container Safetensors landmark matrix saver.
    """
    return SafetensorsLandmarkMatrixSaveCollector[K]()

lmsc_aliases.update({
    ".safetensors": safetensors_lmsc_creator,
})
