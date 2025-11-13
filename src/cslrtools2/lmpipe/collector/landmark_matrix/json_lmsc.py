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

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ....typings import NDArrayFloat
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class JsonLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into per-key ``*.json`` or ``*.jsonc`` files inside ``landmarks``."""

    def __init__(self, *, indent: int | None = 2, encoding: str = "utf-8", extension: str = ".json") -> None:
        """Initialize the JSON landmark matrix save collector.
        
        Args:
            indent (:class:`int | None`, optional): JSON indentation level. Defaults to 2.
            encoding (:class:`str`, optional): File encoding. Defaults to "utf-8".
            extension (:class:`str`, optional): File extension. Either ".json" or ".jsonc". Defaults to ".json".
        """
        if extension not in (".json", ".jsonc"):
            raise ValueError(f"Invalid extension: {extension}. Must be '.json' or '.jsonc'.")
        self.indent = indent
        self.encoding = encoding
        self.extension = extension
        self._base_dir: Path | None = None
        self._buffers: dict[str, list[list[float]]] = {}

    @property
    def is_perkey(self) -> bool:
        """Per-key mode: saves individual JSON files."""
        return True

    @property
    def is_container(self) -> bool:
        """Not a container collector."""
        return False

    @property
    def file_ext(self) -> str:
        """File extension for JSON files (.json or .jsonc)."""
        return self.extension

    def _open_file(self, path: Path):
        self._base_dir = self._prepare_landmark_dir(path)
        self._buffers = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, value in result.items():
            key = str(raw_key)
            buffer = self._buffers.setdefault(key, [])
            buffer.append(np.asarray(value).tolist())

    def _close_file(self):
        if self._base_dir is None:
            return
        for key, entries in self._buffers.items():
            file_path = self._base_dir / f"{key}{self.extension}"
            with file_path.open("w", encoding=self.encoding) as fh:
                json.dump(entries, fh, ensure_ascii=False, indent=self.indent)
        self._base_dir = None
        self._buffers = {}


def json_lmsc_creator[K: str](key_type: type[K]) -> JsonLandmarkMatrixSaveCollector[K]:
    """Create a JSON landmark matrix save collector.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`JsonLandmarkMatrixSaveCollector[K]`: JSON landmark matrix saver with .json extension.
    """
    return JsonLandmarkMatrixSaveCollector[K]()

def jsonc_lmsc_creator[K: str](key_type: type[K]) -> JsonLandmarkMatrixSaveCollector[K]:
    """Create a JSONC landmark matrix save collector.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`JsonLandmarkMatrixSaveCollector[K]`: JSON landmark matrix saver with .jsonc extension.
    """
    return JsonLandmarkMatrixSaveCollector[K](extension=".jsonc")

lmsc_aliases.update({
    ".json": json_lmsc_creator,
    ".jsonc": jsonc_lmsc_creator,
})
