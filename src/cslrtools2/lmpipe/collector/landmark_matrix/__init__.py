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

"""Landmark matrix save collectors.

This subpackage provides collectors that save landmark matrices
in various file formats.
"""

from .base import LandmarkMatrixSaveCollector, lmsc_aliases
from .csv_lmsc import CsvLandmarkMatrixSaveCollector
from .json_lmsc import JsonLandmarkMatrixSaveCollector
from .npy_lmsc import NpyLandmarkMatrixSaveCollector
from .npz_lmsc import NpzLandmarkMatrixSaveCollector
from .safetensors_lmsc import SafetensorsLandmarkMatrixSaveCollector
from .torch_lmsc import TorchLandmarkMatrixSaveCollector
from .zarr_lmsc import ZarrLandmarkMatrixSaveCollector

__all__ = [
    "LandmarkMatrixSaveCollector",
    "CsvLandmarkMatrixSaveCollector",
    "JsonLandmarkMatrixSaveCollector",
    "NpyLandmarkMatrixSaveCollector",
    "NpzLandmarkMatrixSaveCollector",
    "SafetensorsLandmarkMatrixSaveCollector",
    "TorchLandmarkMatrixSaveCollector",
    "ZarrLandmarkMatrixSaveCollector",
    "lmsc_aliases",
]
