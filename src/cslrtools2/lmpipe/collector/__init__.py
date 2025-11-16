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

"""Collectors for landmark estimation pipeline results.

This package provides various collectors that process and store results
from landmark estimation pipelines, including landmark matrices and
annotated frames in different formats.
"""

# Base collector
from .base import Collector

# Landmark matrix collectors
from .landmark_matrix import (
    LandmarkMatrixSaveCollector,
    CsvLandmarkMatrixSaveCollector,
    JsonLandmarkMatrixSaveCollector,
    NpyLandmarkMatrixSaveCollector,
    NpzLandmarkMatrixSaveCollector,
    SafetensorsLandmarkMatrixSaveCollector,
    TorchLandmarkMatrixSaveCollector,
    ZarrLandmarkMatrixSaveCollector,
)

# Annotated frames collectors
from .annotated_frames import (
    AnnotatedFramesSaveCollector,
    AnnotatedFramesShowCollector,
    Cv2AnnotatedFramesSaveCollector,
    Cv2AnnotatedFramesShowCollector,
    MatplotlibAnnotatedFramesSaveCollector,
    MatplotlibAnnotatedFramesShowCollector,
    PilAnnotatedFramesShowCollector,
    TorchVisionAnnotatedFramesShowCollector,
)

__all__ = [
    # Base
    "Collector",
    # Landmark matrix base
    "LandmarkMatrixSaveCollector",
    # Landmark matrix concrete
    "CsvLandmarkMatrixSaveCollector",
    "JsonLandmarkMatrixSaveCollector",
    "NpyLandmarkMatrixSaveCollector",
    "NpzLandmarkMatrixSaveCollector",
    "SafetensorsLandmarkMatrixSaveCollector",
    "TorchLandmarkMatrixSaveCollector",
    "ZarrLandmarkMatrixSaveCollector",
    # Annotated frames base
    "AnnotatedFramesSaveCollector",
    "AnnotatedFramesShowCollector",
    # Annotated frames concrete
    "Cv2AnnotatedFramesSaveCollector",
    "Cv2AnnotatedFramesShowCollector",
    "MatplotlibAnnotatedFramesSaveCollector",
    "MatplotlibAnnotatedFramesShowCollector",
    "PilAnnotatedFramesShowCollector",
    "TorchVisionAnnotatedFramesShowCollector",
]
