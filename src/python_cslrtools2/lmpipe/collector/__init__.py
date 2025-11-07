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
