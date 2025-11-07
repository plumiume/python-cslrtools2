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
