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

"""Sign Language Dataset management module.

This module provides core dataset classes and utilities for managing
sign language datasets with efficient storage and PyTorch integration.
"""

from __future__ import annotations

# Key holder for type guards
from .holder import SLKeyHolder

# Dataset item classes and type aliases
from .item import (
    SLDatasetItem,
    DefaultSLDatasetItem,
    TensorSLDatasetItem,
    ZarrSLDatasetItem,
)

# Core dataset classes and utilities
from .core import (
    # Type aliases
    DefaultSLDataset,
    DefaultIterableSLDataset,
    # Base classes
    Dataset,
    IterableDataset,
    # Batch class
    SLDatasetBatch,
    # Main dataset classes
    SLDataset,
    IterableSLDataset,
    # Utility functions
    dataset_to_zarr,
)

__all__ = [
    # Key holder
    "SLKeyHolder",
    # Type aliases
    "DefaultSLDatasetItem",
    "TensorSLDatasetItem",
    "ZarrSLDatasetItem",
    "DefaultSLDataset",
    "DefaultIterableSLDataset",
    # Dataset item
    "SLDatasetItem",
    # Base classes
    "Dataset",
    "IterableDataset",
    # Batch class
    "SLDatasetBatch",
    # Main dataset classes
    "SLDataset",
    "IterableSLDataset",
    # Utility functions
    "dataset_to_zarr",
]
