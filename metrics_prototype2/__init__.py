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

"""Metrics plugin system prototype v2 - Integrated with SLDataset.

**STATUS**: PROTOTYPE v2 - Improved version using cslrtools2.sldataset

Changes from v1
---------------

1. **SLDataset Integration**: Uses ``SLDataset.from_zarr()`` for data loading
2. **Utility Functions**: New ``utils.py`` for landmark processing helpers
3. **Simplified Loading**: Removed redundant zarr access code
4. **Type Safety**: Better integration with ``DefaultSLDatasetItem``

Purpose
-------

This prototype demonstrates:
    1. Plugin architecture for metrics
    2. Integration with SLDataset loading system
    3. Helper functions for multi-part landmark processing
    4. Efficient batch processing with zarr lazy loading

Quick Start
-----------

Use MetricCalculator with SLDataset::

    >>> import numpy as np
    >>> import zarr
    >>> from metrics_prototype2 import MetricCalculator
    >>> from cslrtools2.sldataset import SLDataset
    >>>
    >>> # Create calculator and register metrics
    >>> calculator = MetricCalculator()
    >>> calculator.add_metric("completeness.nan_rate")
    >>> calculator.add_metric("temporal.smoothness")
    >>>
    >>> # Load dataset (zarr.Array references only)
    >>> root = zarr.open_group("dataset.zarr", mode="r")
    >>> dataset = SLDataset.from_zarr(root)
    >>>
    >>> # Process items with __array__ protocol
    >>> item = dataset[0]
    >>> landmarks_np = {k: np.asarray(v) for k, v in item.landmarks.items()}
    >>> results = calculator.calculate(landmarks_np["mediapipe.pose"])

Future Integration
------------------

When implementing the real metrics system in ``src/cslrtools2/sldataset/metrics/``:

1. Move utility functions to ``sldataset/metrics/utils.py``
2. Register metrics via pyproject.toml Entry Points
3. Add CLI integration (``sldataset metrics`` command)
4. Comprehensive tests and documentation

External Plugin Registration
-----------------------------

External projects can register metrics in their pyproject.toml::

    [project.entry-points."cslrtools2.sldataset.metrics"]
    "custom.my_metric" = "my_package.metrics:MyMetric"

See Also
--------
:mod:`.base` : Metric base class
:mod:`.loader` : Plugin discovery system
:mod:`.utils` : Landmark processing utilities
:mod:`.plugins` : Metric implementations
"""

from __future__ import annotations

from .base import Metric, LandmarkMetric, MetricResult
from .loader import (
    load_metrics,
    list_metric_names,
    get_metric_info,
    create_metric,
    MetricCalculator,
    MetricInfo,
)

__all__ = [
    # Base classes
    "Metric",
    "LandmarkMetric",
    "MetricResult",
    # Loader functions
    "load_metrics",
    "list_metric_names",
    "get_metric_info",
    "create_metric",
    "MetricCalculator",
    "MetricInfo",
]
