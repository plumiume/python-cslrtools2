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

"""Metrics plugin system prototype.

**STATUS**: PROTOTYPE - NOT part of main cslrtools2 package.

This module demonstrates the proposed plugin architecture for the future
`src/cslrtools2/sldataset/metrics/` module.

Purpose
-------

This prototype serves as:
    1. Design reference for the plugin architecture
    2. Demo of NaN metric implementation
    3. Testing ground for Entry Points simulation

Quick Start
-----------

Register and use a metric::

    >>> from metrics_prototype import register_metric, create_metric
    >>> from metrics_prototype.plugins.completeness import NaNRateMetric
    >>> import numpy as np
    >>> 
    >>> # Register plugin (simulates Entry Points)
    >>> register_metric("completeness.nan_rate", NaNRateMetric, {})
    >>> 
    >>> # Create metric instance
    >>> metric = create_metric("completeness.nan_rate")
    >>> 
    >>> # Calculate on data
    >>> data = np.random.rand(100, 33, 3)
    >>> data[10:20] = np.nan
    >>> result = metric.calculate(data)
    >>> print(f"NaN rate: {result['values']['nan_rate']:.2%}")

Future Integration
------------------

When implementing the real metrics system in `src/cslrtools2/sldataset/`:

1. Copy architecture from this prototype
2. Replace simulated registry with real Entry Points
3. Add more sophisticated metrics (temporal, anatomical)
4. Integrate with CLI (`sldataset metrics` command)
5. Add comprehensive tests

See Also
--------
:mod:`.base` : Metric base class
:mod:`.loader` : Plugin discovery system
:mod:`.plugins.completeness` : NaN metric implementation
"""

from __future__ import annotations

from .base import Metric, MetricResult
from .loader import (
    load_metrics,
    list_metric_names,
    get_metric_info,
    create_metric,
    register_metric,
    MetricInfo,
)

__all__ = [
    # Base classes
    "Metric",
    "MetricResult",
    # Loader functions
    "load_metrics",
    "list_metric_names",
    "get_metric_info",
    "create_metric",
    "register_metric",
    "MetricInfo",
]
