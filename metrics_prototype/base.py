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

"""Base classes and types for metrics evaluation system (PROTOTYPE).

**STATUS**: This is a prototype demonstrating the plugin architecture for
metrics. This code is NOT part of the main cslrtools2 package and serves
as a design reference for future implementation in `src/cslrtools2/sldataset/metrics/`.

Key Design Principles
---------------------

1. **Engine Agnostic**: Metrics work with any landmark estimation engine
   (MediaPipe, OpenPose, etc.) by operating on raw NumPy arrays.

2. **Ground Truth Free**: All metrics can be computed without reference data,
   making them suitable for production environments.

3. **Plugin Architecture**: Metrics are discoverable via Entry Points, allowing
   third-party extensions.

4. **Type Safety**: Strict typing with PEP 695 generics and Pyright compatibility.

Architecture
------------

::

    Metric (ABC)
    ├── calculate(data) → MetricResult
    ├── validate_inputs(data) → bool
    └── get_description() → str

    MetricResult (TypedDict)
    ├── metric_name: str
    ├── values: Mapping[str, float]
    └── metadata: Mapping[str, Any]

Example
-------

Implementing a custom metric::

    >>> from metrics_prototype.base import Metric, MetricResult
    >>> import numpy as np
    >>> 
    >>> class MyMetric(Metric):
    ...     def calculate(self, data: np.ndarray, **kwargs) -> MetricResult:
    ...         # data shape: (frames, keypoints, coordinates)
    ...         value = np.mean(data)
    ...         return MetricResult(
    ...             metric_name="my_metric",
    ...             values={"mean": float(value)},
    ...             metadata={"shape": data.shape}
    ...         )
    ...     
    ...     def get_description(self) -> str:
    ...         return "Example metric calculating mean"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypedDict

import numpy as np
from numpy.typing import NDArray


class MetricResult(TypedDict):
    """Result dictionary returned by metric calculations.
    
    Attributes:
        metric_name: Identifier for the metric (e.g., "nan_rate").
        values: Computed metric values. Keys are sub-metric names, values are scores.
        metadata: Additional context information such as input shape, parameters.
    """

    metric_name: str
    values: Mapping[str, float]
    metadata: Mapping[str, Any]


class Metric(ABC):
    """Abstract base class for all landmark quality metrics.
    
    All metrics in the plugin system must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across different
    metric implementations.
    
    Expected Input Format
    ---------------------
    
    Landmark data should be a NumPy array with shape:
        ``(frames, keypoints, coordinates)``
    
    Where:
        - frames: Number of video frames (T)
        - keypoints: Number of body landmarks (K), e.g., 33 for MediaPipe Pose
        - coordinates: Dimension of each landmark (D), typically 3 for (x, y, z)
          or 4 for (x, y, z, visibility)
    
    Example shapes:
        - (300, 33, 3): 300 frames, 33 keypoints, xyz coordinates
        - (150, 21, 4): 150 frames, 21 hand keypoints, xyz + visibility
    """

    @abstractmethod
    def calculate(self, data: NDArray[np.float32], **kwargs: Any) -> MetricResult:
        """Calculate the metric on the given landmark data.
        
        Args:
            data: Landmark data array with shape (frames, keypoints, coordinates).
                May contain NaN values for missing landmarks.
            **kwargs: Additional metric-specific parameters.
        
        Returns:
            Dictionary containing metric name, computed values, and metadata.
        
        Raises:
            ValueError: If input data has invalid shape or all values are NaN.
            TypeError: If input is not a NumPy array.
        """
        pass

    def validate_inputs(self, data: NDArray[np.float32]) -> bool:
        """Validate input data format and shape.
        
        Default implementation checks:
            - Data is a NumPy array
            - Data has 3 dimensions: (frames, keypoints, coordinates)
            - Data has at least 1 frame
        
        Args:
            data: Input landmark data to validate.
        
        Returns:
            True if input is valid, False otherwise.
        """
        if data.ndim != 3:
            return False

        frames, keypoints, coordinates = data.shape
        if frames < 1 or keypoints < 1 or coordinates < 1:
            return False

        return True

    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of the metric.
        
        Returns:
            Description of the metric purpose and interpretation.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the metric."""
        return f"{self.__class__.__name__}()"
