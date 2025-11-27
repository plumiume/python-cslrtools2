# Copyright 2025 cslrtools2 Contributors
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

"""Temporal consistency metrics for landmark quality evaluation.

This module implements metrics that evaluate the temporal coherence of
landmark trajectories across frames. These metrics are engine-agnostic
and do not require ground truth data.

**Purpose**: Measures data completeness by calculating the proportion of
frames that contain at least one NaN value (frame-level missing rate).

**Interpretation**:
    - Lower values indicate smoother, more consistent motion
    - Higher values suggest jittery or unstable tracking
    - Useful for comparing different pose estimation engines

**Engine Agnostic**: ✅ Yes - Works with any landmark estimation engine

**Ground Truth Required**: ❌ No

Mathematical Definition
-----------------------

Given landmark data X with shape (T, K, D):
    - T: number of frames
    - K: number of keypoints
    - D: coordinate dimensions (typically 2 or 3)

We compute:
    1. **Velocity** (frame-to-frame displacement)::
    
        V[t] = X[t+1] - X[t]
        shape: (T-1, K, D)
    
    2. **Acceleration** (velocity change)::
    
        A[t] = V[t+1] - V[t]
        shape: (T-2, K, D)
    
    3. **Smoothness** (acceleration standard deviation)::
    
        smoothness = std(A)
        
        Lower values indicate smoother motion.

Usage
-----

Basic usage::

    >>> import numpy as np
    >>> from metrics_prototype.plugins.temporal import TemporalConsistencyMetric
    >>> 
    >>> # Create sample data with smooth motion
    >>> landmarks = np.random.rand(100, 33, 3)
    >>> 
    >>> # Calculate metric
    >>> metric = TemporalConsistencyMetric()
    >>> result = metric.calculate(landmarks)
    >>> 
    >>> print(f"Mean velocity: {result['values']['mean_velocity']:.4f}")
    Mean velocity: 0.0523
    >>> print(f"Smoothness: {result['values']['smoothness']:.4f}")
    Smoothness: 0.0087

References
----------
.. [1] Liu & Yuan, "Recognizing Human Actions as the Evolution of Pose 
       Estimation Maps", CVPR 2018
.. [2] Güler et al., "DensePose: Dense Human Pose Estimation In The Wild", 
       CVPR 2018

See Also
--------
:class:`~metrics_prototype.base.Metric` : Base class for all metrics
:class:`~metrics_prototype.plugins.completeness.NaNRateMetric` : Completeness metric

Notes
-----
- This metric requires at least 3 frames to compute acceleration
- NaN values in input data will be propagated to output
- Consider filtering or interpolating NaN values before calculation
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from metrics_prototype.base import Metric, MetricResult


class TemporalConsistencyMetric(Metric):
    """Calculate temporal consistency of landmark trajectories.
    
    This metric evaluates motion smoothness by analyzing frame-to-frame
    velocity and acceleration patterns. Smooth, stable tracking produces
    low acceleration variance, while jittery or unstable tracking shows
    high variance.
    
    Formula:
        velocity = X[t+1] - X[t]
        acceleration = velocity[t+1] - velocity[t]
        smoothness = std(acceleration)
    
    Attributes:
        None (stateless metric)
    
    Notes:
        - Requires minimum 3 frames (for acceleration calculation)
        - NaN values in input data affect output statistics
        - Higher smoothness values indicate more jitter
        - Metric is scale-dependent (units of landmark coordinates)
    
    See Also:
        :class:`~metrics_prototype.base.Metric` : Base class
    """

    def validate_inputs(self, data: NDArray[np.float32]) -> bool:
        """Validate input landmark data.
        
        Args:
            data: Landmark array with shape (frames, keypoints, coordinates).
        
        Returns:
            True if data is valid (3D array with at least 3 frames).
        
        Raises:
            ValueError: If data has invalid shape or too few frames.
        """
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, keypoints, coords), "
                f"got {data.ndim}D array with shape {data.shape}"
            )
        
        if data.shape[0] < 3:
            raise ValueError(
                f"Temporal consistency requires at least 3 frames, "
                f"got {data.shape[0]} frames"
            )
        
        return True

    def calculate(self, data: NDArray[np.float32], **kwargs: Any) -> MetricResult:
        """Calculate temporal consistency metrics.
        
        Args:
            data: Landmark array with shape (frames, keypoints, coordinates).
                Expected to be float32.
            **kwargs: Unused for this metric (maintained for interface compatibility).
        
        Returns:
            MetricResult containing:
                - metric_name: "temporal_consistency"
                - values:
                    - mean_velocity: Average frame-to-frame displacement
                    - std_velocity: Standard deviation of velocity
                    - mean_acceleration: Average acceleration magnitude
                    - std_acceleration: Standard deviation of acceleration
                    - smoothness: Overall smoothness (std of acceleration)
                - metadata:
                    - total_frames: Total number of frames
                    - velocity_frames: Number of velocity samples (T-1)
                    - acceleration_frames: Number of acceleration samples (T-2)
                    - shape: Original data shape
        
        Raises:
            ValueError: If data has invalid shape or too few frames.
            TypeError: If data is not a NumPy array.
        
        Example:
            >>> import numpy as np
            >>> metric = TemporalConsistencyMetric()
            >>> data = np.random.rand(100, 33, 3)
            >>> result = metric.calculate(data)
            >>> print(result['values']['smoothness'])
            0.08234
        """
        # Validate inputs
        self.validate_inputs(data)

        # Calculate velocity (frame-to-frame displacement)
        # Shape: (T-1, K, D)
        velocity = data[1:] - data[:-1]
        
        # Calculate acceleration (velocity change)
        # Shape: (T-2, K, D)
        acceleration = velocity[1:] - velocity[:-1]
        
        # Compute statistics
        # Use nanmean/nanstd to handle NaN values in data
        mean_velocity = float(np.nanmean(np.abs(velocity)))
        std_velocity = float(np.nanstd(velocity))
        
        mean_acceleration = float(np.nanmean(np.abs(acceleration)))
        std_acceleration = float(np.nanstd(acceleration))
        
        # Smoothness: lower is better (less jitter)
        smoothness = std_acceleration

        return MetricResult(
            metric_name="temporal_consistency",
            values={
                "mean_velocity": mean_velocity,
                "std_velocity": std_velocity,
                "mean_acceleration": mean_acceleration,
                "std_acceleration": std_acceleration,
                "smoothness": smoothness,
            },
            metadata={
                "total_frames": data.shape[0],
                "velocity_frames": velocity.shape[0],
                "acceleration_frames": acceleration.shape[0],
                "shape": data.shape,
            },
        )

    def get_description(self) -> str:
        """Return description of the temporal consistency metric.
        
        Returns:
            Human-readable description of the metric.
        """
        return (
            "Evaluates temporal consistency by analyzing velocity and acceleration. "
            "Lower 'smoothness' values indicate more stable tracking with less jitter. "
            "Useful for comparing pose estimation engines on motion quality."
        )


# ============================================================================
# Plugin Registration (for simulated Entry Points)
# ============================================================================

temporal_consistency_info = (TemporalConsistencyMetric, {})  # pyright: ignore[reportUnknownVariableType]
