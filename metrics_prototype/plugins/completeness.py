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

"""NaN rate metric plugin (PROTOTYPE).

This module implements a completeness metric that calculates the rate of
missing (NaN) values in landmark data. This is a demo implementation showing
how the plugin system would work.

Metric Description
------------------

**Name**: ``completeness.nan_rate``

**Category**: Completeness (Phase 1)

**Purpose**: Measures data completeness by calculating the proportion of
frames that contain at least one NaN value (frame-level missing rate).

**Interpretation**:
    - 0.0 (0%): Perfect completeness, no frames with missing data
    - 0.1 (10%): 10% of frames contain at least one NaN
    - 1.0 (100%): All frames contain NaN (invalid data)

**Recommended Threshold**: < 0.2 (20%) for production use

**Engine Agnostic**: ✅ Yes - Works with any landmark estimation engine

**Ground Truth Required**: ❌ No

Mathematical Definition
-----------------------

Given landmark data X with shape (T, K, D):
    - T: number of frames
    - K: number of keypoints
    - D: coordinate dimensions

The metric calculates the expectation of frame-level NaN presence::

    m = E_t [ ∨^K ∨^D isNaN(X_t,k,d) ]

Where:
    - ∨ denotes logical OR across keypoints and dimensions
    - E_t denotes expectation (mean) across all frames
    - For each frame t, we check if ANY value is NaN

Equivalently::

    frame_has_nan[t] = any(isnan(X[t, :, :]))  # per-frame boolean
    nan_rate = mean(frame_has_nan)              # average across frames

Example
-------

Basic usage::

    >>> import numpy as np
    >>> from metrics_prototype.plugins.completeness import NaNRateMetric
    >>>
    >>> # Create sample data
    >>> landmarks = np.random.rand(100, 33, 3)
    >>> landmarks[10:20, :, :] = np.nan  # 10 frames with NaN
    >>>
    >>> # Calculate metric
    >>> metric = NaNRateMetric()
    >>> result = metric.calculate(landmarks)
    >>>
    >>> print(f"NaN rate: {result['values']['nan_rate']:.2%}")
    NaN rate: 10.00%
    >>> print(f"Frames with NaN: {result['metadata']['frames_with_nan']}")
    Frames with NaN: 10
    >>> print(f"Total frames: {result['metadata']['total_frames']}")
    Total frames: 100

References
----------

This metric is recommended in Phase 1 (Basic Metrics) with the highest
priority rating (⭐⭐⭐⭐⭐) for engine-agnostic evaluation.

See: `pose_estimation_metrics_analysis.md` Section 6.1
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from metrics_prototype.base import Metric, MetricResult


class NaNRateMetric(Metric):
    """Calculate the proportion of frames containing NaN (missing) values.

    This metric measures frame-level completeness by calculating what fraction
    of frames contain at least one NaN value. Unlike element-wise NaN counting,
    this provides a more practical measure: a single missing keypoint makes
    the entire frame incomplete.

    Formula:
        nan_rate = mean(any(isnan(X[t, :, :])) for all frames t)

    Attributes:
        None (stateless metric)

    Notes:
        - Frame-level measurement: one NaN in a frame marks it as incomplete
        - Vectorized using NumPy for efficiency
        - Handles edge cases: empty arrays, all-NaN frames
        - Returns metadata about frames for debugging

    See Also:
        :class:`~metrics_prototype.base.Metric` : Base class
    """

    def calculate(self, data: NDArray[np.float32], **kwargs: Any) -> MetricResult:
        """Calculate frame-level NaN rate for the given landmark data.

        Args:
            data: Landmark array with shape (frames, keypoints, coordinates).
                Expected to be float32 with possible NaN values.
            **kwargs: Unused for this metric (maintained for interface compatibility).

        Returns:
            MetricResult containing:
                - metric_name: "nan_rate"
                - values:
                    - nan_rate: Proportion of frames with NaN (0.0 to 1.0)
                - metadata:
                    - total_frames: Total number of frames
                    - frames_with_nan: Count of frames containing at least one NaN
                    - shape: Original data shape

        Raises:
            ValueError: If data has invalid shape (not 3D) or all frames contain NaN.
            TypeError: If data is not a NumPy array.

        Example:
            >>> import numpy as np
            >>> metric = NaNRateMetric()
            >>> data = np.zeros((100, 33, 3))
            >>> data[0:10] = np.nan  # 10 frames with NaN
            >>> result = metric.calculate(data)
            >>> print(result['values']['nan_rate'])
            0.1
        """
        # Validate inputs
        if not self.validate_inputs(data):
            raise ValueError(
                f"Invalid input shape. Expected 3D array (frames, keypoints, coords), "
                f"got shape {data.shape}"
            )

        # Calculate frame-level NaN presence
        # For each frame, check if ANY value is NaN
        # Shape: (T, K, D) -> (T,) boolean array
        frame_has_nan = np.any(np.isnan(data), axis=(1, 2))

        frames_with_nan = int(np.sum(frame_has_nan))
        total_frames = data.shape[0]

        # Check if all frames contain NaN (invalid data)
        if frames_with_nan == total_frames:
            raise ValueError(
                "All frames in the input data contain NaN. "
                "Cannot compute meaningful metric."
            )

        # Calculate frame-level NaN rate
        nan_rate = float(frames_with_nan / total_frames) if total_frames > 0 else 0.0

        return MetricResult(
            metric_name="nan_rate",
            values={
                "nan_rate": nan_rate,
            },
            metadata={
                "total_frames": total_frames,
                "frames_with_nan": frames_with_nan,
                "shape": data.shape,
            },
        )

    def get_description(self) -> str:
        """Return description of the NaN rate metric.

        Returns:
            Human-readable description of the metric.
        """
        return (
            "Calculates the proportion of frames containing at least one NaN value. "
            "A frame is considered incomplete "
            "if ANY keypoint or coordinate is missing. "
            "Lower values indicate better frame-level completeness. "
            "Recommended threshold: < 0.2 (20%) for production use."
        )


# Plugin info tuple (simulates Entry Point return value)
# In real implementation, this would be registered in pyproject.toml:
# [project.entry-points."cslrtools2.sldataset.metrics"]
# "completeness.nan_rate" = "cslrtools2.plugins.metrics.completeness:nan_rate_info"
nan_rate_info = (NaNRateMetric, {})
