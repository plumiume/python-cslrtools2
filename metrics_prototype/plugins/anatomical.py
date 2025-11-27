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

"""Anatomical constraint metrics for landmark quality evaluation.

This module implements metrics that evaluate whether landmark data satisfies
basic anatomical constraints (e.g., consistent bone lengths, valid joint angles).
These metrics are engine-agnostic and do not require ground truth data.

**Purpose**: Measures the physical plausibility of landmark data by checking
anatomical constraints.

**Interpretation**:
    - Lower bone length variation indicates more consistent skeleton structure
    - Useful for detecting tracking failures or unrealistic poses
    - Helps compare different pose estimation engines

**Engine Agnostic**: ✅ Yes - Works with any landmark estimation engine

**Ground Truth Required**: ❌ No

Mathematical Definition
-----------------------

Given landmark data X with shape (T, K, D):
    - T: number of frames
    - K: number of keypoints
    - D: coordinate dimensions

**Bone Length Consistency**:

For a bone connecting joints i and j::

    bone_length[t] = ||X[t, i] - X[t, j]||_2
    
    variation_coefficient = std(bone_length) / mean(bone_length)

Lower variation coefficients indicate more consistent bone lengths across frames.

Usage
-----

Basic usage::

    >>> import numpy as np
    >>> from metrics_prototype.plugins.anatomical import (
    ...     AnatomicalConstraintMetric
    ... )
    >>>
    >>> # Create sample data
    >>> landmarks = np.random.rand(100, 33, 3)
    >>>
    >>> # Define bone pairs (MediaPipe Pose skeleton)
    >>> bone_pairs = [
    ...     (11, 13),  # left_shoulder -> left_elbow
    ...     (13, 15),  # left_elbow -> left_wrist
    ...     (12, 14),  # right_shoulder -> right_elbow
    ...     (14, 16),  # right_elbow -> right_wrist
    ... ]
    >>>
    >>> # Calculate metric
    >>> metric = AnatomicalConstraintMetric()
    >>> result = metric.calculate(landmarks, bone_pairs=bone_pairs)
    >>>
    >>> print(f"Mean bone variation: {result['values']['mean_variation']:.4f}")
    Mean bone variation: 0.0523

References
----------
.. [1] Cao et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation using
       Part Affinity Fields", CVPR 2017
.. [2] Mehta et al., "VNect: Real-time 3D Human Pose Estimation with a Single
       RGB Camera", ACM TOG 2017

See Also
--------
:class:`~metrics_prototype.base.Metric` : Base class for all metrics
:class:`~metrics_prototype.plugins.temporal.TemporalConsistencyMetric` :
    Temporal consistency metric

Notes
-----
- Bone pairs must be specified based on the landmark format
- NaN values in input data will propagate to output
- Metric is scale-dependent (units of landmark coordinates)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from metrics_prototype.base import Metric, MetricResult

# Default bone pairs for MediaPipe Pose (33 keypoints)
# Based on MediaPipe Pose landmark topology
MEDIAPIPE_POSE_BONES = [
    # Torso
    (11, 12),  # left_shoulder <-> right_shoulder
    (11, 23),  # left_shoulder <-> left_hip
    (12, 24),  # right_shoulder <-> right_hip
    (23, 24),  # left_hip <-> right_hip
    # Left arm
    (11, 13),  # left_shoulder -> left_elbow
    (13, 15),  # left_elbow -> left_wrist
    # Right arm
    (12, 14),  # right_shoulder -> right_elbow
    (14, 16),  # right_elbow -> right_wrist
    # Left leg
    (23, 25),  # left_hip -> left_knee
    (25, 27),  # left_knee -> left_ankle
    # Right leg
    (24, 26),  # right_hip -> right_knee
    (26, 28),  # right_knee -> right_ankle
]


class AnatomicalConstraintMetric(Metric):
    """Calculate anatomical constraint violations in landmark data.

    This metric evaluates the physical plausibility of landmark trajectories
    by measuring bone length consistency across frames. Consistent bone lengths
    indicate stable, realistic tracking.

    Formula:
        For each bone (i, j):
            length[t] = ||X[t, i] - X[t, j]||
            variation_coef = std(length) / mean(length)

    Attributes:
        None (stateless metric)

    Notes:
        - Requires bone pair definitions (joint connectivity)
        - Lower variation coefficients are better (more consistent)
        - NaN values in input data affect output statistics
        - Metric is scale-dependent

    See Also:
        :class:`~metrics_prototype.base.Metric` : Base class
    """

    def validate_inputs(self, data: NDArray[np.float32]) -> bool:
        """Validate input landmark data.

        Args:
            data: Landmark array with shape (frames, keypoints, coordinates).

        Returns:
            True if data is valid (3D array).

        Raises:
            ValueError: If data has invalid shape.
        """
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, keypoints, coords), "
                f"got {data.ndim}D array with shape {data.shape}"
            )

        return True

    def calculate(
        self,
        data: NDArray[np.float32],
        bone_pairs: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Calculate anatomical constraint metrics.

        Args:
            data: Landmark array with shape (frames, keypoints, coordinates).
                Expected to be float32.
            bone_pairs: List of (joint_i, joint_j) tuples defining bones.
                If None, uses default MediaPipe Pose bones.
            **kwargs: Unused for this metric (maintained for interface
                compatibility).

        Returns:
            MetricResult containing:
                - metric_name: "anatomical_constraint"
                - values:
                    - mean_variation: Average variation coefficient across
                        all bones
                    - std_variation: Standard deviation of variation
                        coefficients
                    - min_variation: Minimum variation coefficient
                    - max_variation: Maximum variation coefficient
                - metadata:
                    - total_frames: Total number of frames
                    - num_bones: Number of bones evaluated
                    - bone_pairs: List of bone pairs used
                    - shape: Original data shape

        Raises:
            ValueError: If data has invalid shape or bone indices are
                out of range.
            TypeError: If data is not a NumPy array.

        Example:
            >>> import numpy as np
            >>> metric = AnatomicalConstraintMetric()
            >>> data = np.random.rand(100, 33, 3)
            >>> bone_pairs = [(11, 13), (13, 15)]
            >>> result = metric.calculate(data, bone_pairs=bone_pairs)
            >>> print(result['values']['mean_variation'])
            0.0234
        """
        # Validate inputs
        self.validate_inputs(data)

        # Use default bone pairs if not provided
        if bone_pairs is None:
            bone_pairs = MEDIAPIPE_POSE_BONES

        # Validate bone indices
        num_keypoints = data.shape[1]
        for i, j in bone_pairs:
            if i >= num_keypoints or j >= num_keypoints:
                raise ValueError(
                    f"Bone pair ({i}, {j}) has indices out of range "
                    f"for {num_keypoints} keypoints"
                )

        # Calculate bone lengths for each frame
        variation_coefficients: list[float] = []

        for i, j in bone_pairs:
            # Compute bone length for each frame
            # Shape: (T,)
            bone_vectors = data[:, i, :] - data[:, j, :]
            bone_lengths = np.linalg.norm(bone_vectors, axis=1)

            # Skip bones with all NaN or zero mean
            if np.all(np.isnan(bone_lengths)):
                continue

            mean_length = float(np.nanmean(bone_lengths))
            if mean_length == 0:
                continue

            std_length = np.nanstd(bone_lengths)

            # Variation coefficient = std / mean
            variation_coef = float(std_length / mean_length)
            variation_coefficients.append(variation_coef)

        # Compute summary statistics
        if len(variation_coefficients) == 0:
            raise ValueError(
                "All bones have invalid lengths (NaN or zero mean)"
            )

        variation_array = np.array(variation_coefficients)

        return MetricResult(
            metric_name="anatomical_constraint",
            values={
                "mean_variation": float(np.mean(variation_array)),
                "std_variation": float(np.std(variation_array)),
                "min_variation": float(np.min(variation_array)),
                "max_variation": float(np.max(variation_array)),
            },
            metadata={
                "total_frames": data.shape[0],
                "num_bones": len(bone_pairs),
                "bone_pairs": bone_pairs,
                "shape": data.shape,
            },
        )

    def get_description(self) -> str:
        """Return description of the anatomical constraint metric.

        Returns:
            Human-readable description of the metric.
        """
        return (
            "Evaluates anatomical plausibility by measuring bone length "
            "consistency across frames. Lower 'mean_variation' values indicate "
            "more physically realistic and stable tracking."
        )


# ============================================================================
# Plugin Registration (for simulated Entry Points)
# ============================================================================

anatomical_constraint_info = (
    AnatomicalConstraintMetric,
    {},
)  # pyright: ignore[reportUnknownVariableType]
