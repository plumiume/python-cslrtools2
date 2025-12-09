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

# pyright: reportUnknownMemberType=false
# Reason: DTypeLike parameter handling in NumPy has complex type inference

"""Utility functions for landmark processing (PROTOTYPE v2).

**STATUS**: PROTOTYPE - Helper functions for metrics calculation.

These utilities are specific to metrics calculation and should remain in the
metrics module. They handle common tasks like:
    - Converting zarr.Array to NumPy arrays
    - Categorizing landmarks by body part
    - Combining multiple landmark parts

Design Decision
---------------

These functions are **NOT** added to ``cslrtools2.sldataset.utils`` because:
    1. They are metrics-specific logic (not core dataset functionality)
    2. Categorization logic is use-case dependent
    3. Future metrics may need different categorization schemes

When integrating into production, these should go into:
    ``src/cslrtools2/sldataset/metrics/utils.py``
"""

from __future__ import annotations

from typing import Mapping, Any, Iterable

import numpy as np


def categorize_landmarks(landmark_keys: Iterable[str]) -> dict[str, list[str]]:
    """Categorize landmark keys by body part based on suffix after last dot.

    Extracts the part after the last '.' in the key name and categorizes:
        - **pose** or **body**: Pose category
        - **left_hand** or **l_hand**: Left Hand category
        - **right_hand** or **r_hand**: Right Hand category
        - **face** or **facial**: Face category

    Args:
        landmark_keys: Iterator of landmark key names

    Returns:
        Dictionary mapping category names to lists of keys.
        Empty categories are excluded.

    Example:
        >>> keys = ["mediapipe.pose", "mediapipe.left_hand", "openpose.right_hand"]
        >>> categories = categorize_landmarks(keys)
        >>> print(categories)
        {
            "Pose": ["mediapipe.pose"],
            "Left Hand": ["mediapipe.left_hand"],
            "Right Hand": ["openpose.right_hand"]
        }

    Note:
        Uses `key.rsplit(".", 1)[-1]` to extract suffix for categorization.
        This makes categorization independent of the landmark engine prefix.
    """
    categories: dict[str, list[str]] = {
        "Pose": [],
        "Left Hand": [],
        "Right Hand": [],
        "Face": [],
    }

    for key in landmark_keys:
        # Extract suffix after last '.'
        suffix = key.rsplit(".", 1)[-1].lower()

        # Check for pose/body keypoints
        if suffix in ("pose", "body"):
            categories["Pose"].append(key)

        # Check for left hand
        elif suffix in ("left_hand", "l_hand"):
            categories["Left Hand"].append(key)

        # Check for right hand
        elif suffix in ("right_hand", "r_hand"):
            categories["Right Hand"].append(key)

        # Check for face
        elif suffix in ("face", "facial"):
            categories["Face"].append(key)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def combine_landmarks(
    landmarks: Mapping[str, np.ndarray], keys: list[str], axis: int = 1
) -> np.ndarray:
    """Combine multiple landmark arrays along specified axis.

    Useful for creating combined representations like "Both Hands" or "All Parts".

    Args:
        landmarks: Mapping of landmark key to NumPy array
        keys: Keys to combine
        axis: Axis to concatenate along (default: 1 for keypoints dimension)

    Returns:
        Combined NumPy array

    Raises:
        ValueError: If no valid keys found or arrays have incompatible shapes

    Example:
        >>> landmarks = {
        ...     "mediapipe.left_hand": np.zeros((300, 21, 3)),
        ...     "mediapipe.right_hand": np.zeros((300, 21, 3))
        ... }
        >>> combined = combine_landmarks(
        ...     landmarks,
        ...     ["mediapipe.left_hand", "mediapipe.right_hand"],
        ...     axis=1
        ... )
        >>> combined.shape
        (300, 42, 3)

    Example - Combining all parts:
        >>> all_keys = list(landmarks.keys())
        >>> all_combined = combine_landmarks(landmarks, all_keys, axis=1)
    """
    arrays = [landmarks[key] for key in keys if key in landmarks]

    if not arrays:
        raise ValueError(
            f"No valid keys found in landmarks. "
            f"Requested: {keys}, Available: {list(landmarks.keys())}"
        )

    if len(arrays) == 1:
        return arrays[0]

    try:
        return np.concatenate(arrays, axis=axis)
    except ValueError as e:
        shapes = [arr.shape for arr in arrays]
        raise ValueError(
            f"Cannot concatenate arrays along axis {axis}. " f"Shapes: {shapes}"
        ) from e


def calculate_multi_part_nan_rates(
    landmarks: Mapping[str, np.ndarray], metric_calculate_func: Any
) -> dict[str, list[float]]:
    """Calculate NaN rates for multiple landmark parts including combinations.

    This is a convenience function that:
        1. Categorizes landmarks by body part
        2. Calculates metrics for each individual part
        3. Calculates metrics for combined parts (Hands, All)

    Args:
        landmarks: Mapping of landmark keys to NumPy arrays
        metric_calculate_func: Function to calculate metric (e.g., metric.calculate)

    Returns:
        Dictionary mapping part names to metric values

    Example:
        >>> from metrics_prototype2 import create_metric
        >>> metric = create_metric("completeness.nan_rate")
        >>> # Use np.asarray() with __array__ protocol for conversion
        >>> landmarks_np = {k: np.asarray(v) for k, v in item.landmarks.items()}
        >>> results = calculate_multi_part_nan_rates(landmarks_np, metric.calculate)
        >>> print(results)
        {
            "Pose": [0.0],
            "Left Hand": [0.048],
            "Right Hand": [0.154],
            "Hands": [0.101],
            "All": [0.051]
        }
    """
    categories = categorize_landmarks(landmarks.keys())
    results: dict[str, list[float]] = {}

    # Calculate for individual categories
    for category, keys in categories.items():
        if not keys:
            continue

        # Use first key for single-part categories
        if len(keys) == 1:
            result = metric_calculate_func(landmarks[keys[0]])
            results[category] = [result["values"]["nan_rate"]]

    # Calculate for combined hands
    if "Left Hand" in categories and "Right Hand" in categories:
        hands_keys = categories["Left Hand"] + categories["Right Hand"]
        if hands_keys:
            hands_combined = combine_landmarks(landmarks, hands_keys, axis=1)
            result = metric_calculate_func(hands_combined)
            results["Hands"] = [result["values"]["nan_rate"]]

    # Calculate for all parts combined
    all_keys = list(landmarks.keys())
    if all_keys:
        all_combined = combine_landmarks(landmarks, all_keys, axis=1)
        result = metric_calculate_func(all_combined)
        results["All"] = [result["values"]["nan_rate"]]

    return results
