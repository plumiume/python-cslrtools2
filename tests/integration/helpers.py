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

"""Helper functions for integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def verify_npz_structure(npz_path: Path, expected_keys: Sequence[str]) -> None:
    """Verify NPZ file structure and keys.

    Args:
        npz_path: Path to NPZ file.
        expected_keys: Expected keys in the NPZ file.

    Raises:
        AssertionError: If file doesn't exist or keys are missing.
    """
    assert npz_path.exists(), f"NPZ file not found: {npz_path}"

    data = np.load(npz_path)
    actual_keys = set(data.keys())
    expected_keys_set = set(expected_keys)

    assert expected_keys_set.issubset(actual_keys), (
        f"Missing keys in NPZ file. "
        f"Expected: {expected_keys_set}, "
        f"Actual: {actual_keys}"
    )

    data.close()


def verify_zarr_structure(zarr_path: Path) -> None:
    """Verify Zarr directory structure.

    Args:
        zarr_path: Path to Zarr directory.

    Raises:
        AssertionError: If Zarr structure is invalid.
    """
    assert zarr_path.exists(), f"Zarr directory not found: {zarr_path}"
    assert zarr_path.is_dir(), f"Zarr path is not a directory: {zarr_path}"

    # Check for .zgroup file (Zarr v2 format)
    zgroup_file = zarr_path / ".zgroup"
    assert zgroup_file.exists(), f"Zarr group file not found: {zgroup_file}"


def count_video_frames(video_path: Path) -> int:
    """Count number of frames in video file.

    Args:
        video_path: Path to video file.

    Returns:
        Number of frames in the video.

    Raises:
        ImportError: If cv2 is not installed.
        AssertionError: If video file doesn't exist.
    """
    import cv2  # pyright: ignore[reportMissingImports]

    assert video_path.exists(), f"Video file not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def get_video_info(video_path: Path) -> dict[str, int | float]:
    """Get video file information.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary with video info (width, height, fps, frame_count).

    Raises:
        ImportError: If cv2 is not installed.
        AssertionError: If video file doesn't exist.
    """
    import cv2  # pyright: ignore[reportMissingImports]

    assert video_path.exists(), f"Video file not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()

    return info


def verify_landmark_shape(
    landmarks: np.ndarray,
    expected_num_points: int,
    expected_coords: int = 3,
) -> None:
    """Verify landmark array shape.

    Args:
        landmarks: Landmark array to verify.
        expected_num_points: Expected number of landmark points.
        expected_coords: Expected number of coordinates per point
            (default: 3 for x,y,z).

    Raises:
        AssertionError: If shape doesn't match expectations.
    """
    assert (
        landmarks.ndim == 3
    ), f"Expected 3D array (frames, points, coords), got {landmarks.ndim}D"

    _, num_points, num_coords = landmarks.shape

    assert (
        num_points == expected_num_points
    ), f"Expected {expected_num_points} points, got {num_points}"

    assert (
        num_coords == expected_coords
    ), f"Expected {expected_coords} coordinates, got {num_coords}"
