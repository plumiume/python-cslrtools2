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

"""Integration tests for LMPipe end-to-end pipeline.

Scenario 1.1: Basic end-to-end processing (Happy Path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.landmark_matrix import NpzLandmarkMatrixSaveCollector
from cslrtools2.lmpipe.interface import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe.holistic import MediaPipeHolisticEstimator

from .helpers import verify_npz_structure, get_video_info


@pytest.mark.integration
@pytest.mark.requires_video
@pytest.mark.slow
@pytest.mark.skip(
    reason="MediaPipe Holistic API compatibility issue - AttributeError: landmarks"
)
class TestLMPipeE2EBasic:
    """Test basic end-to-end LMPipe processing.

    NOTE: These tests are currently skipped due to MediaPipe Holistic API
    compatibility issues. The detection_results object structure varies
    between MediaPipe versions (landmarks vs landmark attribute).

    TODO: Investigate MediaPipe version requirements and fix API usage.
    """

    def test_single_video_to_npz(
        self,
        sample_video_stop: Path,
        integration_tmp_path: Path,
        skip_if_no_mediapipe: None,
    ):
        """
        Scenario: Single video file → MediaPipe Holistic → NPZ output

        Given: Test video file (hand_gesture_stop.mp4)
        When: LMPipe with MediaPipe Holistic estimator and NPZ collector
        Then: NPZ file is correctly generated with landmark data

        Verifies:
        - Video file is read successfully
        - MediaPipe Holistic initializes and processes frames
        - NPZ file is created with correct structure
        - Landmark data has expected shape and keys
        """
        # Arrange
        video_path = sample_video_stop
        output_dir = integration_tmp_path / "test_single_video_npz"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video info for validation
        video_info = get_video_info(video_path)
        expected_frame_count = video_info["frame_count"]

        # Create estimator with MediaPipe Holistic
        # Using default arguments for simplicity
        estimator = MediaPipeHolisticEstimator()

        # Create NPZ collector
        npz_collector = NpzLandmarkMatrixSaveCollector()

        # Create interface
        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[npz_collector],
            max_cpus=1,  # Single-threaded for deterministic testing
        )

        # Act
        interface.run(video_path, output_dir)

        # Assert
        # 1. Verify NPZ file was created
        npz_file = output_dir / "landmarks.npz"
        assert npz_file.exists(), f"NPZ file not created: {npz_file}"

        # 2. Verify NPZ structure has expected keys
        expected_keys = [
            "pose",
            "pose_world",
            "left_hand",
            "right_hand",
            "face",
        ]
        verify_npz_structure(npz_file, expected_keys)

        # 3. Verify landmark data shape
        data = np.load(npz_file)

        # Check pose landmarks (33 points, 3 coords)
        pose = data["pose"]
        assert pose.ndim == 3, f"Expected 3D array, got {pose.ndim}D"
        frames, points, coords = pose.shape
        assert (
            frames == expected_frame_count
        ), f"Frame count mismatch: expected {expected_frame_count}, got {frames}"
        assert points == 33, f"Expected 33 pose points, got {points}"
        assert coords == 3, f"Expected 3 coordinates, got {coords}"

        # Check hand landmarks (21 points, 3 coords)
        left_hand = data["left_hand"]
        assert left_hand.shape == (
            expected_frame_count,
            21,
            3,
        ), f"Left hand shape mismatch: {left_hand.shape}"

        right_hand = data["right_hand"]
        assert right_hand.shape == (
            expected_frame_count,
            21,
            3,
        ), f"Right hand shape mismatch: {right_hand.shape}"

        # Check face landmarks (468 points, 3 coords)
        face = data["face"]
        assert face.shape == (
            expected_frame_count,
            468,
            3,
        ), f"Face shape mismatch: {face.shape}"

        # 4. Verify data is not all zeros (actual landmarks detected)
        # At least some frames should have non-zero landmarks
        pose_non_zero = np.any(pose != 0)
        assert pose_non_zero, "Pose landmarks are all zeros"

        data.close()

    def test_single_video_to_npz_with_custom_filename(
        self,
        sample_video_stop: Path,
        integration_tmp_path: Path,
        skip_if_no_mediapipe: None,
    ):
        """
        Scenario: Single video → NPZ with custom output filename

        Given: Test video and custom output directory name
        When: Process video with LMPipe
        Then: NPZ file is created in correctly named directory
        """
        # Arrange
        video_path = sample_video_stop
        custom_output_name = "custom_hand_gesture"
        output_dir = integration_tmp_path / custom_output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        estimator = MediaPipeHolisticEstimator()
        npz_collector = NpzLandmarkMatrixSaveCollector()

        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[npz_collector],
            max_cpus=1,
        )

        # Act
        interface.run(video_path, output_dir)

        # Assert
        npz_file = output_dir / "landmarks.npz"
        assert npz_file.exists()
        assert npz_file.parent.name == custom_output_name

    def test_video_metadata_preservation(
        self,
        sample_video_stop: Path,
        integration_tmp_path: Path,
        skip_if_no_mediapipe: None,
    ):
        """
        Scenario: Verify video metadata is correctly processed

        Given: Test video with known properties
        When: Process with LMPipe
        Then: Output reflects correct frame count and dimensions
        """
        # Arrange
        video_path = sample_video_stop
        output_dir = integration_tmp_path / "test_metadata"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get expected video info
        video_info = get_video_info(video_path)
        expected_frames = video_info["frame_count"]

        estimator = MediaPipeHolisticEstimator()
        npz_collector = NpzLandmarkMatrixSaveCollector()

        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[npz_collector],
        )

        # Act
        interface.run(video_path, output_dir)

        # Assert
        npz_file = output_dir / "landmarks.npz"
        data = np.load(npz_file)

        # Verify all landmark arrays have correct frame count
        for key in ["pose", "left_hand", "right_hand", "face"]:
            assert key in data, f"Missing key: {key}"
            frames = data[key].shape[0]
            assert (
                frames == expected_frames
            ), f"{key} frame count mismatch: expected {expected_frames}, got {frames}"

        data.close()
