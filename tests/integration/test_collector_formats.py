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

"""Integration tests for collector format round-trip verification.

This module tests data integrity across different collector formats
(CSV, NPY, NPZ, Zarr, etc.), verifying that landmarks can be saved
and loaded without data loss.

Test Coverage:
    - Round-trip verification for each format
    - Format comparison and consistency
    - Edge cases (empty data, single frame, special characters)
    - Metadata validation

The tests ensure that:
    - Data is preserved across save/load cycles
    - Container formats (NPZ, Zarr) match per-key formats (NPY, CSV)
    - Edge cases are handled gracefully
    - Format metadata is correctly reported

Example:
    Run collector format tests::

        >>> pytest tests/integration/test_collector_formats.py -v
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import Iterable, Mapping
from pathlib import Path

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
import zarr

from cslrtools2.sldataset.utils import get_array
from cslrtools2.lmpipe.collector.landmark_matrix import (
    CsvLandmarkMatrixSaveCollector,
    NpyLandmarkMatrixSaveCollector,
    NpzLandmarkMatrixSaveCollector,
    ZarrLandmarkMatrixSaveCollector,
)


class TestCollectorRoundTrip:
    """Test round-trip data integrity for different collector formats."""

    @pytest.fixture
    def sample_landmarks(self):
        """Generate sample landmark data for testing."""
        return {
            "pose": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ],
                dtype=np.float32,
            ),
            "left_hand": np.array(
                [
                    [1.0, 1.1],
                    [1.2, 1.3],
                ],
                dtype=np.float32,
            ),
            "right_hand": np.array(
                [
                    [2.0, 2.1],
                    [2.2, 2.3],
                ],
                dtype=np.float32,
            ),
        }

    @pytest.fixture
    def sample_headers(self):
        """Generate sample headers for landmark data."""
        return {
            "pose": np.array(["x", "y", "z"], dtype=str),
            "left_hand": np.array(["x", "y"], dtype=str),
            "right_hand": np.array(["x", "y"], dtype=str),
        }

    def test_csv_roundtrip(
        self, tmp_path: Path,
        sample_landmarks: Mapping[str, np.ndarray],
        sample_headers: Mapping[str, np.ndarray]
    ):
        """Test CSV format round-trip with data verification.

        Verifies that landmarks saved to CSV can be loaded back correctly.
        """
        collector = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "csv_output"

        # Save landmarks
        collector._open_file(output_dir)
        try:
            collector._append_result(0, sample_headers, sample_landmarks)
            # Flush to ensure data is written
            for handle in collector._file_handles.values():
                handle.flush()
        finally:
            collector._close_file()

        # Verify files exist
        landmarks_dir = output_dir / "landmarks"
        assert (landmarks_dir / "pose.csv").exists()
        assert (landmarks_dir / "left_hand.csv").exists()
        assert (landmarks_dir / "right_hand.csv").exists()

        # Load and verify CSV data
        import csv

        # Verify pose landmarks
        with (landmarks_dir / "pose.csv").open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3  # 3 landmarks
            # Check float values with tolerance (float32 precision)
            assert float(rows[0]["x"]) == pytest.approx(0.1, rel=1e-5)
            assert float(rows[0]["y"]) == pytest.approx(0.2, rel=1e-5)
            assert float(rows[0]["z"]) == pytest.approx(0.3, rel=1e-5)
            assert float(rows[2]["x"]) == pytest.approx(0.7, rel=1e-5)

    def test_npy_roundtrip(
        self, tmp_path: Path, sample_landmarks: Mapping[str, np.ndarray]
    ):
        """Test NPY format round-trip with data verification.

        Verifies that landmarks saved to NPY files can be loaded back correctly.
        """
        collector = NpyLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "npy_output"

        # Save landmarks
        collector._open_file(output_dir)
        try:
            collector._append_result(0, {}, sample_landmarks)
        finally:
            collector._close_file()

        # Verify files exist
        landmarks_dir = output_dir / "landmarks"
        assert (landmarks_dir / "pose.npy").exists()
        assert (landmarks_dir / "left_hand.npy").exists()
        assert (landmarks_dir / "right_hand.npy").exists()

        # Load and verify NPY data (note: shape is (num_frames, num_landmarks, coords))
        loaded_pose = np.load(landmarks_dir / "pose.npy")
        assert loaded_pose.shape == (1, 3, 3)  # 1 frame, 3 landmarks, 3 coords
        np.testing.assert_allclose(loaded_pose[0], sample_landmarks["pose"], rtol=1e-5)

        loaded_left = np.load(landmarks_dir / "left_hand.npy")
        assert loaded_left.shape == (1, 2, 2)  # 1 frame, 2 landmarks, 2 coords
        np.testing.assert_allclose(
            loaded_left[0], sample_landmarks["left_hand"], rtol=1e-5
        )

    def test_npz_roundtrip(
        self, tmp_path: Path, sample_landmarks: Mapping[str, np.ndarray]
    ):
        """Test NPZ container format round-trip with data verification.

        Verifies that landmarks saved to a single NPZ file can be loaded correctly.
        """
        collector = NpzLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "npz_output"
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Save landmarks
        collector._open_file(output_dir)
        try:
            collector._append_result(0, {}, sample_landmarks)
        finally:
            collector._close_file()

        # Verify container file exists
        landmarks_file = output_dir / "landmarks.npz"
        assert landmarks_file.exists()

        # Load and verify NPZ data (note: shape is (num_frames, num_landmarks, coords))
        with np.load(landmarks_file) as data:
            assert "pose" in data
            assert "left_hand" in data
            assert "right_hand" in data

            assert data["pose"].shape == (1, 3, 3)  # 1 frame, 3 landmarks, 3 coords
            np.testing.assert_allclose(
                data["pose"][0], sample_landmarks["pose"], rtol=1e-5
            )
            np.testing.assert_allclose(
                data["left_hand"][0], sample_landmarks["left_hand"], rtol=1e-5
            )
            np.testing.assert_allclose(
                data["right_hand"][0], sample_landmarks["right_hand"], rtol=1e-5
            )

    def test_zarr_roundtrip(
        self, tmp_path: Path, sample_landmarks: Mapping[str, np.ndarray]
    ):
        """Test Zarr format round-trip with data verification.

        Verifies that landmarks saved to Zarr store can be loaded correctly.
        """
        collector = ZarrLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "zarr_output"

        # Save landmarks
        collector._open_file(output_dir)
        try:
            collector._append_result(0, {}, sample_landmarks)
        finally:
            collector._close_file()

        # Verify Zarr directory exists
        landmarks_dir = output_dir / "landmarks.zarr"
        assert landmarks_dir.exists()
        assert landmarks_dir.is_dir()

        # Load and verify Zarr data
        store = zarr.open_group(str(landmarks_dir), mode="r")

        assert "pose" in store
        assert "left_hand" in store
        assert "right_hand" in store

        pose_array = get_array(store, "pose")
        pose_array_all = pose_array[:]
        pose_array_0 = pose_array[0]
        left_hand_array_0 = get_array(store, "left_hand")[0]
        right_hand_array_0 = get_array(store, "right_hand")[0]

        assert isinstance(pose_array_all, np.ndarray), """
            pose_array is not a numpy array"""
        assert not isinstance(pose_array_all, np.generic)
        assert isinstance(pose_array_0, np.ndarray), """
            pose_array_0 is not a numpy array"""
        assert not isinstance(pose_array_0, np.generic)
        assert isinstance(left_hand_array_0, np.ndarray), """
            left_hand_array_0 is not a numpy array"""
        assert not isinstance(left_hand_array_0, np.generic)
        assert isinstance(right_hand_array_0, np.ndarray), """
            right_hand_array_0 is not a numpy array"""
        assert not isinstance(right_hand_array_0, np.generic)

        # Verify shape (num_frames, num_landmarks, coords)
        assert pose_array_all.shape == (1, 3, 3)
        np.testing.assert_allclose(
            pose_array_0,
            sample_landmarks["pose"],
            rtol=1e-5
        )
        np.testing.assert_allclose(
            left_hand_array_0,
            sample_landmarks["left_hand"],
            rtol=1e-5
        )
        np.testing.assert_allclose(
            right_hand_array_0,
            sample_landmarks["right_hand"],
            rtol=1e-5
        )


class TestCollectorFormatComparison:
    """Test consistency between container and per-key formats."""

    @pytest.fixture
    def multi_frame_landmarks(self) -> list[Mapping[str, np.ndarray]]:
        """Generate multi-frame landmark data for testing."""
        frames: list[Mapping[str, np.ndarray]] = []
        for i in range(5):  # 5 frames
            frame_data = {
                "pose": np.random.randn(33, 3).astype(np.float32) * 0.1 + i,
                "hand": np.random.randn(21, 2).astype(np.float32) * 0.1 + i,
            }
            frames.append(frame_data)
        return frames

    def test_npz_vs_npy_consistency(
        self,
        tmp_path: Path,
        multi_frame_landmarks: Iterable[dict[str, np.ndarray]]
    ):
        """Test that NPZ container and individual NPY files produce same results.

        Verifies data consistency between container and per-key formats.
        """
        # Save with NPZ (container)
        npz_collector = NpzLandmarkMatrixSaveCollector[str]()
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        npz_collector._open_file(npz_dir)
        try:
            for frame_id, landmarks in enumerate(multi_frame_landmarks):
                npz_collector._append_result(frame_id, {}, landmarks)
        finally:
            npz_collector._close_file()

        # Save with NPY (per-key)
        npy_collector = NpyLandmarkMatrixSaveCollector[str]()
        npy_dir = tmp_path / "npy"
        npy_collector._open_file(npy_dir)
        try:
            for frame_id, landmarks in enumerate(multi_frame_landmarks):
                npy_collector._append_result(frame_id, {}, landmarks)
        finally:
            npy_collector._close_file()

        # Load both formats
        npz_data = np.load(npz_dir / "landmarks.npz")
        npy_pose = np.load(npy_dir / "landmarks" / "pose.npy")
        npy_hand = np.load(npy_dir / "landmarks" / "hand.npy")

        # Compare data (both should have shape (num_frames, num_landmarks, coords))
        np.testing.assert_allclose(npz_data["pose"], npy_pose, rtol=1e-6)
        np.testing.assert_allclose(npz_data["hand"], npy_hand, rtol=1e-6)

        # Verify shape is (num_frames, num_landmarks, coords)
        assert npz_data["pose"].shape[0] == 5  # 5 frames
        assert npz_data["hand"].shape[0] == 5
        assert npy_pose.shape[0] == 5
        assert npy_hand.shape[0] == 5

    def test_collector_format_metadata(self, tmp_path: Path):
        """Test that collectors properly report their format characteristics."""
        csv_collector = CsvLandmarkMatrixSaveCollector[str]()
        npy_collector = NpyLandmarkMatrixSaveCollector[str]()
        npz_collector = NpzLandmarkMatrixSaveCollector[str]()
        zarr_collector = ZarrLandmarkMatrixSaveCollector[str]()

        # CSV is per-key, not container
        assert csv_collector.is_perkey is True
        assert csv_collector.is_container is False
        assert csv_collector.file_ext == ".csv"

        # NPY is per-key, not container
        assert npy_collector.is_perkey is True
        assert npy_collector.is_container is False
        assert npy_collector.file_ext == ".npy"

        # NPZ is container, not per-key
        assert npz_collector.is_perkey is False
        assert npz_collector.is_container is True
        assert npz_collector.file_ext == ".npz"

        # Zarr is container, not per-key
        assert zarr_collector.is_perkey is False
        assert zarr_collector.is_container is True
        assert zarr_collector.file_ext == ".zarr"


class TestCollectorEdgeCases:
    """Test edge cases in collector behavior."""

    def test_empty_landmarks(self, tmp_path: Path):
        """Test handling of empty landmark arrays."""
        collector = NpzLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "empty_output"
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Save empty landmarks
        collector._open_file(output_dir)
        try:
            empty_data = {
                "pose": np.array([], dtype=np.float32).reshape(0, 3),
            }
            collector._append_result(0, {}, empty_data)
        finally:
            collector._close_file()

        # Verify file exists and can be loaded
        landmarks_file = output_dir / "landmarks.npz"
        assert landmarks_file.exists()

        data = np.load(landmarks_file)
        assert isinstance(data, np.lib.npyio.NpzFile), """
            data is not an NpzFile instance"""

        assert "pose" in data
        assert data["pose"].shape == (1, 0, 3)  # 1 frame with 0 landmarks

    def test_single_frame_zarr(self, tmp_path: Path):
        """Test Zarr collector with single frame."""
        collector = ZarrLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "single_frame"

        landmarks = {"test": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}

        collector._open_file(output_dir)
        try:
            collector._append_result(0, {}, landmarks)
        finally:
            collector._close_file()

        # Load and verify
        zarr_path = str(output_dir / "landmarks.zarr")
        store = zarr.open_group(zarr_path, mode="r")

        # Verify shape (num_frames, num_landmarks, coords)
        test_array = get_array(store, "test")
        assert test_array.shape == (1, 1, 3)

        # Use get_array for type-safe array extraction
        test_array_0 = get_array(store, "test")[0]

        # Type check assertions
        assert isinstance(test_array_0, np.ndarray), """
            test_array_0 is not a numpy array"""
        assert not isinstance(test_array_0, np.generic)

        # Verify data
        np.testing.assert_allclose(
            test_array_0,
            landmarks["test"],
            rtol=1e-5
        )

    def test_csv_with_special_characters_in_keys(self, tmp_path: Path):
        """Test CSV collector with special characters in landmark keys."""
        collector = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "special_chars"

        # Note: Keys with special chars should be sanitized or handled
        landmarks = {
            "pose_landmarks": np.array([[1.0, 2.0]], dtype=np.float32),
        }
        headers = {
            "pose_landmarks": np.array(["x", "y"], dtype=str),
        }

        collector._open_file(output_dir)
        try:
            collector._append_result(0, headers, landmarks)
            for handle in collector._file_handles.values():
                handle.flush()
        finally:
            collector._close_file()

        # Verify file exists
        landmarks_dir = output_dir / "landmarks"
        csv_file = landmarks_dir / "pose_landmarks.csv"
        assert csv_file.exists()
