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

"""Unit tests for lmpipe/collector/landmark_matrix/npy_lmsc.py

Tests for NpyLandmarkMatrixSaveCollector.
Coverage target: 39% → 85%+
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
from pathlib import Path
from typing import Literal, Mapping

from cslrtools2.lmpipe.collector.landmark_matrix.npy_lmsc import (
    NpyLandmarkMatrixSaveCollector,
    npy_lmsc_creator,
)
from cslrtools2.typings import NDArrayFloat, NDArrayStr


def _make_empty_headers[K: str](
    landmarks: Mapping[K, NDArrayFloat],
) -> dict[K, NDArrayStr]:
    """Create empty header mappings for landmarks (used when headers are not needed)."""
    return {key: np.array([], dtype=str) for key in landmarks.keys()}


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_single_key_result() -> dict[Literal["pose"], NDArrayFloat]:
    """Create a sample result with a single key."""
    np.random.seed(42)
    return {"pose": np.random.rand(10, 33, 3).astype(np.float32)}


@pytest.fixture
def sample_multi_key_result() -> dict[Literal["pose", "left_hand"], NDArrayFloat]:
    """Create a sample result with multiple keys."""
    np.random.seed(42)
    return {
        "pose": np.random.rand(5, 33, 3).astype(np.float32),
        "left_hand": np.random.rand(5, 21, 3).astype(np.float32),
    }


class TestNPYLMSCInitialization:
    """Tests for NpyLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.is_perkey is True
        assert collector.is_container is False
        assert collector.file_ext == ".npy"


class TestNPYLMSCFileOperations:
    """Tests for NPY file operations."""

    def test_save_single_key_npy(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key to NPY file."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify file exists
        npy_file = temp_output_dir / "landmarks" / "pose.npy"
        assert npy_file.exists()

        # Verify content
        data = np.load(npy_file)
        assert data.shape == (1, 10, 33, 3)  # (num_appends, frames, landmarks, coords)

    def test_save_multiple_keys_npy(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys to separate NPY files."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify both files exist
        pose_file = temp_output_dir / "landmarks" / "pose.npy"
        hand_file = temp_output_dir / "landmarks" / "left_hand.npy"
        assert pose_file.exists()
        assert hand_file.exists()

        # Verify pose content
        pose_data = np.load(pose_file)
        assert pose_data.shape == (1, 5, 33, 3)

        # Verify hand content
        hand_data = np.load(hand_file)
        assert hand_data.shape == (1, 5, 21, 3)

    def test_multiple_append_calls(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple append calls stack arrays."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
            collector._append_result(
                1,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )  # Append again
            collector._append_result(
                2,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )  # And again
        finally:
            collector._close_file()

        npy_file = temp_output_dir / "landmarks" / "pose.npy"
        data = np.load(npy_file)
        # Should stack 3 arrays
        assert data.shape == (3, 10, 33, 3)


class TestNPYLMSCErrorHandling:
    """Tests for error handling."""

    def test_inconsistent_shapes_raises_error(self, temp_output_dir: Path):
        """Test that inconsistent shapes raise ValueError with helpful message."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        result1: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(10, 33, 3).astype(np.float32)
        }
        result2: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(5, 33, 3).astype(np.float32)  # Different shape
        }

        collector._open_file(temp_output_dir)
        try:
            with pytest.raises(ValueError, match="with shape.*at key 'pose'"):
                collector._append_result(0, _make_empty_headers(result1), result1)
                collector._append_result(0, _make_empty_headers(result2), result2)
                collector._close_file()
        finally:
            # Ensure cleanup even if test fails
            if collector._base_dir is not None:
                collector._base_dir = None
                collector._buffer = {}


class TestNPYLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_landmark_array(self, temp_output_dir: Path):
        """Test handling empty landmark arrays."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        npy_file = temp_output_dir / "landmarks" / "pose.npy"
        assert npy_file.exists()

        data = np.load(npy_file)
        # Stacked empty array
        assert data.shape == (1, 0, 33, 3)

    def test_no_appends_creates_empty_file(self, temp_output_dir: Path):
        """Test that no appends creates an empty NPY file."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        # Open and close without appending
        collector._open_file(temp_output_dir)
        # Manually add empty buffer entry
        collector._buffer["pose"] = []
        collector._close_file()

        npy_file = temp_output_dir / "landmarks" / "pose.npy"
        assert npy_file.exists()

        data = np.load(npy_file)
        assert data.shape == (0,)  # Empty 1D array

    def test_single_frame_landmark(self, temp_output_dir: Path):
        """Test handling single frame."""
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        single_frame: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(1, 33, 3).astype(np.float32)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(single_frame), single_frame)
        finally:
            collector._close_file()

        npy_file = temp_output_dir / "landmarks" / "pose.npy"
        data = np.load(npy_file)
        assert data.shape == (1, 1, 33, 3)

    def test_nested_output_directory_creation(self, temp_output_dir: Path):
        """Test creating nested output directories."""
        nested_path = temp_output_dir / "a" / "b" / "c" / "test.npy"
        collector = NpyLandmarkMatrixSaveCollector[Literal["pose"]]()

        result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32)
        }

        collector._open_file(nested_path.parent)
        try:
            collector._append_result(0, _make_empty_headers(result), result)
        finally:
            collector._close_file()

        npy_file = temp_output_dir / "a" / "b" / "c" / "landmarks" / "pose.npy"
        assert npy_file.exists()


class TestNPYLMSCCreators:
    """Tests for creator functions."""

    def test_npy_lmsc_creator(self):
        """Test npy_lmsc_creator function."""
        collector = npy_lmsc_creator(str)
        assert isinstance(collector, NpyLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".npy"
