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

"""Unit tests for lmpipe/collector/landmark_matrix/zarr_lmsc.py

Tests for ZarrLandmarkMatrixSaveCollector (dual mode: per-key and container).
Coverage target: 35% → 85%+
"""

from __future__ import annotations

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
import zarr
from pathlib import Path
from typing import Literal, Mapping

from cslrtools2.lmpipe.collector.landmark_matrix.zarr_lmsc import (
    ZarrLandmarkMatrixSaveCollector,
    zarr_lmsc_creator,
)
from cslrtools2.sldataset.utils import get_array
from cslrtools2.typings import NDArrayFloat, NDArrayStr


def _make_empty_headers[K: str](
    landmarks: Mapping[K, NDArrayFloat],
) -> dict[K, NDArrayStr]:
    """Create empty header mappings for landmarks (used when headers are not needed)."""
    return {key: np.array([], dtype=str) for key in landmarks.keys()}


def get_zarr_data(store: zarr.Group, key: str) -> np.ndarray:
    """Helper to get Zarr array data with proper type annotation."""
    return np.asarray(get_array(store, key)[:])  # type: ignore[index]


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


class TestZarrLMSCInitialization:
    """Tests for ZarrLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization_container_mode(self):
        """Test default initialization (container mode)."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.is_perkey is False
        assert collector.is_container is True
        assert collector.file_ext == ".zarr"
        assert collector.USE_LANDMARK_DIR is False

    def test_per_key_mode_initialization(self):
        """Test initialization with per_key=True."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)
        assert collector.is_perkey is True
        assert collector.is_container is False
        assert collector.USE_LANDMARK_DIR is True


class TestZarrLMSCContainerMode:
    """Tests for Zarr container mode (per_key=False)."""

    def test_save_single_key_container(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key in container mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify zarr directory exists
        zarr_dir = temp_output_dir / "landmarks.zarr"
        assert zarr_dir.exists()
        assert zarr_dir.is_dir()

        # Verify content
        store = zarr.open_group(zarr_dir, mode="r")
        assert "pose" in store
        data = get_zarr_data(store, "pose")

        assert data.shape == (1, 10, 33, 3)

    def test_save_multiple_keys_container(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys in container mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]](
            per_key=False
        )

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify single zarr directory with both keys
        zarr_dir = temp_output_dir / "landmarks.zarr"
        assert zarr_dir.exists()

        store = zarr.open_group(zarr_dir, mode="r")
        assert "pose" in store
        assert "left_hand" in store

        assert get_zarr_data(store, "pose").shape == (1, 5, 33, 3)
        assert get_zarr_data(store, "left_hand").shape == (1, 5, 21, 3)

    def test_multiple_appends_container(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple appends in container mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        zarr_dir = temp_output_dir / "landmarks.zarr"
        store = zarr.open_group(zarr_dir, mode="r")
        data = get_zarr_data(store, "pose")
        assert data.shape == (2, 10, 33, 3)


class TestZarrLMSCPerKeyMode:
    """Tests for Zarr per-key mode (per_key=True)."""

    def test_save_single_key_perkey(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key in per-key mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify landmarks directory and key-specific zarr directory
        landmarks_dir = temp_output_dir / "landmarks"
        assert landmarks_dir.exists()

        zarr_dir = landmarks_dir / "pose.zarr"
        assert zarr_dir.exists()
        assert zarr_dir.is_dir()

        # Verify content
        store = zarr.open_group(zarr_dir, mode="r")
        assert "data" in store
        data = get_zarr_data(store, "data")
        assert data.shape == (1, 10, 33, 3)

    def test_save_multiple_keys_perkey(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys in per-key mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]](
            per_key=True
        )

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify separate zarr directories for each key
        landmarks_dir = temp_output_dir / "landmarks"
        pose_zarr = landmarks_dir / "pose.zarr"
        hand_zarr = landmarks_dir / "left_hand.zarr"

        assert pose_zarr.exists()
        assert hand_zarr.exists()

        # Verify pose content
        pose_store = zarr.open_group(pose_zarr, mode="r")
        assert get_zarr_data(pose_store, "data").shape == (1, 5, 33, 3)

        # Verify hand content
        hand_store = zarr.open_group(hand_zarr, mode="r")
        assert get_zarr_data(hand_store, "data").shape == (1, 5, 21, 3)

    def test_multiple_appends_perkey(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple appends in per-key mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        zarr_dir = temp_output_dir / "landmarks" / "pose.zarr"
        store = zarr.open_group(zarr_dir, mode="r")
        data = get_zarr_data(store, "data")
        assert data.shape == (2, 10, 33, 3)


class TestZarrLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_array_container_mode(self, temp_output_dir: Path):
        """Test empty array in container mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        zarr_dir = temp_output_dir / "landmarks.zarr"
        store = zarr.open_group(zarr_dir, mode="r")
        data = get_zarr_data(store, "pose")
        assert data.shape == (1, 0, 33, 3)

    def test_empty_array_perkey_mode(self, temp_output_dir: Path):
        """Test empty array in per-key mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        zarr_dir = temp_output_dir / "landmarks" / "pose.zarr"
        store = zarr.open_group(zarr_dir, mode="r")
        data = get_zarr_data(store, "data")
        assert data.shape == (1, 0, 33, 3)

    def test_no_appends_container_mode(self, temp_output_dir: Path):
        """Test no appends in container mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        collector._open_file(temp_output_dir)
        # Don't append anything
        collector._close_file()

        zarr_dir = temp_output_dir / "landmarks.zarr"
        assert zarr_dir.exists()

        store = zarr.open_group(zarr_dir, mode="r")
        # Empty store has no keys
        assert len(list(store.keys())) == 0

    def test_no_appends_perkey_mode(self, temp_output_dir: Path):
        """Test no appends in per-key mode."""
        collector = ZarrLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        collector._open_file(temp_output_dir)
        # Manually add empty buffer to trigger file creation
        collector._buffer["pose"] = []
        collector._close_file()

        zarr_dir = temp_output_dir / "landmarks" / "pose.zarr"
        assert zarr_dir.exists()

        store = zarr.open_group(zarr_dir, mode="r")
        data = get_zarr_data(store, "data")
        assert data.shape == (0,)


class TestZarrLMSCCreators:
    """Tests for creator functions."""

    def test_zarr_lmsc_creator(self):
        """Test zarr_lmsc_creator function (default container mode)."""
        collector = zarr_lmsc_creator(str)
        assert isinstance(collector, ZarrLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".zarr"
        assert collector.is_container is True
        assert collector.is_perkey is False


class TestZarrLMSCEmptyBuffer:
    """Tests for empty buffer edge cases."""

    def test_close_file_with_empty_buffer_per_key_mode(self, temp_output_dir: Path):
        """Test _close_file in per-key mode with empty buffer."""
        from cslrtools2.lmpipe.runspec import RunSpec

        collector = ZarrLandmarkMatrixSaveCollector[str](per_key=True)

        video_file = temp_output_dir / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, temp_output_dir)

        # Collect with empty results
        collector.collect_results(runspec, [])

        # Verify landmarks directory was created but empty
        landmarks_dir = temp_output_dir / "landmarks"
        assert landmarks_dir.exists()
        assert landmarks_dir.is_dir()

        # No zarr files should be created for empty buffer
        assert len(list(landmarks_dir.glob("*.zarr"))) == 0

    def test_close_file_with_empty_buffer_container_mode(self, temp_output_dir: Path):
        """Test _close_file in container mode with empty buffer."""
        from cslrtools2.lmpipe.runspec import RunSpec

        collector = ZarrLandmarkMatrixSaveCollector[str](per_key=False)

        video_file = temp_output_dir / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, temp_output_dir)

        # Collect with empty results
        collector.collect_results(runspec, [])

        # Verify empty zarr directory was created
        zarr_dir = temp_output_dir / "landmarks.zarr"
        assert zarr_dir.exists()

        # Verify it's a valid empty zarr store
        store = zarr.open_group(zarr_dir, mode="r")
        assert len(list(store.keys())) == 0
