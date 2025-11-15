"""Unit tests for lmpipe/collector/landmark_matrix/npz_lmsc.py

Tests for NpzLandmarkMatrixSaveCollector (container format).
Coverage target: 50% → 85%+
"""
from __future__ import annotations

import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
from pathlib import Path
from typing import Literal

from cslrtools2.lmpipe.collector.landmark_matrix.npz_lmsc import (
    NpzLandmarkMatrixSaveCollector,
    npz_lmsc_creator,
)
from cslrtools2.typings import NDArrayFloat


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
    return {
        "pose": np.random.rand(10, 33, 3).astype(np.float32)
    }


@pytest.fixture
def sample_multi_key_result() -> dict[Literal["pose", "left_hand"], NDArrayFloat]:
    """Create a sample result with multiple keys."""
    np.random.seed(42)
    return {
        "pose": np.random.rand(5, 33, 3).astype(np.float32),
        "left_hand": np.random.rand(5, 21, 3).astype(np.float32),
    }


class TestNPZLMSCInitialization:
    """Tests for NpzLandmarkMatrixSaveCollector initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.is_perkey is False  # Container format
        assert collector.is_container is True  # All keys in one file
        assert collector.file_ext == ".npz"
        assert collector.USE_LANDMARK_DIR is False


class TestNPZLMSCFileOperations:
    """Tests for NPZ file operations."""
    
    def test_save_single_key_npz(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat]
    ):
        """Test saving single key to NPZ file."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(sample_single_key_result)
        finally:
            collector._close_file()

        # Verify file exists (landmarks.npz, not landmarks/pose.npz)
        npz_file = temp_output_dir / "landmarks.npz"
        assert npz_file.exists()

        # Verify content
        with np.load(npz_file, allow_pickle=True) as data:
            assert "pose" in data
            assert data["pose"].shape == (1, 10, 33, 3)  # (num_appends, frames, landmarks, coords)
    
    def test_save_multiple_keys_npz(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat]
    ):
        """Test saving multiple keys to single NPZ file."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(sample_multi_key_result)
        finally:
            collector._close_file()

        # Verify single file with both keys
        npz_file = temp_output_dir / "landmarks.npz"
        assert npz_file.exists()

        with np.load(npz_file, allow_pickle=True) as data:
            assert "pose" in data
            assert "left_hand" in data
            assert data["pose"].shape == (1, 5, 33, 3)
            assert data["left_hand"].shape == (1, 5, 21, 3)
    
    def test_multiple_append_calls(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat]
    ):
        """Test multiple append calls stack arrays."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(sample_single_key_result)
            collector._append_result(sample_single_key_result)  # Append again
            collector._append_result(sample_single_key_result)  # And again
        finally:
            collector._close_file()

        npz_file = temp_output_dir / "landmarks.npz"
        with np.load(npz_file, allow_pickle=True) as data:
            # Should stack 3 arrays
            assert data["pose"].shape == (3, 10, 33, 3)


class TestNPZLMSCEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_landmark_array(self, temp_output_dir: Path):
        """Test handling empty landmark arrays."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()
        
        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }
        
        collector._open_file(temp_output_dir)
        try:
            collector._append_result(empty_result)
        finally:
            collector._close_file()
        
        npz_file = temp_output_dir / "landmarks.npz"
        assert npz_file.exists()
        
        with np.load(npz_file, allow_pickle=True) as data:
            assert "pose" in data
            # Stacked empty array
            assert data["pose"].shape == (1, 0, 33, 3)
    
    def test_no_appends_creates_empty_npz(self, temp_output_dir: Path):
        """Test that no appends creates an empty NPZ file."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()
        
        # Open and close without appending
        collector._open_file(temp_output_dir)
        # Don't append anything
        collector._close_file()
        
        npz_file = temp_output_dir / "landmarks.npz"
        assert npz_file.exists()
        
        with np.load(npz_file, allow_pickle=True) as data:
            # Empty NPZ file has no keys
            assert len(data.files) == 0
    
    def test_single_frame_landmark(self, temp_output_dir: Path):
        """Test handling single frame."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()

        single_frame: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(1, 33, 3).astype(np.float32)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(single_frame)
        finally:
            collector._close_file()

        npz_file = temp_output_dir / "landmarks.npz"
        with np.load(npz_file, allow_pickle=True) as data:
            assert data["pose"].shape == (1, 1, 33, 3)
    
    def test_nested_output_directory_creation(self, temp_output_dir: Path):
        """Test creating nested output directories."""
        nested_path = temp_output_dir / "a" / "b" / "c"
        nested_path.mkdir(parents=True)  # NPZ doesn't use landmarks/ dir, so create parent
        
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose"]]()
        
        result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32)
        }
        
        collector._open_file(nested_path)
        try:
            collector._append_result(result)
        finally:
            collector._close_file()
        
        # NPZ is a container file, so it goes directly to the path
        npz_file = nested_path / "landmarks.npz"
        assert npz_file.exists()
    
    def test_mixed_keys_across_appends(self, temp_output_dir: Path):
        """Test appending different keys across multiple append calls."""
        collector = NpzLandmarkMatrixSaveCollector[Literal["pose", "left_hand", "right_hand"]]()
        
        result1: dict[Literal["pose", "left_hand", "right_hand"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32),
            "left_hand": np.random.rand(2, 21, 3).astype(np.float32),
        }
        result2: dict[Literal["pose", "left_hand", "right_hand"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32),
            "right_hand": np.random.rand(2, 21, 3).astype(np.float32),
        }
        
        collector._open_file(temp_output_dir)
        try:
            collector._append_result(result1)
            collector._append_result(result2)
        finally:
            collector._close_file()
        
        npz_file = temp_output_dir / "landmarks.npz"
        with np.load(npz_file, allow_pickle=True) as data:
            # pose should be stacked twice
            assert data["pose"].shape == (2, 2, 33, 3)
            # left_hand and right_hand each once
            assert data["left_hand"].shape == (1, 2, 21, 3)
            assert data["right_hand"].shape == (1, 2, 21, 3)


class TestNPZLMSCCreators:
    """Tests for creator functions."""
    
    def test_npz_lmsc_creator(self):
        """Test npz_lmsc_creator function."""
        collector = npz_lmsc_creator(str)
        assert isinstance(collector, NpzLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".npz"
        assert collector.is_container is True
