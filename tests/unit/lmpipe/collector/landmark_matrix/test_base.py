"""Unit tests for LandmarkMatrixSaveCollector base class."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]

# Direct import to avoid safetensors initialization issue
from cslrtools2.lmpipe.collector.landmark_matrix import base as lm_base
from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.options import DEFAULT_LMPIPE_OPTIONS
from cslrtools2.lmpipe.runspec import RunSpec
from cslrtools2.typings import NDArrayFloat


class MockPerKeyCollector(lm_base.LandmarkMatrixSaveCollector[str]):
    """Mock per-key collector for testing."""
    
    def __init__(self):
        self.collected_results: list[Mapping[str, NDArrayFloat]] = []
        self.file_opened = False
        self.file_closed = False
    
    @property
    def is_perkey(self) -> bool:
        return True
    
    @property
    def is_container(self) -> bool:
        return False
    
    @property
    def file_ext(self) -> str:
        return ".csv"
    
    def _open_file(self, path: Path):
        self.file_opened = True
    
    def _append_result(self, result: Mapping[str, NDArrayFloat]):
        self.collected_results.append(result)
    
    def _close_file(self):
        self.file_closed = True


class MockContainerCollector(lm_base.LandmarkMatrixSaveCollector[str]):
    """Mock container collector for testing."""
    
    def __init__(self):
        self.collected_results: list[Mapping[str, NDArrayFloat]] = []
        self.file_opened = False
        self.file_closed = False
    
    @property
    def is_perkey(self) -> bool:
        return False
    
    @property
    def is_container(self) -> bool:
        return True
    
    @property
    def file_ext(self) -> str:
        return ".npz"
    
    def _open_file(self, path: Path):
        self.file_opened = True
    
    def _append_result(self, result: Mapping[str, NDArrayFloat]):
        self.collected_results.append(result)
    
    def _close_file(self):
        self.file_closed = True


class TestLandmarkMatrixSaveCollectorPerKey:
    """Test per-key mode of LandmarkMatrixSaveCollector."""

    def test_configure_from_options(self):
        """Test configure_from_options sets exist_rule."""
        collector = MockPerKeyCollector()
        options = DEFAULT_LMPIPE_OPTIONS
        
        collector.configure_from_options(options)
        
        assert hasattr(collector, 'exist_rule')
        assert collector.exist_rule == "skip"  # Default value

    def test_get_output_path_perkey(self, tmp_path: Path):
        """Test _get_output_path returns directory for per-key mode."""
        collector = MockPerKeyCollector()
        collector.exist_rule = "overwrite"
        
        output_path = collector._get_output_path(tmp_path)
        
        assert output_path == tmp_path / "landmarks"
        assert output_path.is_dir()

    def test_collect_results_perkey(self, tmp_path: Path):
        """Test collect_results processes results correctly."""
        collector = MockPerKeyCollector()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        results = [
            ProcessResult(
                frame_id=0,
                headers={"pose": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"pose": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            ProcessResult(
                frame_id=1,
                headers={"pose": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"pose": np.array([[4.0, 5.0, 6.0]], dtype=np.float32)},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8)
            )
        ]
        
        collector.collect_results(runspec, results)
        
        assert collector.file_opened
        assert collector.file_closed
        assert len(collector.collected_results) == 2

    def test_apply_exist_rule_skip(self, tmp_path: Path):
        """Test apply_exist_rule with 'skip' mode."""
        collector = MockPerKeyCollector()
        collector.exist_rule = "skip"
        
        # Create existing directory
        landmarks_dir = tmp_path / "landmarks"
        landmarks_dir.mkdir()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should skip because directory exists
        assert collector.apply_exist_rule(runspec) is False

    def test_apply_exist_rule_overwrite(self, tmp_path: Path):
        """Test apply_exist_rule with 'overwrite' mode."""
        collector = MockPerKeyCollector()
        collector.exist_rule = "overwrite"
        
        # Create existing directory
        landmarks_dir = tmp_path / "landmarks"
        landmarks_dir.mkdir()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should overwrite
        assert collector.apply_exist_rule(runspec) is True

    def test_apply_exist_rule_error(self, tmp_path: Path):
        """Test apply_exist_rule with 'error' mode raises FileExistsError."""
        collector = MockPerKeyCollector()
        collector.exist_rule = "error"
        
        # Create existing directory
        landmarks_dir = tmp_path / "landmarks"
        landmarks_dir.mkdir()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should raise error
        with pytest.raises(FileExistsError):
            collector.apply_exist_rule(runspec)

    def test_apply_exist_rule_no_conflict(self, tmp_path: Path):
        """Test apply_exist_rule when landmarks directory doesn't exist initially."""
        collector = MockPerKeyCollector()
        collector.exist_rule = "overwrite"  # Use overwrite to always proceed
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        runspec = RunSpec(video_file, output_dir)
        
        # Should proceed (overwrite mode always returns True)
        assert collector.apply_exist_rule(runspec) is True


class TestLandmarkMatrixSaveCollectorContainer:
    """Test container mode of LandmarkMatrixSaveCollector."""

    def test_get_output_path_container(self, tmp_path: Path):
        """Test _get_output_path returns file for container mode."""
        collector = MockContainerCollector()
        collector.exist_rule = "overwrite"
        
        output_path = collector._get_output_path(tmp_path)
        
        assert output_path == tmp_path / "landmarks.npz"
        assert not output_path.exists()  # File not created yet

    def test_collect_results_container(self, tmp_path: Path):
        """Test collect_results for container mode."""
        collector = MockContainerCollector()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        results = [
            ProcessResult(
                frame_id=0,
                headers={"pose": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"pose": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8)
            )
        ]
        
        collector.collect_results(runspec, results)
        
        assert collector.file_opened
        assert collector.file_closed
        assert len(collector.collected_results) == 1

    def test_apply_exist_rule_container_skip(self, tmp_path: Path):
        """Test apply_exist_rule for container with existing file."""
        collector = MockContainerCollector()
        collector.exist_rule = "skip"
        
        # Create existing file
        landmarks_file = tmp_path / "landmarks.npz"
        landmarks_file.touch()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should skip because file exists
        assert collector.apply_exist_rule(runspec) is False

    def test_apply_exist_rule_container_overwrite(self, tmp_path: Path):
        """Test apply_exist_rule for container with overwrite."""
        collector = MockContainerCollector()
        collector.exist_rule = "overwrite"
        
        # Create existing file
        landmarks_file = tmp_path / "landmarks.npz"
        landmarks_file.touch()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should overwrite
        assert collector.apply_exist_rule(runspec) is True

    def test_apply_exist_rule_container_error(self, tmp_path: Path):
        """Test apply_exist_rule for container with error mode."""
        collector = MockContainerCollector()
        collector.exist_rule = "error"
        
        # Create existing file
        landmarks_file = tmp_path / "landmarks.npz"
        landmarks_file.touch()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Should raise error
        with pytest.raises(FileExistsError):
            collector.apply_exist_rule(runspec)


class TestLandmarkMatrixSaveCollectorHelpers:
    """Test helper methods of LandmarkMatrixSaveCollector."""

    def test_prepare_landmark_dir_with_use_landmark_dir(self, tmp_path: Path):
        """Test _prepare_landmark_dir creates landmark directory."""
        collector = MockPerKeyCollector()
        collector.USE_LANDMARK_DIR = True
        
        result = collector._prepare_landmark_dir(tmp_path)
        
        assert result == tmp_path / "landmarks"
        assert result.is_dir()

    def test_prepare_landmark_dir_without_use_landmark_dir(self, tmp_path: Path):
        """Test _prepare_landmark_dir uses dst directly."""
        # Create collector that doesn't use landmark dir
        class NoLandmarkDirCollector(MockContainerCollector):
            USE_LANDMARK_DIR = False
        
        collector = NoLandmarkDirCollector()
        
        result = collector._prepare_landmark_dir(tmp_path)
        
        assert result == tmp_path
        assert result.is_dir()

    def test_get_landmark_file_path_with_landmark_dir(self, tmp_path: Path):
        """Test _get_landmark_file_path with landmark directory."""
        collector = MockPerKeyCollector()
        collector.USE_LANDMARK_DIR = True
        
        result = collector._get_landmark_file_path(tmp_path, "test.csv")
        
        assert result == tmp_path / "landmarks" / "test.csv"

    def test_get_landmark_file_path_without_landmark_dir(self, tmp_path: Path):
        """Test _get_landmark_file_path without landmark directory."""
        # Create collector that doesn't use landmark dir
        class NoLandmarkDirCollector(MockContainerCollector):
            USE_LANDMARK_DIR = False
        
        collector = NoLandmarkDirCollector()
        
        result = collector._get_landmark_file_path(tmp_path, "test.npz")
        
        assert result == tmp_path / "test.npz"


class TestLandmarkMatrixSaveCollectorSuffix:
    """Test suffix mode of exist_rule (TODO implementation)."""

    def test_apply_exist_rule_suffix(self, tmp_path: Path):
        """Test apply_exist_rule with 'suffix' mode (currently overwrites)."""
        collector = MockContainerCollector()
        collector.exist_rule = "suffix"
        
        # Create existing file
        landmarks_file = tmp_path / "landmarks.npz"
        landmarks_file.touch()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)
        
        # Current implementation returns True (TODO: implement suffix logic)
        assert collector.apply_exist_rule(runspec) is True
