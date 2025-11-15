"""Unit tests for lmpipe/runspec.py

Tests for RunSpec class creation and validation.
Coverage target: 52% → 100%
"""
from __future__ import annotations

import pytest
from pathlib import Path

from cslrtools2.lmpipe.runspec import RunSpec
from cslrtools2.exceptions import VideoProcessingError


@pytest.fixture
def temp_video_file(tmp_path: Path) -> Path:
    """Fixture for temporary video file."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_text("fake video content")
    return video_path


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Fixture for temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


class TestRunSpecDirectCreation:
    """Test direct RunSpec instantiation."""

    def test_create_with_path_source(self, temp_video_file: Path, temp_output_dir: Path):
        """Test creating RunSpec with Path source."""
        runspec = RunSpec(temp_video_file, temp_output_dir)
        
        assert runspec.src == temp_video_file
        assert runspec.dst == temp_output_dir
        assert isinstance(runspec.src, Path)

    def test_create_with_int_source(self, temp_output_dir: Path):
        """Test creating RunSpec with int (camera) source."""
        runspec = RunSpec(0, temp_output_dir)
        
        assert runspec.src == 0
        assert runspec.dst == temp_output_dir
        assert isinstance(runspec.src, int)


class TestRunSpecFromPathLikes:
    """Test RunSpec.from_pathlikes() factory method."""

    def test_from_pathlikes_with_existing_file(
        self,
        temp_video_file: Path,
        temp_output_dir: Path
    ):
        """Test creating RunSpec from path-like objects with existing file."""
        runspec = RunSpec.from_pathlikes(temp_video_file, temp_output_dir)
        
        assert runspec.src == temp_video_file
        assert runspec.dst == temp_output_dir
        assert isinstance(runspec.src, Path)
        assert isinstance(runspec.dst, Path)

    def test_from_pathlikes_with_string_paths(
        self,
        temp_video_file: Path,
        temp_output_dir: Path
    ):
        """Test creating RunSpec from string paths."""
        runspec = RunSpec.from_pathlikes(str(temp_video_file), str(temp_output_dir))
        
        assert runspec.src == temp_video_file
        assert runspec.dst == temp_output_dir

    def test_from_pathlikes_nonexistent_source_raises_error(
        self,
        tmp_path: Path
    ):
        """Test that nonexistent source path raises VideoProcessingError."""
        nonexistent_src = tmp_path / "nonexistent_video.mp4"
        output_dir = tmp_path / "output"
        
        with pytest.raises(VideoProcessingError, match="Source path does not exist"):
            RunSpec.from_pathlikes(nonexistent_src, output_dir)

    def test_from_pathlikes_with_directory_source(
        self,
        tmp_path: Path,
        temp_output_dir: Path
    ):
        """Test creating RunSpec with directory as source."""
        src_dir = tmp_path / "videos"
        src_dir.mkdir()
        
        runspec = RunSpec.from_pathlikes(src_dir, temp_output_dir)
        
        assert runspec.src == src_dir
        assert runspec.src.is_dir()


class TestRunSpecFromIndex:
    """Test RunSpec.from_index() factory method."""

    def test_from_index_with_default_camera(self, temp_output_dir: Path):
        """Test creating RunSpec from default camera index (0)."""
        runspec = RunSpec.from_index(0, temp_output_dir)
        
        assert runspec.src == 0
        assert runspec.dst == temp_output_dir
        assert isinstance(runspec.src, int)

    def test_from_index_with_secondary_camera(self, temp_output_dir: Path):
        """Test creating RunSpec from secondary camera index (1)."""
        runspec = RunSpec.from_index(1, temp_output_dir)
        
        assert runspec.src == 1
        assert runspec.dst == temp_output_dir

    def test_from_index_with_string_dst(self, tmp_path: Path):
        """Test creating RunSpec from index with string destination."""
        dst_path = tmp_path / "camera_output"
        runspec = RunSpec.from_index(2, str(dst_path))
        
        assert runspec.src == 2
        assert runspec.dst == dst_path
        assert isinstance(runspec.dst, Path)


class TestRunSpecEdgeCases:
    """Test edge cases and special scenarios."""

    def test_dst_path_not_required_to_exist(self, temp_video_file: Path, tmp_path: Path):
        """Test that destination path doesn't need to exist beforehand."""
        nonexistent_dst = tmp_path / "nonexistent_output"
        
        # Should not raise error even if dst doesn't exist
        runspec = RunSpec.from_pathlikes(temp_video_file, nonexistent_dst)
        
        assert runspec.dst == nonexistent_dst
        assert not runspec.dst.exists()  # Destination can be created later

    def test_nested_destination_path(self, temp_video_file: Path, tmp_path: Path):
        """Test RunSpec with deeply nested destination path."""
        nested_dst = tmp_path / "a" / "b" / "c" / "output"
        
        runspec = RunSpec.from_pathlikes(temp_video_file, nested_dst)
        
        assert runspec.dst == nested_dst

