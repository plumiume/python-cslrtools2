"""Tests for Matplotlib annotated frames collector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.annotated_frames.matplotlib_af import (
    MatplotlibAnnotatedFramesSaveCollector,
)
from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.runspec import RunSpec


@pytest.fixture
def sample_process_results() -> list[ProcessResult[str]]:
    """Create sample process results with annotated frames."""
    results: list[ProcessResult[str]] = []
    for idx in range(3):
        # Create 50x50 RGB frame
        frame = np.ones((50, 50, 3), dtype=np.uint8) * (idx * 80)
        result = ProcessResult[str](
            frame_id=idx,
            headers={"pose": np.array(["x", "y", "z"], dtype=str)},
            landmarks={"pose": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)},
            annotated_frame=frame,
        )
        results.append(result)
    return results


class TestMatplotlibAFImageSequence:
    """Test Matplotlib annotated frames collector."""

    def test_png_initialization(self):
        """Test initialization with PNG format."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](
            extension=".png", dpi=100
        )

        assert collector.is_image_sequence
        assert not collector.is_video
        assert collector.file_ext == ".png"
        assert collector.dpi == 100

    def test_jpg_initialization(self):
        """Test initialization with JPG format."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](
            extension=".jpg", dpi=150
        )

        assert collector.is_image_sequence
        assert collector.file_ext == ".jpg"
        assert collector.dpi == 150

    def test_save_png_sequence(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving frames as PNG using Matplotlib."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](extension=".png")

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, sample_process_results)

        frames_dir = output_dir / "annotated_frames"
        assert frames_dir.exists()
        assert frames_dir.is_dir()

        frame_files = sorted(frames_dir.glob("*.png"))
        assert len(frame_files) == 3
        assert frame_files[0].name == "frame_000000.png"
        assert frame_files[2].name == "frame_000002.png"

    def test_save_jpg_sequence(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving frames as JPG."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](extension=".jpg")

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, sample_process_results)

        frames_dir = output_dir / "annotated_frames"
        frame_files = sorted(frames_dir.glob("*.jpg"))
        assert len(frame_files) == 3

    def test_custom_dpi(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving with custom DPI."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](
            extension=".png", dpi=200
        )

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, sample_process_results)

        frames_dir = output_dir / "annotated_frames"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("*.png"))) == 3


class TestMatplotlibAFEdgeCases:
    """Test edge cases for Matplotlib collector."""

    def test_single_frame(self, tmp_path: Path):
        """Test saving single frame."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](extension=".png")

        frame = np.ones((30, 30, 3), dtype=np.uint8) * 200
        result = ProcessResult[str](
            frame_id=0,
            headers={},
            landmarks={},
            annotated_frame=frame,
        )

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, [result])

        frames_dir = output_dir / "annotated_frames"
        frame_files = list(frames_dir.glob("*.png"))
        assert len(frame_files) == 1

    def test_empty_results(self, tmp_path: Path):
        """Test handling empty results list."""
        collector = MatplotlibAnnotatedFramesSaveCollector[str](extension=".png")

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, [])

        frames_dir = output_dir / "annotated_frames"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("*.png"))) == 0
