"""Tests for OpenCV annotated frames collector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.annotated_frames.cv2_af import (
    Cv2AnnotatedFramesSaveCollector,
    Cv2AnnotatedFramesShowCollector,
    cv2_af_show_collector_creator,
)
from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.runspec import RunSpec


@pytest.fixture
def sample_process_results() -> list[ProcessResult[str]]:
    """Create sample process results with annotated frames."""
    results = []
    for idx in range(5):
        # Create 100x100 RGB frame
        frame = np.ones((100, 100, 3), dtype=np.uint8) * (idx * 50)
        result = ProcessResult[str](
            frame_id=idx,
            headers={"pose": np.array(["x", "y", "z"], dtype=str)},
            landmarks={"pose": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)},
            annotated_frame=frame,
        )
        results.append(result)
    return results


class TestCv2AFImageSequenceMode:
    """Test OpenCV annotated frames collector in image sequence mode."""

    def test_png_image_sequence_initialization(self):
        """Test initialization with PNG image sequence."""
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".png")

        assert collector.is_image_sequence
        assert not collector.is_video
        assert collector.file_ext == ".png"
        assert collector.extension == ".png"

    def test_jpg_image_sequence_initialization(self):
        """Test initialization with JPG image sequence."""
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".jpg")

        assert collector.is_image_sequence
        assert not collector.is_video
        assert collector.file_ext == ".jpg"

    def test_save_png_sequence(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving frames as PNG image sequence."""
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".png")

        # Create runspec
        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        # Collect results
        collector.collect_results(runspec, sample_process_results)

        # Check output directory created
        frames_dir = output_dir / "annotated_frames"
        assert frames_dir.exists()
        assert frames_dir.is_dir()

        # Check all frames saved
        frame_files = sorted(frames_dir.glob("*.png"))
        assert len(frame_files) == 5

        # Check frame naming
        assert frame_files[0].name == "frame_000000.png"
        assert frame_files[4].name == "frame_000004.png"

    def test_save_jpg_sequence(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving frames as JPG image sequence."""
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".jpg")

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, sample_process_results)

        frames_dir = output_dir / "annotated_frames"
        frame_files = sorted(frames_dir.glob("*.jpg"))
        assert len(frame_files) == 5


class TestCv2AFVideoMode:
    """Test OpenCV annotated frames collector in video mode."""

    def test_mp4_video_initialization(self):
        """Test initialization with MP4 video."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mp4",
            height=100,
            width=100,
            fps=30,
        )

        assert collector.is_video
        assert not collector.is_image_sequence
        assert collector.file_ext == ".mp4"
        assert collector.height == 100
        assert collector.width == 100
        assert collector.fps == 30

    def test_avi_video_initialization(self):
        """Test initialization with AVI video."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".avi",
            height=200,
            width=300,
            fps=25,
            fourcc="XVID",
        )

        assert collector.is_video
        assert collector.file_ext == ".avi"
        assert collector.fourcc == "XVID"

    def test_video_requires_dimensions_and_fps(self):
        """Test that video mode requires height, width, and fps."""
        with pytest.raises(ValueError, match="requires height, width, and fps"):
            Cv2AnnotatedFramesSaveCollector[str](
                extension=".mp4",
                height=None,
                width=None,
                fps=None,
            )

    def test_save_mp4_video(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test saving frames as MP4 video."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mp4",
            height=100,
            width=100,
            fps=30,
        )

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, sample_process_results)

        # Check video file created
        video_output = output_dir / "annotated_frames.mp4"
        assert video_output.exists()
        assert video_output.is_file()
        assert video_output.stat().st_size > 0


class TestCv2AFEdgeCases:
    """Test edge cases for OpenCV annotated frames collector."""

    def test_single_frame(self, tmp_path: Path):
        """Test saving single frame."""
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".png")

        frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
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
        collector = Cv2AnnotatedFramesSaveCollector[str](extension=".png")

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)

        collector.collect_results(runspec, [])

        frames_dir = output_dir / "annotated_frames"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("*.png"))) == 0


class TestCv2AFFourCCCodecSelection:
    """Test FourCC codec selection for different video formats."""

    def test_fourcc_avi_codec(self):
        """Test FourCC codec selection for AVI format."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".avi", width=640, height=480, fps=30
        )

        # pyright: ignore[reportAttributeAccessIssue]
        fourcc = collector._get_fourcc_code()

        # Verify it returns an integer (FourCC code)
        assert isinstance(fourcc, int)
        # XVID codec for AVI: fourcc(*"XVID")
        assert fourcc != 0

    def test_fourcc_mov_codec(self):
        """Test FourCC codec selection for MOV format."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mov", width=640, height=480, fps=30
        )

        # pyright: ignore[reportAttributeAccessIssue]
        fourcc = collector._get_fourcc_code()

        # Verify it returns an integer (FourCC code)
        assert isinstance(fourcc, int)
        # mp4v codec for MOV
        assert fourcc != 0

    def test_fourcc_mkv_codec(self):
        """Test FourCC codec selection for MKV format."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mkv", width=640, height=480, fps=30
        )

        # pyright: ignore[reportAttributeAccessIssue]
        fourcc = collector._get_fourcc_code()

        # Verify it returns an integer (FourCC code)
        assert isinstance(fourcc, int)
        # X264 codec for MKV
        assert fourcc != 0

    def test_fourcc_default_fallback(self):
        """Test FourCC codec selection for unknown format falls back to mp4v."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".webm",  # Unknown format
            width=640,
            height=480,
            fps=30,
        )

        # pyright: ignore[reportAttributeAccessIssue]
        fourcc = collector._get_fourcc_code()

        # Verify it returns an integer (FourCC code)
        assert isinstance(fourcc, int)
        # Default fallback: mp4v
        assert fourcc != 0


class TestCv2AFCreatorFunctions:
    """Test creator functions for cv2 annotated frames collectors."""

    def test_cv2_af_show_collector_creator(self):
        """Test cv2_af_show_collector_creator returns correct type."""
        collector = cv2_af_show_collector_creator(str)

        # Verify it returns a Cv2AnnotatedFramesShowCollector instance
        assert isinstance(collector, Cv2AnnotatedFramesShowCollector)
