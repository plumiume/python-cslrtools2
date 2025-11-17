"""Tests for CV2 collector parameter validation and edge cases."""

from __future__ import annotations

import pytest

from cslrtools2.lmpipe.collector.annotated_frames.cv2_af import (
    Cv2AnnotatedFramesSaveCollector,
)


class TestCv2CollectorValidation:
    """Test parameter validation for CV2 collectors."""

    def test_video_mode_requires_dimensions(self):
        """Test that video mode requires height, width, and fps."""
        with pytest.raises(
            ValueError, match="Video mode.*requires height, width, and fps"
        ):
            Cv2AnnotatedFramesSaveCollector[str](
                extension=".mp4",
                height=None,  # Missing
                width=640,
                fps=30,
            )

    def test_video_mode_requires_width(self):
        """Test that video mode requires width."""
        with pytest.raises(
            ValueError, match="Video mode.*requires height, width, and fps"
        ):
            Cv2AnnotatedFramesSaveCollector[str](
                extension=".avi",
                height=480,
                width=None,  # Missing
                fps=30,
            )

    def test_video_mode_requires_fps(self):
        """Test that video mode requires fps."""
        with pytest.raises(
            ValueError, match="Video mode.*requires height, width, and fps"
        ):
            Cv2AnnotatedFramesSaveCollector[str](
                extension=".mp4",
                height=480,
                width=640,
                fps=None,  # Missing
            )

    def test_video_mode_with_all_params(self):
        """Test that video mode works with all required parameters."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mp4",
            height=480,
            width=640,
            fps=30,
        )

        assert collector.is_video
        assert not collector.is_image_sequence
        assert collector.height == 480
        assert collector.width == 640
        assert collector.fps == 30

    def test_image_sequence_mode_no_dimensions_required(self):
        """Test that image sequence mode doesn't require dimensions."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".png",
            # height, width, fps are optional
        )

        assert collector.is_image_sequence
        assert not collector.is_video
        assert collector.height is None
        assert collector.width is None
        assert collector.fps is None

    def test_custom_fourcc(self):
        """Test custom FourCC codec specification."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mp4",
            height=480,
            width=640,
            fps=30,
            fourcc="avc1",
        )

        assert collector.fourcc == "avc1"

    def test_default_fourcc_is_none(self):
        """Test that fourcc defaults to None (auto-selected later)."""
        collector = Cv2AnnotatedFramesSaveCollector[str](
            extension=".mp4",
            height=480,
            width=640,
            fps=30,
        )

        # Default fourcc is None, actual codec selected during VideoWriter
        # initialization
        assert collector.fourcc is None
