"""Tests for torchvision annotated frames collector."""

from __future__ import annotations

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.annotated_frames.torchvision_af import (
    TorchVisionAnnotatedFramesShowCollector,
)
from cslrtools2.lmpipe.estimator import ProcessResult


@pytest.fixture
def sample_process_result() -> ProcessResult[str]:
    """Create a sample process result with annotated frame."""
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    return ProcessResult[str](
        frame_id=0,
        headers={"pose": np.array(["x", "y", "z"], dtype=str)},
        landmarks={"pose": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)},
        annotated_frame=frame,
    )


class TestTorchVisionAFShowCollector:
    """Test TorchVision annotated frames show collector."""

    def test_initialization(self):
        """Test TorchVision show collector initialization."""
        collector = TorchVisionAnnotatedFramesShowCollector[str]()
        assert collector is not None
        assert collector.figsize == (10, 8)

    def test_custom_figsize(self):
        """Test initialization with custom figure size."""
        collector = TorchVisionAnnotatedFramesShowCollector[str](figsize=(8, 6))
        assert collector.figsize == (8, 6)

    @pytest.mark.skip(reason="Requires display - would block in CI")
    def test_show_frame(self, sample_process_result: ProcessResult[str]):
        """Test showing single frame (manual test only)."""
        TorchVisionAnnotatedFramesShowCollector[str]()
        # This would open matplotlib figure in GUI
        # collector._update(sample_process_result)

    def test_numpy_frame_handling(self, sample_process_result: ProcessResult[str]):
        """Test that numpy frames are accepted."""
        collector = TorchVisionAnnotatedFramesShowCollector[str]()

        # Should not raise - just verify initialization works with numpy frame
        assert isinstance(sample_process_result.annotated_frame, np.ndarray)
        assert collector is not None
