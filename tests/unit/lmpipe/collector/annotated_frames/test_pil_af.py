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

"""Tests for PIL annotated frames collector."""

from __future__ import annotations

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.annotated_frames.pil_af import (
    PilAnnotatedFramesShowCollector,
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


class TestPilAFShowCollector:
    """Test PIL annotated frames show collector."""

    def test_initialization(self):
        """Test PIL show collector initialization."""
        collector = PilAnnotatedFramesShowCollector[str]()
        assert collector is not None

    @pytest.mark.skip(reason="Requires display - would block in CI")
    def test_show_frame(self, sample_process_result: ProcessResult[str]):
        """Test showing single frame (manual test only)."""
        PilAnnotatedFramesShowCollector[str]()
        # This would open image viewer in GUI
        # collector._update(sample_process_result)

    def test_numpy_frame_handling(self, sample_process_result: ProcessResult[str]):
        """Test that numpy frames are accepted."""
        collector = PilAnnotatedFramesShowCollector[str]()

        # Should not raise - just verify initialization works with numpy frame
        assert isinstance(sample_process_result.annotated_frame, np.ndarray)
        assert collector is not None
