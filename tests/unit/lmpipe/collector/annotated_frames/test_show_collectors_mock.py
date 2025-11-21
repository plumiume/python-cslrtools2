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

"""Tests for annotated frames show collectors using mocks (CI-safe)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.runspec import RunSpec


@pytest.fixture
def sample_process_results() -> list[ProcessResult[str]]:
    """Create sample process results with annotated frames."""
    results = []
    for idx in range(3):
        frame = np.ones((50, 50, 3), dtype=np.uint8) * (idx * 80)
        result = ProcessResult[str](
            frame_id=idx,
            headers={"pose": np.array(["x", "y", "z"], dtype=str)},
            landmarks={"pose": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)},
            annotated_frame=frame,
        )
        results.append(result)
    return results


class TestCv2ShowCollectorMocked:
    """Test OpenCV show collector with mocked cv2 functions (CI-safe)."""

    def test_cv2_show_with_mock(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test cv2 show collector calls correct cv2 functions."""
        from cslrtools2.lmpipe.collector.annotated_frames.cv2_af import (
            Cv2AnnotatedFramesShowCollector,
        )

        collector = Cv2AnnotatedFramesShowCollector[str]()

        # Mock the cv2 methods on the collector's _cv2 attribute
        with patch.object(collector._cv2, "namedWindow") as mock_named_window:
            with patch.object(collector._cv2, "imshow") as mock_imshow:
                with patch.object(
                    collector._cv2, "waitKey", return_value=ord("q")
                ) as mock_waitkey:
                    with patch.object(collector._cv2, "destroyWindow") as mock_destroy:
                        video_file = tmp_path / "input.mp4"
                        video_file.touch()
                        runspec = RunSpec(video_file, tmp_path)

                        # Collect results (mocked, won't actually display)
                        collector.collect_results(runspec, sample_process_results)

                        # Verify namedWindow was called in setup
                        mock_named_window.assert_called_once()

                        # Verify imshow was called for each frame
                        assert mock_imshow.call_count == 3

                        # waitKey should be called for each frame
                        assert mock_waitkey.call_count == 3

                        # destroyWindow should be called at cleanup
                        mock_destroy.assert_called_once()


class TestMatplotlibShowCollectorMocked:
    """Test matplotlib show collector with mocked plt methods (CI-safe)."""

    def test_matplotlib_show_with_mock(
        self,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test matplotlib show collector calls plt methods correctly."""
        from cslrtools2.lmpipe.collector.annotated_frames.matplotlib_af import (
            MatplotlibAnnotatedFramesShowCollector,
        )

        collector = MatplotlibAnnotatedFramesShowCollector[str](figsize=(8, 6))

        # Mock the plt methods on the collector's _plt attribute
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_canvas = MagicMock()
        mock_fig.canvas = mock_canvas

        with patch.object(collector._plt, "ion") as mock_ion:
            with patch.object(
                collector._plt, "subplots", return_value=(mock_fig, mock_ax)
            ) as mock_subplots:
                with patch.object(collector._plt, "pause") as mock_pause:
                    with patch.object(collector._plt, "ioff") as mock_ioff:
                        with patch.object(collector._plt, "close") as mock_close:
                            video_file = tmp_path / "input.mp4"
                            video_file.touch()
                            runspec = RunSpec(video_file, tmp_path)

                            # Collect results (mocked, won't actually display)
                            collector.collect_results(runspec, sample_process_results)

                            # Verify interactive mode was enabled
                            mock_ion.assert_called_once()

                            # Verify subplots was called with correct figsize
                            mock_subplots.assert_called_once_with(figsize=(8, 6))

                            # pause should be called for each frame
                            assert mock_pause.call_count == 3

                            # Cleanup: ioff and close should be called
                            mock_ioff.assert_called_once()
                            mock_close.assert_called_once()


class TestPilShowCollectorMocked:
    """Test PIL show collector with mocked Image.show (CI-safe)."""

    @patch("PIL.Image.fromarray")
    def test_pil_show_with_mock(
        self,
        mock_fromarray: MagicMock,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test PIL show collector calls Image.show correctly."""
        from cslrtools2.lmpipe.collector.annotated_frames.pil_af import (
            PilAnnotatedFramesShowCollector,
        )

        # Mock PIL Image
        mock_image = MagicMock()
        mock_fromarray.return_value = mock_image

        collector = PilAnnotatedFramesShowCollector[str]()

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        # Collect results (mocked, won't actually display)
        collector.collect_results(runspec, sample_process_results)

        # Verify fromarray was called for each frame
        assert mock_fromarray.call_count == 3

        # Verify show was called for each image
        assert mock_image.show.call_count == 3

        # Check that title was passed
        first_show_call = mock_image.show.call_args_list[0]
        if first_show_call[1]:  # kwargs exist
            assert "title" in first_show_call[1]


class TestTorchvisionShowCollectorMocked:
    """Test torchvision show collector with mocked matplotlib (CI-safe)."""

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.axis")
    @patch("matplotlib.pyplot.show")
    def test_torchvision_show_with_mock(
        self,
        mock_show: MagicMock,
        mock_axis: MagicMock,
        mock_title: MagicMock,
        mock_imshow: MagicMock,
        mock_figure: MagicMock,
        tmp_path: Path,
        sample_process_results: list[ProcessResult[str]],
    ):
        """Test torchvision show collector calls plt.show correctly."""
        from cslrtools2.lmpipe.collector.annotated_frames.torchvision_af import (
            TorchVisionAnnotatedFramesShowCollector,
        )

        collector = TorchVisionAnnotatedFramesShowCollector[str](figsize=(10, 8))

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        # Collect results (mocked, won't actually display)
        collector.collect_results(runspec, sample_process_results)

        # Verify figure was created
        assert mock_figure.call_count == 3

        # plt.show should be called for each frame
        assert mock_show.call_count == 3

        # title should be set
        assert mock_title.call_count == 3

        # axis('off') should be called
        assert mock_axis.call_count == 3


class TestShowCollectorErrorHandling:
    """Test error handling in show collectors."""

    def test_cv2_show_handles_display_error(
        self,
        tmp_path: Path,
    ):
        """Test that cv2 show collector handles display errors gracefully."""
        from cslrtools2.lmpipe.collector.annotated_frames.cv2_af import (
            Cv2AnnotatedFramesShowCollector,
        )

        collector = Cv2AnnotatedFramesShowCollector[str]()

        frame = np.ones((50, 50, 3), dtype=np.uint8)
        result = ProcessResult[str](
            frame_id=0,
            headers={},
            landmarks={},
            annotated_frame=frame,
        )

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        # Mock imshow to raise an error
        with patch.object(collector._cv2, "namedWindow"):
            with patch.object(
                collector._cv2, "imshow", side_effect=Exception("Display error")
            ):
                with patch.object(collector._cv2, "destroyWindow") as mock_destroy:
                    # Should raise the exception
                    with pytest.raises(Exception, match="Display error"):
                        collector.collect_results(runspec, [result])

                    # cleanup should still be called due to try/finally
                    mock_destroy.assert_called_once()
