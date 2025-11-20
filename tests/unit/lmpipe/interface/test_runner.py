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

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, cast
from pathlib import Path
from concurrent.futures import Future

import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
from cv2 import VideoCapture
from unittest.mock import Mock, MagicMock, patch

from cslrtools2.lmpipe.interface.runner import (
    LMPipeInterface, LMPipeRunner, _runner_public_api
)
from cslrtools2.lmpipe.collector import Collector
from cslrtools2.lmpipe.estimator import Estimator, ProcessResult
from cslrtools2.lmpipe.options import (
    DEFAULT_LMPIPE_OPTIONS,
    LMPipeOptions,
)
from cslrtools2.lmpipe.runspec import RunSpec
from cslrtools2.typings import NDArrayFloat, NDArrayStr, MatLike


def dummy_enter(self: Any) -> Any:
    return self


class dummy_estimator_enter:

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def __call__(self, *args: Any) -> Any:
        return self.estimator


def dummy_exit(self: Any, *args: Any) -> None:
    return None


class DummyEstimator(Estimator[Literal["test"]]):
    """Dummy estimator for testing."""

    def __init__(self):
        self._setup_called = False
        self._estimate_called = False
        self._annotate_called = False

    def setup(self) -> None:
        self._setup_called = True

    @property
    def shape(self) -> Mapping[Literal["test"], tuple[int, int]]:
        return {"test": (1, 3)}

    def configure_estimator_name(self) -> Literal["test"]:
        return "test"

    @property
    def headers(self) -> Mapping[Literal["test"], NDArrayStr]:
        return {"test": np.array(["x", "y", "z"], dtype=str)}

    def estimate(
        self, frame_src: MatLike | None, frame_idx: int
    ) -> Mapping[Literal["test"], NDArrayFloat]:
        self._estimate_called = True
        return {"test": np.array([[1.0, 2.0, 3.0]])}

    def annotate(
        self,
        frame_src: MatLike | None,
        frame_idx: int,
        landmarks: Mapping[Literal["test"], NDArrayFloat],
    ) -> MatLike:
        self._annotate_called = True
        return (
            frame_src if frame_src is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        )


@pytest.fixture
def mock_interface():
    """Create a mock LMPipeInterface."""
    interface = Mock()
    interface.estimator = DummyEstimator()
    interface.collectors_or_factory = []
    return interface


@pytest.fixture
def runner(mock_interface: LMPipeInterface[Literal["test"]]):
    """Create a LMPipeRunner instance."""
    return LMPipeRunner(mock_interface, DEFAULT_LMPIPE_OPTIONS)


class TestLMPipeRunnerInitialization:
    """Tests for LMPipeRunner initialization."""

    def test_initialization(self, mock_interface: LMPipeInterface[Literal["test"]]):
        """Test basic initialization."""
        runner = LMPipeRunner(mock_interface, DEFAULT_LMPIPE_OPTIONS)

        assert runner.lmpipe_interface is mock_interface
        assert runner.lmpipe_options == DEFAULT_LMPIPE_OPTIONS
        assert runner.toplevel_call is True
        assert runner.executors == {}
        assert isinstance(runner.collectors, list)

    def test_getstate_excludes_executors(self, runner: LMPipeRunner[Literal["test"]]):
        """Test __getstate__ excludes executors from serialization."""
        runner.executors["batch"] = Mock()
        state = runner.__getstate__()

        assert "executors" in state
        assert state["executors"] == {}
        assert "lmpipe_interface" in state

    def test_setstate_restores_state(self, runner: LMPipeRunner[Literal["test"]]):
        """Test __setstate__ restores state."""
        state = {
            "lmpipe_interface": Mock(),
            "lmpipe_options": DEFAULT_LMPIPE_OPTIONS,
            "toplevel_call": False,
        }
        runner.__setstate__(state)

        assert runner.lmpipe_interface is state["lmpipe_interface"]
        assert runner.toplevel_call is False

    def test_configure_collectors_with_list(
        self, mock_interface: LMPipeInterface[Literal["test"]]
    ):
        """Test collector configuration with pre-configured list."""
        collector1 = cast(Collector[Literal["test"]], Mock())
        collector2 = cast(Collector[Literal["test"]], Mock())
        mock_interface.collectors_or_factory = [collector1, collector2]

        runner = LMPipeRunner(mock_interface, DEFAULT_LMPIPE_OPTIONS)

        assert len(runner.collectors) == 2
        assert collector1 in runner.collectors
        assert collector2 in runner.collectors

    def test_configure_collectors_with_factory(
        self, mock_interface: LMPipeInterface[Literal["test"]]
    ):
        """Test collector configuration with factory function."""
        collector1 = Mock()
        collector2 = Mock()
        factory = Mock(return_value=[collector1, collector2])
        mock_interface.collectors_or_factory = factory

        runner = LMPipeRunner(mock_interface, DEFAULT_LMPIPE_OPTIONS)

        factory.assert_called_once()
        assert len(runner.collectors) == 2
        assert collector1 in runner.collectors

    def test_configure_collectors_empty(
        self, mock_interface: LMPipeInterface[Literal["test"]]
    ):
        """Test collector configuration with empty list."""
        mock_interface.collectors_or_factory = []

        runner = LMPipeRunner(mock_interface, DEFAULT_LMPIPE_OPTIONS)

        assert runner.collectors == []


class TestRunMethodDispatch:
    """Tests for run method dispatch logic."""

    def test_run_dispatches_to_run_batch_for_directory(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test run() dispatches to run_batch() for directory."""
        src_dir = tmp_path / "input"
        src_dir.mkdir()
        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        with patch.object(runner, "run_batch") as mock_run_batch:
            runner.run(runspec)
            mock_run_batch.assert_called_once_with(runspec)

    def test_run_dispatches_to_run_single_for_file(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test run() dispatches to run_single() for file."""
        src_file = tmp_path / "input.mp4"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        with patch.object(runner, "run_single") as mock_run_single:
            runner.run(runspec)
            mock_run_single.assert_called_once_with(runspec)

    def test_run_raises_for_nonexistent_path(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test run() raises FileNotFoundError for nonexistent path."""
        src_path = tmp_path / "nonexistent"
        dst_path = tmp_path / "output"
        runspec = RunSpec(src_path, dst_path)

        with pytest.raises(FileNotFoundError, match="Source path does not exist"):
            runner.run(runspec)

    @patch("cslrtools2.lmpipe.interface.runner.is_video_file", return_value=True)
    def test_run_single_dispatches_to_run_video(
        self,
        mock_is_video: Callable[[Path], bool],
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_single() dispatches to run_video() for video file."""
        src_file = tmp_path / "video.mp4"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        with patch.object(runner, "run_video") as mock_run_video:
            runner.run_single(runspec)
            mock_run_video.assert_called_once_with(runspec)

    @patch("cslrtools2.lmpipe.interface.runner.is_images_dir", return_value=True)
    @patch("cslrtools2.lmpipe.interface.runner.is_video_file", return_value=False)
    def test_run_single_dispatches_to_run_sequence_images(
        self,
        mock_is_video: Callable[[Path], bool],
        mock_is_images_dir: Callable[[Path], bool],
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_single() dispatches to run_sequence_images() for image directory."""
        src_dir = tmp_path / "images"
        src_dir.mkdir()
        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        with patch.object(runner, "run_sequence_images") as mock_run_seq:
            runner.run_single(runspec)
            mock_run_seq.assert_called_once_with(runspec)

    @patch("cslrtools2.lmpipe.interface.runner.is_image_file", return_value=True)
    @patch("cslrtools2.lmpipe.interface.runner.is_images_dir", return_value=False)
    @patch("cslrtools2.lmpipe.interface.runner.is_video_file", return_value=False)
    def test_run_single_dispatches_to_run_single_image(
        self,
        mock_is_video: Callable[[Path], bool],
        mock_is_images_dir: Callable[[Path], bool],
        mock_is_image: Callable[[Path], bool],
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_single() dispatches to run_single_image() for image file."""
        src_file = tmp_path / "image.jpg"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        with patch.object(runner, "run_single_image") as mock_run_image:
            runner.run_single(runspec)
            mock_run_image.assert_called_once_with(runspec)

    @patch("cslrtools2.lmpipe.interface.runner.is_image_file", return_value=False)
    @patch("cslrtools2.lmpipe.interface.runner.is_images_dir", return_value=False)
    @patch("cslrtools2.lmpipe.interface.runner.is_video_file", return_value=False)
    def test_run_single_raises_for_unsupported_type(
        self,
        mock_is_video: Callable[[Path], bool],
        mock_is_images_dir: Callable[[Path], bool],
        mock_is_image: Callable[[Path], bool],
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_single() raises ValueError for unsupported file type."""
        src_file = tmp_path / "unsupported.xyz"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        with pytest.raises(ValueError, match="Unsupported source path"):
            runner.run_single(runspec)


class TestVideoProcessing:
    """Tests for video processing methods."""

    @patch("cslrtools2.lmpipe.interface.runner.cv2.VideoCapture")
    @patch("cslrtools2.lmpipe.interface.runner.capture_to_frames")
    def test_run_video_processes_frames(
        self,
        mock_capture_to_frames: MagicMock,
        mock_VideoCapture: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_video() processes video frames."""
        src_file = tmp_path / "video.mp4"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        # Mock video capture
        mock_capture = MagicMock(spec=VideoCapture)
        mock_capture.isOpened.return_value = True

        def get_side_effect(prop: int) -> int | float:
            return {2: 100, 5: 30.0}.get(prop, 0)

        mock_capture.get.side_effect = get_side_effect
        mock_VideoCapture.return_value = mock_capture

        # Mock frames
        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        mock_capture_to_frames.return_value = mock_frames

        with (
            patch.object(runner, "process_frames") as mock_process_frames,
            patch.object(runner, "_collect_results") as mock_collect,
        ):
            mock_process_frames.return_value = []
            runner.run_video(runspec)

            mock_VideoCapture.assert_called_once()
            mock_capture.isOpened.assert_called()
            mock_process_frames.assert_called_once()
            mock_collect.assert_called_once()

    @patch("cslrtools2.lmpipe.interface.runner.cv2.VideoCapture")
    def test_run_video_raises_if_cannot_open(
        self,
        mock_VideoCapture: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_video() raises ValueError if video cannot be opened."""
        src_file = tmp_path / "video.mp4"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_VideoCapture.return_value = mock_capture

        with pytest.raises(ValueError, match="Cannot open video file"):
            runner.run_video(runspec)

    @patch("cslrtools2.lmpipe.interface.runner.seq_imgs_to_frames")
    def test_run_sequence_images_processes_frames(
        self,
        mock_seq_imgs_to_frames: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_sequence_images() processes image sequence."""
        src_dir = tmp_path / "images"
        src_dir.mkdir()
        (src_dir / "img1.jpg").touch()
        (src_dir / "img2.jpg").touch()
        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        mock_seq_imgs_to_frames.return_value = mock_frames

        with (
            patch.object(runner, "process_frames") as mock_process_frames,
            patch.object(runner, "_collect_results") as mock_collect,
        ):
            mock_process_frames.return_value = []
            runner.run_sequence_images(runspec)

            mock_seq_imgs_to_frames.assert_called_once_with(src_dir)
            mock_process_frames.assert_called_once()
            mock_collect.assert_called_once()

    @patch("cslrtools2.lmpipe.interface.runner.image_file_to_frame")
    def test_run_single_image_processes_frame(
        self,
        mock_image_file_to_frame: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_single_image() processes single image."""
        src_file = tmp_path / "image.jpg"
        src_file.touch()
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_file_to_frame.return_value = mock_frame

        with (
            patch.object(runner, "process_frames") as mock_process_frames,
            patch.object(runner, "_collect_results") as mock_collect,
        ):
            mock_process_frames.return_value = []
            runner.run_single_image(runspec)

            mock_image_file_to_frame.assert_called_once_with(src_file)
            mock_process_frames.assert_called_once_with([mock_frame])
            mock_collect.assert_called_once()

    @patch("cslrtools2.lmpipe.interface.runner.cv2.VideoCapture")
    @patch("cslrtools2.lmpipe.interface.runner.capture_to_frames")
    def test_run_stream_processes_stream(
        self,
        mock_capture_to_frames: MagicMock,
        mock_VideoCapture: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_stream() processes video stream."""
        dst_file = tmp_path / "output"
        runspec = RunSpec(0, dst_file)  # Device index 0

        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_VideoCapture.return_value = mock_capture

        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        mock_capture_to_frames.return_value = mock_frames

        with (
            patch.object(runner, "process_frames") as mock_process_frames,
            patch.object(runner, "_collect_results") as mock_collect,
        ):
            mock_process_frames.return_value = []
            runner.run_stream(runspec)

            mock_VideoCapture.assert_called_once_with(0)
            mock_process_frames.assert_called_once()
            mock_collect.assert_called_once()

    @patch("cslrtools2.lmpipe.interface.runner.cv2.VideoCapture")
    def test_run_stream_raises_if_cannot_open(
        self,
        mock_VideoCapture: MagicMock,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test run_stream() raises ValueError if stream cannot be opened."""
        dst_file = tmp_path / "output"
        runspec = RunSpec(0, dst_file)

        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_VideoCapture.return_value = mock_capture

        with pytest.raises(ValueError, match="Cannot open video stream"):
            runner.run_stream(runspec)


class TestFrameProcessing:
    """Tests for frame processing methods."""

    def test_call_estimator_first_frame(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _call_estimator() sets up estimator on first frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        estimator = runner.lmpipe_interface.estimator
        assert isinstance(estimator, DummyEstimator), "not DummyEstimator"

        result = runner._call_estimator(frame, 0)

        assert estimator._setup_called is True
        assert estimator._estimate_called is True
        assert estimator._annotate_called is True
        assert isinstance(result, ProcessResult)
        assert result.frame_id == 0
        assert "test" in result.headers
        assert "test" in result.landmarks

    def test_call_estimator_subsequent_frames(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _call_estimator() skips setup on subsequent frames."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        estimator = runner.lmpipe_interface.estimator
        assert isinstance(estimator, DummyEstimator), "not DummyEstimator"

        # First frame to trigger setup
        runner._call_estimator(frame, 0)
        estimator._setup_called = False  # Reset flag

        # Second frame
        result = runner._call_estimator(frame, 1)

        assert estimator._setup_called is False  # Should not call setup again
        assert estimator._estimate_called is True
        assert result.frame_id == 1

    def test_call_estimator_returns_process_result(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _call_estimator() returns ProcessResult with correct structure."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = runner._call_estimator(frame, 5)

        assert isinstance(result, ProcessResult)
        assert result.frame_id == 5
        assert "test" in result.headers
        assert np.array_equal(
            result.headers["test"], np.array(["x", "y", "z"], dtype=str)
        )
        assert "test" in result.landmarks
        assert np.array_equal(result.landmarks["test"], np.array([[1.0, 2.0, 3.0]]))
        assert result.annotated_frame is frame

    def test_process_frames_yields_results(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test process_frames() yields ProcessResult for each frame."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_get_executor_resources") as mock_resources,
        ):
            # Mock executor to use DummyExecutor (synchronous)
            from cslrtools2.lmpipe.interface.executor import DummyExecutor

            mock_executor = DummyExecutor(
                initializer=runner._default_executor_initializer
            )

            def enter_mock(s: Any) -> Any:
                return mock_executor

            mock_init_executor.return_value.__enter__ = enter_mock
            mock_init_executor.return_value.__exit__ = dummy_exit
            mock_resources.return_value.__enter__ = dummy_enter
            mock_resources.return_value.__exit__ = dummy_exit

            results = list(runner.process_frames(frames))

            assert len(results) == 2
            assert all(isinstance(r, ProcessResult) for r in results)
            assert results[0].frame_id == 0
            assert results[1].frame_id == 1


class TestCollectorIntegration:
    """Tests for collector integration."""

    def test_collect_results_calls_all_collectors(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _collect_results() calls all configured collectors."""
        collector1 = cast(Collector[Literal["test"]], Mock())
        collector1.collect_results = MagicMock()
        collector2 = cast(Collector[Literal["test"]], Mock())
        collector2.collect_results = MagicMock()
        runner.collectors = [collector1, collector2]

        src_file = tmp_path / "input.mp4"
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        results = [
            ProcessResult[Literal["test"]](
                frame_id=0,
                headers={"test": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"test": np.array([[1.0, 2.0, 3.0]])},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            )
        ]

        runner._collect_results(runspec, results)

        collector1.collect_results.assert_called_once()
        collector2.collect_results.assert_called_once()

    def test_collect_results_with_no_collectors(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _collect_results() handles empty collector list."""
        runner.collectors = []

        src_file = tmp_path / "input.mp4"
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        results = [
            ProcessResult[Literal["test"]](
                frame_id=0,
                headers={"test": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"test": np.array([[1.0, 2.0, 3.0]])},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            )
        ]

        # Should not raise
        runner._collect_results(runspec, results)


class TestExecutorConfiguration:
    """Tests for executor configuration."""

    def test_configure_executor_returns_dummy_when_mode_mismatch(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test configure_executor() returns DummyExecutor when mode doesn't match."""
        from cslrtools2.lmpipe.interface.executor import DummyExecutor

        # Options set to batch mode, request frames mode
        runner.lmpipe_options = LMPipeOptions(
            {**DEFAULT_LMPIPE_OPTIONS, "executor_mode": "batch"}
        )

        executor = runner.configure_executor("frames", lambda: None)

        assert isinstance(executor, DummyExecutor)

    def test_configure_executor_returns_dummy_when_max_cpus_zero(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test configure_executor() returns DummyExecutor when max_cpus is 0."""
        from cslrtools2.lmpipe.interface.executor import DummyExecutor

        runner.lmpipe_options = LMPipeOptions({**DEFAULT_LMPIPE_OPTIONS, "max_cpus": 0})
        executor = runner.configure_executor("batch", lambda: None)

        assert isinstance(executor, DummyExecutor)

    def test_configure_executor_process_pool(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test configure_executor() creates ProcessPoolExecutor."""
        from cslrtools2.lmpipe.interface.executor import ProcessPoolExecutor

        runner.lmpipe_options = LMPipeOptions({
            **DEFAULT_LMPIPE_OPTIONS,
            "executor_mode": "batch",
            "executor_type": "process",
            "max_cpus": 4,
            "cpu": 1,
        })

        executor = runner.configure_executor("batch", lambda: None)

        assert isinstance(executor, ProcessPoolExecutor)

    def test_configure_executor_thread_pool(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test configure_executor() creates ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        runner.lmpipe_options = LMPipeOptions({
            **DEFAULT_LMPIPE_OPTIONS,
            "executor_mode": "batch",
            "executor_type": "thread",
            "max_cpus": 4,
            "cpu": 1,
        })

        executor = runner.configure_executor("batch", lambda: None)

        assert isinstance(executor, ThreadPoolExecutor)

    def test_init_executor_caches_executor(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _init_executor() caches executor instances."""
        # Ensure executors dict is empty
        runner.executors = {}

        with patch.object(runner, "configure_executor") as mock_configure:
            mock_executor = Mock()
            mock_configure.return_value = mock_executor

            # First call - should configure and cache
            result1 = runner._init_executor("batch")

            # Second call - should return cached executor
            result2 = runner._init_executor("batch")

            # Should only configure once
            mock_configure.assert_called_once_with(
                "batch", runner._default_executor_initializer
            )
            assert result1 is result2
            # Both should be the configured executor
            assert runner.executors["batch"] is mock_executor
            assert result1 is mock_executor


class TestEventSystem:
    """Tests for event system."""

    def test_events_ctxmgr_calls_start_and_end(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _events_ctxmgr() calls start and end handlers."""
        on_start = Mock()
        on_end = Mock()
        args = ("arg1", "arg2")

        with runner._events_ctxmgr(on_start, on_end, *args):
            on_start.assert_called_once_with(*args)
            on_end.assert_not_called()

        on_end.assert_called_once_with(*args)

    def test_events_ctxmgr_calls_end_on_exception(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _events_ctxmgr() calls end handler even on exception."""
        on_start = Mock()
        on_end = Mock()

        with pytest.raises(RuntimeError):
            with runner._events_ctxmgr(on_start, on_end):
                raise RuntimeError("Test error")

        on_start.assert_called_once()
        on_end.assert_called_once()

    def test_submit_with_events_calls_on_submit(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _submit_with_events() calls on_submit immediately."""
        future = Future[Any]()
        on_submit = Mock()
        on_success = Mock()
        on_failure = Mock()

        runner._submit_with_events(  # pyright: ignore[reportArgumentType]
            future, on_submit, on_success, on_failure, "arg1"
        )

        on_submit.assert_called_once_with("arg1")

    def test_callback_with_events_calls_on_success(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _CallbackWithEvents calls on_success for successful future."""
        on_success = Mock()
        on_failure = Mock()
        callback = runner._CallbackWithEvents(
            on_success, on_failure, ("arg1",)  # pyright: ignore[reportArgumentType]
        )

        future = Future[str]()
        future.set_result("result")

        callback(future)

        on_success.assert_called_once_with("arg1", "result")
        on_failure.assert_not_called()

    def test_callback_with_events_calls_on_failure(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _CallbackWithEvents calls on_failure for failed future."""
        on_success = Mock()
        on_failure = Mock()
        callback = runner._CallbackWithEvents(
            on_success, on_failure, ("arg1",)  # pyright: ignore[reportArgumentType]
        )

        future = Future[Any]()
        error = RuntimeError("Test error")
        future.set_exception(error)

        callback(future)

        on_failure.assert_called_once()
        call_args = on_failure.call_args[0]
        assert call_args[0] == "arg1"
        assert isinstance(call_args[1], RuntimeError)
        on_success.assert_not_called()

    def test_task_with_events_calls_handlers(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _task_with_events calls start/end handlers."""
        task = Mock(return_value="result")
        on_start = Mock()
        on_end = Mock()

        task_wrapper = runner._task_with_events(task, on_start, on_end)
        result = task_wrapper()

        on_start.assert_called_once()
        task.assert_called_once()
        on_end.assert_called_once()
        assert result == "result"


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_runspecs_finds_video_files(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _get_runspecs() finds video files in directory."""
        src_dir = tmp_path / "input"
        src_dir.mkdir()
        (src_dir / "video1.mp4").touch()
        (src_dir / "video2.avi").touch()
        (src_dir / "notes.txt").touch()  # Should be ignored

        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        with (
            patch.object(runner, "_apply_exist_rule", return_value=True),
            patch("cslrtools2.lmpipe.interface.runner.is_video_file") as mock_is_video,
            patch(
                "cslrtools2.lmpipe.interface.runner.is_image_file", return_value=False
            ),
        ):
            def is_video_check(p: Path) -> bool:
                return p.suffix in [".mp4", ".avi"]

            mock_is_video.side_effect = is_video_check

            runspecs = list(runner._get_runspecs(runspec))

            assert len(runspecs) == 2
            assert all(isinstance(rs, RunSpec) for rs in runspecs)

    def test_get_runspecs_detects_image_sequences(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _get_runspecs() detects image sequence directories."""
        src_dir = tmp_path / "input"
        images_dir = src_dir / "sequence1"
        images_dir.mkdir(parents=True)
        (images_dir / "img1.jpg").touch()
        (images_dir / "img2.jpg").touch()

        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        with (
            patch.object(runner, "_apply_exist_rule", return_value=True),
            patch(
                "cslrtools2.lmpipe.interface.runner.is_image_file", return_value=True
            ),
            patch(
                "cslrtools2.lmpipe.interface.runner.is_video_file", return_value=False
            ),
        ):
            runspecs = list(runner._get_runspecs(runspec))

            # Should create one runspec for the image directory
            assert len(runspecs) == 1
            assert runspecs[0].src == images_dir

    def test_get_runspecs_skips_hidden_directories(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _get_runspecs() skips hidden directories."""
        src_dir = tmp_path / "input"
        src_dir.mkdir()
        hidden_dir = src_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "video.mp4").touch()

        dst_dir = tmp_path / "output"
        runspec = RunSpec(src_dir, dst_dir)

        with patch.object(runner, "_apply_exist_rule", return_value=True):
            runspecs = list(runner._get_runspecs(runspec))

            # Should not find files in hidden directory
            assert len(runspecs) == 0

    def test_apply_exist_rule_returns_true_if_any_collector_allows(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _apply_exist_rule() returns True if any collector allows processing."""
        collector1 = cast(Collector[Literal["test"]], Mock())
        collector1.apply_exist_rule = MagicMock(return_value=False)
        collector2 = cast(Collector[Literal["test"]], Mock())
        collector2.apply_exist_rule = MagicMock(return_value=True)
        runner.collectors = [collector1, collector2]

        src_file = tmp_path / "input.mp4"
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        result = runner._apply_exist_rule(runspec)

        assert result is True
        collector1.apply_exist_rule.assert_called_once_with(runspec)
        collector2.apply_exist_rule.assert_called_once_with(runspec)

    def test_apply_exist_rule_returns_false_if_all_collectors_reject(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test _apply_exist_rule() returns False if all collectors reject."""
        collector1 = cast(Collector[Literal["test"]], Mock())
        collector1.apply_exist_rule = MagicMock(return_value=False)
        collector2 = cast(Collector[Literal["test"]], Mock())
        collector2.apply_exist_rule = MagicMock(return_value=False)
        runner.collectors = [collector1, collector2]

        src_file = tmp_path / "input.mp4"
        dst_file = tmp_path / "output"
        runspec = RunSpec(src_file, dst_file)

        result = runner._apply_exist_rule(runspec)

        assert result is False


class TestRunnerPublicApiDecorator:
    """Tests for _runner_public_api decorator."""

    def test_decorator_calls_on_complete_on_success(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator calls on_complete() on successful execution."""
        runner.on_complete = Mock()

        @_runner_public_api
        def test_method(self: Any):
            return "success"

        result = test_method(runner)

        assert result == "success"
        runner.on_complete.assert_called_once()

    def test_decorator_calls_on_keyboard_interrupt(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator calls on_keyboard_interrupt() on KeyboardInterrupt."""
        runner.on_keyboard_interrupt = Mock(side_effect=KeyboardInterrupt)

        @_runner_public_api
        def test_method(self: Any):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            test_method(runner)

        runner.on_keyboard_interrupt.assert_called_once()

    def test_decorator_calls_on_general_exception(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator calls on_general_exception() on Exception."""
        runner.on_general_exception = Mock(side_effect=RuntimeError)

        @_runner_public_api
        def test_method(self: Any):
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            test_method(runner)

        runner.on_general_exception.assert_called_once()

    def test_decorator_calls_on_finally(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator calls on_finally() in finally block."""
        runner.on_finally = Mock()

        @_runner_public_api
        def test_method(self: Any):
            raise RuntimeError("Test error")

        runner.on_general_exception = Mock(side_effect=RuntimeError)

        with pytest.raises(RuntimeError):
            test_method(runner)

        runner.on_finally.assert_called_once()

    def test_decorator_shuts_down_executors(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator shuts down executors in finally block."""
        mock_executor = Mock()
        runner.executors["batch"] = mock_executor

        @_runner_public_api
        def test_method(self: Any):
            return "success"

        test_method(runner)

        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_decorator_skips_wrapper_for_non_toplevel_calls(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test decorator skips wrapper logic for nested calls."""
        runner.toplevel_call = False
        runner.on_complete = Mock()

        @_runner_public_api
        def test_method(self: Any):
            return "success"

        result = test_method(runner)

        assert result == "success"
        # Should not call on_complete for nested calls
        runner.on_complete.assert_not_called()


class TestLocalRunnerMethod:
    """Tests for _local_runner_method wrapper."""

    def test_local_runner_method_stores_method_info(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _local_runner_method stores method and runner info."""

        def test_method(self: Any, arg1: Any, arg2: Any):
            return f"{arg1}+{arg2}"

        wrapper = LMPipeRunner._local_runner_method(test_method, runner, "val1", "val2")

        assert wrapper.method is test_method
        assert wrapper.runner_type is type(runner)
        assert wrapper.runner_id == runner._id
        assert wrapper.args == ("val1", "val2")  # pyright: ignore[reportUnknownMemberType] # noqa: E501

    def test_local_runner_method_executes_in_local_context(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _local_runner_method executes method with local runner instance."""
        # Register runner in thread-local storage
        LMPipeRunner._local.instances[runner._id] = runner

        def test_method(self: Any, arg1: Any):
            return f"executed with {arg1}"

        wrapper = LMPipeRunner._local_runner_method(test_method, runner, "test_arg")
        result = wrapper()

        assert result == "executed with test_arg"

    def test_local_runner_method_validates_runner_type(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _local_runner_method validates runner type."""
        # Register wrong type of runner
        wrong_runner = Mock()
        LMPipeRunner._local.instances[runner._id] = wrong_runner

        def test_method(self: Any, arg1: Any):
            return arg1

        wrapper = LMPipeRunner._local_runner_method(test_method, runner, "arg")

        with pytest.raises(TypeError, match="Expected runner of type"):
            wrapper()

    def test_local_runner_method_validates_runner_id(
        self, runner: LMPipeRunner[Literal["test"]]
    ):
        """Test _local_runner_method validates runner ID."""
        # Create another runner with different ID
        other_runner = LMPipeRunner(runner.lmpipe_interface, DEFAULT_LMPIPE_OPTIONS)
        LMPipeRunner._local.instances[runner._id] = other_runner

        def test_method(self: Any, arg1: Any):
            return arg1

        wrapper = LMPipeRunner._local_runner_method(test_method, runner, "arg")

        with pytest.raises(ValueError, match="Runner ID mismatch"):
            wrapper()


class TestLMPipeRunnerBatchProcessing:
    """Tests for LMPipeRunner batch processing with mocked methods."""

    def test_run_batch_calls_required_methods(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test that run_batch calls all required methods in correct order."""
        # Create test video files
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test1.mp4").touch()
        (video_dir / "test2.mp4").touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_dir, output_dir)

        # Track method calls

        # Mock all event methods
        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "_submit_with_events") as mock_submit,
            patch.object(runner, "on_determined_batch_task_count") as mock_determined,
            patch.object(runner, "run_single"),
        ):
            # Setup mocks
            mock_executor = MagicMock()
            mock_init_executor.return_value.__enter__ = dummy_estimator_enter(
                mock_executor
            )
            mock_init_executor.return_value.__exit__ = dummy_exit

            mock_events_ctx.return_value.__enter__ = dummy_enter
            mock_events_ctx.return_value.__exit__ = dummy_exit

            # Create mock runspecs for 2 tasks
            task_runspec1 = RunSpec(video_dir / "test1.mp4", output_dir)
            task_runspec2 = RunSpec(video_dir / "test2.mp4", output_dir)
            mock_get_runspecs.return_value = [task_runspec1, task_runspec2]

            # Setup future mock
            mock_future = MagicMock(spec=Future)
            mock_future.result.return_value = None
            mock_submit.return_value = mock_future

            # Execute
            runner.run_batch(runspec)

            # Verify _init_executor was called
            mock_init_executor.assert_called_once_with("batch")

            # Verify _events_ctxmgr was called with event handlers
            mock_events_ctx.assert_called_once()

            # Verify _get_runspecs was called
            mock_get_runspecs.assert_called_once_with(runspec)

            # Verify _submit_with_events was called for each task
            assert mock_submit.call_count == 2

            # Verify on_determined_batch_task_count was called
            mock_determined.assert_called_once_with(runspec, 2)

            # Verify futures were awaited
            assert mock_future.result.call_count == 2

    def test_run_batch_with_single_task(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test run_batch with a single task."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_file, output_dir)

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "_submit_with_events") as mock_submit,
            patch.object(runner, "on_determined_batch_task_count") as mock_determined,
        ):
            # Setup mocks
            mock_executor = MagicMock()
            mock_init_executor.return_value.__enter__ = dummy_estimator_enter(
                mock_executor
            )
            mock_init_executor.return_value.__exit__ = dummy_exit

            mock_events_ctx.return_value.__enter__ = dummy_enter
            mock_events_ctx.return_value.__exit__ = dummy_exit

            mock_get_runspecs.return_value = [runspec]

            mock_future = MagicMock(spec=Future)
            mock_future.result.return_value = None
            mock_submit.return_value = mock_future

            # Execute
            runner.run_batch(runspec)

            # Verify task count
            mock_determined.assert_called_once_with(runspec, 1)

            # Verify single task submitted
            assert mock_submit.call_count == 1

    def test_run_batch_handles_task_failure(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test that run_batch handles task failures gracefully."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test.mp4").touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_dir, output_dir)

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "_submit_with_events") as mock_submit,
        ):
            # Setup mocks
            mock_executor = MagicMock()
            mock_init_executor.return_value.__enter__ = dummy_estimator_enter(
                mock_executor
            )
            mock_init_executor.return_value.__exit__ = dummy_exit

            mock_events_ctx.return_value.__enter__ = dummy_enter
            mock_events_ctx.return_value.__exit__ = dummy_exit

            mock_get_runspecs.return_value = [runspec]

            # Setup future that raises exception
            mock_future = MagicMock(spec=Future)
            mock_future.result.side_effect = RuntimeError("Task failed")
            mock_submit.return_value = mock_future

            # Execute - should not raise exception
            runner.run_batch(runspec)

            # Verify future.result was called (exception was handled)
            mock_future.result.assert_called_once()

    def test_run_batch_with_no_tasks(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test run_batch with empty task list."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_dir, output_dir)

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "on_determined_batch_task_count") as mock_determined,
        ):
            # Setup mocks
            mock_executor = MagicMock()
            mock_init_executor.return_value.__enter__ = dummy_estimator_enter(
                mock_executor
            )
            mock_init_executor.return_value.__exit__ = dummy_exit

            mock_events_ctx.return_value.__enter__ = dummy_enter
            mock_events_ctx.return_value.__exit__ = dummy_exit

            # No tasks
            mock_get_runspecs.return_value = []

            # Execute
            runner.run_batch(runspec)

            # Verify task count is 0
            mock_determined.assert_called_once_with(runspec, 0)

    def test_run_batch_executor_context_management(
        self, runner: LMPipeRunner[Literal["test"]], tmp_path: Path
    ):
        """Test that run_batch properly manages executor context."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_file, output_dir)

        enter_called = False
        exit_called = False

        class MockContextManager:
            def __enter__(self):
                nonlocal enter_called
                enter_called = True
                return MagicMock()

            def __exit__(self, *args: Any):
                nonlocal exit_called
                exit_called = True
                return None

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "_submit_with_events") as mock_submit,
        ):
            mock_init_executor.return_value = MockContextManager()

            mock_events_ctx.return_value.__enter__ = dummy_enter
            mock_events_ctx.return_value.__exit__ = dummy_exit

            mock_get_runspecs.return_value = [runspec]

            mock_future = MagicMock(spec=Future)
            mock_future.result.return_value = None
            mock_submit.return_value = mock_future

            # Execute
            runner.run_batch(runspec)

            # Verify context manager was used
            assert enter_called, "Executor context manager __enter__ was not called"
            assert exit_called, "Executor context manager __exit__ was not called"

    def test_run_batch_events_context_management(
        self,
        runner: LMPipeRunner[Literal["test"]],
        tmp_path: Path
    ):
        """Test that run_batch properly manages events context."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runspec = RunSpec(video_file, output_dir)

        event_enter_called = False
        event_exit_called = False

        class MockEventContextManager:
            def __enter__(self):
                nonlocal event_enter_called
                event_enter_called = True
                return None

            def __exit__(self, *args: Any):
                nonlocal event_exit_called
                event_exit_called = True
                return None

        with (
            patch.object(runner, "_init_executor") as mock_init_executor,
            patch.object(runner, "_events_ctxmgr") as mock_events_ctx,
            patch.object(runner, "_get_runspecs") as mock_get_runspecs,
            patch.object(runner, "_submit_with_events") as mock_submit,
        ):
            mock_executor = MagicMock()
            mock_init_executor.return_value.__enter__ = dummy_estimator_enter(
                mock_executor
            )
            mock_init_executor.return_value.__exit__ = dummy_exit

            mock_events_ctx.return_value = MockEventContextManager()

            mock_get_runspecs.return_value = [runspec]

            mock_future = MagicMock(spec=Future)
            mock_future.result.return_value = None
            mock_submit.return_value = mock_future

            # Execute
            runner.run_batch(runspec)

            # Verify events context was used
            assert event_enter_called, "Events context manager __enter__ was not called"
            assert event_exit_called, "Events context manager __exit__ was not called"
