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

"""LMPipe runner module for pipeline execution coordination.

This module provides the LMPipeRunner class and related utilities for coordinating
pipeline execution, managing executors, and handling events during processing.

The module contains:
    - LMPipeRunner: Internal runner class for handling execution and coordination
    - _runner_public_api: Decorator for runner public API methods
    - Utility classes for event handling and executor management

Example:
    Creating a custom runner:

    >>> from cslrtools2.lmpipe.interface.runner import LMPipeRunner
    >>> class MyRunner(LMPipeRunner):
    ...     def on_start_batch_job(self, runspec):
    ...         print(f"Starting batch job: {runspec.src}")
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any,
    Callable, Concatenate,
    Iterable
)
from types import FrameType
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import signal
from threading import local, get_ident
from concurrent.futures import Future, Executor, ThreadPoolExecutor

import cv2

from ...typings import MatLike
from ..logger import lmpipe_logger
from ..typings import ExecutorMode
from ..options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS
from ..estimator import Estimator, ProcessResult
from .executor import DummyExecutor, ProcessPoolExecutor
from ..collector import Collector
from ..runspec import RunSpec
from ..utils import (
    capture_to_frames,
    image_file_to_frame,
    is_image_file,
    is_images_dir,
    is_video_file,
    seq_imgs_to_frames,
)

if TYPE_CHECKING:
    # このファイルからはエクスポートされていないため、
    # これを別のファイルからインポートしたら違反
    from . import LMPipeInterface


__all__ = ["LMPipeRunner", "_runner_public_api"]


def _runner_public_api[S: LMPipeRunner[Any], **P, R](
    func: Callable[Concatenate[S, P], R]
) -> Callable[Concatenate[S, P], R | None]:

    @wraps(func)
    def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R | None:
        if not self.toplevel_call:
            return func(self, *args, **kwargs)
        self.toplevel_call = False

        try:
            ret = func(self, *args, **kwargs)
            self.on_complete()
            lmpipe_logger.info(f"Task completed successfully: {func.__name__}")
            return ret
        except KeyboardInterrupt as e:
            lmpipe_logger.warning(f"Task interrupted by user: {func.__name__}")
            self.on_keyboard_interrupt(e)
        except Exception as e:
            lmpipe_logger.error(
                f"Task failed with exception: {func.__name__}", exc_info=True
            )
            lmpipe_logger.error(
                f"Task failed with exception: {func.__name__}", exc_info=True
            )
            self.on_general_exception(e)
        finally:
            lmpipe_logger.debug(f"Shutting down executors for task: {func.__name__}")
            self.toplevel_call = True
            for executor in self.executors.values():
                executor.shutdown(wait=False, cancel_futures=True)
            self.on_finally()
            lmpipe_logger.info(f"Task finalized: {func.__name__}")

    return wrapper


class LMPipeRunner[K: str]:
    """Internal runner class for coordinating pipeline execution.

    This class is responsible for the actual execution of processing tasks,
    coordinating between estimators, executors, and collectors. It handles
    parallel execution, event management, and resource allocation.

    The runner is typically instantiated by LMPipeInterface and should not
    be used directly by end users. It provides extensibility points through
    method overrides and event handlers for custom processing workflows.

    Note:
        This class is designed for extension. Subclasses can override methods
        like configure_executor() and event handlers to customize behavior.
    """

    toplevel_call: bool = True
    "Indicates if this is the top-level call in nested executions."

    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "executors": {},  # Exclude executors from serialization
        }

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)

    def __init__(
        self,
        interface: LMPipeInterface[K],
        options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS,
    ):
        """Initialize LMPipeRunner with interface and options.

        Sets up the runner with necessary components including collectors,
        thread-local storage, and configuration. This constructor is typically
        called internally by :class:`LMPipeInterface` methods.

        Args:
            interface (:class:`LMPipeInterface`\\[:obj:`K`\\]):
                Parent :class:`LMPipeInterface` instance.
            options (:class:`~cslrtools2.lmpipe.options.LMPipeOptions`):
                Configuration options for execution.
                Defaults to :obj:`DEFAULT_LMPIPE_OPTIONS`.

        Example:
            Typically instantiated internally::

                # Internal usage by LMPipeInterface
                runner = LMPipeRunner(interface, options)

            Custom runner subclass::

                class MyCustomRunner(LMPipeRunner):
                    def __init__(self, interface, options):
                        super().__init__(interface, options)
                        # Additional initialization
                        self.custom_state = {}
        """

        self._main_tid = get_ident()
        "Main thread identifier for thread-local storage."
        self._main_pid = os.getpid()
        "Main process identifier for process-local storage."

        self._id = id(self)
        "Unique identifier used to bind runners in thread-local storage."

        self.lmpipe_interface = interface
        "Parent interface coordinating execution."
        self.lmpipe_options = options
        "Runner-specific configuration snapshot."

        self.executors: dict[ExecutorMode, Executor] = {}
        "Executor instances used for parallel processing, keyed by execution mode."

        self.collectors = self._configure_collectors()
        "Active collectors receiving processing results."

        lmpipe_logger.debug(
            f"LMPipeRunner initialized: id={self._id}, "
            f"collectors={len(self.collectors)}, "
            f"runner_type={type(self).__name__}"
        )

    @_runner_public_api
    def run(self, runspec: RunSpec[Path]):
        """Execute processing based on run specification with automatic type detection.

        Analyzes the source path in the run specification and delegates to the
        appropriate processing method (:meth:`run_batch` for directories,
        :meth:`run_single` for files).
        This is the main entry point for runner-level execution.

        Args:
            runspec (`RunSpec[Path]`):
                Run specification containing source and destination paths.

        Raises:
            ValueError: If source path type is not supported.
            FileNotFoundError: If source path does not exist.

        Example:
            Internal usage by LMPipeInterface::

                runspec = RunSpec.from_pathlikes('input/', 'output/')
                runner = LMPipeRunner(interface, options)
                runner.run(runspec)
        """
        lmpipe_logger.info(f"Runner.run: processing {runspec.src}")

        if runspec.src.is_dir():
            lmpipe_logger.debug("Detected directory, delegating to run_batch")
            return self.run_batch(runspec)

        if runspec.src.is_file():
            lmpipe_logger.debug("Detected file, delegating to run_single")
            return self.run_single(runspec)

        if runspec.src.exists():
            lmpipe_logger.error(f"Unsupported source path type: {runspec.src}")
            raise ValueError(f"Unsupported source path: {runspec.src}")

        lmpipe_logger.error(f"Source path does not exist: {runspec.src}")
        raise FileNotFoundError(f"Source path does not exist: {runspec.src}")

    @_runner_public_api
    def run_batch(self, runspec: RunSpec[Path]):
        """Execute batch processing on multiple files using parallel execution.

        Processes multiple files in the source directory using the configured
        executor for parallel processing. Handles task distribution, event
        management, and result collection across all batch tasks. Each file
        is processed independently, enabling efficient parallelization.

        Args:
            runspec (`RunSpec[Path]`):
                Run specification with source directory and destination.

        Example:
            Process multiple files in parallel::

                # Internally called by LMPipeInterface.run_batch()
                runspec = RunSpec.from_pathlikes('videos/', 'results/')
                runner = LMPipeRunner(interface, options)
                runner.run_batch(runspec)
        """
        lmpipe_logger.info(f"Starting batch processing: {runspec.src}")

        with (
            self._init_executor("batch") as executor,
            self._events_ctxmgr(self.on_start_batch_job, self.on_end_batch_job, runspec)
        ):

            lmpipe_logger.debug(f"Batch executor configured: {type(executor).__name__}")

            futures = [
                self._submit_with_events(
                    executor.submit(
                        self._task_with_events(
                            task=self._local_runner_method(
                                type(self).run_single, self, task_runspec
                            ),
                            on_start=self._local_runner_method(
                                type(self).on_start_batch_task,
                                self,
                                task_runspec,
                                task_id,
                            ),
                            on_end=self._local_runner_method(
                                type(self).on_end_batch_task,
                                self,
                                task_runspec,
                                task_id,
                            ),
                        )
                    ),
                    self.on_submit_batch_task,
                    self.on_success_batch_task,
                    self.on_failure_batch_task,
                    task_runspec,
                    task_id,
                )
                for task_id, task_runspec in enumerate(self._get_runspecs(runspec))
            ]

            self.on_determined_batch_task_count(runspec, len(futures))

            for ftr in futures:
                try:
                    ftr.result()  # Propagate exceptions
                    lmpipe_logger.debug(f"Batch task completed successfully: {ftr}")
                except Exception as e:
                    import traceback
                    lmpipe_logger.error(f"Batch task failed: {ftr} with exception: {e}")
                    lmpipe_logger.error(f"Traceback:\n{traceback.format_exc()}")
                    pass  # Handled in on_failure_batch_task
                    pass  # Handled in on_failure_batch_task

    @_runner_public_api
    def run_single(self, runspec: RunSpec[Path]):
        """Execute processing on a single file with automatic type detection.

        Analyzes the file type and delegates to the appropriate specialized
        processing method (video, image sequence, or single image).

        Args:
            runspec (`RunSpec[Path]`):
                Run specification with source file and destination.

        Raises:
            ValueError: If file type is not supported.

        Subclass Development:
            Override to add custom file type detection or preprocessing::

                class MyRunner(LMPipeRunner):
                    def run_single(self, runspec):
                        # Custom preprocessing
                        self.preprocess_file(runspec.src)

                        # Call parent implementation
                        return super().run_single(runspec)
        """
        lmpipe_logger.info(f"Processing single file: {runspec.src}")

        if is_video_file(runspec.src):
            lmpipe_logger.debug("Detected video file, delegating to run_video")
            return self.run_video(runspec)

        if is_images_dir(runspec.src):
            lmpipe_logger.debug(
                "Detected image directory, delegating to run_sequence_images"
            )
            lmpipe_logger.debug(
                "Detected image directory, delegating to run_sequence_images"
            )
            return self.run_sequence_images(runspec)

        if is_image_file(runspec.src):
            lmpipe_logger.debug("Detected single image, delegating to run_single_image")
            return self.run_single_image(runspec)

        lmpipe_logger.error(f"Unsupported file type: {runspec.src}")
        raise ValueError(f"Unsupported source path: {runspec.src}")

    @_runner_public_api
    def run_video(self, runspec: RunSpec[Path]):
        """Execute processing on a video file.

        Opens the video file, extracts frames sequentially, and processes
        each frame through the estimator pipeline.

        Args:
            runspec (`RunSpec[Path]`):
                Run specification with source video file and destination.

        Raises:
            ValueError: If video file cannot be opened.

        Subclass Development:
            Override to add custom video processing logic::

                class MyRunner(LMPipeRunner):
                    def run_video(self, runspec):
                        # Custom video opening with additional parameters
                        capture = cv2.VideoCapture(str(runspec.src))
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

                        if not capture.isOpened():
                            raise ValueError(f"Cannot open: {runspec.src}")

                        # Custom frame processing
                        results = self.process_frames(
                            self.preprocess_video_frames(
                                capture_to_frames(capture)
                            )
                        )
                        self._collect_results(runspec, results)
        """
        lmpipe_logger.info(f"Processing video file: {runspec.src}")
        capture = cv2.VideoCapture(str(runspec.src))

        if not capture.isOpened():
            lmpipe_logger.error(f"Cannot open video file: {runspec.src}")
            raise ValueError(f"Cannot open video file: {runspec.src}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        lmpipe_logger.debug(f"Video properties: frames={frame_count}, fps={fps}")

        results = self.process_frames(capture_to_frames(capture))

        self._collect_results(runspec, results)
        lmpipe_logger.info(f"Video processing completed: {runspec.src}")

    @_runner_public_api
    def run_sequence_images(self, runspec: RunSpec[Path]):
        """Execute processing on an image sequence directory.

        Processes all images in the directory as a sequence, maintaining
        temporal order for frame-based analysis.

        Args:
            runspec (:class:`RunSpec`[Path]):
                Run specification with source directory and destination.
        """
        lmpipe_logger.info(f"Processing image sequence: {runspec.src}")
        image_count = len(list(runspec.src.glob("*")))
        lmpipe_logger.debug(f"Image sequence contains {image_count} files")

        results = self.process_frames(seq_imgs_to_frames(runspec.src))

        self._collect_results(runspec, results)
        lmpipe_logger.info(f"Image sequence processing completed: {runspec.src}")

    @_runner_public_api
    def run_single_image(self, runspec: RunSpec[Path]):
        """Execute processing on a single image file.

        Loads and processes a single image through the estimator pipeline.

        Args:
            runspec (:class:`RunSpec`[Path]):
                Run specification with source image file and destination.
        """
        lmpipe_logger.info(f"Processing single image: {runspec.src}")
        results = self.process_frames([image_file_to_frame(runspec.src)])

        self._collect_results(runspec, results)
        lmpipe_logger.info(f"Single image processing completed: {runspec.src}")

    @_runner_public_api
    def run_stream(self, runspec: RunSpec[int]):
        """Execute processing on a live video stream.

        Captures frames from a live video stream (e.g., camera) and processes
        them in real-time through the estimator pipeline.

        Args:
            runspec (`RunSpec[int]`):
                Run specification with stream index and destination.

        Raises:
            ValueError: If video stream cannot be opened.

        Subclass Development:
            Override to add stream-specific handling::

                class MyRunner(LMPipeRunner):
                    def run_stream(self, runspec):
                        # Configure camera settings
                        capture = cv2.VideoCapture(runspec.src)
                        capture.set(cv2.CAP_PROP_FPS, 30)
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

                        if not capture.isOpened():
                            raise ValueError(f"Cannot open: {runspec.src}")

                        # Add frame buffering
                        results = self.process_frames(
                            self.buffer_stream_frames(
                                capture_to_frames(capture)
                            )
                        )
                        self._collect_results(runspec, results)
        """
        lmpipe_logger.info(f"Starting stream processing: device={runspec.src}")
        capture = cv2.VideoCapture(runspec.src)

        if not capture.isOpened():
            lmpipe_logger.error(f"Cannot open video stream: device={runspec.src}")
            raise ValueError(f"Cannot open video stream: {runspec.src}")

        lmpipe_logger.debug(f"Stream opened successfully: device={runspec.src}")
        results = self.process_frames(capture_to_frames(capture))

        self._collect_results(runspec, results)
        lmpipe_logger.info(f"Stream processing completed: device={runspec.src}")

    def process_frames(self, frames: Iterable[MatLike]) -> Iterable[ProcessResult[K]]:
        """Process a sequence of frames through the estimator pipeline.

        Coordinates the processing of multiple frames using parallel execution.
        Manages resource allocation, task submission, and result collection
        for efficient frame-by-frame processing.

        Args:
            frames (`Iterable[MatLike]`): Iterable of frame matrices to process.

        Yields:
            :class:`ProcessResult[K]`:
                ProcessResult objects containing landmarks and annotated frames.

        Subclass Development:
            Override to implement custom frame processing strategies::

                class MyRunner(LMPipeRunner):
                    def process_frames(self, frames):
                        # Add frame preprocessing
                        preprocessed = (self.preprocess(f) for f in frames)

                        # Use parent's parallel processing
                        for result in super().process_frames(preprocessed):
                            # Add postprocessing
                            yield self.postprocess_result(result)

            Or implement completely custom parallel processing::

                class MyRunner(LMPipeRunner):
                    def process_frames(self, frames):
                        frame_list = list(frames)
                        with self.configure_executor(
                            "frames",
                            self._default_executor_initializer
                        ) as executor:
                            futures = [
                                executor.submit(self._call_estimator, frame, idx)
                                for idx, frame in enumerate(frame_list)
                            ]
                            for future in futures:
                                yield future.result()
        """
        lmpipe_logger.debug("Starting frame processing pipeline")

        with (
            self._get_executor_resources(self.lmpipe_interface.estimator),
            self._init_executor("frames") as executor,
            self._events_ctxmgr(
                self.on_start_frames_job,
                self.on_end_frames_job,
            )
        ):

            lmpipe_logger.debug(f"Frame executor configured: {type(executor).__name__}")

            futures = [
                self._submit_with_events(
                    executor.submit(
                        self._task_with_events(
                            self._local_runner_method(
                                type(self)._call_estimator, self, frame_src, frame_idx
                            ),
                            self._local_runner_method(
                                type(self).on_start_frames_task, self, task_id=frame_idx
                            ),
                            self._local_runner_method(
                                type(self).on_end_frames_task, self, task_id=frame_idx
                            ),
                        )
                    ),
                    self.on_submit_frames_task,
                    self.on_success_frames_task,
                    self.on_failure_frames_task,
                    frame_idx,
                )
                for frame_idx, frame_src in enumerate(frames)
            ]

            self.on_determined_frames_task_count(len(futures))
            lmpipe_logger.debug(f"Processing {len(futures)} frames")

            for ftr in futures:
                yield ftr.result()

            lmpipe_logger.debug(
                f"Frame processing completed: {len(futures)} frames processed"
            )
            lmpipe_logger.debug(
                f"Frame processing completed: {len(futures)} frames processed"
            )

    def _call_estimator(self, frame_src: MatLike, frame_idx: int) -> ProcessResult[K]:
        """Execute estimator on a single frame.

        Processes a single frame through the configured estimator, performing
        setup on the first frame, landmark estimation, and frame annotation.

        Args:
            frame_src (`MatLike`): Input frame matrix to process.
            frame_idx (`int`): Index/ID of the frame in the sequence.

        Returns:
            :class:`ProcessResult[K]`:
                ProcessResult containing frame ID, headers, landmarks,
                and annotated frame.

        Subclass Development:
            Override to customize single frame processing::

                class MyRunner(LMPipeRunner):
                    def _call_estimator(self, frame_src, frame_idx):
                        # Add frame-level preprocessing
                        preprocessed = self.denoise(frame_src)

                        # Custom setup logic
                        if frame_idx == 0:
                            self.lmpipe_interface.estimator.setup()
                            self.initialize_tracking()

                        # Get estimator results
                        headers = self.lmpipe_interface.estimator.headers
                        landmarks = self.lmpipe_interface.estimator.estimate(
                            preprocessed, frame_idx
                        )

                        # Custom annotation
                        annotated_frame = self.custom_annotate(
                            frame_src, landmarks
                        )

                        return ProcessResult(
                            frame_id=frame_idx,
                            headers=headers,
                            landmarks=landmarks,
                            annotated_frame=annotated_frame
                        )
        """
        if frame_idx == 0:
            lmpipe_logger.debug("Setting up estimator for first frame")
            self.lmpipe_interface.estimator.setup()

        headers = self.lmpipe_interface.estimator.headers
        landmarks = self.lmpipe_interface.estimator.estimate(frame_src, frame_idx)
        annotated_frame = self.lmpipe_interface.estimator.annotate(
            frame_src, frame_idx, landmarks
        )

        if frame_idx % 100 == 0:
            lmpipe_logger.debug(f"Processed frame {frame_idx}")

        return ProcessResult(
            frame_id=frame_idx,
            headers=headers,
            landmarks=landmarks,
            annotated_frame=annotated_frame,
        )

    @contextmanager
    def _get_executor_resources(self, estimator: Estimator[K]):
        """Context manager for allocating executor resources.

        Args:
            estimator (:class:`Estimator`[K]):
                The estimator that will use the resources.

        Note:
            This is a placeholder for future resource allocation logic.
        """
        yield  # TODO: implement resource allocation logic

    def _collect_results(
        self,
        runspec: RunSpec[Any],
        results: Iterable[ProcessResult[K]]
    ):
        """Collect processing results using configured collectors.

        Args:
            runspec (`RunSpec[Any]`): Run specification for context.
            results (`Iterable[ProcessResult[K]]`):
                Iterable of processing results to collect.

        Subclass Development:
            Override to add custom result processing before collection::

                class MyRunner(LMPipeRunner):
                    def _collect_results(self, runspec, results):
                        # Filter or transform results
                        filtered = (r for r in results if self.is_valid(r))

                        # Call parent's collection
                        super()._collect_results(runspec, filtered)

                        # Add custom post-collection processing
                        self.generate_summary_report(runspec)
        """
        lmpipe_logger.debug(
            f"Collecting results with {len(self.collectors)} collectors"
        )

        # Consume the iterable to allow multiple passes
        collected_results = list(results)

        for collector in self.collectors:
            collector_name = type(collector).__name__
            lmpipe_logger.debug(f"Running collector: {collector_name}")
            collector.collect_results(runspec, collected_results)
        lmpipe_logger.debug("Result collection completed")

    # ============================================================
    # Event Handlers
    # ============================================================

    def on_start_batch_job(self, runspec: RunSpec[Path]):
        """Event handler called when a batch job starts.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.

        Subclass Development:
            Override to add custom batch job initialization::

                class MyRunner(LMPipeRunner):
                    def on_start_batch_job(self, runspec):
                        print(f"Starting batch processing: {runspec.src}")
                        self.batch_start_time = time.time()
                        self.initialize_progress_bar()
        """
        ...

    def on_submit_batch_task(self, runspec: RunSpec[Path], task_id: int):
        """Event handler called when a batch task is submitted.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_id (`int`): Unique identifier for the task.
        """
        ...

    def on_start_batch_task(self, runspec: RunSpec[Path], task_id: int):
        """Event handler called when a batch task starts execution.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_id (`int`): Unique identifier for the task.
        """
        ...

    def on_end_batch_task(self, runspec: RunSpec[Path], task_id: int):
        """Event handler called when a batch task completes.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_id (`int`): Unique identifier for the task.
        """
        ...

    def on_success_batch_task(self, runspec: RunSpec[Path], task_id: int, result: Any):
        """Event handler called when a batch task completes successfully.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_id (`int`): Unique identifier for the task.
            result (`Any`): Result returned by the successful task.

        Subclass Development:
            Override to track successful task completions::

                class MyRunner(LMPipeRunner):
                    def on_success_batch_task(self, runspec, task_id, result):
                        self.completed_tasks += 1
                        self.update_progress(self.completed_tasks, self.total_tasks)
                        self.log_success(task_id, result)
        """
        ...

    def on_failure_batch_task(
        self,
        runspec: RunSpec[Path],
        task_id: int,
        error: Exception
    ):
        """Event handler called when a batch task fails.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_id (`int`): Unique identifier for the task.
            error (`Exception`): Exception that caused the task failure.

        Subclass Development:
            Override to implement custom error handling::

                class MyRunner(LMPipeRunner):
                    def on_failure_batch_task(self, runspec, task_id, error):
                        self.failed_tasks.append((task_id, error))
                        self.logger.error(f"Task {task_id} failed: {error}")

                        # Optionally retry
                        if self.should_retry(error):
                            self.retry_task(task_id)
        """
        ...

    def on_end_batch_job(self, runspec: RunSpec[Path]):
        """Event handler called when a batch job ends.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
        """
        ...

    def on_determined_batch_task_count(self, runspec: RunSpec[Path], task_count: int):
        """Event handler called when batch task count is determined.

        Args:
            runspec (`RunSpec[Path]`): Run specification for the batch job.
            task_count (`int`): Total number of tasks in the batch.
        """
        ...

    ###### Frame Events ######

    def on_start_frames_job(self):
        """Event handler called when frame processing job starts."""
        ...

    def on_submit_frames_task(self, task_id: int):
        """Event handler called when a frame processing task is submitted.

        Args:
            task_id (`int`): Unique identifier for the frame task.
        """
        ...

    def on_start_frames_task(self, task_id: int):
        """Event handler called when a frame processing task starts.

        Args:
            task_id (`int`): Unique identifier for the frame task.
        """
        ...

    def on_end_frames_task(self, task_id: int):
        """Event handler called when a frame processing task completes.

        Args:
            task_id (`int`): Unique identifier for the frame task.
        """
        ...

    def on_success_frames_task(self, task_id: int, result: Any):
        """Event handler called when a frame processing task succeeds.

        Args:
            task_id (`int`): Unique identifier for the frame task.
            result (`Any`): Result returned by the successful task.
        """
        ...

    def on_failure_frames_task(self, task_id: int, error: Exception):
        """Event handler called when a frame processing task fails.

        Args:
            task_id (`int`): Unique identifier for the frame task.
            error (`Exception`): :class:`Exception` that caused the task failure.
        """
        ...

    def on_end_frames_job(self):
        """Event handler called when frame processing job ends."""
        ...

    def on_determined_frames_task_count(self, task_count: int):
        """Event handler called when frame task count is determined.

        Args:
            task_count (`int`): Total number of frame processing tasks.
        """
        ...

    def on_keyboard_interrupt(self, error: KeyboardInterrupt) -> None:
        """Event handler called when a keyboard interrupt occurs.

        Args:
            error (`KeyboardInterrupt`): The keyboard interrupt exception.

        Raises:
            :class:`KeyboardInterrupt`:
                Re-raises the keyboard interrupt to halt execution.
        """
        raise error

    def on_general_exception(self, error: Exception) -> None:
        """Event handler called when a general exception occurs.

        Args:
            error (`Exception`): The exception that occurred.

        Raises:
            :class:`Exception`: Re-raises the exception to propagate the error.
        """
        raise error

    def on_complete(self) -> None:
        """Event handler called when all processing is complete."""
        ...

    def on_finally(self) -> None:
        """Event handler called in the finally block after processing."""
        ...

    # ============================================================
    # Executor Configuration
    # ============================================================

    @contextmanager
    def _events_ctxmgr[*Ts](
        self,
        on_start: Callable[[*Ts], Any],
        on_end: Callable[[*Ts], Any],
        *args: *Ts
    ):
        """Context manager for wrapping execution blocks with start/end event handlers.

        Provides a convenient way to ensure event handlers are called at the beginning
        and end of an execution block, with guaranteed cleanup even if exceptions occur.

        Args:
            on_start (`((*Ts) -> Any)`):
                Event handler to call when entering the context.
            on_end (`((*Ts) -> Any)`):
                Event handler to call when exiting the context.
            *args (`*Ts`): Arguments to pass to both event handlers.

        Yields:
            None: Control is yielded to the wrapped execution block.
        """
        on_start(*args)
        try:
            yield
        finally:
            on_end(*args)

    def _submit_with_events[*Ts, R](
        self,
        ftr: Future[R],
        on_submit: Callable[[*Ts], Any],
        on_success: Callable[[*Ts, R], Any],
        on_failure: Callable[[*Ts, Exception], Any],
        *args: *Ts
    ) -> Future[R]:
        """Submit a task with event callbacks for submission, success, and failure.

        Wraps a Future with event handlers that are triggered at different stages
        of task execution: immediately on submission, on successful completion, or
        on failure. This provides a unified event-driven interface for task monitoring.

        Args:
            ftr (`Future[R]`): The future representing the submitted task.
            on_submit (`((*Ts) -> Any)`)
                Event handler called immediately when task is submitted.
            on_success (`((*Ts, R) -> Any)`):
                Event handler called when task completes successfully.
            on_failure (`((*Ts, Exception) -> Any)`):
                Event handler called when task fails.
            *args (`*Ts`): Arguments to pass to the event handlers.

        Returns:
            :class:`Future[R]`: The same future with event callbacks attached.
        """

        on_submit(*args)
        ftr.add_done_callback(self._CallbackWithEvents(on_success, on_failure, args))

        return ftr

    class _CallbackWithEvents[*Ts, R]:
        """Internal callback wrapper for handling task success and failure events.

        This class wraps success and failure callbacks to be executed when a
        Future completes, providing a clean interface for event-driven task
        completion handling in the executor framework.
        """

        def __init__(
            self,
            on_success: Callable[[*Ts, R], Any],
            on_failure: Callable[[*Ts, Exception], Any],
            args: tuple[*Ts]
        ):
            self.on_success = on_success
            "Callback invoked when the task succeeds."
            self.on_failure = on_failure
            "Callback invoked when the task fails."
            self.args = args
            "Arguments to pass to the callbacks along with result or error."

        def __call__(self, ftr: Future[R]):
            """Execute the appropriate callback based on task completion status.

            Args:
                ftr (`Future[R]`): The completed future to process.
            """
            try:
                result = ftr.result()
                self.on_success(*self.args, result)
            except Exception as e:
                self.on_failure(*self.args, e)

    class _task_with_events[R]:
        def __init__(
            self,
            task: Callable[[], R],
            on_start: Callable[[], Any],
            on_end: Callable[[], Any],
        ):
            self.task = task
            self.on_start = on_start
            self.on_end = on_end

        def __call__(self) -> R:
            """Execute the task with the stored arguments.

            Returns:
                :class:`R`: Result of the task execution.
            """
            self.on_start()
            result = self.task()
            self.on_end()
            return result

    class _local_runner_method[T: LMPipeRunner[Any], **P, R]:
        """Internal callable wrapper for executing runner methods in worker.

        This class serializes runner method calls for execution across process/thread
        boundaries by storing the method reference, runner ID, and arguments. When
        called in a worker, it retrieves the correct runner instance from thread-local
        storage and invokes the method with the stored arguments.

        This pattern enables safe execution of instance methods in parallel executors
        where the runner instance may not be directly accessible.
        """

        def __init__(
            self,
            method: Callable[Concatenate[T, P], R],
            runner: T, *args: P.args, **kwargs: P.kwargs
        ):

            self.method = method
            "The runner method to be invoked."
            self.runner_type = type(runner)
            "Expected type of the runner instance."
            self.runner_id = runner._id
            "Unique identifier of the runner instance."
            self.args = args
            "Positional arguments to pass to the method."
            self.kwargs = kwargs
            "Keyword arguments to pass to the method."

        def __call__(self) -> R:
            """Execute the stored method with the stored arguments.

            Retrieves the runner instance from thread-local storage and invokes
            the method. Validates runner type and ID to ensure correct instance.

            Returns:
                :class:`R`: Result of the method invocation.

            Raises:
                :class:`TypeError`: If retrieved runner is not of expected type.
                :class:`ValueError`: If runner ID does not match expected ID.
            """

            runner = LMPipeRunner._local.instances[self.runner_id]

            if not isinstance(runner, self.runner_type):
                raise TypeError(
                    f"Expected runner of type {self.runner_type.__name__}, "
                    f"got {type(runner).__name__}."
                )

            if runner._id != self.runner_id:
                raise ValueError(
                    f"Runner ID mismatch: expected {self.runner_id}, got {runner._id}."
                )

            return self.method(runner, *self.args, **self.kwargs)

    class _local(local):
        """Thread-local storage container for runner instances.

        Provides a thread-local dictionary mapping runner IDs to runner instances,
        enabling worker processes/threads to access the appropriate runner instance
        for method execution. This is essential for parallel execution where each
        worker needs its own runner context.
        """

        instances: dict[int, "LMPipeRunner[Any]"] = {}
        "Mapping of runner IDs to runner instances in current thread."

    def _init_executor(self, mode: ExecutorMode):
        if mode in self.executors:
            return self.executors[mode]

        self.executors[mode] = self.configure_executor(
            mode, self._default_executor_initializer
        )

        return self.executors[mode]

    def configure_executor(
        self,
        mode: ExecutorMode,
        initializer: Callable[[], Any]
    ) -> Executor:
        """Configure and return an appropriate executor for the given mode.

        This method can be overridden in subclasses to customize executor
        selection and configuration based on specific requirements.

        Args:
            mode (`ExecutorMode`): Execution mode ("batch" or "frames").
            initializer (`(() -> Any)`): Function to initialize executor workers.

        Returns:
            :class:`Executor`: Configured executor instance based on options and mode.

        Note:
            Returns DummyExecutor if the current mode doesn't match the
            requested mode or if no specific executor type is configured.

        Subclass Development:
            Override this method to implement custom executor logic::

                class MyRunner(LMPipeRunner):
                    def configure_executor(self, mode, initializer):
                        if mode == "batch":
                            return MyCustomBatchExecutor(
                                max_workers=self.lmpipe_options['max_cpus'],
                                initializer=initializer
                            )
                        return super().configure_executor(mode, initializer)
        """
        lmpipe_logger.debug(f"Configuring executor: mode={mode}")

        if (
            self.lmpipe_options['executor_mode'] != mode
            or self.lmpipe_options['max_cpus'] == 0
        ):
            lmpipe_logger.debug(
                f"Using DummyExecutor: executor_mode={
                    self.lmpipe_options['executor_mode']
                }, "
                f"max_cpus={self.lmpipe_options['max_cpus']}"
            )
            return DummyExecutor(initializer=initializer)

        max_cpus = self.lmpipe_options["max_cpus"]
        require_cpus = self.lmpipe_options["cpu"]
        max_workers = int(max(max_cpus / require_cpus, 1))

        if self.lmpipe_options["executor_type"] == "process":
            lmpipe_logger.info(
                f"Using ProcessPoolExecutor: max_workers={max_workers}, "
                f"max_cpus={max_cpus}, require_cpus={require_cpus}"
            )
            # TODO: cpu_tagsを考慮したリソース割り当てロジックを実装する
            lmpipe_logger.debug(
                f"cpu_tags not yet implemented: {
                    self.lmpipe_options.get('cpu_tags', [])
                }"
            )
            return ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=initializer
            )

        if self.lmpipe_options["executor_type"] == "thread":
            lmpipe_logger.info(f"Using ThreadPoolExecutor: max_workers={max_cpus}")
            return ThreadPoolExecutor(max_workers=max_cpus, initializer=initializer)

        lmpipe_logger.debug("No specific executor type configured, using DummyExecutor")
        return DummyExecutor(initializer=initializer)

    def _default_executor_initializer(self):
        """Default initializer function for executor workers.

        Registers this runner instance in thread-local storage for
        access by worker processes/threads.
        """

        self._local.instances[self._id] = self

        if get_ident() == self._main_tid and os.getpid() == self._main_pid:
            return

        signal.signal(
            signal.SIGINT,
            self._sigint_handler_not_for_main_thread(self._main_pid)
        )

    class _sigint_handler_not_for_main_thread:

        def __init__(self, main_pid: int):
            self._main_pid = main_pid

        def __call__(self, signum: int, frame: FrameType | None):
            # Re-raise KeyboardInterrupt in the main process
            os.kill(self._main_pid, signal.SIGINT)

    # ============================================================
    # Batch Processing Configuration
    # ============================================================

    def _get_runspecs(self, runspec: RunSpec[Path]) -> Iterable[RunSpec[Path]]:
        """Generate run specifications for batch processing.

        Walks through the source directory structure and creates individual
        run specifications for each file or image directory that should be processed.

        Args:
            runspec (`RunSpec[Path]`): Base run specification with source directory.

            Yields:
            Individual run specifications for each processable item.
        """
        lmpipe_logger.debug(f"Scanning directory for batch processing: {runspec.src}")
        spec_count = 0

        for dirpath, _, filenames in runspec.src.walk():
            rel_dst = runspec.dst / dirpath.relative_to(runspec.src)

            if any(part.startswith(".") for part in rel_dst.parts):
                lmpipe_logger.debug(f"Skipping hidden directory: {rel_dst}")
                continue

            files = [dirpath / f for f in filenames if not f.startswith(".")]

            all_image_files = files and all(
                is_image_file(file_path) for file_path in files
            )

            if all_image_files:
                lmpipe_logger.debug(
                    f"Processing directory as image sequence: {dirpath}"
                )
                lmpipe_logger.debug(
                    f"Processing directory as image sequence: {dirpath}"
                )
                iterable = [RunSpec(dirpath, rel_dst)]
            else:
                lmpipe_logger.debug(
                    f"Processing individual files in directory: {dirpath}"
                )
                lmpipe_logger.debug(
                    f"Processing individual files in directory: {dirpath}"
                )
                iterable = (
                    RunSpec(file_path, rel_dst / file_path.name)
                    for file_path in files
                    if is_video_file(file_path) or is_image_file(file_path)
                )

            for spec in iterable:
                if self._apply_exist_rule(spec):
                    spec_count += 1
                    yield spec

        lmpipe_logger.debug(f"Found {spec_count} files/directories to process")

    def _configure_collectors(self) -> list[Collector[K]]:
        """Configure and return list of result collectors.

        Returns:
            `list[Collector[K]]`: List of configured collector instances.

        Note:
            This is a placeholder for future collector configuration logic.

        Subclass Development:
            Override to add custom collector initialization::

                class MyRunner(LMPipeRunner):
                    def _configure_collectors(self):
                        # Get default collectors
                        collectors = super()._configure_collectors()

                        # Add custom collectors based on options
                        if self.lmpipe_options.get('enable_csv_export'):
                            collectors.append(CsvLandmarkMatrixSaveCollector())

                        return collectors
        """
        lmpipe_logger.debug("Configuring collectors")

        if callable(self.lmpipe_interface.collectors_or_factory):
            lmpipe_logger.debug("Using collector factory function")
            collectors = self.lmpipe_interface.collectors_or_factory()
        else:
            lmpipe_logger.debug("Using pre-configured collectors")
            collectors = self.lmpipe_interface.collectors_or_factory

        if not collectors:
            lmpipe_logger.warning(
                "No collectors configured - results will not be saved"
            )
            lmpipe_logger.warning(
                "No collectors configured - results will not be saved"
            )
        else:
            collector_names = [type(c).__name__ for c in collectors]
            lmpipe_logger.info(
                f"Configured {len(collectors)} collectors: {', '.join(collector_names)}"
            )
            lmpipe_logger.info(
                f"Configured {len(collectors)} collectors: {', '.join(collector_names)}"
            )

        return collectors

    def _apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        """
        Apply existence rules to determine if a runspec should be processed.`

        Args:
            runspec (`RunSpec[Any]`): Run specification to evaluate.

        Returns:
            :code:`bool`: True if the runspec should be processed, False otherwise.
        """
        result = any(cllctr.apply_exist_rule(runspec) for cllctr in self.collectors)
        if not result:
            lmpipe_logger.warning(
                f"Skipping existing file per collector rules: {runspec.src}"
            )
            lmpipe_logger.warning(
                f"Skipping existing file per collector rules: {runspec.src}"
            )
        return result
