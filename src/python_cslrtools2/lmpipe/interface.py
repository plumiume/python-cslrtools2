"""LMPipe interface module for computer vision processing pipeline.

This module provides the main interface and runner classes for the LMPipe system,
which enables batch processing of images, videos, and streams using machine learning
estimators with configurable execution modes and collection strategies.

The module contains:
    - LMPipeInterface: Main user interface for running CV processing tasks
    - LMPipeRunner: Internal runner class for handling execution and coordination
    - Utility functions for file type detection and frame processing

Example:
    Basic usage with an estimator:
    
    >>> from python_cslrtools2.lmpipe import LMPipeInterface
    >>> interface = LMPipeInterface(my_estimator)
    >>> interface.run('input.mp4', 'output/')
"""

from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from contextlib import contextmanager
from pathlib import Path
from threading import local
from concurrent.futures import wait, Future, Executor, ThreadPoolExecutor

import cv2

from .typings import PathLike, MatLike, ExecutorMode
from .options import LMPipeOptions, LMPipeOptionsPartial, DEFAULT_LMPIPE_OPTIONS
from .estimator import Estimator, ProcessResult
from .executor import DummyExecutor, ProcessPoolExecutor
from .collector import Collector
from .runspec import RunSpec
from .utils import (
    capture_to_frames,
    image_file_to_frame,
    is_image_file,
    is_images_dir,
    is_video_file,
    seq_imgs_to_frames,
)

# cp314 ready
type LMPipeRunner[K: str] = "LMPipeRunner[K]" # pyright: ignore[reportRedeclaration]

def _update_lmpipe_options(
    base: LMPipeOptions,
    *updates: LMPipeOptionsPartial
    ) -> LMPipeOptions:
    """Update LMPipe options by merging partial options into base options.
    
    Args:
        base (LMPipeOptions): Base options to start with.
        *updates (LMPipeOptionsPartial): Variable number of partial options to merge.
        
    Returns:
        :class:`LMPipeOptions`: Updated options with all partial options merged.
    """
    ret = base.copy()
    for opt in updates:
        ret.update(opt)
    return ret

class LMPipeInterface[K: str]:
    """Main interface for LMPipe computer vision processing pipeline.
    
    This class provides the primary interface for users to execute various types
    of computer vision processing tasks including single images, image sequences,
    videos, and live streams. It handles automatic detection of input types and
    delegates to appropriate processing methods.
    
    The interface supports multiple execution modes and can be configured with
    various options for parallel processing, output collection, and error handling.

    Example:
        Basic usage with video processing:
        
        >>> interface = LMPipeInterface(my_estimator)
        >>> interface.run('input.mp4', 'output_dir/')
        
        Custom options:
        
        >>> interface = LMPipeInterface(
        ...     my_estimator,
        ...     max_cpus=4,
        ...     executor_type='thread'
        ... )
        >>> interface.run_batch('images/', 'results/')
    """

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    def __init__(self, estimator: Estimator[K], **options: Unpack[LMPipeOptionsPartial]):
        """Initialize LMPipeInterface with estimator and options.
        
        Args:
            estimator (:class:`Estimator`[K]): The machine learning estimator to use for processing.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override defaults.
                Common options include max_cpus, executor_type, executor_mode.
        """
        self.estimator = estimator
        "Estimator[K]: Core estimator implementation provided by the caller."
        self.lmpipe_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        "LMPipeOptions: Effective interface configuration after overrides."


    def run(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing pipeline with automatic input type detection.
        
        Automatically detects the type of input (file, directory, video, images)
        and delegates to the appropriate specialized run method.
        
        Args:
            src (PathLike): Source path (file or directory) to process.
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
            
        Returns:
            Processing results from the appropriate runner method.
            
        Raises:
            FileNotFoundError: If source path does not exist.
            ValueError: If source path type is not supported.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run(runspec)
    
    def run_batch(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run batch processing on multiple files in a directory.
        
        Processes all supported files in the source directory using parallel
        execution based on the configured executor settings.
        
        Args:
            src (PathLike): Source directory containing files to process.
            dst (PathLike): Destination directory for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_batch(runspec)
    
    def run_single(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing on a single file with automatic type detection.
        
        Automatically detects if the file is a video, image directory, or single image
        and processes accordingly.
        
        Args:
            src (PathLike): Source file path to process.
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
            
        Raises:
            ValueError: If file type is not supported.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_single(runspec)
    
    def run_video(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing on a video file.
        
        Processes each frame of the video file sequentially or in parallel
        depending on configuration.
        
        Args:
            src (PathLike): Source video file path.
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_video(runspec)
    
    def run_sequence_images(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing on a directory of image sequences.
        
        Processes image files in the directory as a sequence, maintaining
        temporal order where relevant.
        
        Args:
            src (PathLike): Source directory containing image sequence.
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_sequence_images(runspec)
    
    def run_single_image(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing on a single image file.
        
        Processes a single image file using the configured estimator.
        
        Args:
            src (PathLike): Source image file path.
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_single_image(runspec)
    
    def run_stream(self, src: int, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        """Run processing on a live video stream.
        
        Processes frames from a live video stream (e.g., webcam) in real-time.
        
        Args:
            src (int): Video stream index (e.g., 0 for default camera).
            dst (PathLike): Destination path for output results.
            **options (Unpack[LMPipeOptionsPartial]): Optional configuration parameters to override instance defaults.
            
        Raises:
            ValueError: If video stream cannot be opened.
        """
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_index(src, dst)
        return LMPipeRunner(self, updated_options).run_stream(runspec)


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
    
    def __init__(
        self,
        interface: LMPipeInterface[K],
        options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS,
        ):
        """Initialize LMPipeRunner with interface and options.
        
        Args:
            interface (:class:`LMPipeInterface`[K]): Parent LMPipeInterface instance.
            options (:class:`LMPipeOptions`): Configuration options for execution.
        """
        self._id = id(self)
        """int: Unique identifier used to bind runners in thread-local storage."""

        self.lmpipe_interface = interface
        "LMPipeInterface[K]: Parent interface coordinating execution."
        self.lmpipe_options = options
        "LMPipeOptions: Runner-specific configuration snapshot."

        self.collectors = self._configure_collectors()
        """list[Collector[K]]: Active collectors receiving processing results."""


    def run(self, runspec: RunSpec[Path]):
        """Execute processing based on run specification with automatic type detection.
        
        Analyzes the source path in the run specification and delegates to the
        appropriate processing method (batch for directories, single for files).
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification containing source and destination paths.
            
        Returns:
            Processing results from the delegated method.
            
        Raises:
            ValueError: If source path type is not supported.
            FileNotFoundError: If source path does not exist.
        """
        if runspec.src.is_dir():
            return self.run_batch(runspec)
        
        if runspec.src.is_file():
            return self.run_single(runspec)
        
        if runspec.src.exists():
            raise ValueError(f"Unsupported source path: {runspec.src}")
        raise FileNotFoundError(f"Source path does not exist: {runspec.src}")


    def run_batch(self, runspec: RunSpec[Path]):
        """Execute batch processing on multiple files using parallel execution.
        
        Processes multiple files in the source directory using the configured
        executor for parallel processing. Handles task distribution, event
        management, and result collection across all batch tasks.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification with source directory and destination.
        """
        with (
            self.configure_executor(
                "batch", self._default_executor_initializer
            ) as executor,
            self._events_ctxmgr(self.on_start_batch_job, self.on_end_batch_job, runspec)
            ):

            futures = [
                self._submit_with_events(
                    executor.submit(self._local_runner_method(
                        type(self).run_single, self,
                        task_runspec
                    )),
                    self.on_submit_batch_task,
                    self.on_success_batch_task,
                    self.on_failure_batch_task,
                    runspec, task_id
                )
                for task_id, task_runspec in enumerate(
                    self._get_runspecs(runspec)
                )
            ]

            self.on_determined_batch_task_count(runspec, len(futures))

    def run_single(self, runspec: RunSpec[Path]):
        """Execute processing on a single file with automatic type detection.
        
        Analyzes the file type and delegates to the appropriate specialized
        processing method (video, image sequence, or single image).
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification with source file and destination.
            
        Returns:
            Processing results from the delegated method.
            
        Raises:
            ValueError: If file type is not supported.
        """
        if is_video_file(runspec.src):
            return self.run_video(runspec)
        
        if is_images_dir(runspec.src):
            return self.run_sequence_images(runspec)
        
        if is_image_file(runspec.src):
            return self.run_single_image(runspec)
        
        raise ValueError(f"Unsupported source path: {runspec.src}")


    def run_video(self, runspec: RunSpec[Path]):
        """Execute processing on a video file.
        
        Opens the video file, extracts frames sequentially, and processes
        each frame through the estimator pipeline.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification with source video file and destination.
            
        Raises:
            ValueError: If video file cannot be opened.
        """
        capture = cv2.VideoCapture(str(runspec.src))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {runspec.src}")
        
        results = self.process_frames(capture_to_frames(capture))
        
        self._collect_results(runspec, results)

    def run_sequence_images(self, runspec: RunSpec[Path]):
        """Execute processing on an image sequence directory.
        
        Processes all images in the directory as a sequence, maintaining
        temporal order for frame-based analysis.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification with source directory and destination.
        """
        results = self.process_frames(seq_imgs_to_frames(runspec.src))

        self._collect_results(runspec, results)

    def run_single_image(self, runspec: RunSpec[Path]):
        """Execute processing on a single image file.
        
        Loads and processes a single image through the estimator pipeline.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification with source image file and destination.
        """
        results = self.process_frames([image_file_to_frame(runspec.src)])

        self._collect_results(runspec, results)

    def run_stream(self, runspec: RunSpec[int]):
        """Execute processing on a live video stream.
        
        Captures frames from a live video stream (e.g., camera) and processes
        them in real-time through the estimator pipeline.
        
        Args:
            runspec (:class:`RunSpec`[int]): Run specification with stream index and destination.
            
        Raises:
            ValueError: If video stream cannot be opened.
        """
        capture = cv2.VideoCapture(runspec.src)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video stream: {runspec.src}")
        
        results = self.process_frames(capture_to_frames(capture))

        self._collect_results(runspec, results)


    def process_frames(self, frames: Iterable[MatLike]) -> Iterable[ProcessResult[K]]:
        """Process a sequence of frames through the estimator pipeline.
        
        Coordinates the processing of multiple frames using parallel execution.
        Manages resource allocation, task submission, and result collection
        for efficient frame-by-frame processing.
        
        Args:
            frames (Iterable[:class:`MatLike`]): Iterable of frame matrices to process.
            
        Yields:
            :class:`ProcessResult`[K]: ProcessResult objects containing landmarks and annotated frames.
        """
        with (
            self._get_executor_resources(self.lmpipe_interface.estimator),
            self.configure_executor(
                "frames", self._default_executor_initializer
            ) as executor,
            self._events_ctxmgr(
                self.on_start_frames_job,
                self.on_end_frames_job,
            )):
            
            futures = [
                self._submit_with_events(
                    executor.submit(self._local_runner_method(
                        type(self)._call_estimator, self,
                        frame_src, frame_idx
                    )),
                    self.on_submit_frames_task,
                    self.on_success_frames_task,
                    self.on_failure_frames_task,
                    frame_idx
                )
                for frame_idx, frame_src in enumerate(frames)
            ]

            self.on_determined_frames_task_count(len(futures))

            for ftr in futures:
                completes, _ = wait([ftr], return_when='FIRST_COMPLETED')
                yield from (c.result() for c in completes)


    def _call_estimator(self, frame_src: MatLike, frame_idx: int) -> ProcessResult[K]:
        """Execute estimator on a single frame.
        
        Processes a single frame through the configured estimator, performing
        setup on the first frame, landmark estimation, and frame annotation.
        
        Args:
            frame_src (:class:`MatLike`): Input frame matrix to process.
            frame_idx (int): Index/ID of the frame in the sequence.
            
        Returns:
            :class:`ProcessResult`[K]: ProcessResult containing frame ID, headers, landmarks, and annotated frame.
        """
        if frame_idx == 0:
            self.lmpipe_interface.estimator.setup()

        headers = self.lmpipe_interface.estimator.headers
        landmarks = self.lmpipe_interface.estimator.estimate(frame_src, frame_idx)
        annotated_frame = self.lmpipe_interface.estimator.annotate(
            frame_src, frame_idx, landmarks
        )

        return ProcessResult(
            frame_id=frame_idx,
            headers=headers,
            landmarks=landmarks,
            annotated_frame=annotated_frame
        )
    
    @contextmanager
    def _get_executor_resources(self, estimator: Estimator[K]):
        """Context manager for allocating executor resources.
        
        Args:
            estimator (:class:`Estimator`[K]): The estimator that will use the resources.

        Note:
            This is a placeholder for future resource allocation logic.
        """
        yield # TODO: implement resource allocation logic

    def _collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect processing results using configured collectors.
        
        Args:
            runspec (:class:`RunSpec`[Any]): Run specification for context.
            results (Iterable[:class:`ProcessResult`[K]]): Iterable of processing results to collect.
        """
        for collector in self.collectors:
            collector.collect_results(runspec, results)

    ###### Events ######

    def on_start_batch_job(self, runspec: RunSpec[Path]): 
        """Event handler called when a batch job starts.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
        """
        ...
        
    def on_submit_batch_task(self, runspec: RunSpec[Path], task_id: int): 
        """Event handler called when a batch task is submitted.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_id (int): Unique identifier for the task.
        """
        ...
        
    def on_start_batch_task(self, runspec: RunSpec[Path], task_id: int): 
        """Event handler called when a batch task starts execution.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_id (int): Unique identifier for the task.
        """
        ...
        
    def on_end_batch_task(self, runspec: RunSpec[Path], task_id: int): 
        """Event handler called when a batch task completes.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_id (int): Unique identifier for the task.
        """
        ...
        
    def on_success_batch_task(self, runspec: RunSpec[Path], task_id: int, result: Any): 
        """Event handler called when a batch task completes successfully.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_id (int): Unique identifier for the task.
            result (Any): Result returned by the successful task.
        """
        ...
        
    def on_failure_batch_task(self, runspec: RunSpec[Path], task_id: int, error: Exception): 
        """Event handler called when a batch task fails.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_id (int): Unique identifier for the task.
            error (Exception): Exception that caused the task failure.
        """
        ...
        
    def on_end_batch_job(self, runspec: RunSpec[Path]): 
        """Event handler called when a batch job ends.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
        """
        ...
        
    def on_determined_batch_task_count(self, runspec: RunSpec[Path], task_count: int): 
        """Event handler called when batch task count is determined.
        
        Args:
            runspec (:class:`RunSpec`[Path]): Run specification for the batch job.
            task_count (int): Total number of tasks in the batch.
        """
        ...


    def on_start_frames_job(self): 
        """Event handler called when frame processing job starts."""
        ...
        
    def on_submit_frames_task(self, task_id: int): 
        """Event handler called when a frame processing task is submitted.
        
        Args:
            task_id (int): Unique identifier for the frame task.
        """
        ...
        
    def on_start_frames_task(self, task_id: int): 
        """Event handler called when a frame processing task starts.
        
        Args:
            task_id (int): Unique identifier for the frame task.
        """
        ...
        
    def on_end_frames_task(self, task_id: int): 
        """Event handler called when a frame processing task completes.
        
        Args:
            task_id (int): Unique identifier for the frame task.
        """
        ...
        
    def on_success_frames_task(self, task_id: int, result: Any): 
        """Event handler called when a frame processing task succeeds.
        
        Args:
            task_id (int): Unique identifier for the frame task.
            result (Any): Result returned by the successful task.
        """
        ...
        
    def on_failure_frames_task(self, task_id: int, error: Exception): 
        """Event handler called when a frame processing task fails.
        
        Args:
            task_id (int): Unique identifier for the frame task.
            error (Exception): Exception that caused the task failure.
        """
        ...
        
    def on_end_frames_job(self): 
        """Event handler called when frame processing job ends."""
        ...
        
    def on_determined_frames_task_count(self, task_count: int): 
        """Event handler called when frame task count is determined.
        
        Args:
            task_count (int): Total number of frame processing tasks.
        """
        ...

    ###### Executor Configuration ######

    @contextmanager
    def _events_ctxmgr[*Ts](
        self,
        on_start: Callable[[*Ts], Any],
        on_end: Callable[[*Ts], Any],
        *args: *Ts
        ):
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

        on_submit(*args)
        ftr.add_done_callback(
            self.CallbackWithEvents(
                on_success,
                on_failure,
                args
            )
        )

        return ftr

    class CallbackWithEvents[*Ts, R]:
        def __init__(
            self,
            on_success: Callable[[*Ts, R], Any],
            on_failure: Callable[[*Ts, Exception], Any],
            args: tuple[*Ts]
            ):
            self.on_success = on_success
            self.on_failure = on_failure
            self.args = args
        def __call__(self, ftr: Future[R]):
            try:
                result = ftr.result()
                self.on_success(*self.args, result)
            except Exception as e:
                self.on_failure(*self.args, e)

    class _local_runner_method[T: LMPipeRunner[Any], **P, R]:

        def __init__(
            self,
            method: Callable[Concatenate[T, P], R],
            runner: T, *args: P.args, **kwargs: P.kwargs
            ):

            self.method = method
            self.runner_type = type(runner)
            self.runner_id = runner._id
            self.args = args
            self.kwargs = kwargs

        def __call__(self) -> R:

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
        instances: dict[int, LMPipeRunner[Any]] = {}

    def configure_executor(self, mode: ExecutorMode, initializer: Callable[[], Any]) -> Executor:
        """Configure and return an appropriate executor for the given mode.
        
        This method can be overridden in subclasses to customize executor
        selection and configuration based on specific requirements.
        
        Args:
            mode (ExecutorMode): Execution mode ("batch" or "frames").
            initializer (Callable[[], Any]): Function to initialize executor workers.
            
        Returns:
            :class:`Executor`: Configured executor instance based on options and mode.
            
        Note:
            Returns DummyExecutor if the current mode doesn't match the
            requested mode or if no specific executor type is configured.
        """
        if self.lmpipe_options['executor_mode'] != mode:
            return DummyExecutor(initializer=initializer)
        
        if self.lmpipe_options['executor_type'] == 'process':
            return ProcessPoolExecutor(
                max_workers=self.lmpipe_options['max_cpus'],
                initializer=initializer
            )
        
        if self.lmpipe_options['executor_type'] == 'thread':
            return ThreadPoolExecutor(
                max_workers=int(self.lmpipe_options['max_cpus']),
                initializer=initializer
            )
        
        return DummyExecutor(initializer=initializer)

    def _default_executor_initializer(self):
        """Default initializer function for executor workers.
        
        Registers this runner instance in thread-local storage for
        access by worker processes/threads.
        """
        self._local.instances[self._id] = self

    def _get_runspecs(self, runspec: RunSpec[Path]) -> Iterable[RunSpec[Path]]:
        """Generate run specifications for batch processing.
        
        Walks through the source directory structure and creates individual
        run specifications for each file or image directory that should be processed.
        
        Args:
            runspec: Base run specification with source directory.
            
        Yields:
            Individual run specifications for each processable item.
        """
        for dirpath, dirnames, filenames in runspec.src.walk():

            if dirpath.name.startswith('.') or '/.' in str(dirpath):
                continue

            if any(not d.startswith('.') for d in dirnames):
                continue

            res_dst = runspec.dst / dirpath.relative_to(runspec.src)
            files = [dirpath / f for f in filenames if not f.startswith('.')]

            if is_images_dir(dirpath):
                iterable = [RunSpec(dirpath, res_dst)]
            else:
                iterable = (
                    RunSpec(file_path, res_dst / file_path.name)
                    for file_path in files
                )

            for spec in iterable:
                if self._apply_exist_rule(spec):
                    yield spec

    def _configure_collectors(self) -> list[Collector[K]]:
        """Configure and return list of result collectors.
        
        Returns:
            :code:`list`[:class:`Collector`[K]]: List of configured collector instances.
            
        Note:
            This is a placeholder for future collector configuration logic.
        """
        # TODO: implement collector configuration logic
        return []

    def _apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        """Apply existence rules to determine if a runspec should be processed.
        
        Args:
            runspec (:class:`RunSpec`[Any]): Run specification to evaluate.
            
        Returns:
            :code:`bool`: True if the runspec should be processed, False otherwise.
        """
        return any(
            cllctr.apply_exist_rule(runspec)
            for cllctr in self.collectors
        )
