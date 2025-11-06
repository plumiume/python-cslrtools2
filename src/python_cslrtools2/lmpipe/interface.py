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

# cp314 ready
type LMPipeRunner[K: str] = "LMPipeRunner[K]" # pyright: ignore[reportRedeclaration]

def is_video_file(path: Path) -> bool: ...
def is_images_dir(path: Path) -> bool: ...
def is_image_file(path: Path) -> bool: ...
def capture_to_frames(capture: cv2.VideoCapture) -> Iterable[MatLike]: ...
def seq_imgs_to_frames(src: Path) -> Iterable[MatLike]: ...
def image_file_to_frame(src: Path) -> MatLike: ...



def _update_lmpipe_options(
    base: LMPipeOptions,
    *updates: LMPipeOptionsPartial
    ) -> LMPipeOptions:
    ret = base.copy()
    for opt in updates:
        ret.update(opt)
    return ret



class LMPipeInterface[K: str]:

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    def __init__(self, estimator: Estimator[K], **options: Unpack[LMPipeOptionsPartial]):

        self.estimator = estimator
        self.lmpipe_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )


    def run(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run(runspec)
    
    def run_batch(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_batch(runspec)
    
    def run_single(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_single(runspec)
    
    def run_video(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_video(runspec)
    
    def run_sequence_images(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_sequence_images(runspec)
    
    def run_single_image(self, src: PathLike, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_pathlikes(src, dst)
        return LMPipeRunner(self, updated_options).run_single_image(runspec)
    
    def run_stream(self, src: int, dst: PathLike, **options: Unpack[LMPipeOptionsPartial]):
        updated_options = _update_lmpipe_options(
            self.lmpipe_options,
            options
        )
        runspec = RunSpec.from_index(src, dst)
        return LMPipeRunner(self, updated_options).run_stream(runspec)


class LMPipeRunner[K: str]:
    
    def __init__(
        self,
        interface: LMPipeInterface[K],
        options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS,
        ):

        self._id = id(self)

        self.lmpipe_interface = interface
        self.lmpipe_options = options

        self.collectors = self._configure_collectors()


    def run(self, runspec: RunSpec[Path]):
        
        if runspec.src.is_dir():
            return self.run_batch(runspec)
        
        if runspec.src.is_file():
            return self.run_single(runspec)
        
        if runspec.src.exists():
            raise ValueError(f"Unsupported source path: {runspec.src}")
        raise FileNotFoundError(f"Source path does not exist: {runspec.src}")


    def run_batch(self, runspec: RunSpec[Path]):

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
        
        if is_video_file(runspec.src):
            return self.run_video(runspec)
        
        if is_images_dir(runspec.src):
            return self.run_sequence_images(runspec)
        
        if is_image_file(runspec.src):
            return self.run_single_image(runspec)
        
        raise ValueError(f"Unsupported source path: {runspec.src}")


    def run_video(self, runspec: RunSpec[Path]):
        
        capture = cv2.VideoCapture(str(runspec.src))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {runspec.src}")
        
        results = self.process_frames(capture_to_frames(capture))
        
        self._collect_results(runspec, results)

    def run_sequence_images(self, runspec: RunSpec[Path]):
        
        results = self.process_frames(seq_imgs_to_frames(runspec.src))

        self._collect_results(runspec, results)

    def run_single_image(self, runspec: RunSpec[Path]):
        
        results = self.process_frames([image_file_to_frame(runspec.src)])

        self._collect_results(runspec, results)

    def run_stream(self, runspec: RunSpec[int]):
        
        capture = cv2.VideoCapture(runspec.src)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video stream: {runspec.src}")
        
        results = self.process_frames(capture_to_frames(capture))

        self._collect_results(runspec, results)


    def process_frames(self, frames: Iterable[MatLike]) -> Iterable[ProcessResult[K]]:
        
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
        yield # TODO: implement resource allocation logic

    def _collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        for collector in self.collectors:
            collector.collect_results(runspec, results)

    ###### Events ######

    def on_start_batch_job(self, runspec: RunSpec[Path]): ...
    def on_submit_batch_task(self, runspec: RunSpec[Path], task_id: int): ...
    def on_start_batch_task(self, runspec: RunSpec[Path], task_id: int): ...
    def on_end_batch_task(self, runspec: RunSpec[Path], task_id: int): ...
    def on_success_batch_task(self, runspec: RunSpec[Path], task_id: int, result: Any): ...
    def on_failure_batch_task(self, runspec: RunSpec[Path], task_id: int, error: Exception): ...
    def on_end_batch_job(self, runspec: RunSpec[Path]): ...
    def on_determined_batch_task_count(self, runspec: RunSpec[Path], task_count: int): ...


    def on_start_frames_job(self): ...
    def on_submit_frames_task(self, task_id: int): ...
    def on_start_frames_task(self, task_id: int): ...
    def on_end_frames_task(self, task_id: int): ...
    def on_success_frames_task(self, task_id: int, result: Any): ...
    def on_failure_frames_task(self, task_id: int, error: Exception): ...
    def on_end_frames_job(self): ...
    def on_determined_frames_task_count(self, task_count: int): ...

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

    # overridable method
    def configure_executor(self, mode: ExecutorMode, initializer: Callable[[], Any]) -> Executor:
        
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
        self._local.instances[self._id] = self

    def _get_runspecs(self, runspec: RunSpec[Path]) -> Iterable[RunSpec[Path]]:

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
        # TODO: implement collector configuration logic
        return []

    def _apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        return any(
            cllctr.apply_exist_rule(runspec)
            for cllctr in self.collectors
        )
