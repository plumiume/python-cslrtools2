from typing import Any
from pathlib import Path
from threading import local
from rich.markup import escape
from rich.console import Group
from rich.live import Live
from rich.progress import (
    Progress, TaskID,
    TextColumn, SpinnerColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn,
    MofNCompleteColumn
)

from ..options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS
from ..runspec import RunSpec
from ..interface import LMPipeInterface, LMPipeRunner
from .mp_rich import RichManager

DESC_TEMPLATE = "[{deco}]{desc}[/{deco}]"

class _Local(local):
    manager_map: dict[int, RichManager] = {}
    search_progress_task_ids: dict[int, TaskID] = {}
    batch_progress_task_ids: dict[int, TaskID] = {}
    frames_progress_task_ids: dict[int, TaskID] = {}

_local = _Local()

class CliAppRunner[K: str](LMPipeRunner[K]):

    def __init__(
        self,
        interface: LMPipeInterface[K],
        options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS
        ):

        super().__init__(interface, options)

        manager = RichManager()
        manager.start()
        _local.manager_map[id(self)] = manager
        self.rich_client = manager.client()

        self.search_progress_ref = self.rich_client.initialize(
            Progress,
            TextColumn(DESC_TEMPLATE.format(
                deco="bold blue",
                desc="{task.description}"
            )),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
        )

        self.batch_progress_ref = self.rich_client.initialize(
            Progress,
            TextColumn(DESC_TEMPLATE.format(
                deco="bold blue",
                desc="{task.description}"
            )),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn()
        )

        self.frames_progress_ref = self.rich_client.initialize(
            Progress,
            TextColumn(DESC_TEMPLATE.format(
                deco="bold blue",
                desc="{task.description}"
            )),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn()
        )

        self.rich_group_ref = self.rich_client.initialize(
            Group,
            self.frames_progress_ref,
            self.search_progress_ref,
            self.batch_progress_ref,
        )

        self.rich_live_ref = self.rich_client.initialize(
            Live,
            self.rich_group_ref,
            refresh_per_second=4
        )

        self.rich_client.call_method( # Live.start
            self.rich_live_ref,
            Live.start
        )


    def __del__(self):

        self.rich_client.call_method( # Live.stop
            self.rich_live_ref,
            Live.stop
        )

        manager = _local.manager_map.pop(id(self), None)
        if manager is not None:
            manager.stop()

    @property
    def search_progress_finish_task(self):
        return Progress.stop_task
    @property
    def batch_progress_finish_task(self):
        return Progress.stop_task
    @property
    def frames_progress_finish_task(self):
        return Progress.remove_task

    def on_start_batch_job(self, runspec: RunSpec[Path]):

        search_progress_task_id = _local.search_progress_task_ids.pop(id(self), None)
        if search_progress_task_id:
            self.rich_client.call_method(
                self.search_progress_ref,
                self.search_progress_finish_task,
                search_progress_task_id
            )

        _local.search_progress_task_ids[id(self)] = self.rich_client.call_method(
            self.search_progress_ref,
            Progress.add_task,
            DESC_TEMPLATE.format(
                deco="bold blue",
                desc=f"Searching batch tasks ..."
            ),
            total=None
        )

        batch_progress_task_id = _local.batch_progress_task_ids.pop(id(self), None)
        if batch_progress_task_id:
            self.rich_client.call_method(
                self.batch_progress_ref,
                self.batch_progress_finish_task,
                batch_progress_task_id
            )

        _local.batch_progress_task_ids[id(self)] = self.rich_client.call_method(
            self.batch_progress_ref,
            Progress.add_task,
            DESC_TEMPLATE.format(
                deco="bold blue",
                desc=f"Running batch job ..."
            ),
            total=None
        )

    def on_submit_batch_task(self, runspec: RunSpec[Path], task_id: int):

        search_progress_task_id = _local.search_progress_task_ids.get(id(self))
        if search_progress_task_id is not None:

            self.rich_client.call_method(
                self.search_progress_ref,
                Progress.advance,
                search_progress_task_id
            )

    def on_start_batch_task(self, runspec: RunSpec[Path], task_id: int):
        return

    def on_determined_batch_task_count(self, runspec: RunSpec[Path], task_count: int):

        search_progress_task_id = _local.search_progress_task_ids.get(id(self))
        if search_progress_task_id is not None:
            self.rich_client.call_method(
                self.search_progress_ref,
                Progress.update,
                search_progress_task_id,
                total=task_count
            )
            self.rich_client.call_method(
                self.search_progress_ref,
                Progress.stop_task,
                search_progress_task_id
            )

        batch_progress_task_id = _local.batch_progress_task_ids.get(id(self))
        if batch_progress_task_id is not None:
            self.rich_client.call_method(
                self.batch_progress_ref,
                Progress.update,
                batch_progress_task_id,
                total=task_count
            )

    def on_end_batch_task(self, runspec: RunSpec[Path], task_id: int):
        return

    def on_success_batch_task(self, runspec: RunSpec[Path], task_id: int, result: Any):

        batch_progress_task_id = _local.batch_progress_task_ids.get(id(self))
        if batch_progress_task_id is not None:

            self.rich_client.call_method(
                self.batch_progress_ref,
                Progress.advance,
                batch_progress_task_id
            )

    def on_failure_batch_task(self, runspec: RunSpec[Path], task_id: int, error: Exception):

        raise error

        batch_progress_task_id = _local.batch_progress_task_ids.get(id(self))
        if batch_progress_task_id is not None:

            self.rich_client.call_method(
                self.batch_progress_ref,
                Progress.advance,
                batch_progress_task_id
            )

    def on_end_batch_job(self, runspec: RunSpec[Path]):

        search_progress_task_id = _local.search_progress_task_ids.pop(id(self), None)
        if search_progress_task_id is not None:
            self.rich_client.call_method(
                self.search_progress_ref,
                self.search_progress_finish_task,
                search_progress_task_id
            )

        batch_progress_task_id = _local.batch_progress_task_ids.pop(id(self), None)
        if batch_progress_task_id is not None:
            self.rich_client.call_method(
                self.batch_progress_ref,
                self.batch_progress_finish_task,
                batch_progress_task_id
            )


    def on_start_frames_job(self):

        frames_progress_task_id = _local.frames_progress_task_ids.pop(id(self), None)
        if frames_progress_task_id is not None:
            self.rich_client.call_method(
                self.frames_progress_ref,
                self.frames_progress_finish_task,
                frames_progress_task_id
            )

        # _local.frames_progress_task_ids[id(self)] = self.rich_client.call_method(
        #     self.frames_progress_ref,
        #     Progress.add_task,
        #     DESC_TEMPLATE.format(
        #         deco="bold blue",
        #         desc=f"Processing frames ..."
        #     ),
        #     total=None
        # )

    def on_submit_frames_task(self, task_id: int):
        ...

    def on_start_frames_task(self, task_id: int):
        ...

    def on_determined_frames_task_count(self, task_count: int):

        frames_progress_task_id = _local.frames_progress_task_ids.get(id(self))
        if frames_progress_task_id is not None:
            self.rich_client.call_method(
                self.frames_progress_ref,
                Progress.update,
                frames_progress_task_id,
                total=task_count
            )

    def on_end_frames_task(self, task_id: int):
        ...

    def on_success_frames_task(self, task_id: int, result: Any):

        frames_progress_task_id = _local.frames_progress_task_ids.get(id(self))
        if frames_progress_task_id is not None:

            self.rich_client.call_method(
                self.frames_progress_ref,
                Progress.advance,
                frames_progress_task_id
            )

    def on_failure_frames_task(self, task_id: int, error: Exception):

        raise error

        frames_progress_task_id = _local.frames_progress_task_ids.get(id(self))
        if frames_progress_task_id is not None:

            self.rich_client.call_method(
                self.frames_progress_ref,
                Progress.advance,
                frames_progress_task_id
            )

    def on_end_frames_job(self):

        frames_progress_task_id = _local.frames_progress_task_ids.pop(id(self), None)
        if frames_progress_task_id is not None:
            self.rich_client.call_method(
                self.frames_progress_ref,
                self.frames_progress_finish_task,
                frames_progress_task_id
            )

    def _close_tasks(self, deco: str, desc: str) -> None:

        search_progress_task_id = _local.search_progress_task_ids.get(id(self))
        if search_progress_task_id is not None:
            self.rich_client.call_method(
                self.search_progress_ref,
                Progress.update,
                search_progress_task_id,
                description=DESC_TEMPLATE.format(deco=deco, desc=desc)
            )
            self.rich_client.call_method(
                self.search_progress_ref,
                self.search_progress_finish_task,
                search_progress_task_id
            )

        frames_progress_task_id = _local.frames_progress_task_ids.get(id(self))
        if frames_progress_task_id is not None:
            self.rich_client.call_method(
                self.frames_progress_ref,
                Progress.update,
                frames_progress_task_id,
                description=DESC_TEMPLATE.format(deco=deco, desc=desc)
            )
            self.rich_client.call_method(
                self.frames_progress_ref,
                self.frames_progress_finish_task,
                frames_progress_task_id
            )

        batch_progress_task_id = _local.batch_progress_task_ids.get(id(self))
        if batch_progress_task_id is not None:
            self.rich_client.call_method(
                self.batch_progress_ref,
                Progress.update,
                batch_progress_task_id,
                description=DESC_TEMPLATE.format(deco=deco, desc=desc)
            )
            self.rich_client.call_method(
                self.batch_progress_ref,
                self.batch_progress_finish_task,
                batch_progress_task_id
            )

    def on_keyboard_interrupt(self, error: KeyboardInterrupt) -> None:

        self._close_tasks(
            deco="bold red",
            desc="Processing interrupted by user."
        )

        error.__suppress_context__ = True
        raise error

    def on_exception(self, error: Exception):

        self._close_tasks(
            deco="bold red",
            desc=escape(f"Error during processing: {error}")
        )

        raise error

    def on_complete(self):

        self._close_tasks(
            deco="bold green",
            desc="Processing complete."
        )
