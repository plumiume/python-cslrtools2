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


from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Protocol, final
from pathlib import Path

from ...typings import ExistRule
from ...options import LMPipeOptions
from ...runspec import RunSpec

from ..base import Collector
from ...estimator import ProcessResult


class AnnotatedFramesSaveCollector[K: str](Collector[K]):
    """Base collector for saving annotated frames to files."""

    ANNOTATED_FRAMES_NAME = "annotated_frames"

    exist_rule: ExistRule
    """Existence rule for handling existing output files."""

    @property
    @abstractmethod
    def is_video(self) -> bool:
        """Whether this collector saves as a video file (True)
        or image sequence (False)."""
        ...

    @property
    @abstractmethod
    def is_image_sequence(self) -> bool:
        """Whether this collector saves as an image sequence (True)
        or video file (False)."""
        ...

    @property
    @abstractmethod
    def file_ext(self) -> str:
        """File extension for this collector's output format.

        For image sequence collectors, this is the extension of individual frames.
        For video collectors, this is the extension of the video file.
        """
        ...

    def configure_from_options(self, options: LMPipeOptions) -> None:
        """Configure collector from LMPipe options.

        Args:
            options (`LMPipeOptions`): The pipeline options containing configuration.
        """
        self.exist_rule = options["annotated_frames_save_exist_rule"]

    @final
    def _get_output_path(self, dst: Path) -> Path:
        """Get the output path for annotated frames.

        This method is final and uses the abstract properties (is_video,
        is_image_sequence, file_ext) to determine the correct output path.

        Args:
            dst: The destination directory path.

        Returns:
            :class:`Path`: The full output path.
                - For image sequence collectors: dst/annotated_frames/ (directory)
                - For video collectors: dst/annotated_frames{file_ext}
        """
        if self.is_video:
            # Video mode: dst/annotated_frames{file_ext}
            return dst / f"{self.ANNOTATED_FRAMES_NAME}{self.file_ext}"
        else:
            # Image sequence mode: dst/annotated_frames/ directory
            return dst / self.ANNOTATED_FRAMES_NAME

    def apply_exist_rule(self, runspec: "RunSpec[Any]") -> bool:
        """Apply the existence rule to determine if processing should continue.

        This method can be overridden by subclasses for custom existence checking logic.

        Args:
            runspec: The RunSpec instance containing input/output path information.

        Returns:
            :class:`bool`: True if processing should continue, False if it
                should be skipped.

        Raises:
            FileExistsError: If exist_rule is "error" and the output already exists.
        """
        output_path = self._get_output_path(runspec.dst)

        if self.is_video:
            # Video mode: check if the video file exists
            exists = output_path.exists()
        else:
            # Image sequence mode: check if the directory exists
            exists = output_path.exists() and output_path.is_dir()

        if not exists:
            return True  # No conflict, proceed

        if self.exist_rule == "skip":
            return False  # Skip processing
        elif self.exist_rule == "overwrite":
            return True  # Overwrite existing file/directory
        elif self.exist_rule == "suffix":
            # TODO: Implement suffix logic (e.g., annotated_frames_1.mp4,
            # annotated_frames_2.mp4)
            return True  # For now, just overwrite
        elif self.exist_rule == "error":
            raise FileExistsError(f"Output already exists: {output_path}")

        return True  # Default: proceed

    # overridable hook
    def _open_file(self, path: Path):
        """Prepare for writing annotated frames.

        Args:
            path (`Path`): The destination directory path.
        """
        pass

    # overridable hook
    def _append_result(self, result: ProcessResult[K]):
        """Process and save a single annotated frame result.

        Args:
            result (`ProcessResult[K]`): The result containing the annotated frame.
        """
        pass

    # overridable hook
    def _close_file(self):
        """Finalize and clean up after writing annotated frames."""
        pass

    def collect_results(
        self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]
    ):
        """Collect annotated frame results and save them.

        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of
                :class:`ProcessResult` objects.
        """
        self._open_file(runspec.dst)
        try:
            for result in results:
                self._append_result(result)
        finally:
            self._close_file()


class AnnotatedFramesShowCollector[K: str](Collector[K]):
    """Base collector for displaying annotated frames interactively."""

    def configure_from_options(self, options: LMPipeOptions) -> None:
        """Configure collector from LMPipe options.

        Args:
            options (`LMPipeOptions`): The pipeline options containing configuration.
        """
        # Show collectors typically don't use exist_rule, but implement for consistency
        pass

    # overridable hook
    def _setup(self):
        """Initialize display resources before showing frames."""
        pass

    # overridable hook
    def _update(self, result: ProcessResult[K]):
        """Update the display with a new annotated frame.

        Args:
            result (`ProcessResult[K]`): The result containing the annotated
                frame to display.
        """
        pass

    # overridable hook
    def _cleanup(self):
        """Clean up display resources after showing all frames."""
        pass

    def collect_results(
        self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]
    ):
        """Display annotated frame results interactively.

        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of
                :class:`ProcessResult` objects.
        """
        self._setup()
        try:
            for result in results:
                self._update(result)
        finally:
            self._cleanup()


class AFSaveCollectorCreator(Protocol):
    def __call__[K: str](
        self, file_ext: str, key_type: type[K]
    ) -> AnnotatedFramesSaveCollector[K]: ...


class AFShowCollectorCreator(Protocol):
    def __call__[K: str](
        self, key_type: type[K]
    ) -> AnnotatedFramesShowCollector[K]: ...


af_save_aliases: dict[str, AFSaveCollectorCreator] = {}
af_show_aliases: dict[str, AFShowCollectorCreator] = {}
