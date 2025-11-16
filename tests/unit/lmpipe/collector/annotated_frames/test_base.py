"""Tests for annotated frames collector base classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from cslrtools2.lmpipe.collector.annotated_frames.base import (
    AnnotatedFramesSaveCollector,
    AnnotatedFramesShowCollector,
)
from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.options import DEFAULT_LMPIPE_OPTIONS, LMPipeOptions
from cslrtools2.lmpipe.runspec import RunSpec


class DummyImageSequenceSaveCollector(AnnotatedFramesSaveCollector[str]):
    """Dummy collector for testing image sequence mode."""

    def __init__(self, extension: str = ".png"):
        self._extension = extension
        self.opened = False
        self.closed = False
        self.appended_results = []

    @property
    def is_video(self) -> bool:
        return False

    @property
    def is_image_sequence(self) -> bool:
        return True

    @property
    def file_ext(self) -> str:
        return self._extension

    def _open_file(self, path: Path):
        self.opened = True

    def _append_result(self, result: ProcessResult[str]):
        self.appended_results.append(result)

    def _close_file(self):
        self.closed = True


class DummyVideoSaveCollector(AnnotatedFramesSaveCollector[str]):
    """Dummy collector for testing video mode."""

    def __init__(self, extension: str = ".mp4"):
        self._extension = extension
        self.opened = False
        self.closed = False
        self.appended_results = []

    @property
    def is_video(self) -> bool:
        return True

    @property
    def is_image_sequence(self) -> bool:
        return False

    @property
    def file_ext(self) -> str:
        return self._extension

    def _open_file(self, path: Path):
        self.opened = True

    def _append_result(self, result: ProcessResult[str]):
        self.appended_results.append(result)

    def _close_file(self):
        self.closed = True


class DummyShowCollector(AnnotatedFramesShowCollector[str]):
    """Dummy collector for testing show mode."""

    def __init__(self):
        self.setup_called = False
        self.cleanup_called = False
        self.updated_results = []

    def _setup(self):
        self.setup_called = True

    def _update(self, result: ProcessResult[str]):
        self.updated_results.append(result)

    def _cleanup(self):
        self.cleanup_called = True


class TestAnnotatedFramesSaveCollectorPaths:
    """Test output path generation for save collectors."""

    def test_image_sequence_path(self, tmp_path: Path):
        """Test output path for image sequence collector."""
        collector = DummyImageSequenceSaveCollector(extension=".png")

        output_path = collector._get_output_path(tmp_path)

        assert output_path == tmp_path / "annotated_frames"
        assert not output_path.suffix  # Directory has no extension

    def test_video_path(self, tmp_path: Path):
        """Test output path for video collector."""
        collector = DummyVideoSaveCollector(extension=".mp4")

        output_path = collector._get_output_path(tmp_path)

        assert output_path == tmp_path / "annotated_frames.mp4"
        assert output_path.suffix == ".mp4"

    def test_different_extensions(self, tmp_path: Path):
        """Test output paths with different extensions."""
        collectors = [
            DummyImageSequenceSaveCollector(extension=".jpg"),
            DummyVideoSaveCollector(extension=".avi"),
        ]

        for collector in collectors:
            output_path = collector._get_output_path(tmp_path)
            if collector.is_video:
                assert output_path.suffix == collector.file_ext
            else:
                assert not output_path.suffix


class TestAnnotatedFramesSaveCollectorExistRule:
    """Test existence rule handling."""

    def test_skip_rule_with_existing_video(self, tmp_path: Path):
        """Test skip rule when video file exists."""
        collector = DummyVideoSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "skip"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_file = tmp_path / "annotated_frames.mp4"
        output_file.touch()  # Create existing file

        runspec = RunSpec(video_file, tmp_path)

        assert not collector.apply_exist_rule(runspec)

    def test_skip_rule_with_nonexistent_video(self, tmp_path: Path):
        """Test skip rule when video file doesn't exist."""
        collector = DummyVideoSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "skip"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        assert collector.apply_exist_rule(runspec)

    def test_overwrite_rule(self, tmp_path: Path):
        """Test overwrite rule."""
        collector = DummyVideoSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "overwrite"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_file = tmp_path / "annotated_frames.mp4"
        output_file.touch()

        runspec = RunSpec(video_file, tmp_path)

        assert collector.apply_exist_rule(runspec)

    def test_error_rule_with_existing_file(self, tmp_path: Path):
        """Test error rule raises when file exists."""
        collector = DummyVideoSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "error"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_file = tmp_path / "annotated_frames.mp4"
        output_file.touch()

        runspec = RunSpec(video_file, tmp_path)

        with pytest.raises(FileExistsError):
            collector.apply_exist_rule(runspec)

    def test_error_rule_with_nonexistent_file(self, tmp_path: Path):
        """Test error rule doesn't raise when file doesn't exist."""
        collector = DummyVideoSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "error"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        assert collector.apply_exist_rule(runspec)

    def test_skip_rule_with_existing_directory(self, tmp_path: Path):
        """Test skip rule when image sequence directory exists."""
        collector = DummyImageSequenceSaveCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}
        options["annotated_frames_save_exist_rule"] = "skip"
        collector.configure_from_options(options)

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        output_dir = tmp_path / "annotated_frames"
        output_dir.mkdir()  # Create existing directory

        runspec = RunSpec(video_file, tmp_path)

        assert not collector.apply_exist_rule(runspec)


class TestAnnotatedFramesSaveCollectorWorkflow:
    """Test collector workflow (open, append, close)."""

    def test_collect_results_workflow(self, tmp_path: Path):
        """Test complete collection workflow."""
        import numpy as np

        collector = DummyImageSequenceSaveCollector()

        results = [
            ProcessResult[str](
                frame_id=i,
                headers={},
                landmarks={},
                annotated_frame=np.zeros((10, 10, 3), dtype=np.uint8),
            )
            for i in range(3)
        ]

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        collector.collect_results(runspec, results)

        assert collector.opened
        assert collector.closed
        assert len(collector.appended_results) == 3

    def test_collect_results_empty(self, tmp_path: Path):
        """Test collection with empty results."""
        collector = DummyImageSequenceSaveCollector()

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        collector.collect_results(runspec, [])

        assert collector.opened
        assert collector.closed
        assert len(collector.appended_results) == 0

    def test_collect_results_cleanup_on_error(self, tmp_path: Path):
        """Test that _close_file is called even on error."""
        import numpy as np

        collector = DummyImageSequenceSaveCollector()

        # Override _append_result to raise an error
        def error_append(result):
            raise RuntimeError("Test error")

        collector._append_result = error_append

        result = ProcessResult[str](
            frame_id=0,
            headers={},
            landmarks={},
            annotated_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        )

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        with pytest.raises(RuntimeError):
            collector.collect_results(runspec, [result])

        assert collector.opened
        assert collector.closed  # Should still be closed


class TestAnnotatedFramesShowCollector:
    """Test show collector."""

    def test_show_workflow(self, tmp_path: Path):
        """Test show collector workflow."""
        import numpy as np

        collector = DummyShowCollector()

        results = [
            ProcessResult[str](
                frame_id=i,
                headers={},
                landmarks={},
                annotated_frame=np.zeros((10, 10, 3), dtype=np.uint8),
            )
            for i in range(3)
        ]

        video_file = tmp_path / "input.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, tmp_path)

        collector.collect_results(runspec, results)

        assert collector.setup_called
        assert collector.cleanup_called
        assert len(collector.updated_results) == 3

    def test_configure_from_options(self):
        """Test configure_from_options does nothing for show collectors."""
        collector = DummyShowCollector()
        options: LMPipeOptions = {**DEFAULT_LMPIPE_OPTIONS}

        # Should not raise
        collector.configure_from_options(options)
