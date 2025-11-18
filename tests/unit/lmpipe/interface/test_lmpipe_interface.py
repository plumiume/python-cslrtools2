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

import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
from pathlib import Path
from typing import Literal, Mapping
from unittest.mock import Mock, patch

from cslrtools2.lmpipe.interface import LMPipeInterface, LMPipeRunner
from cslrtools2.lmpipe.interface import _update_lmpipe_options
from cslrtools2.lmpipe.estimator import Estimator
from cslrtools2.lmpipe.collector import Collector
from cslrtools2.lmpipe.options import (
    DEFAULT_LMPIPE_OPTIONS,
    LMPipeOptionsPartial,
    LMPipeOptions
)
from cslrtools2.lmpipe.runspec import RunSpec
from cslrtools2.typings import NDArrayFloat, NDArrayStr, MatLike


class DummyEstimator(Estimator[Literal["test"]]):
    """Dummy estimator for testing."""

    def setup(self) -> None:
        pass

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
        return {"test": np.array([[1.0, 2.0, 3.0]])}

    def annotate(
        self,
        frame_src: MatLike | None,
        frame_idx: int,
        landmarks: Mapping[Literal["test"], NDArrayFloat],
    ) -> MatLike:
        return (
            frame_src if frame_src is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        )


@pytest.fixture
def dummy_estimator():
    """Create a dummy estimator."""
    return DummyEstimator()


@pytest.fixture
def mock_collector():
    """Create a mock collector."""
    return Mock(spec=Collector)


class TestUpdateLMPipeOptions:
    """Tests for _update_lmpipe_options function."""

    def test_update_with_empty_updates(self):
        """Test update with no updates returns copy of base."""
        base = DEFAULT_LMPIPE_OPTIONS
        result = _update_lmpipe_options(base)

        assert result == base
        assert result is not base  # Should be a copy

    def test_update_with_single_partial(self):
        """Test update with single partial options."""
        base = DEFAULT_LMPIPE_OPTIONS
        partial: LMPipeOptionsPartial = {"max_cpus": 8}

        result = _update_lmpipe_options(base, partial)

        assert result["max_cpus"] == 8
        assert result["executor_type"] == base["executor_type"]

    def test_update_with_multiple_partials(self):
        """Test update with multiple partial options."""
        base = DEFAULT_LMPIPE_OPTIONS
        partial1: LMPipeOptionsPartial = {"max_cpus": 8}
        partial2: LMPipeOptionsPartial = {"executor_type": "thread"}

        result = _update_lmpipe_options(base, partial1, partial2)

        assert result["max_cpus"] == 8
        assert result["executor_type"] == "thread"

    def test_update_preserves_base_options(self):
        """Test that base options are not modified."""
        base = DEFAULT_LMPIPE_OPTIONS.copy()
        original_max_cpus = base["max_cpus"]
        partial: LMPipeOptionsPartial = {"max_cpus": 99}

        _update_lmpipe_options(base, partial)

        assert base["max_cpus"] == original_max_cpus

    def test_later_updates_override_earlier(self):
        """Test that later updates override earlier ones."""
        base = DEFAULT_LMPIPE_OPTIONS
        partial1: LMPipeOptionsPartial = {"max_cpus": 4}
        partial2: LMPipeOptionsPartial = {"max_cpus": 8}

        result = _update_lmpipe_options(base, partial1, partial2)

        assert result["max_cpus"] == 8


class TestLMPipeInterfaceInitialization:
    """Tests for LMPipeInterface initialization."""

    def test_initialization_with_estimator_only(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test basic initialization with only an estimator."""
        interface = LMPipeInterface(dummy_estimator)

        assert interface.estimator is dummy_estimator
        assert interface.lmpipe_options == DEFAULT_LMPIPE_OPTIONS
        assert interface.collectors_or_factory == []
        assert interface.runner_type is LMPipeRunner

    def test_initialization_with_collectors_list(
        self,
        dummy_estimator: Estimator[Literal["test"]],
        mock_collector: Collector[Literal["test"]]
    ):
        """Test initialization with list of collectors."""
        collectors = [mock_collector]
        interface = LMPipeInterface(dummy_estimator, collectors=collectors)

        assert interface.collectors_or_factory is collectors
        assert isinstance(interface.collectors_or_factory, list)

    def test_initialization_with_collectors_factory(
        self,
        dummy_estimator: Estimator[Literal["test"]],
        mock_collector: Collector[Literal["test"]]
    ):
        """Test initialization with collector factory function."""
        factory = Mock(return_value=[mock_collector])
        interface = LMPipeInterface(dummy_estimator, collectors=factory)

        assert interface.collectors_or_factory is factory

    def test_initialization_with_options_dict(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test initialization with options dictionary."""
        options: LMPipeOptionsPartial = {"max_cpus": 8, "executor_type": "thread"}
        interface = LMPipeInterface(dummy_estimator, options=options)

        assert interface.lmpipe_options["max_cpus"] == 8
        assert interface.lmpipe_options["executor_type"] == "thread"

    def test_initialization_with_kwargs(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test initialization with keyword arguments."""
        interface = LMPipeInterface(
            dummy_estimator, max_cpus=4, executor_type="process"
        )

        assert interface.lmpipe_options["max_cpus"] == 4
        assert interface.lmpipe_options["executor_type"] == "process"

    def test_initialization_kwargs_override_options(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test that kwargs override options dict."""
        options: LMPipeOptionsPartial = {"max_cpus": 4}
        interface = LMPipeInterface(dummy_estimator, options=options, max_cpus=8)

        assert interface.lmpipe_options["max_cpus"] == 8

    def test_initialization_with_custom_runner_type(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test initialization with custom runner type."""

        class CustomRunner(LMPipeRunner[Literal["test"]]):
            pass

        interface = LMPipeInterface(dummy_estimator, runner_type=CustomRunner)

        assert interface.runner_type is CustomRunner


class TestLMPipeInterfaceRunMethods:
    """Tests for LMPipeInterface run methods."""

    def test_run_creates_runner_and_calls_run(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run() creates runner and delegates to runner.run()."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run") as mock_run:
            interface.run(src, dst)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert isinstance(call_args[0], RunSpec)
            assert call_args[0].src == src
            assert call_args[0].dst == dst

    def test_run_with_options_override(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run() with option overrides."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator, max_cpus=4)

        with (
            patch.object(LMPipeRunner, "__init__", return_value=None) as mock_init,
            patch.object(LMPipeRunner, "run", return_value=None),
        ):
            interface.run(src, dst, max_cpus=8)

            # Check that runner was initialized with updated options
            call_args = mock_init.call_args[0]
            runner_options = call_args[1]
            assert runner_options["max_cpus"] == 8

    def test_run_batch_creates_runner_and_calls_run_batch(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_batch() creates runner and delegates to runner.run_batch()."""
        src = tmp_path / "input"
        src.mkdir()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_batch") as mock_run_batch:
            interface.run_batch(src, dst)

            mock_run_batch.assert_called_once()
            call_args = mock_run_batch.call_args[0]
            assert isinstance(call_args[0], RunSpec)
            assert call_args[0].src == src

    def test_run_single_creates_runner_and_calls_run_single(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_single() creates runner and delegates to runner.run_single()."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_single") as mock_run_single:
            interface.run_single(src, dst)

            mock_run_single.assert_called_once()
            call_args = mock_run_single.call_args[0]
            assert isinstance(call_args[0], RunSpec)

    def test_run_video_creates_runner_and_calls_run_video(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_video() creates runner and delegates to runner.run_video()."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_video") as mock_run_video:
            interface.run_video(src, dst)

            mock_run_video.assert_called_once()
            call_args = mock_run_video.call_args[0]
            assert isinstance(call_args[0], RunSpec)

    def test_run_sequence_images_creates_runner_and_calls_run_sequence_images(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_sequence_images() creates runner and delegates correctly."""
        src = tmp_path / "images"
        src.mkdir()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_sequence_images") as mock_run_seq:
            interface.run_sequence_images(src, dst)

            mock_run_seq.assert_called_once()
            call_args = mock_run_seq.call_args[0]
            assert isinstance(call_args[0], RunSpec)

    def test_run_single_image_creates_runner_and_calls_run_single_image(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_single_image() creates runner and delegates correctly."""
        src = tmp_path / "image.jpg"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_single_image") as mock_run_img:
            interface.run_single_image(src, dst)

            mock_run_img.assert_called_once()
            call_args = mock_run_img.call_args[0]
            assert isinstance(call_args[0], RunSpec)

    def test_run_stream_creates_runner_and_calls_run_stream(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test run_stream() creates runner and delegates to runner.run_stream()."""
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_stream") as mock_run_stream:
            interface.run_stream(0, dst)

            mock_run_stream.assert_called_once()
            call_args = mock_run_stream.call_args[0]
            assert isinstance(call_args[0], RunSpec)
            assert call_args[0].src == 0


class TestLMPipeInterfaceRunSpecCreation:
    """Tests for RunSpec creation in interface methods."""

    def test_run_creates_runspec_from_pathlikes(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that run() creates RunSpec from path-like objects."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run") as mock_run:
            interface.run(src, dst)

            runspec = mock_run.call_args[0][0]
            assert isinstance(runspec.src, Path)
            assert isinstance(runspec.dst, Path)
            assert runspec.src == src
            assert runspec.dst == dst

    def test_run_stream_creates_runspec_from_index(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that run_stream() creates RunSpec from integer index."""
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run_stream") as mock_run_stream:
            interface.run_stream(0, dst)

            runspec = mock_run_stream.call_args[0][0]
            assert runspec.src == 0
            assert isinstance(runspec.dst, Path)


class TestLMPipeInterfaceOptionsHandling:
    """Tests for options handling in interface methods."""

    def test_each_method_updates_options(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that each run method properly updates options."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator, max_cpus=4)

        methods_to_test = [
            ("run", (src, dst)),
            ("run_single", (src, dst)),
            ("run_video", (src, dst)),
            ("run_single_image", (src, dst)),
        ]

        for method_name, args in methods_to_test:
            method = getattr(interface, method_name)

            with (
                patch.object(LMPipeRunner, "__init__", return_value=None) as mock_init,
                patch.object(LMPipeRunner, method_name, return_value=None),
            ):
                method(*args, max_cpus=8)

                # Verify runner was initialized with updated options
                runner_options = mock_init.call_args[0][1]
                assert runner_options["max_cpus"] == 8, (
                    f"Failed for method {method_name}"
                )

    def test_batch_method_updates_options(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that run_batch() properly updates options."""
        src = tmp_path / "input"
        src.mkdir()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator, max_cpus=4)

        with (
            patch.object(LMPipeRunner, "__init__", return_value=None) as mock_init,
            patch.object(LMPipeRunner, "run_batch", return_value=None),
        ):
            interface.run_batch(src, dst, executor_type="thread")

            runner_options = mock_init.call_args[0][1]
            assert runner_options["executor_type"] == "thread"

    def test_sequence_images_method_updates_options(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that run_sequence_images() properly updates options."""
        src = tmp_path / "images"
        src.mkdir()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator, max_cpus=4)

        with (
            patch.object(LMPipeRunner, "__init__", return_value=None) as mock_init,
            patch.object(LMPipeRunner, "run_sequence_images", return_value=None),
        ):
            interface.run_sequence_images(src, dst, max_cpus=16)

            runner_options = mock_init.call_args[0][1]
            assert runner_options["max_cpus"] == 16

    def test_stream_method_updates_options(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that run_stream() properly updates options."""
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator, cpu=1)

        with (
            patch.object(LMPipeRunner, "__init__", return_value=None) as mock_init,
            patch.object(LMPipeRunner, "run_stream", return_value=None),
        ):
            interface.run_stream(0, dst, cpu=2)

            runner_options = mock_init.call_args[0][1]
            assert runner_options["cpu"] == 2


class TestLMPipeInterfaceCustomRunner:
    """Tests for custom runner type support."""

    def test_custom_runner_is_used(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that custom runner type is instantiated."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        class CustomRunner(LMPipeRunner[Literal["test"]]):
            pass

        interface = LMPipeInterface(dummy_estimator, runner_type=CustomRunner)

        with (
            patch.object(CustomRunner, "__init__", return_value=None) as mock_init,
            patch.object(CustomRunner, "run", return_value=None),
        ):
            interface.run(src, dst)

            mock_init.assert_called_once()

    def test_custom_runner_receives_interface_and_options(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that custom runner receives correct interface and options."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        class CustomRunner(LMPipeRunner[Literal["test"]]):
            pass

        interface = LMPipeInterface(
            dummy_estimator, runner_type=CustomRunner, max_cpus=8
        )

        with (
            patch.object(CustomRunner, "__init__", return_value=None) as mock_init,
            patch.object(CustomRunner, "run", return_value=None),
        ):
            interface.run(src, dst)

            call_args = mock_init.call_args[0]
            assert call_args[0] is interface
            assert call_args[1]["max_cpus"] == 8


class TestLMPipeInterfaceIntegration:
    """Integration tests for LMPipeInterface."""

    def test_interface_preserves_collectors(
        self,
        dummy_estimator: Estimator[Literal["test"]],
        mock_collector: Collector[Literal["test"]]
    ):
        """Test that interface preserves collectors configuration."""
        collectors = [mock_collector]
        interface = LMPipeInterface(dummy_estimator, collectors=collectors)

        assert interface.collectors_or_factory is collectors

    def test_interface_preserves_estimator(
        self, dummy_estimator: Estimator[Literal["test"]]
    ):
        """Test that interface preserves estimator reference."""
        interface = LMPipeInterface(dummy_estimator)

        assert interface.estimator is dummy_estimator

    def test_multiple_run_calls_create_separate_runners(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that multiple run calls create separate runner instances."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        runner_instances: list[LMPipeRunner[Literal["test"]]] = []

        def track_runner(
            iface: LMPipeInterface[Literal["test"]],
            options: LMPipeOptions
        ) -> LMPipeRunner[Literal["test"]]:
            runner = LMPipeRunner(iface, options)
            runner_instances.append(runner)
            return runner

        with (
            patch.object(interface, "runner_type", side_effect=track_runner),
            patch.object(LMPipeRunner, "run", return_value=None),
        ):
            interface.run(src, dst)
            interface.run(src, dst)

            assert len(runner_instances) == 2
            assert runner_instances[0] is not runner_instances[1]

    def test_interface_with_string_paths(
        self, dummy_estimator: Estimator[Literal["test"]], tmp_path: Path
    ):
        """Test that interface accepts string paths."""
        src = tmp_path / "input.mp4"
        src.touch()
        dst = tmp_path / "output"

        interface = LMPipeInterface(dummy_estimator)

        with patch.object(LMPipeRunner, "run") as mock_run:
            interface.run(str(src), str(dst))

            runspec = mock_run.call_args[0][0]
            assert isinstance(runspec.src, Path)
            assert isinstance(runspec.dst, Path)
