"""Unit tests for lmpipe collector base classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pytest  # pyright: ignore[reportUnusedImport]

from cslrtools2.lmpipe.collector.base import Collector
from cslrtools2.lmpipe.estimator import ProcessResult
from cslrtools2.lmpipe.options import LMPipeOptions
from cslrtools2.lmpipe.runspec import RunSpec


class MockCollector(Collector[str]):
    """Mock collector for testing base class functionality."""
    
    def __init__(self):
        self.configured = False
        self.collected_results: list[ProcessResult[str]] = []
        self.collected_runspecs: list[RunSpec[Any]] = []
    
    def configure_from_options(self, options: LMPipeOptions) -> None:
        """Store configuration."""
        self.configured = True
        self.options = options
    
    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[str]]):
        """Store collected results."""
        self.collected_runspecs.append(runspec)
        self.collected_results.extend(results)


class TestCollectorBase:
    """Test Collector abstract base class."""

    def test_abstract_methods_enforced(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            Collector()  # type: ignore[abstract]

    def test_mock_collector_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        collector = MockCollector()
        assert isinstance(collector, Collector)
        assert not collector.configured

    def test_configure_from_options(self):
        """Test configure_from_options method."""
        from cslrtools2.lmpipe.options import DEFAULT_LMPIPE_OPTIONS
        
        collector = MockCollector()
        # Use default options
        options = DEFAULT_LMPIPE_OPTIONS
        
        collector.configure_from_options(options)
        
        assert collector.configured
        assert collector.options is options

    def test_collect_results(self, tmp_path: Path):
        """Test collect_results method."""
        import numpy as np
        
        collector = MockCollector()
        
        # Create mock runspec with correct API
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)
        
        # Create mock results with correct ProcessResult structure
        results = [
            ProcessResult(
                frame_id=0,
                headers={"pose": np.array(["x", "y", "z"], dtype=str)},
                landmarks={"pose": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)},
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8)
            )
        ]
        
        collector.collect_results(runspec, results)
        
        assert len(collector.collected_runspecs) == 1
        assert collector.collected_runspecs[0] == runspec
        assert len(collector.collected_results) == 1

    def test_apply_exist_rule_default(self, tmp_path: Path):
        """Test default apply_exist_rule returns True."""
        collector = MockCollector()
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)
        
        # Default implementation should return True
        assert collector.apply_exist_rule(runspec) is True


class CustomCollector(Collector[str]):
    """Collector with custom apply_exist_rule implementation."""
    
    def __init__(self, should_process: bool):
        self.should_process = should_process
    
    def configure_from_options(self, options: LMPipeOptions) -> None:
        pass
    
    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[str]]):
        pass
    
    def apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        """Custom rule that uses stored flag."""
        return self.should_process


class TestCustomCollector:
    """Test custom collector with overridden methods."""

    def test_apply_exist_rule_custom_true(self, tmp_path: Path):
        """Test custom apply_exist_rule returns True."""
        collector = CustomCollector(should_process=True)
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)
        
        assert collector.apply_exist_rule(runspec) is True

    def test_apply_exist_rule_custom_false(self, tmp_path: Path):
        """Test custom apply_exist_rule returns False."""
        collector = CustomCollector(should_process=False)
        
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"
        runspec = RunSpec(video_file, output_dir)
        
        assert collector.apply_exist_rule(runspec) is False

