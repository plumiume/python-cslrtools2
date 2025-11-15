"""Unit tests for lmpipe/options.py

Tests for options group to dict conversion functions.
Coverage target: 96% → 100%
"""
from __future__ import annotations

import pytest # pyright: ignore[reportUnusedImport]
from pathlib import Path

from cslrtools2.lmpipe.options import (
    LoggingOptionsGroup,
    RuntimeResourcesGroup,
    ExecutorOptionsGroup,
    CollectorOptionsGroup,
    LMPipeOptionsGroup,
    logging_options_group_to_dict,
    runtime_resources_group_to_dict,
    executor_options_group_to_dict,
    collector_options_group_to_dict,
    lm_pipe_options_group_to_dict,
)


class TestLoggingOptionsGroupConversion:
    """Test logging_options_group_to_dict()."""

    def test_logging_options_default(self):
        """Test conversion with default values."""
        group = LoggingOptionsGroup.T()
        result = logging_options_group_to_dict(group)
        
        assert result["log_level"] == "info"
        assert result["log_file"] is None

    def test_logging_options_custom(self):
        """Test conversion with custom values."""
        group = LoggingOptionsGroup.T()
        group.log_level = "debug"
        group.log_file = Path("test.log")
        
        result = logging_options_group_to_dict(group)
        
        assert result["log_level"] == "debug"
        assert result["log_file"] == Path("test.log")


class TestRuntimeResourcesGroupConversion:
    """Test runtime_resources_group_to_dict()."""

    def test_runtime_resources_default(self):
        """Test conversion with default values."""
        group = RuntimeResourcesGroup.T()
        result = runtime_resources_group_to_dict(group)
        
        assert result["cpu"] == 1.0
        assert result["cpu_tags"] == []
        assert result["gpu"] == 0.0
        assert result["gpu_tags"] == []

    def test_runtime_resources_custom(self):
        """Test conversion with custom values."""
        group = RuntimeResourcesGroup.T()
        group.cpu = 2.5
        group.cpu_tags = ["fast", "x86"]
        group.gpu = 1.0
        group.gpu_tags = ["cuda", "tensor"]
        
        result = runtime_resources_group_to_dict(group)
        
        assert result["cpu"] == 2.5
        assert result["cpu_tags"] == ["fast", "x86"]
        assert result["gpu"] == 1.0
        assert result["gpu_tags"] == ["cuda", "tensor"]


class TestExecutorOptionsGroupConversion:
    """Test executor_options_group_to_dict()."""

    def test_executor_options_default(self):
        """Test conversion with default values."""
        group = ExecutorOptionsGroup.T()
        result = executor_options_group_to_dict(group)
        
        assert result["max_cpus"] == 1
        assert result["executor_mode"] is None
        assert result["executor_type"] is None

    def test_executor_options_custom(self):
        """Test conversion with custom values."""
        group = ExecutorOptionsGroup.T()
        group.max_cpus = 4
        group.executor_mode = "serial"
        group.executor_type = "thread"
        
        result = executor_options_group_to_dict(group)
        
        assert result["max_cpus"] == 4
        assert result["executor_mode"] == "serial"
        assert result["executor_type"] == "thread"


class TestCollectorOptionsGroupConversion:
    """Test collector_options_group_to_dict()."""

    def test_collector_options_default(self):
        """Test conversion with default values."""
        group = CollectorOptionsGroup.T()
        result = collector_options_group_to_dict(group)
        
        assert result["landmark_matrix_save_file_format"] is None
        assert result["landmark_matrix_save_exist_rule"] == "skip"
        assert result["annotated_frames_save_file_format"] == ".mp4"
        assert result["annotated_frames_save_framework"] is None
        assert result["annotated_frames_save_exist_rule"] == "skip"
        assert result["annotated_frames_show_framework"] is None

    def test_collector_options_custom(self):
        """Test conversion with custom values."""
        group = CollectorOptionsGroup.T()
        group.landmark_matrix_save_file_format = "csv"
        group.landmark_matrix_save_exist_rule = "overwrite"
        group.annotated_frames_save_file_format = "mp4"
        group.annotated_frames_save_framework = "matplotlib"
        group.annotated_frames_save_exist_rule = "skip"
        group.annotated_frames_show_framework = "matplotlib"
        
        result = collector_options_group_to_dict(group)
        
        assert result["landmark_matrix_save_file_format"] == "csv"
        assert result["landmark_matrix_save_exist_rule"] == "overwrite"
        assert result["annotated_frames_save_file_format"] == "mp4"
        assert result["annotated_frames_save_framework"] == "matplotlib"
        assert result["annotated_frames_save_exist_rule"] == "skip"
        assert result["annotated_frames_show_framework"] == "matplotlib"


class TestLMPipeOptionsGroupConversion:
    """Test lm_pipe_options_group_to_dict()."""

    def test_lmpipe_options_default(self):
        """Test full LMPipeOptionsGroup conversion with defaults."""
        group = LMPipeOptionsGroup.T()
        result = lm_pipe_options_group_to_dict(group)
        
        # Logging options
        assert result["log_level"] == "info"
        assert result["log_file"] is None
        
        # Runtime resources
        assert result["cpu"] == 1.0
        assert result["cpu_tags"] == []
        assert result["gpu"] == 0.0
        assert result["gpu_tags"] == []
        
        # Executor options
        assert result["max_cpus"] == 1
        assert result["executor_mode"] is None
        assert result["executor_type"] is None
        
        # Collector options
        assert result["landmark_matrix_save_file_format"] is None
        assert result["landmark_matrix_save_exist_rule"] == "skip"
        assert result["annotated_frames_save_file_format"] == ".mp4"
        assert result["annotated_frames_save_framework"] is None
        assert result["annotated_frames_save_exist_rule"] == "skip"
        assert result["annotated_frames_show_framework"] is None

    def test_lmpipe_options_custom(self):
        """Test full LMPipeOptionsGroup conversion with custom values."""
        group = LMPipeOptionsGroup.T()
        
        # Set custom values
        group.log_level = "debug"
        group.log_file = Path("custom.log")
        group.cpu = 4.0
        group.gpu = 1.0
        group.max_cpus = 8
        group.executor_mode = "serial"
        group.landmark_matrix_save_file_format = "zarr"
        
        result = lm_pipe_options_group_to_dict(group)
        
        # Verify merged result
        assert result["log_level"] == "debug"
        assert result["log_file"] == Path("custom.log")
        assert result["cpu"] == 4.0
        assert result["gpu"] == 1.0
        assert result["max_cpus"] == 8
        assert result["executor_mode"] == "serial"
        assert result["landmark_matrix_save_file_format"] == "zarr"
