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

"""Configuration options for the LMPipe pipeline.

This module defines TypedDict classes and default configurations for
logging, runtimeresources, executor, collector, lmpipe options used throughout the pipeline processing.

Attributes:
    DEFAULT_LOGGING_OPTIONS (LoggingOptions): Default Logging configuration for pipeline execution..
    DEFAULT_RUNTIME_RESOURCES (RuntimeResources): Default Runtime resource allocation for parallel execution..
    DEFAULT_EXECUTOR_OPTIONS (ExecutorOptions): Default Parallel execution configuration..
    DEFAULT_COLLECTOR_OPTIONS (CollectorOptions): Default Output collection and saving configuration..
    DEFAULT_LMPIPE_OPTIONS (LMPipeOptions): Combined default lmpipe configuration.
"""

from clipar import group, mixin
from typing import TypedDict
from .typings import ExecutorMode, ExecutorType, ExistRule

class LoggingOptions(TypedDict):
    """Logging configuration for pipeline execution."""
    log_level: str
    "Logging verbosity level. Choices: 'debug', 'info', 'warning', 'error', 'critical'."
    log_file: str | None
    "Optional path to save log output. If None, logs are written to console only."

class LoggingOptionsPartial(TypedDict, total=False):
    """Partial Logging configuration for pipeline execution.
    
    Same as LoggingOptions but with all fields optional.
    """
    log_level: str
    "Logging verbosity level. Choices: 'debug', 'info', 'warning', 'error', 'critical'."
    log_file: str | None
    "Optional path to save log output. If None, logs are written to console only."

@group
class LoggingOptionsGroup(mixin.ReprMixin):
    """Logging configuration for pipeline execution."""
    log_level: str = 'info'
    "Logging verbosity level. Choices: 'debug', 'info', 'warning', 'error', 'critical'."
    log_file: str | None = None
    "Optional path to save log output. If None, logs are written to console only."

DEFAULT_LOGGING_OPTIONS: LoggingOptions = {
    "log_level": 'info',
    "log_file": None
}

class RuntimeResources(TypedDict):
    """Runtime resource allocation for parallel execution."""
    cpu: float
    "Number of CPU cores to allocate per worker (float allows fractional allocation)."
    cpu_tags: list[str]
    "Tags for CPU resource management (used by advanced schedulers)."
    gpu: float
    "Number of GPU devices to allocate per worker (0.0 = CPU-only execution)."
    gpu_tags: list[str]
    "Tags for GPU resource management (used by advanced schedulers)."

class RuntimeResourcesPartial(TypedDict, total=False):
    """Partial Runtime resource allocation for parallel execution.
    
    Same as RuntimeResources but with all fields optional.
    """
    cpu: float
    "Number of CPU cores to allocate per worker (float allows fractional allocation)."
    cpu_tags: list[str]
    "Tags for CPU resource management (used by advanced schedulers)."
    gpu: float
    "Number of GPU devices to allocate per worker (0.0 = CPU-only execution)."
    gpu_tags: list[str]
    "Tags for GPU resource management (used by advanced schedulers)."

@group
class RuntimeResourcesGroup(mixin.ReprMixin):
    """Runtime resource allocation for parallel execution."""
    cpu: float = 1.0
    "Number of CPU cores to allocate per worker (float allows fractional allocation)."
    cpu_tags: list[str] = []
    "Tags for CPU resource management (used by advanced schedulers)."
    gpu: float = 0.0
    "Number of GPU devices to allocate per worker (0.0 = CPU-only execution)."
    gpu_tags: list[str] = []
    "Tags for GPU resource management (used by advanced schedulers)."

DEFAULT_RUNTIME_RESOURCES: RuntimeResources = {
    "cpu": 1.0,
    "cpu_tags": [],
    "gpu": 0.0,
    "gpu_tags": []
}

class ExecutorOptions(TypedDict):
    """Parallel execution configuration."""
    max_cpus: int
    "Maximum number of parallel workers. Set to 1 for sequential processing."
    executor_mode: ExecutorMode
    "Execution mode: 'loky' for robust multiprocessing, 'thread' for I/O bound tasks, or None for auto-detection."
    executor_type: ExecutorType
    "Executor backend: 'process' for CPU-bound tasks, 'thread' for I/O-bound, or None for auto-selection."

class ExecutorOptionsPartial(TypedDict, total=False):
    """Partial Parallel execution configuration.
    
    Same as ExecutorOptions but with all fields optional.
    """
    max_cpus: int
    "Maximum number of parallel workers. Set to 1 for sequential processing."
    executor_mode: ExecutorMode
    "Execution mode: 'loky' for robust multiprocessing, 'thread' for I/O bound tasks, or None for auto-detection."
    executor_type: ExecutorType
    "Executor backend: 'process' for CPU-bound tasks, 'thread' for I/O-bound, or None for auto-selection."

@group
class ExecutorOptionsGroup(mixin.ReprMixin):
    """Parallel execution configuration."""
    max_cpus: int = 1
    "Maximum number of parallel workers. Set to 1 for sequential processing."
    executor_mode: ExecutorMode = None
    "Execution mode: 'loky' for robust multiprocessing, 'thread' for I/O bound tasks, or None for auto-detection."
    executor_type: ExecutorType = None
    "Executor backend: 'process' for CPU-bound tasks, 'thread' for I/O-bound, or None for auto-selection."

DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
    "max_cpus": 1,
    "executor_mode": None,
    "executor_type": None
}

class CollectorOptions(TypedDict):
    """Output collection and saving configuration."""
    landmark_matrix_save_file_format: str | None
    "File format for landmark coordinates. Supported: '.csv', '.json', '.npy', '.npz', '.pt', '.safetensors', '.zarr'. None = disabled."
    landmark_matrix_save_exist_rule: ExistRule
    "Behavior when output file exists: 'skip' (keep existing), 'overwrite' (replace), or 'error' (raise exception)."
    annotated_frames_save_file_format: str
    "Video format for annotated output. Supported: '.mp4', '.avi', '.mov', '.mkv'."
    annotated_frames_save_framework: str | None
    "Framework for video encoding: 'cv2' (OpenCV), 'matplotlib', or None for auto-detection."
    annotated_frames_save_exist_rule: ExistRule
    "Behavior when annotated output exists: 'skip', 'overwrite', or 'error'."
    annotated_frames_show_framework: str | None
    "Framework for live preview display: 'cv2' (OpenCV window), 'pil' (PIL), 'matplotlib', 'torchvision', or None to disable."

class CollectorOptionsPartial(TypedDict, total=False):
    """Partial Output collection and saving configuration.
    
    Same as CollectorOptions but with all fields optional.
    """
    landmark_matrix_save_file_format: str | None
    "File format for landmark coordinates. Supported: '.csv', '.json', '.npy', '.npz', '.pt', '.safetensors', '.zarr'. None = disabled."
    landmark_matrix_save_exist_rule: ExistRule
    "Behavior when output file exists: 'skip' (keep existing), 'overwrite' (replace), or 'error' (raise exception)."
    annotated_frames_save_file_format: str
    "Video format for annotated output. Supported: '.mp4', '.avi', '.mov', '.mkv'."
    annotated_frames_save_framework: str | None
    "Framework for video encoding: 'cv2' (OpenCV), 'matplotlib', or None for auto-detection."
    annotated_frames_save_exist_rule: ExistRule
    "Behavior when annotated output exists: 'skip', 'overwrite', or 'error'."
    annotated_frames_show_framework: str | None
    "Framework for live preview display: 'cv2' (OpenCV window), 'pil' (PIL), 'matplotlib', 'torchvision', or None to disable."

@group
class CollectorOptionsGroup(mixin.ReprMixin):
    """Output collection and saving configuration."""
    landmark_matrix_save_file_format: str | None = None
    "File format for landmark coordinates. Supported: '.csv', '.json', '.npy', '.npz', '.pt', '.safetensors', '.zarr'. None = disabled."
    landmark_matrix_save_exist_rule: ExistRule = 'skip'
    "Behavior when output file exists: 'skip' (keep existing), 'overwrite' (replace), or 'error' (raise exception)."
    annotated_frames_save_file_format: str = '.mp4'
    "Video format for annotated output. Supported: '.mp4', '.avi', '.mov', '.mkv'."
    annotated_frames_save_framework: str | None = None
    "Framework for video encoding: 'cv2' (OpenCV), 'matplotlib', or None for auto-detection."
    annotated_frames_save_exist_rule: ExistRule = 'skip'
    "Behavior when annotated output exists: 'skip', 'overwrite', or 'error'."
    annotated_frames_show_framework: str | None = None
    "Framework for live preview display: 'cv2' (OpenCV window), 'pil' (PIL), 'matplotlib', 'torchvision', or None to disable."

DEFAULT_COLLECTOR_OPTIONS: CollectorOptions = {
    "landmark_matrix_save_file_format": None,
    "landmark_matrix_save_exist_rule": 'skip',
    "annotated_frames_save_file_format": '.mp4',
    "annotated_frames_save_framework": None,
    "annotated_frames_save_exist_rule": 'skip',
    "annotated_frames_show_framework": None
}

class LMPipeOptions(
    LoggingOptions,
    RuntimeResources,
    ExecutorOptions,
    CollectorOptions,
    ):
    """LMPipeOptions configuration options."""
    pass

class LMPipeOptionsPartial(
    LoggingOptionsPartial,
    RuntimeResourcesPartial,
    ExecutorOptionsPartial,
    CollectorOptionsPartial,
    total=False
    ):
    """Partial lmpipeoptions configuration options.
    
    Same as LMPipeOptions but with all fields optional, useful for
    overriding specific configuration values.
    """
    pass

@group
class LMPipeOptionsGroup(
    LoggingOptionsGroup.T,
    RuntimeResourcesGroup.T,
    ExecutorOptionsGroup.T,
    CollectorOptionsGroup.T,
    ):
    """CLI argument group combining logging and runtimeresources and executor and collector options."""
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_LOGGING_OPTIONS,
    **DEFAULT_RUNTIME_RESOURCES,
    **DEFAULT_EXECUTOR_OPTIONS,
    **DEFAULT_COLLECTOR_OPTIONS,
}

def logging_options_group_to_dict(group: LoggingOptionsGroup.T) -> LoggingOptions:
    """Convert LoggingOptionsGroup instance to LoggingOptions TypedDict."""
    return {
        "log_level": group.log_level,
        "log_file": group.log_file
    }

def runtime_resources_group_to_dict(group: RuntimeResourcesGroup.T) -> RuntimeResources:
    """Convert RuntimeResourcesGroup instance to RuntimeResources TypedDict."""
    return {
        "cpu": group.cpu,
        "cpu_tags": group.cpu_tags,
        "gpu": group.gpu,
        "gpu_tags": group.gpu_tags
    }

def executor_options_group_to_dict(group: ExecutorOptionsGroup.T) -> ExecutorOptions:
    """Convert ExecutorOptionsGroup instance to ExecutorOptions TypedDict."""
    return {
        "max_cpus": group.max_cpus,
        "executor_mode": group.executor_mode,
        "executor_type": group.executor_type
    }

def collector_options_group_to_dict(group: CollectorOptionsGroup.T) -> CollectorOptions:
    """Convert CollectorOptionsGroup instance to CollectorOptions TypedDict."""
    return {
        "landmark_matrix_save_file_format": group.landmark_matrix_save_file_format,
        "landmark_matrix_save_exist_rule": group.landmark_matrix_save_exist_rule,
        "annotated_frames_save_file_format": group.annotated_frames_save_file_format,
        "annotated_frames_save_framework": group.annotated_frames_save_framework,
        "annotated_frames_save_exist_rule": group.annotated_frames_save_exist_rule,
        "annotated_frames_show_framework": group.annotated_frames_show_framework
    }

def lm_pipe_options_group_to_dict(group: LMPipeOptionsGroup.T) -> LMPipeOptions:
    """Convert LMPipeOptionsGroup instance to LMPipeOptions TypedDict."""
    return {
        **logging_options_group_to_dict(group),
        **runtime_resources_group_to_dict(group),
        **executor_options_group_to_dict(group),
        **collector_options_group_to_dict(group)
    }
