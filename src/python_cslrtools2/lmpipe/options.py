"""Configuration options for the LMPipe pipeline.

This module defines TypedDict classes and default configurations for
runtimeresources, executor, collector, lmpipe options used throughout the pipeline processing.

Attributes:
    DEFAULT_RUNTIME_RESOURCES (RuntimeResources): Default specification of runtime resource allocation.
    DEFAULT_EXECUTOR_OPTIONS (ExecutorOptions): Default options for executor configuration.
    DEFAULT_COLLECTOR_OPTIONS (CollectorOptions): Default options for collector configuration.
    DEFAULT_LMPIPE_OPTIONS (LMPipeOptions): Combined default lmpipe configuration.
"""

from clipar import group, mixin
from typing import TypedDict
from .typings import ExecutorMode, ExecutorType, ExistRule

class RuntimeResources(TypedDict):
    """specification of runtime resource allocation"""
    cpu: float
    'amount of CPU resources to allocate'
    cpu_tags: list[str]
    'tags for CPU resource allocation'
    gpu: float
    'amount of GPU resources to allocate'
    gpu_tags: list[str]
    'tags for GPU resource allocation'

class RuntimeResourcesPartial(TypedDict, total=False):
    """Partial specification of runtime resource allocation
    
    Same as RuntimeResources but with all fields optional.
    """
    cpu: float
    'amount of CPU resources to allocate'
    cpu_tags: list[str]
    'tags for CPU resource allocation'
    gpu: float
    'amount of GPU resources to allocate'
    gpu_tags: list[str]
    'tags for GPU resource allocation'

@group
class RuntimeResourcesGroup(mixin.ReprMixin):
    """specification of runtime resource allocation"""
    cpu: float = 1.0
    'amount of CPU resources to allocate'
    cpu_tags: list[str] = []
    'tags for CPU resource allocation'
    gpu: float = 0.0
    'amount of GPU resources to allocate'
    gpu_tags: list[str] = []
    'tags for GPU resource allocation'

DEFAULT_RUNTIME_RESOURCES: RuntimeResources = {
    'cpu': 1.0,
    'cpu_tags': [],
    'gpu': 0.0,
    'gpu_tags': []
}

class ExecutorOptions(TypedDict):
    """options for executor configuration"""
    max_cpus: int
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode
    'execution mode for the executor'
    executor_type: ExecutorType
    'type of executor to use'

class ExecutorOptionsPartial(TypedDict, total=False):
    """Partial options for executor configuration
    
    Same as ExecutorOptions but with all fields optional.
    """
    max_cpus: int
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode
    'execution mode for the executor'
    executor_type: ExecutorType
    'type of executor to use'

@group
class ExecutorOptionsGroup(mixin.ReprMixin):
    """options for executor configuration"""
    max_cpus: int = 1
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode = None
    'execution mode for the executor'
    executor_type: ExecutorType = None
    'type of executor to use'

DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
    'max_cpus': 1,
    'executor_mode': None,
    'executor_type': None
}

class CollectorOptions(TypedDict):
    """options for collector configuration"""
    landmark_matrix_save_file_format: str | None
    'file format for saving landmark matrix files'
    landmark_matrix_save_exist_rule: ExistRule
    'existence rule for saving landmark matrix files'
    annotated_frames_save_file_format: str
    'file format for saving annotated frame files'
    annotated_frames_save_framework: str | None
    'framework to use for saving annotated frames (e.g., 'cv2')'
    annotated_frames_save_exist_rule: ExistRule
    'existence rule for saving annotated frame files'
    annotated_frames_show_framework: str | None
    'framework to use for displaying annotated frames (e.g., 'cv2')'

class CollectorOptionsPartial(TypedDict, total=False):
    """Partial options for collector configuration
    
    Same as CollectorOptions but with all fields optional.
    """
    landmark_matrix_save_file_format: str | None
    'file format for saving landmark matrix files'
    landmark_matrix_save_exist_rule: ExistRule
    'existence rule for saving landmark matrix files'
    annotated_frames_save_file_format: str
    'file format for saving annotated frame files'
    annotated_frames_save_framework: str | None
    'framework to use for saving annotated frames (e.g., 'cv2')'
    annotated_frames_save_exist_rule: ExistRule
    'existence rule for saving annotated frame files'
    annotated_frames_show_framework: str | None
    'framework to use for displaying annotated frames (e.g., 'cv2')'

@group
class CollectorOptionsGroup(mixin.ReprMixin):
    """options for collector configuration"""
    landmark_matrix_save_file_format: str | None = None
    'file format for saving landmark matrix files'
    landmark_matrix_save_exist_rule: ExistRule = 'skip'
    'existence rule for saving landmark matrix files'
    annotated_frames_save_file_format: str = '.mp4'
    'file format for saving annotated frame files'
    annotated_frames_save_framework: str | None = None
    'framework to use for saving annotated frames (e.g., 'cv2')'
    annotated_frames_save_exist_rule: ExistRule = 'skip'
    'existence rule for saving annotated frame files'
    annotated_frames_show_framework: str | None = None
    'framework to use for displaying annotated frames (e.g., 'cv2')'

DEFAULT_COLLECTOR_OPTIONS: CollectorOptions = {
    'landmark_matrix_save_file_format': None,
    'landmark_matrix_save_exist_rule': 'skip',
    'annotated_frames_save_file_format': '.mp4',
    'annotated_frames_save_framework': None,
    'annotated_frames_save_exist_rule': 'skip',
    'annotated_frames_show_framework': None
}

class LMPipeOptions(
    RuntimeResources,
    ExecutorOptions,
    CollectorOptions,
    ):
    """LMPipeOptions configuration options."""
    pass

class LMPipeOptionsPartial(
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
    RuntimeResourcesGroup.T,
    ExecutorOptionsGroup.T,
    CollectorOptionsGroup.T,
    ):
    """CLI argument group combining runtimeresources and executor and collector options."""
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_RUNTIME_RESOURCES,
    **DEFAULT_EXECUTOR_OPTIONS,
    **DEFAULT_COLLECTOR_OPTIONS,
}

def runtime_resources_group_to_dict(group: RuntimeResourcesGroup.T) -> RuntimeResources:
    """Convert RuntimeResourcesGroup instance to RuntimeResources TypedDict."""
    return {
        'cpu': group.cpu,
        'cpu_tags': group.cpu_tags,
        'gpu': group.gpu,
        'gpu_tags': group.gpu_tags
    }

def executor_options_group_to_dict(group: ExecutorOptionsGroup.T) -> ExecutorOptions:
    """Convert ExecutorOptionsGroup instance to ExecutorOptions TypedDict."""
    return {
        'max_cpus': group.max_cpus,
        'executor_mode': group.executor_mode,
        'executor_type': group.executor_type
    }

def collector_options_group_to_dict(group: CollectorOptionsGroup.T) -> CollectorOptions:
    """Convert CollectorOptionsGroup instance to CollectorOptions TypedDict."""
    return {
        'landmark_matrix_save_file_format': group.landmark_matrix_save_file_format,
        'landmark_matrix_save_exist_rule': group.landmark_matrix_save_exist_rule,
        'annotated_frames_save_file_format': group.annotated_frames_save_file_format,
        'annotated_frames_save_framework': group.annotated_frames_save_framework,
        'annotated_frames_save_exist_rule': group.annotated_frames_save_exist_rule,
        'annotated_frames_show_framework': group.annotated_frames_show_framework
    }

def lm_pipe_options_group_to_dict(group: LMPipeOptionsGroup.T) -> LMPipeOptions:
    """Convert LMPipeOptionsGroup instance to LMPipeOptions TypedDict."""
    return {
        **runtime_resources_group_to_dict(group),
        **executor_options_group_to_dict(group),
        **collector_options_group_to_dict(group)
    }
