"""Configuration options for the LMPipe pipeline.

This module defines TypedDict classes and default configurations for
runtimeresuorces, executor, lmpipe options used throughout the pipeline processing.

Attributes:
    DEFAULT_RUNTIME_RESUORCES (RuntimeResuorces): Default specification of runtime resource allocation.
    DEFAULT_EXECUTOR_OPTIONS (ExecutorOptions): Default executor configuration options.
    DEFAULT_LMPIPE_OPTIONS (LMPipeOptions): Combined default lmpipe configuration.
"""

from clipar import group, mixin
from typing import TypedDict
from .typings import ExecutorMode, ExecutorType

class RuntimeResuorces(TypedDict):
    """specification of runtime resource allocation"""
    cpu: float
    'amount of CPU resources to allocate'
    cpu_tags: list[str]
    'tags for CPU resource allocation'
    gpu: float
    'amount of GPU resources to allocate'
    gpu_tags: list[str]
    'tags for GPU resource allocation'

class RuntimeResuorcesPartial(TypedDict, total=False):
    """Partial specification of runtime resource allocation
    
    Same as RuntimeResuorces but with all fields optional.
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
class RuntimeResuorcesGroup(mixin.ReprMixin):
    """specification of runtime resource allocation"""
    cpu: float = 1.0
    'amount of CPU resources to allocate'
    cpu_tags: list[str] = []
    'tags for CPU resource allocation'
    gpu: float = 0.0
    'amount of GPU resources to allocate'
    gpu_tags: list[str] = []
    'tags for GPU resource allocation'

DEFAULT_RUNTIME_RESUORCES: RuntimeResuorces = {
    'cpu': 1.0,
    'cpu_tags': [],
    'gpu': 0.0,
    'gpu_tags': []
}

class ExecutorOptions(TypedDict):
    """ExecutorOptions configuration options."""
    max_cpus: float
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode
    'execution mode for the executor'
    executor_type: ExecutorType
    'type of executor to use'

class ExecutorOptionsPartial(TypedDict, total=False):
    """Partial ExecutorOptions configuration options.
    
    Same as ExecutorOptions but with all fields optional.
    """
    max_cpus: float
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode
    'execution mode for the executor'
    executor_type: ExecutorType
    'type of executor to use'

@group
class ExecutorOptionsGroup(mixin.ReprMixin):
    """ExecutorOptions configuration options."""
    max_cpus: float = 1.0
    'maximum number of CPUs to allocate for the executor'
    executor_mode: ExecutorMode = None
    'execution mode for the executor'
    executor_type: ExecutorType = None
    'type of executor to use'

DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
    'max_cpus': 1.0,
    'executor_mode': None,
    'executor_type': None
}

class LMPipeOptions(
    RuntimeResuorces,
    ExecutorOptions,
    ):
    """LMPipeOptions configuration options."""
    pass

class LMPipeOptionsPartial(
    RuntimeResuorcesPartial,
    ExecutorOptionsPartial,
    total=False
    ):
    """Partial lmpipeoptions configuration options.
    
    Same as LMPipeOptions but with all fields optional, useful for
    overriding specific configuration values.
    """
    pass

@group
class LMPipeOptionsGroup(
    RuntimeResuorcesGroup.T,
    ExecutorOptionsGroup.T,
    ):
    """CLI argument group combining runtimeresuorces and executor options."""
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_RUNTIME_RESUORCES,
    **DEFAULT_EXECUTOR_OPTIONS,
}
