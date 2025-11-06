from typing import Literal

from zarr.api.synchronous import (
    array,
    consolidate_metadata,
    copy,
    copy_all,
    copy_store,
    create,
    create_array,
    create_group,
    create_hierarchy,
    empty,
    empty_like,
    from_array,
    full,
    full_like,
    group,
    load,
    ones,
    ones_like,
    open,
    open_array,
    open_consolidated,
    open_group,
    open_like,
    save,
    save_array,
    save_group,
    tree,
    zeros,
    zeros_like,
)

from zarr.core.array import Array, AsyncArray
from zarr.core.config import config
from zarr.core.group import AsyncGroup, Group

def print_debug_info() -> None: ...

def set_log_level(
    level: Literal[
        'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
    ]) -> None: ...

def set_format(log_format: str) -> None: ...

__version__: str

__all__ = [
        "Array",
    "AsyncArray",
    "AsyncGroup",
    "Group",
    "__version__",
    "array",
    "config",
    "consolidate_metadata",
    "copy",
    "copy_all",
    "copy_store",
    "create",
    "create_array",
    "create_group",
    "create_hierarchy",
    "empty",
    "empty_like",
    "from_array",
    "full",
    "full_like",
    "group",
    "load",
    "ones",
    "ones_like",
    "open",
    "open_array",
    "open_consolidated",
    "open_group",
    "open_like",
    "print_debug_info",
    "save",
    "save_array",
    "save_group",
    "tree",
    "zeros",
    "zeros_like",
]
