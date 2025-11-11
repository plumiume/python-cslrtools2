from typing import Any, Callable, TypedDict, TypeGuard, Protocol, runtime_checkable
import importlib.metadata
from clipar.entities import NamespaceWrapper

def _is_tuple(obj: object) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)

def _is_nswrapper(obj: object) -> TypeGuard[NamespaceWrapper[Any]]:
    return isinstance(obj, NamespaceWrapper)

type Info[T] = tuple[
    NamespaceWrapper[T],
    Callable[[T], None]
]

@runtime_checkable
class _Processor[T](Protocol):
    def __call__(self, ns: T) -> None: ...

def _is_processor(obj: object) -> TypeGuard[_Processor[Any]]:
    return isinstance(obj, _Processor)

class PluginInfo[T](TypedDict):
    name: str
    nswrapper: NamespaceWrapper[T]
    processor: _Processor[T]

def loader() -> dict[str, PluginInfo[Any]]:

    entry_points = importlib.metadata.entry_points(
        group="cslrtools2.sldataset.plugins"
    )

    plugins: dict[str, PluginInfo[Any]] = {}

    for ep in entry_points:

        info = ep.load()

        if not _is_tuple(info):
            raise TypeError(
                f"Plugin entry point {ep.name} does not return a tuple"
            )
        if len(info) != 2:
            raise ValueError(
                f"Plugin entry point {ep.name} does not return a tuple of length 2"
            )
    
        nswrapper, processor = info

        if not _is_nswrapper(nswrapper):
            raise TypeError(
                f"First element of plugin entry point {ep.name} is not a NamespaceWrapper"
            )
        
        if not _is_processor(processor):
            raise TypeError(
                f"Second element of plugin entry point {ep.name} is not a processor callable"
            )
        
        plugins[ep.name] = PluginInfo(
            name=ep.name,
            nswrapper=nswrapper,
            processor=processor
        )

    return plugins
