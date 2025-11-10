from typing import Any, TYPE_CHECKING, TypeGuard, Protocol, TypedDict, runtime_checkable
import importlib.metadata
from clipar.entities import NamespaceWrapper

if TYPE_CHECKING:
    from ..estimator import Estimator
else:
    Estimator = tuple

def _is_tuple(obj: object) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)

def _is_nswrapper(obj: object) -> TypeGuard[NamespaceWrapper[Any]]:
    return isinstance(obj, NamespaceWrapper)

@runtime_checkable
class _EstimatorCreator[T, K: str](Protocol):
    def __call__(self, ns: T) -> Estimator[K]: ...

def _is_estimator_creator(obj: object) -> TypeGuard[_EstimatorCreator[Any, Any]]:
    return isinstance(obj, _EstimatorCreator)

type Info[T, K: str] = tuple[
    NamespaceWrapper[T],
    _EstimatorCreator[T, K]
]

class PluginInfo[T, K: str](TypedDict):
    name: str
    type: str
    nswrapper: NamespaceWrapper[T]
    creator: _EstimatorCreator[T, K]

def loader() -> dict[str, dict[str, PluginInfo[Any, Any]]]:

    entry_points = importlib.metadata.entry_points(
        group="cslrtools2.lmpipe.plugins"
    )

    plugins: dict[str, dict[str, PluginInfo[Any, Any]]] = {}

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

        nswrapper, creator = info

        if not _is_nswrapper(nswrapper):
            raise TypeError(
                f"First element of plugin entry point {ep.name} is not a NamespaceWrapper"
            )

        if not _is_estimator_creator(creator):
            raise TypeError(
                f"Second element of plugin entry point {ep.name} is not callable"
            )

        name_, type_ = ep.name.rsplit(".", 1)

        plugins.setdefault(type_, {})[name_] = PluginInfo(
            name=name_,
            type=type_,
            nswrapper=nswrapper,
            creator=creator
        )

    return plugins