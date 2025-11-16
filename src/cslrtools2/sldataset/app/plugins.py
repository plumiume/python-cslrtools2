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


from __future__ import annotations

from typing import Any, Callable, TypedDict, TypeGuard, Protocol, runtime_checkable
import importlib.metadata
from clipar.entities import NamespaceWrapper

from ...exceptions import PluginError


def _is_tuple(obj: object) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)


def _is_nswrapper(obj: object) -> TypeGuard[NamespaceWrapper[Any]]:
    return isinstance(obj, NamespaceWrapper)


type Info[T] = tuple[NamespaceWrapper[T], Callable[[T], None]]


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
    entry_points = importlib.metadata.entry_points(group="cslrtools2.sldataset.plugins")

    plugins: dict[str, PluginInfo[Any]] = {}

    for ep in entry_points:
        info = ep.load()

        if not _is_tuple(info):
            raise PluginError(
                f"Plugin entry point {ep.name} does not return a tuple. "
                f"Plugin must return (NamespaceWrapper, processor_callable)."
            )
        if len(info) != 2:
            raise PluginError(
                f"Plugin entry point {ep.name} does not return a tuple of "
                f"length 2. Expected (NamespaceWrapper, processor_callable), "
                f"got {len(info)} elements."
            )

        nswrapper, processor = info

        if not _is_nswrapper(nswrapper):
            raise PluginError(
                f"First element of plugin entry point {ep.name} is not a "
                f"NamespaceWrapper. Got {type(nswrapper)}."
            )

        if not _is_processor(processor):
            raise PluginError(
                f"Second element of plugin entry point {ep.name} is not a "
                f"processor callable. Expected a Callable[[T], None], "
                f"got {type(processor)}."
            )

        plugins[ep.name] = PluginInfo(
            name=ep.name, nswrapper=nswrapper, processor=processor
        )

    return plugins
