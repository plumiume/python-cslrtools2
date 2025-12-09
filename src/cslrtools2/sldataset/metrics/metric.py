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

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypedDict, Literal

import numpy as np
from numpy.typing import NDArray

from ...typings import ArrayLike
from ..dataset.item import SLDatasetItem
from .schema import MetricSpec
from .stat import StatResult

type CategoryGroup = Literal["videos", "landmarks", "targets"]


class CachedMapping[K](Mapping[K, NDArray[Any]]):

    def __init__(self, base: Mapping[K, ArrayLike]):
        self._base = base
        self._cache: dict[K, NDArray[Any]] = {}

    def __getitem__(self, key: K) -> NDArray[Any]:
        if key not in self._cache:
            self._cache[key] = np.asarray(self._base[key])
        return self._cache[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)


class CachedSLDatasetItem[
    Kvid: str,
    Klm: str,
    Ktgt: str,
](SLDatasetItem[
    Kvid, NDArray[Any],
    Klm, NDArray[Any],
    Ktgt, NDArray[Any],
]):

    def __init__(
        self,
        videos: Mapping[Kvid, ArrayLike],
        landmarks: Mapping[Klm, ArrayLike],
        targets: Mapping[Ktgt, ArrayLike],
    ):

        super().__init__(
            CachedMapping(videos),
            CachedMapping(landmarks),
            CachedMapping(targets),
        )


class MetricResult(TypedDict):
    metric_ident: str  # Calculator set this
    metric_name: str   # Calculator set this
    metric_fqn: str    # Calculator set this
    sample_idx: int    # Calculator set this
    categories_name: str
    categories_values: list[str]
    values: Mapping[str, Any]    # Metric.calculate() set this
    metadata: Mapping[str, Any]  # Metric set this


class Metric(ABC):

    def __init__(
        self,
        spec: MetricSpec
    ):
        self._spec = spec

    @abstractmethod
    def calculate(
        self,
        category_group: CategoryGroup,
        stats: StatResult,
        data: NDArray[Any],
    ) -> Mapping[str, Any]:
        ...

    def get_metadata(self) -> Mapping[str, Any]:
        return {}

    def get_cli_description(self) -> str:
        return "N/A"

    def get_cli_detail(self) -> str:
        return "N/A"
