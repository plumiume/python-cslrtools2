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
from typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import torch

from ...typings import ArrayLike
from ..utils import as_tensor
from ..dataset.core import Dataset, SLDataset
from ..dataset.core import SLDatasetItem, DefaultSLDatasetItem, TensorSLDatasetItem

class Transform[Kvid: str, Klm: str, Ktgt: str](ABC):
    def _ensure_generator(self, gen: torch.Generator | None) -> torch.Generator:
        return gen or torch.default_generator
    @abstractmethod
    def __call__(
        self, item: TensorSLDatasetItem[Kvid, Klm, Ktgt]
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]: ...

class TransformSLDataset[Kmeta: str, Kvid: str, Klm: str, Ktgt: str, Vconn: ArrayLike](
    SLDataset[Kmeta, Kvid, Klm, Ktgt, Vconn]
    ):

    def __init__(
        self,
        dataset: Dataset[DefaultSLDatasetItem[Kvid, Klm, Ktgt]],
        compose: Sequence[Callable[
            [TensorSLDatasetItem[Kvid, Klm, Ktgt]],
            TensorSLDatasetItem[Kvid, Klm, Ktgt]
        ]]
        ):

        self.dataset = dataset
        self.compose = compose

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:

        item = self.dataset[index]

        tensor_item = SLDatasetItem(
            videos={
                kvid: as_tensor(v)
                for kvid, v in item.videos.items()
            },
            landmarks={
                klm: as_tensor(v)
                for klm, v in item.landmarks.items()
            },
            targets={
                ktgt: as_tensor(v)
                for ktgt, v in item.targets.items()
            }
        )

        for transform in self.compose:
            tensor_item = transform(tensor_item)

        return tensor_item
