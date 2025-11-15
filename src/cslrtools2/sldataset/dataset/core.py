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

# (root) / metadata / {km: Any}
# (root) / connections / {klm}.{klm}
# (root) / items / {idx} / videos / {kvid}
# (root) / items / {idx} / landmarks /{klm}
# (root) / items / {idx} / targets / {ktgt}

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import (
    Dataset as _Dataset,
    IterableDataset as _IterableDataset
)
import zarr
if TYPE_CHECKING:
    from zarr.api.synchronous import StoreLike
else:
    StoreLike = object

from ...typings import ArrayLike
from ..logger import sldataset_logger
from ..utils import as_tensor, get_group
from .holder import SLKeyHolder
from .item import (
    SLDatasetItem,
    DefaultSLDatasetItem,
    TensorSLDatasetItem,
    ZarrSLDatasetItem,
)

######## Type Aliases ########

type DefaultSLDataset[
    Kvid: str, Klm: str, Ktgt: str
] = Dataset[
    DefaultSLDatasetItem[Kvid, Klm, Ktgt]
]

type DefaultIterableSLDataset[
    Kvid: str, Klm: str, Ktgt: str
] = IterableDataset[
    DefaultSLDatasetItem[Kvid, Klm, Ktgt]
]


######## Dataset ABC Class ########

class Dataset[Item](_Dataset[Item], ABC):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, index: int) -> Item: ...


class IterableDataset[Item](_IterableDataset[Item], ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Item]: ...


class SLDatasetBatch[Kmeta: str, Kvid: str, Klm: str, Ktgt: str](
    SLKeyHolder[Kmeta, Kvid, Klm, Ktgt]
    ):

    def __init__(
        self,
        dataset: SLDataset[Kmeta, Kvid, Klm, Ktgt, Tensor],
        item: TensorSLDatasetItem[Kvid, Klm, Ktgt],
        ):
        self.dataset = dataset
        self.item = item

        sldataset_logger.debug(
            f"Created SLDatasetBatch with {len(item.videos)} videos, "
            f"{len(item.landmarks)} landmarks, {len(item.targets)} targets"
        )

    def to(self, device: torch.device) -> Self:

        return type(self)(
            dataset=self.dataset.to_partially(device),
            item=self.item.to(device)
        )

    @classmethod
    def from_batch(
        cls,
        dataset: SLDataset[Kmeta, Kvid, Klm, Ktgt, Any],
        batch: Sequence[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]
        ) -> Self:

        b0 = batch[0]

        return cls(
            dataset=SLDataset(
                metadata=dataset.metadata,
                connections={
                    k: as_tensor(v) for k, v in dataset.connections.items()
                },
                items=dataset.items
            ),
            item=SLDatasetItem(
                videos={
                    kvid: torch.cat([
                        as_tensor(item.videos[kvid]) for item in batch
                    ])
                    for kvid in b0.videos.keys()
                },
                landmarks={
                    klm: torch.cat([
                        as_tensor(item.landmarks[klm]) for item in batch
                    ])
                    for klm in b0.landmarks.keys()
                },
                targets={
                    ktgt: torch.cat([
                        as_tensor(item.targets[ktgt]) for item in batch
                    ])
                    for ktgt in b0.targets.keys()
                }
            )
        )


######## Dataset Classes ########

class SLDataset[Kmeta: str, Kvid: str, Klm: str, Ktgt: str, Vconn: ArrayLike](
    SLKeyHolder[Kmeta, Kvid, Klm, Ktgt],
    Dataset[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]
    ):

    def __init__(
        self,
        metadata: Mapping[Kmeta, Any],
        connections: Mapping[tuple[Klm, Klm], Vconn],
        items: DefaultSLDataset[Kvid, Klm, Ktgt] | Sequence[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]
        ):

        self.metadata = metadata
        self.connections = connections
        self.items = items

        sldataset_logger.info(
            f"Initialized SLDataset with {len(items)} items, "
            f"{len(metadata)} metadata entries, {len(connections)} landmark connections"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DefaultSLDatasetItem[Kvid, Klm, Ktgt]:
        sldataset_logger.debug(f"Fetching dataset item at index {index}")
        return self.items[index]

    def to_partially(self, device: torch.device) -> SLDataset[Kmeta, Kvid, Klm, Ktgt, Tensor]:

        return SLDataset(
            metadata=self.metadata,
            connections={
                k: as_tensor(v).to(device)
                for k, v in self.connections.items()
            },
            items=self.items
        )

    @classmethod
    def from_zarr(
        cls, group: zarr.Group
        ) -> SLDataset[Kmeta, Kvid, Klm, Ktgt, zarr.Array]:

        metadata = {
            k: v
            for k, v in get_group(group, "metadata").attrs.items()
            if cls.is_metadata_key(k)
        }

        connections: Mapping[tuple[Klm, Klm], zarr.Array] = {}
        for kconn, vconn in get_group(group, "connections").arrays():
            klm1, klm2 = kconn.split(".")
            if not cls.is_landmark_key(klm1):
                continue
            if not cls.is_landmark_key(klm2):
                continue
            connections[(klm1, klm2)] = vconn

        items: Sequence[ZarrSLDatasetItem[Kvid, Klm, Ktgt]] = []
        for item_group in get_group(group, "items").group_values():
            items.append(
                SLDatasetItem[
                    Kvid, zarr.Array,
                    Klm, zarr.Array,
                    Ktgt, zarr.Array
                ].from_zarr(item_group)
            )

        return SLDataset(
            metadata=metadata,
            connections=connections,
            items=items
        )



class IterableSLDataset[Kmeta: str, Kvid: str, Klm: str, Ktgt: str, Vconn: ArrayLike](
    SLKeyHolder[Kmeta, Kvid, Klm, Ktgt],
    IterableDataset[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]
    ):

    def __init__(
        self,
        metadata: Mapping[Kmeta, Any],
        connections: Mapping[tuple[Klm, Klm], Vconn],
        items: IterableDataset[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]
        ):

        self.metadata = metadata
        self.connections = connections
        self.items = items

        sldataset_logger.info(
            f"Initialized IterableSLDataset with {len(metadata)} metadata entries, "
            f"{len(connections)} landmark connections"
        )

    def __iter__(self) -> Iterator[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]:
        sldataset_logger.debug("Starting iteration over IterableSLDataset")
        return iter(self.items)

    def to_partially(self, device: torch.device) -> IterableSLDataset[Kmeta, Kvid, Klm, Ktgt, Tensor]:

        return IterableSLDataset(
            metadata=self.metadata,
            connections={
                k: as_tensor(v).to(device)
                for k, v in self.connections.items()
            },
            items=self.items
        )


def dataset_to_zarr(
    dataset: (
        SLDataset[Any, Any, Any, Any, Any] |
        IterableSLDataset[Any, Any, Any, Any, Any]
    ),
    store_or_group: StoreLike | zarr.Group
    ) -> zarr.Group:

    if isinstance(store_or_group, zarr.Group):
        root_group = store_or_group
    else:
        root_group = zarr.create_group(store_or_group)

    metadata_group = root_group.create_group("metadata")
    metadata_group.attrs.update(dataset.metadata)

    connections_group = root_group.create_group("connections")
    for (klm1, klm2), conn in dataset.connections.items():
        connections_group.create_array(
            f"{klm1}.{klm2}", data=np.asarray(conn)
        )

    items_group = root_group.create_group("items")
    item_iter = iter(
        dataset if isinstance(dataset, IterableSLDataset)
        else (dataset[i] for i in range(len(dataset)))
    )
    for idx, item in enumerate(item_iter):
        item.to_zarr(items_group.create_group(str(idx)))

    return root_group
