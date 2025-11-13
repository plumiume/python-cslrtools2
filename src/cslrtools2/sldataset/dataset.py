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
from pathlib import Path

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

from ..typings import ArrayLike, PathLike
from .logger import sldataset_logger
from .utils import as_tensor, get_group, get_array
from .array_loader import (
    prekey_loaders, PrekeyLoadFunc,
    container_loaders, ContainerLoadFunc
)

######## Type Aliases ########

type DefaultSLDatasetItem[
    Kvid: str, Klm: str, Ktgt: str
] = SLDatasetItem[Kvid, Any, Klm, Any, Ktgt, Any]

type TensorSLDatasetItem[
    Kvid: str, Klm: str, Ktgt: str
] = SLDatasetItem[Kvid, Tensor, Klm, Tensor, Ktgt, Tensor]

type ZarrSLDatasetItem[
    Kvid: str, Klm: str, Ktgt: str
] = SLDatasetItem[
    Kvid, zarr.Array,
    Klm, zarr.Array,
    Ktgt, zarr.Array
]

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


######## Key Holder Class ########

class SLKeyHolder[Kmeta: str, Kvid: str, Klm: str, Ktgt: str]:
    """Type guards for sign language dataset keys.
    
    Provides runtime type checking for the different key types used in
    sign language datasets (metadata, video, landmark, target).
    
    Type Parameters:
        Kmeta: String type for metadata keys.
        Kvid: String type for video data keys.
        Klm: String type for landmark data keys.
        Ktgt: String type for target/label keys.
    """

    @classmethod
    def is_metadata_key(cls, obj: object) -> TypeGuard[Kmeta]:
        """Check if object is a valid metadata key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid metadata key.
        """
        return isinstance(obj, str)

    @classmethod
    def is_video_key(cls, obj: object) -> TypeGuard[Kvid]:
        """Check if object is a valid video key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid video key.
        """
        return isinstance(obj, str)
    
    @classmethod
    def is_landmark_key(cls, obj: object) -> TypeGuard[Klm]:
        """Check if object is a valid landmark key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid landmark key.
        """
        return isinstance(obj, str)
    
    @classmethod
    def is_target_key(cls, obj: object) -> TypeGuard[Ktgt]:
        """Check if object is a valid target key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid target key.
        """
        return isinstance(obj, str)

######## Item Class ########

class SLDatasetItem[
    Kvid: str, Vvid: ArrayLike,
    Klm: str, Vlm: ArrayLike,
    Ktgt: str, Vtgt: ArrayLike
    ](SLKeyHolder[Never, Kvid, Klm, Ktgt]):
    """Single item from a sign language dataset.
    
    Contains video data, landmark coordinates, and target labels for
    one sample in the dataset. Supports conversion to different devices
    for PyTorch training.
    
    Type Parameters:
        Kvid: String type for video data keys.
        Vvid: Array-like type for video data values.
        Klm: String type for landmark data keys.
        Vlm: Array-like type for landmark data values.
        Ktgt: String type for target/label keys.
        Vtgt: Array-like type for target/label values.
    
    Attributes:
        videos (:class:`~typing.Mapping`\\[:obj:`Kvid`, :obj:`Vvid`\\]):
            Video data mapping. Each value has shape ``[N, T, H, W, C]``
            where N is batch size, T is time, H is height, W is width, C is channels.
        landmarks (:class:`~typing.Mapping`\\[:obj:`Klm`, :obj:`Vlm`\\]):
            Landmark data mapping. Each value has shape ``[N, T, V, A]``
            where N is batch size, T is time, V is vertices, A is attributes.
        targets (:class:`~typing.Mapping`\\[:obj:`Ktgt`, :obj:`Vtgt`\\]):
            Target/label data mapping. Each value has shape ``[N, S]``
            where N is batch size, S is sequence length.
    """

    def __init__(
        self,
        videos: Mapping[Kvid, Vvid], # (Kvid, [N, T, H, W, C])
        landmarks: Mapping[Klm, Vlm], # (Klm, [N, T, V, A])
        targets: Mapping[Ktgt, Vtgt], # (Ktgt, [N, S])
        ): # N = 1 or Batch size

        self.videos = videos
        self.landmarks = landmarks
        self.targets = targets
        
        sldataset_logger.debug(
            f"Initialized SLDatasetItem with {len(videos)} videos, "
            f"{len(landmarks)} landmark types, {len(targets)} targets"
        )

    def to(self, device: torch.device) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:
        """Move all data to the specified device.
        
        Converts all video, landmark, and target data to :class:`torch.Tensor`
        and moves them to the specified :class:`torch.device`.
        
        Args:
            device (:class:`torch.device`): Target device (e.g., ``'cuda'``, ``'cpu'``).
            
        Returns:
            :class:`TensorSLDatasetItem`\\[:obj:`Kvid`, :obj:`Klm`, :obj:`Ktgt`\\]:
                New dataset item with all data on the specified device.
        """

        return SLDatasetItem(
            videos={
                k: as_tensor(v).to(device)
                for k, v in self.videos.items()
            },
            landmarks={
                k: as_tensor(v).to(device)
                for k, v in self.landmarks.items()
            },
            targets={
                k: as_tensor(v).to(device)
                for k, v in self.targets.items()
            }
        )

    def to_zarr(self, store_or_group: StoreLike | zarr.Group) -> zarr.Group:

        if isinstance(store_or_group, zarr.Group):
            item_group = store_or_group
        else:
            item_group = zarr.create_group(store_or_group)

        video_group = item_group.create_group("videos")
        for kvid, vvid in self.videos.items():
            video_group.create_array(
                kvid, data=np.asarray(vvid)
            )

        landmark_group = item_group.create_group("landmarks")
        for klm, vlm in self.landmarks.items():
            landmark_group.create_array(
                klm, data=np.asarray(vlm)
            )

        target_group = item_group.create_group("targets")
        for ktgt, vtgt in self.targets.items():
            target_group.create_array(
                ktgt, data=np.asarray(vtgt)
            )

        return item_group

    @classmethod
    def from_file_system(
        cls,
        path: PathLike,
        extra_prekey_load_funcs: Mapping[str, PrekeyLoadFunc] = {},
        extra_container_load_funcs: Mapping[str, ContainerLoadFunc] = {}
        ) -> DefaultSLDatasetItem[Kvid, Klm, Ktgt]:

        path = Path(path)

        prekey_load_funcs = {
            **prekey_loaders,
            **extra_prekey_load_funcs
        }
        container_load_funcs = {
            **container_loaders,
            **extra_container_load_funcs
        }

        videos = cls._load_category_from_fs(
            path / "videos",
            cls.is_video_key,
            prekey_load_funcs,
            container_load_funcs
        )
        landmarks = cls._load_category_from_fs(
            path / "landmarks",
            cls.is_landmark_key,
            prekey_load_funcs,
            container_load_funcs
        )
        targets = cls._load_category_from_fs(
            path / "targets",
            cls.is_target_key,
            prekey_load_funcs,
            container_load_funcs
        )

        return SLDatasetItem(
            videos=videos,
            landmarks=landmarks,
            targets=targets
        )

    @classmethod
    def _load_category_from_fs[K: str](
        cls,
        path_without_suffix: Path,
        is_key_func: Callable[[object], TypeGuard[K]],
        prekey_load_funcs: Mapping[str, PrekeyLoadFunc],
        container_load_funcs: Mapping[str, ContainerLoadFunc]
        ) -> Mapping[K, ArrayLike]:

        result: dict[K, ArrayLike] = {}

        if path_without_suffix.exists() and path_without_suffix.is_dir():
            # Load pre-key files
            for file in path_without_suffix.iterdir():
                if not file.is_file():
                    continue
                suffix = file.suffix.lower()
                if suffix not in prekey_load_funcs:
                    continue
                key = file.stem
                if not is_key_func(key):
                    continue
                result[key] = prekey_load_funcs[suffix](file)

        else:
            # Load container files
            for ext in container_load_funcs.keys():
                path_with_suffix = path_without_suffix.with_suffix(ext)
                if not path_with_suffix.exists():
                    continue
                if not path_with_suffix.is_file():
                    continue
                mapping = container_load_funcs[ext](path_with_suffix)
                result.update(
                    (karr, varr)
                    for karr, varr in mapping.items()
                    if is_key_func(karr)
                )

        return result

    @classmethod
    def from_zarr(
        cls, group: zarr.Group
    ) -> ZarrSLDatasetItem[Kvid, Klm, Ktgt]:

        videos = cls._load_category_from_zarr(
            cls.is_video_key,
            get_group(group, "videos")
        )
        landmarks = cls._load_category_from_zarr(
            cls.is_landmark_key,
            get_group(group, "landmarks")
        )
        targets = cls._load_category_from_zarr(
            cls.is_target_key,
            get_group(group, "targets")
        )

        return SLDatasetItem(
            videos=videos,
            landmarks=landmarks,
            targets=targets
        )

    @classmethod
    def _load_category_from_zarr[K: str](
        cls,
        is_key_func: Callable[[object], TypeGuard[K]],
        group: zarr.Group,
        ) -> Mapping[K, zarr.Array]:

        return {
            karr: get_array(group, karr)
            for karr in group.array_keys()
            if is_key_func(karr)
        }


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



