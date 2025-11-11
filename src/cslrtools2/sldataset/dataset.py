# (root) / metadata / {km: Any}
# (root) / connections / {klm}.{klm}
# (root) / items / {idx} / videos / {kvid}
# (root) / items / {idx} / landmarks /{klm}
# (root) / items / {idx} / targets / {ktgt}

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
from zarr.api.synchronous import StoreLike

from ..typings import ArrayLike, PathLike
from .utils import as_tensor, get_group, get_array
from .array_loader import (
    prekey_loaders, PrekeyLoadFunc,
    container_loaders, ContainerLoadFunc
)

######## cp314 Ready ########
type SLDatasetItem[ # pyright: ignore[reportRedeclaration]
    Kvid: str, Vvid: ArrayLike,
    Klm: str, Vlm: ArrayLike,
    Ktgt: str, Vtgt: ArrayLike
] = """SLDatasetItem[
    Kvid, Vvid,
    Klm, Vlm,
    Ktgt, Vtgt
]"""

type SLDataset[ # pyright: ignore[reportRedeclaration]
    Kmeta: str, Kvid: str, Klm: str, Ktgt: str, Vconn: ArrayLike
] = """SLDataset[
    Kmeta, Kvid, Klm, Ktgt, Vconn
]"""

type IterableSLDataset[ # pyright: ignore[reportRedeclaration]
    Kmeta: str, Kvid: str, Klm: str, Ktgt: str, Vconn: ArrayLike
] = """IterableSLDataset[
    Kmeta, Kvid, Klm, Ktgt, Vconn
]"""

######## Type Aliases ########

type DefaultSLDatasetItem[
    Kvid: str, Klm: str, Ktgt: str
] = SLDatasetItem[
    Kvid, Any,
    Klm, Any,
    Ktgt, Any
]

type TensorSLDatasetItem[
    Kvid: str, Klm: str, Ktgt: str
] = SLDatasetItem[
    Kvid, Tensor,
    Klm, Tensor,
    Ktgt, Tensor
]

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

    @classmethod
    def is_metadata_key(cls, obj: object) -> TypeGuard[Kmeta]:
        return isinstance(obj, str)

    @classmethod
    def is_video_key(cls, obj: object) -> TypeGuard[Kvid]:
        return isinstance(obj, str)
    
    @classmethod
    def is_landmark_key(cls, obj: object) -> TypeGuard[Klm]:
        return isinstance(obj, str)
    
    @classmethod
    def is_target_key(cls, obj: object) -> TypeGuard[Ktgt]:
        return isinstance(obj, str)

######## Item Class ########

class SLDatasetItem[
    Kvid: str, Vvid: ArrayLike,
    Klm: str, Vlm: ArrayLike,
    Ktgt: str, Vtgt: ArrayLike
    ](SLKeyHolder[Never, Kvid, Klm, Ktgt]):

    def __init__(
        self,
        videos: Mapping[Kvid, Vvid], # (Kvid, [N, T, H, W, C])
        landmarks: Mapping[Klm, Vlm], # (Klm, [N, T, V, A])
        targets: Mapping[Ktgt, Vtgt], # (Ktgt, [N, S])
        ): # N = 1 or Batch size

        self.videos = videos
        self.landmarks = landmarks
        self.targets = targets

    def to(self, device: torch.device) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:

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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DefaultSLDatasetItem[Kvid, Klm, Ktgt]:
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

    def __iter__(self) -> Iterator[DefaultSLDatasetItem[Kvid, Klm, Ktgt]]:
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



