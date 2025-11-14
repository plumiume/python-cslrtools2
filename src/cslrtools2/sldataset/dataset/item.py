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

"""Dataset item classes for sign language datasets.

This module provides the SLDatasetItem class and related utilities for
representing individual samples in a sign language dataset.
"""

from __future__ import annotations

from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import zarr
if TYPE_CHECKING:
    from zarr.api.synchronous import StoreLike
else:
    StoreLike = object

from ...typings import ArrayLike, PathLike
from ..logger import sldataset_logger
from ..utils import as_tensor, get_group, get_array
from ..array_loader import (
    prekey_loaders, PrekeyLoadFunc,
    container_loaders, ContainerLoadFunc
)
from .holder import SLKeyHolder


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


######## Key Holder Class ########

# Note: SLKeyHolder is now defined in holder.py and imported above
# This section is kept for backward compatibility documentation


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
        """Save item to Zarr storage.
        
        Args:
            store_or_group: Zarr store path or group object.
            
        Returns:
            Zarr group containing the saved item.
        """

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
        """Load item from file system.
        
        Args:
            path: Path to the item directory.
            extra_prekey_load_funcs: Additional per-key file loaders.
            extra_container_load_funcs: Additional container file loaders.
            
        Returns:
            Loaded dataset item.
        """

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
        """Load category data from file system.
        
        Supports both per-key files (one file per key) and container files
        (all keys in one file).
        """

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
        """Load item from Zarr group.
        
        Args:
            group: Zarr group containing the item data.
            
        Returns:
            Loaded dataset item with Zarr arrays.
        """

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
        """Load category data from Zarr group."""

        return {
            karr: get_array(group, karr)
            for karr in group.array_keys()
            if is_key_func(karr)
        }
