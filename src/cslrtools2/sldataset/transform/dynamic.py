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

"""Data augmentation and transformation utilities for sign language datasets."""

# INFO: this module is under construction

from __future__ import annotations

from typing import (
    Sequence,
    Callable
)

import torch
import torchvision.transforms.v2 as transforms_v2

from ..dataset.core import SLDatasetItem, TensorSLDatasetItem
from .core import Transform


def return_true() -> bool:
    return True


if return_true():
    raise NotImplementedError("This module is under construction.")


### see DATA_AUGMENTATION.md
### ランドマーク用を優先して実装
### video: [N, T, C, H, W]
### landmark: [N, T, V, C]
### target: [N, S]
### dynamicでは計算コストが高い変換は避ける
### missing_valueはこのモジュールでは扱わず、
### transform/frozen.pyで補完しておくことを想定


def uniform_rand_factory(gen: torch.Generator) -> torch.Tensor:
    return torch.rand([], generator=gen)


class UniformSpeedChange[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
    def __init__(
        self,
        video_keys: Sequence[Kvid],
        landmark_keys: Sequence[Klm],
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        rand_factory: Callable[[torch.Generator], torch.Tensor] = uniform_rand_factory,
        mode: str = "nearest",
        gen: torch.Generator | None = None,
    ):

        # Validate scale parameters
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError(
                f"Scale values must be positive, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )
        if min_scale > max_scale:
            raise ValueError(
                f"min_scale must be <= max_scale, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )

        self.video_keys = video_keys
        self.landmark_keys = landmark_keys

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rand_factory = rand_factory
        self.gen = self._ensure_generator(gen)
        self.mode = mode

    def __call__(
        self,
        item: TensorSLDatasetItem[Kvid, Klm, Ktgt],
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:
        rand_val = self.rand_factory(self.gen).clamp(0, 1).item()
        scale = self.min_scale + (self.max_scale - self.min_scale) * rand_val

        videos = {**item.videos}

        for kvid in self.video_keys:

            vvid = item.videos[kvid]  # NameError?
            "with shape [N, T, C, H, W]"

            time_indices = torch.arange(0, vvid.shape[1] * scale) / scale

            vvid_ncthw = vvid.permute(0, 2, 1, 3, 4)  # [N, C, T, H, W]
            videos[kvid] = torch.nn.functional.interpolate(
                vvid_ncthw,
                size=(
                    time_indices.shape[0],  # T
                    vvid_ncthw.shape[3],    # H
                    vvid_ncthw.shape[4],    # W
                ),
                mode=self.mode,
            ).permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:

            vlm = item.landmarks[klm]  # NameError?
            "with shape [N, T, V, C]"

            time_indices = torch.arange(0, vlm.shape[1] * scale) / scale

            vlm_ncth = vlm.permute(0, 3, 1, 2)  # [N, C, T, V]
            landmarks[klm] = torch.nn.functional.interpolate(
                vlm_ncth,
                size=(
                    time_indices.shape[0],  # T
                    vlm_ncth.shape[3],      # V
                ),
                mode=self.mode,
            ).permute(0, 2, 3, 1)  # [N, T, V, C]

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)


class RandomResizePaddingCrop[
    Kvid: str, Klm: str, Ktgt: str
](Transform[Kvid, Klm, Ktgt]):
    def __init__(
        self,
        video_keys: Sequence[Kvid],
        landmark_keys: Sequence[Klm],
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        gen: torch.Generator | None = None,
    ):

        # Validate scale parameters
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError(
                f"Scale values must be positive, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )
        if min_scale > max_scale:
            raise ValueError(
                f"min_scale must be <= max_scale, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )

        self.video_keys = video_keys
        self.landmark_keys = landmark_keys

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.gen = self._ensure_generator(gen)

    def __call__(
        self,
        item: TensorSLDatasetItem[Kvid, Klm, Ktgt],
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:

        scale = (
            torch.empty(1)
            .uniform_(self.min_scale, self.max_scale, generator=self.gen)
            .item()
        )

        videos = {**item.videos}

        for kvid in self.video_keys:

            vvid = item.videos[kvid]  # NameError?
            "with shape [N, T, C, H, W]"

            # transforms_v2.functional.affine behavior:
            # - Input: [..., C, H, W] with arbitrary leading batch dimensions
            # - Output: Same shape as input (tensor size unchanged)
            # - scale > 1.0: Image zooms in → crop effect (outer regions cut off)
            # - scale < 1.0: Image zooms out → padding effect (filled with fill value)
            # - Transform is center-invariant by default
            # See: torchvision/transforms/v2/functional/_geometry.py::affine()
            vvid = transforms_v2.functional.affine(
                inpt=vvid,
                angle=0.0,
                translate=[0.0, 0.0],
                scale=scale,
                shear=[0.0, 0.0],
            )

            videos[kvid] = vvid

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:

            vlm = item.landmarks[klm]  # NameError?
            "with shape [N, T, V, C]"
            vlm_xy = vlm[..., :2]
            vlm_other = vlm[..., 2:]

            # simple impletementation
            vlm_xy = (vlm_xy - 0.5) * scale + 0.5
            landmarks[klm] = torch.cat([vlm_xy, vlm_other], dim=-1)

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)
