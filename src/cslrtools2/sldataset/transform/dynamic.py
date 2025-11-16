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


def return_true() -> bool:
    return True


if return_true():
    raise NotImplementedError("This module is under construction.")

from typing import *  # pyright: ignore[reportWildcardImportFromLibrary]

import torch

from ..dataset.core import SLDatasetItem, TensorSLDatasetItem
from .core import Transform

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
            vvid = item.videos[kvid]

            time_indices = torch.arange(0, vvid.shape[1] * scale) / scale

            videos[kvid] = torch.nn.functional.interpolate(
                vvid.permute(0, 2, 1, 3, 4),  # NCTHW
                size=(
                    time_indices.shape[0],
                    vvid.shape[2],
                    vvid.shape[3],
                    vvid.shape[4],
                ),
                mode=self.mode,
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)  # Back to NCTHW

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:
            vlm = item.landmarks[klm]

            time_indices = torch.arange(0, vlm.shape[1] * scale) / scale

            landmarks[klm] = torch.nn.functional.interpolate(
                vlm.permute(0, 2, 1, 3),  # NTVC
                size=(time_indices.shape[0], vlm.shape[2]),
                mode=self.mode,
                align_corners=False,
            ).permute(0, 2, 1, 3)  # Back to NTVC

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)


class RondomResizeCrop[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
    def __init__(
        self,
        video_keys: Sequence[Kvid],
        landmark_keys: Sequence[Klm],
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        gen: torch.Generator | None = None,
    ):
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
            vvid = item.videos[kvid]

            t_new = int(vvid.shape[1] * scale)
            start_idx = torch.randint(
                0, vvid.shape[1] - t_new + 1, (1,), generator=self.gen
            ).item()

            videos[kvid] = vvid[:, start_idx : start_idx + t_new]

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:
            vlm = item.landmarks[klm]

            t_new = int(vlm.shape[1] * scale)
            start_idx = torch.randint(
                0, vlm.shape[1] - t_new + 1, (1,), generator=self.gen
            ).item()

            landmarks[klm] = vlm[:, start_idx : start_idx + t_new]

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)
