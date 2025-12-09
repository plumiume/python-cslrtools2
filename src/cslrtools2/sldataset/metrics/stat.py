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

from typing import Mapping, TypedDict

import numpy as np

from ...typings import (
    ArrayLikeInt, ArrayLikeFloat, ArrayLikeStr
)
from ..dataset.item import DefaultSLDatasetItem


class VideoStats(TypedDict):
    total_frames: int
    data_shape: tuple[int, ...]
    has_nan: bool


class LandmarkStats(TypedDict):
    total_frames: int
    data_shape: tuple[int, ...]
    has_nan: bool
    num_vertices: int
    num_coords: int


class TargetStats(TypedDict):
    length: int
    data_shape: tuple[int, ...]


class StatResult(TypedDict):
    videos_stats: Mapping[str, VideoStats]
    landmark_stats: Mapping[str, LandmarkStats]
    targets_stats: Mapping[str, TargetStats]


class Stats:

    def calculate(
        self,
        item: DefaultSLDatasetItem[str, str, str]
    ) -> StatResult:

        return StatResult({
            "videos_stats": {
                kvid: self.video_stats(vvid)
                for kvid, vvid in item.videos.items()
            },
            "landmark_stats": {
                klm: self.landmark_stats(vlm)
                for klm, vlm in item.landmarks.items()
            },
            "targets_stats": {
                ktgt: self.target_stats(vtgt)
                for ktgt, vtgt in item.targets.items()
            },
        })

    def video_stats(self, data: ArrayLikeInt) -> VideoStats:

        np_data = np.asarray(data)

        total_frames = np_data.shape[0]
        data_shape = np_data.shape
        has_nan = np.isnan(np_data).any().item()

        return {
            "total_frames": total_frames,
            "data_shape": data_shape,
            "has_nan": has_nan,
        }

    def landmark_stats(self, data: ArrayLikeFloat) -> LandmarkStats:

        np_data = np.asarray(data)

        total_frames = np_data.shape[0]
        data_shape = np_data.shape
        has_nan = np.isnan(np_data).any().item()
        num_vertices = np_data.shape[1]
        num_coords = np_data.shape[2]

        return {
            "total_frames": total_frames,
            "data_shape": data_shape,
            "has_nan": has_nan,
            "num_vertices": num_vertices,
            "num_coords": num_coords,
        }

    def target_stats(self, data: ArrayLikeStr) -> TargetStats:

        np_data = np.asarray(data)

        length = np_data.shape[0]
        data_shape = np_data.shape

        return {
            "length": length,
            "data_shape": data_shape,
        }
