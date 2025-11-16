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

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.interpolate import (  # pyright: ignore[reportMissingTypeStubs]
    CubicSpline,
)
import torch

from ...typings import ArrayLike
from ..dataset.core import SLDatasetItem, TensorSLDatasetItem
from .core import Transform

# ランドマーク用

BoundaryConditionLiteral = Literal[
    "not-a-knot",
    "periodic",
    "clamped",
    "natural",
]
BoundaryConditionTuple = (
    tuple[BoundaryConditionLiteral, BoundaryConditionLiteral]
    | tuple[Literal[1, 2], ArrayLike]
)
BoundaryConditionType = BoundaryConditionLiteral | BoundaryConditionTuple

ExtrapolationType = Literal["periodic"] | bool | None


class CubicSplineInterpolateMissingLandmarks[Kvid: str, Klm: str, Ktgt: str](
    Transform[Kvid, Klm, Ktgt]
):
    def __init__(
        self,
        landmark_keys: Sequence[Klm],
        bc_type: BoundaryConditionType = "not-a-knot",
        extrapolation: ExtrapolationType = None,
    ):
        self.landmark_keys = landmark_keys
        self.bc_type = bc_type
        self.extrapolation = extrapolation

    def __call__(
        self, item: TensorSLDatasetItem[Kvid, Klm, Ktgt]
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:
        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:
            vlm = landmarks[klm]  # [N, T, V, C]
            n, t, v, c = vlm.shape

            store: list[np.ndarray] = []

            for array in vlm.reshape(n, t, v * c).unbind():
                x = np.arange(t)
                y = array.detach().cpu().numpy()
                is_not_nan = ~np.isnan(y)

                cs = CubicSpline(
                    x[is_not_nan],
                    y[is_not_nan],
                    bc_type=self.bc_type,  # pyright: ignore[reportArgumentType]
                    extrapolate=self.extrapolation,
                )
                store.append(cs(x))

            landmarks[klm] = torch.tensor(
                np.stack(store).reshape(n, t, v, c), dtype=vlm.dtype, device=vlm.device
            )

        return SLDatasetItem(
            videos=item.videos, landmarks=landmarks, targets=item.targets
        )
