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

from typing import Any, Mapping

import numpy as np

from ..stat import StatResult
from ..metric import CategoryGroup, Metric


class TemporalConsistencyMetric(Metric):

    def get_cli_description(self) -> str:
        return "Calculates the temporal consistency of landmark data."

    def get_cli_detail(self) -> str:
        return (
            "The Temporal Consistency Metric evaluates how consistent "
            "landmark positions are over time in a sequence of frames. "
            "It computes the average displacement of landmarks between "
            "consecutive frames, providing insight into the stability "
            "and reliability of the landmark data across the temporal dimension."
        )

    def calculate(
        self,
        category_group: CategoryGroup,  # should be "landmarks"
        stats: StatResult,
        data: np.ndarray  # with shape [T, V, C]
    ) -> Mapping[str, Any]:

        # Calculate velocity and acceleration
        velocity = np.diff(data, axis=0)
        acceleration = np.diff(velocity, axis=0)

        # Calculate metrics
        smoothness = np.nanstd(acceleration)

        velocity_norm: np.ndarray = np.linalg.norm(velocity, axis=-1)
        velocity_mean = np.nanmean(velocity_norm)
        velocity_std = np.nanstd(velocity_norm)

        acceleration_norm: np.ndarray = np.linalg.norm(acceleration, axis=-1)
        acceleration_mean = np.nanmean(acceleration_norm)
        acceleration_std = np.nanstd(acceleration_norm)

        return {
            "smoothness": smoothness,
            "velocity_mean": velocity_mean,
            "velocity_std": velocity_std,
            "acceleration_mean": acceleration_mean,
            "acceleration_std": acceleration_std
        }
