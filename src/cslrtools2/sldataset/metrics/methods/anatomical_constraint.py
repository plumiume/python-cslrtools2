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

# NOTE: this file is under development and may be subject to changes.

from __future__ import annotations

from typing import Any, Sequence, Mapping

import numpy as np

from ..stat import StatResult
from ..metric import CategoryGroup, Metric


class AnatomicalConstraintMetric(Metric):

    def __init__(
        self,
        edges: Sequence[tuple[int, int]]
    ):

        self.edges = edges
        self.start_indices = np.array([e[0] for e in edges])
        self.end_indices = np.array([e[1] for e in edges])

    def get_cli_description(self) -> str:
        return "Evaluates anatomical constraints of landmark data."

    def get_cli_detail(self) -> str:
        return (
            "The Anatomical Constraint Metric assesses how well "
            "the landmark data adheres to predefined anatomical "
            "constraints. It checks for implausible positions "
            "or movements of landmarks based on known anatomical "
            "relationships, providing insights into the validity "
            "and reliability of the landmark data."
        )

    def calculate(
        self,
        category_group: CategoryGroup,  # should be "landmarks"
        stats: StatResult,
        data: np.ndarray  # with shape [T, V, C]
    ) -> Mapping[str, Any]:

        bone_vectors = (
            data[:, self.end_indices, :]
            - data[:, self.start_indices, :]
        )  # with shape [T, E, C]

        bone_lengths = np.linalg.norm(
            bone_vectors, axis=-1
        )  # with shape [T, E]

        valid_lengths = bone_lengths[~np.isnan(bone_lengths)]

        # with shape [T]
        mean_lengths = np.nanmean(bone_lengths, axis=-1)
        std_lengths = np.nanstd(bone_lengths, axis=-1)
        coef_of_var = (
            std_lengths / mean_lengths
            if mean_lengths.size > 0 else
            np.array([np.nan] * bone_lengths.shape[0])
        )

        mean_cv = np.nanmean(coef_of_var)

        return {
            "mean_bone_length": np.nanmean(valid_lengths),
            "std_bone_length": np.nanstd(valid_lengths),
            "mean_coefficient_of_variation": mean_cv
        }
