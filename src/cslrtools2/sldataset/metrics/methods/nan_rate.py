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


class NaNRateMetric(Metric):

    def get_cli_description(self) -> str:
        return "Calculates the rate of NaN values in the dataset."

    def get_cli_detail(self) -> str:
        return (
            "The NaN Rate Metric computes the proportion of"
            "NaN (Not a Number) values present in the dataset. "
            "This metric is useful "
            "for assessing data quality and completeness, "
            "as a high NaN rate may indicate issues "
            "with data collection or preprocessing."
        )

    def calculate(
        self,
        category_group: CategoryGroup,  # maybe "landmarks"
        stats: StatResult,
        data: np.ndarray  # with shape [T, V, C]
    ) -> Mapping[str, Any]:

        frame_has_nan: np.ndarray = np.any(
            np.isnan(data),
            axis=(1, 2)
        )

        nan_rate = np.mean(frame_has_nan)
        frames_with_nan = np.sum(frame_has_nan)

        return {
            "nan_rate": nan_rate,
            "frames_with_nan": frames_with_nan
        }
