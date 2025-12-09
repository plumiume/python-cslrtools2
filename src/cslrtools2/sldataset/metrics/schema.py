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

from typing import Any
from enum import StrEnum

from pydantic import BaseModel

from .metric import CategoryGroup


type CategoryUnion = (
    str |  # Single category name
    list[str] |  # Non-Named categories
    dict[str, list[str]]  # Named categories
)


class CategorySpec(BaseModel):
    group: CategoryGroup | None = None
    "Category group (landmarks, videos, targets)"
    values: list[CategoryUnion]
    "List of category specifications"


class MetricResolveMode(StrEnum):
    WHITE_LIST = "white-list"
    # Future


class CommonMetricSpec(BaseModel):
    targets: CategorySpec | None = None
    "targets category specification"
    tags: dict[str, Any] | None = None
    "Classification tags (e.g., quality, temporal, anatomical)"
    params: dict[str, Any] | None = None
    "Metric-specific parameters"
    extras: dict[str, Any] | None = None
    "Additional metadata for extensibility"


DEFAULT_METRIC_SPEC = CommonMetricSpec(
    targets=None,
    tags={},
    params={},
    extras={},
)


class MetricSpec(CommonMetricSpec, BaseModel):
    type: str
    "Metric type identifier (e.g., 'completeness.nan_rate')"
    metric_name: str
    "Short name of the metric"
    metric_fqn: str | None = None
    "Fully qualified name of the metric class for plugin loading"
    enabled: bool | None = None
    "Whether this metric is enabled"
    description: str | None = None
    "Short description of the metric for CLI/UI display"
    detail: str | None = None
    "Detailed description of the metric for documentation"


class CalculatorConfig(BaseModel):
    metric_resolve_mode: MetricResolveMode = MetricResolveMode.WHITE_LIST
    "Mode for resolving which metrics to execute (white-list, etc.)"


class CalculatorSpec(BaseModel):
    metrics: dict[str, MetricSpec]
    "Dictionary of metric identifiers to their specifications"
    common_metric: CommonMetricSpec
    "Common specification applied to all metrics"
    config: CalculatorConfig
    "Calculator configuration settings"


'''
```yaml

calculator:
    metrics:
        metric_a:
            type: completeness.nan_rate
            categories:
                - pose
                # pose: [pose]
                - left_hand
                # left_hand: [left_hand]
                - right_hand
                # right_hand: [right_hand]
                - hands:
                    - left_hand
                    - right_hand
                # hands: [left_hand, right_hand]
                - all:
                    - pose
                    - left_hand
                    - right_hand
                # all: [pose, left_hand, right_hand]
            params:
                subparam1: subvalue1
                subparam2: subvalue2
        metric_b: {}
    defaults:
        categories:
            - pose
```
'''
