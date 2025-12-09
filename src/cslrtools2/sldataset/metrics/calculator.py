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

# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from typing import Mapping, Self
from dataclasses import dataclass

import numpy as np

from ...typings import ArrayLike
from ..dataset.item import DefaultSLDatasetItem
from ..logger import sldataset_logger
from .schema import (
    CategorySpec,
    CommonMetricSpec, MetricSpec, DEFAULT_METRIC_SPEC,
    CalculatorConfig,
    MetricResolveMode
)
from .stat import Stats
from .metric import Metric, CategoryGroup, MetricResult
from .metric import CachedSLDatasetItem
from .plugin_util import metric_alias_registry

# Type alias for calculation result (list of MetricResult)
CalculationResult = list[MetricResult]

# TODO: Determine specific usage for CATEGORY_GROUP_CONFIG (e.g., cache control)
CATEGORY_GROUP_CONFIG = {
    "videos": {
        "cache": False
    },
    "landmarks": {
        "cache": True
    },
    "targets": {
        "cache": True
    }
}


@dataclass
class MetricEntry:
    metric_spec: MetricSpec
    metric_instance: Metric
    category_group: CategoryGroup | None
    category_values: Mapping[str, list[str]]


class Calculator:

    def __init__(
        self,
        metrics: Mapping[str, MetricEntry],  # metric-ident: MetricEntry instance
    ):

        self._metrics = metrics
        self._stats = Stats()

    @classmethod
    def from_config(
        cls,
        metric_specs: Mapping[str, MetricSpec],
        common_spec: CommonMetricSpec,
        config: CalculatorConfig
    ) -> Self:
        # 継承: CommonMetricSpec -> MetricSpec
        # 参照: MetricSpec -> CommonMetricSpec (-> Error)
        metrics: dict[str, MetricEntry] = {}

        for metric_ident, metric_spec in metric_specs.items():

            merged_spec = MetricSpec(
                **DEFAULT_METRIC_SPEC.model_dump(),
                **{
                    k: v for k, v in common_spec.model_dump().items()
                    if v is not None
                },
                **{
                    k: v for k, v in metric_spec.model_dump().items()
                    if v is not None
                },
            )

            if merged_spec.targets is None:
                continue
            if not merged_spec.enabled:
                continue

            metric_type = cls._get_metric_type(
                merged_spec.type,
                config
            )

            metric = metric_type(merged_spec)

            category_group, category_values = cls._normalize_target(
                merged_spec.targets
            )

            metrics[metric_ident] = MetricEntry(
                metric_spec=merged_spec,
                metric_instance=metric,
                category_group=category_group,
                category_values=category_values
            )

        return cls(metrics)

    @classmethod
    def _get_metric_type(
        cls,
        metric_type: str,
        config: CalculatorConfig
    ) -> type[Metric]:

        metric_cls = metric_alias_registry.get(
            metric_type, None
        )

        if metric_cls is not None:
            return metric_cls

        if config.metric_resolve_mode == MetricResolveMode.WHITE_LIST:
            raise KeyError(
                f"Metric type '{metric_type}' not found in white-list. "
                f"Available metrics: {list(metric_alias_registry.keys())}"
            )

        # TODO: Implement other resolution modes (e.g., FQCN, plugin discovery)
        sldataset_logger.warning(
            f"Metric type resolution mode '{config.metric_resolve_mode}' "
            f"not implemented. Falling back to WHITE_LIST mode for "
            f"metric '{metric_type}'."
        )
        raise NotImplementedError(
            f"Metric type resolution not implemented for mode "
            f"'{config.metric_resolve_mode}'. Metric: '{metric_type}'"
        )

    @classmethod
    def _normalize_target(
        cls,
        targets: CategorySpec
    ) -> tuple[
        CategoryGroup | None,
        Mapping[str, list[str]]
    ]:

        # 簡易実装

        category_group = targets.group
        named_categories: dict[str, list[str]] = {}

        for union in targets.values:

            if isinstance(union, str):
                # Single category name: "pose" -> {"pose": ["pose"]}
                union = {union: [union]}
            elif isinstance(union, list):
                # Non-Named categories:
                # ["left_hand", "right_hand"] -> {"left_hand--right_hand": [...]}
                union = {'--'.join(str(u) for u in union): union}
            elif not isinstance(union, dict):
                raise ValueError(
                    f"Invalid category union type: {type(union).__name__}. "
                    f"Expected str, list, or dict."
                )

            named_categories.update(union)

        return category_group, named_categories

    def calculate(
        self,
        idx: int,
        item: DefaultSLDatasetItem[str, str, str]
    ) -> CalculationResult:

        cached_item = CachedSLDatasetItem[
            str, str, str
        ].from_item(item)
        # TODO: Determine cache strategy based on CATEGORY_GROUP_CONFIG
        resolved_data: dict[str, np.ndarray] = {}
        stat_result = self._stats.calculate(cached_item)

        results: list[MetricResult] = []
        for ident, entry in self._metrics.items():

            if entry.category_group is None:
                sldataset_logger.warning(
                    f"Skipping metric '{ident}': category_group is None. "
                    f"Check 'targets.group' in metric configuration."
                )
                continue

            data = self._prepare_data(
                entry.category_values,
                cached_item.get_map_from_category_group(
                    entry.category_group
                ),
                resolved_data if CATEGORY_GROUP_CONFIG.get(
                    entry.category_group, {}
                ).get("cache", False) else None
            )

            if not data:
                sldataset_logger.debug(
                    f"No data available for metric '{ident}' "
                    f"(category_group: {entry.category_group}). Skipping calculation."
                )
                continue

            metadata = entry.metric_instance.get_metadata()

            for categories_name, array in data.items():

                try:
                    metric_values = entry.metric_instance.calculate(
                        entry.category_group,
                        stat_result,
                        array
                    )

                except Exception as e:
                    sldataset_logger.error(
                        f"Metric calculation failed: metric='{ident}', "
                        f"category='{categories_name}', error={type(e).__name__}: {e}",
                        exc_info=True
                    )
                    metric_values = {
                        "Exception": str(e),
                        "ExceptionType": type(e).__name__
                    }

                results.append(
                    MetricResult({
                        "metric_ident": ident,
                        "metric_name": entry.metric_spec.metric_name,
                        "metric_fqn": entry.metric_spec.metric_fqn or "N/A",
                        "sample_idx": idx,
                        "categories_name": categories_name,
                        "categories_values": entry.category_values[categories_name],
                        "values": metric_values,
                        "metadata": metadata,
                    })
                )

        return results

    def _prepare_data(
        self,
        category_values: Mapping[str, list[str]],
        array_map: Mapping[str, ArrayLike],
        resolved_data: dict[str, np.ndarray] | None
        # enable_cache -> dict, else -> None
    ) -> dict[str, np.ndarray]:
        """Prepare data by concatenating categories.

        Args:
            category_values: Mapping of category names to their constituent keys.
            array_map: Mapping of raw array keys to arrays.
            resolved_data: Optional cache for resolved data. If provided,
                          results are cached and reused.

        Returns:
            Mapping of category names to concatenated arrays.
        """
        data_map: dict[str, np.ndarray] = {}

        for name, values in category_values.items():
            # Check cache first
            if resolved_data is not None and name in resolved_data:
                data_map[name] = resolved_data[name]
                continue

            catted_data = np.concatenate([
                np.asarray(array_map[value])
                for value in values
            ])

            if resolved_data is not None:
                resolved_data[name] = catted_data

            data_map[name] = catted_data

        return data_map
