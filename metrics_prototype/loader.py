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

"""Plugin loader for metrics discovery (PROTOTYPE).

**STATUS**: This is a prototype. NOT part of main cslrtools2 package.

This module demonstrates how the plugin discovery system would work for metrics,
using Python's Entry Points mechanism for runtime discovery.

Simulated Entry Point Format
-----------------------------

In a real implementation, plugins would be registered in pyproject.toml::

    [project.entry-points."cslrtools2.sldataset.metrics"]
    "completeness.nan_rate" = "cslrtools2.plugins.metrics.completeness:nan_rate_info"

For this prototype, we use a simulated registry instead of actual Entry Points.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Metric
else:
    Metric = type


class MetricInfo(TypedDict):
    """Information about a registered metric plugin.

    Attributes:
        name: Full metric name, e.g., "completeness.nan_rate".
        category: Metric category, e.g., "completeness", "temporal".
        metric_name: Short name without category, e.g., "nan_rate".
        metric_class: The metric class itself.
        default_params: Default parameters for the metric.
    """

    name: str
    category: str
    metric_name: str
    metric_class: type[Metric]
    default_params: Mapping[str, Any]


# Simulated plugin registry (replaces Entry Points for prototype)
_SIMULATED_PLUGINS: dict[str, tuple[type[Metric], dict[str, Any]]] = {}  # pyright: ignore[reportUnknownVariableType] # noqa: E501


def register_metric(
    name: str, metric_class: type[Metric], default_params: dict[str, Any] | None = None
) -> None:
    """Register a metric plugin (prototype simulation of Entry Points).

    In the real implementation, this would be done via pyproject.toml Entry Points.

    Args:
        name: Full metric name, e.g., "completeness.nan_rate".
        metric_class: The metric class.
        default_params: Default parameters.

    Example:
        >>> from metrics_prototype.loader import register_metric
        >>> from metrics_prototype.plugins.completeness import NaNRateMetric
        >>> 
        >>> register_metric("completeness.nan_rate", NaNRateMetric, {})
    """
    _SIMULATED_PLUGINS[name] = (metric_class, default_params or {})


def load_metrics() -> dict[str, MetricInfo]:
    """Load all registered metric plugins.

    In prototype mode, this loads from the simulated registry.
    In real implementation, this would use importlib.metadata.entry_points().

    Returns:
        Dictionary mapping metric names to their information.

    Example:
        >>> metrics = load_metrics()
        >>> print(list(metrics.keys()))
        ['completeness.nan_rate']
    """
    metrics: dict[str, MetricInfo] = {}

    for full_name, (metric_class, default_params) in _SIMULATED_PLUGINS.items():
        # Parse name: "category.metric_name"
        if "." in full_name:
            category, metric_name = full_name.rsplit(".", 1)
        else:
            category = "general"
            metric_name = full_name

        metrics[full_name] = MetricInfo(
            name=full_name,
            category=category,
            metric_name=metric_name,
            metric_class=metric_class,
            default_params=default_params,
        )

    return metrics


def list_metric_names() -> list[str]:
    """List names of all available metric plugins.

    Returns:
        Sorted list of metric names.

    Example:
        >>> print(list_metric_names())
        ['completeness.nan_rate']
    """
    return sorted(load_metrics().keys())


def get_metric_info(name: str) -> MetricInfo:
    """Get information about a specific metric by name.

    Args:
        name: Full metric name, e.g., "completeness.nan_rate".

    Returns:
        Information about the requested metric.

    Raises:
        KeyError: If the metric name is not found.

    Example:
        >>> info = get_metric_info("completeness.nan_rate")
        >>> print(info["category"])
        completeness
    """
    metrics = load_metrics()
    if name not in metrics:
        raise KeyError(
            f"Metric '{name}' not found. Available: {list(metrics.keys())}"
        )
    return metrics[name]


def create_metric(name: str, **kwargs: Any) -> Metric:
    """Create an instance of a metric by name.

    Args:
        name: Full metric name.
        **kwargs: Override default parameters for the metric.

    Returns:
        Instantiated metric object.

    Raises:
        KeyError: If the metric name is not found.

    Example:
        >>> metric = create_metric("completeness.nan_rate")
        >>> result = metric.calculate(data)
    """
    info = get_metric_info(name)
    params = {**info["default_params"], **kwargs}
    return info["metric_class"](**params)
