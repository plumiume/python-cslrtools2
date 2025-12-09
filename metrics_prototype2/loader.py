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

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
# Reason: importlib.metadata.entry_points() has complex type inference.
# The EntryPoints API varies between Python 3.9 and 3.10+, causing type
# checking issues. These suppressions are necessary for cross-version
# compatibility without sacrificing runtime behavior.

"""Plugin loader for metrics discovery (PROTOTYPE v2).

**STATUS**: This is a prototype. NOT part of main cslrtools2 package.

This module provides the plugin discovery system for metrics using
importlib.metadata Entry Points for external project integration.

Entry Point Format
------------------

External projects can register metrics in their pyproject.toml::

    [project.entry-points."cslrtools2.sldataset.metrics"]
    "completeness.nan_rate" = "your_package.metrics:NaNRateMetric"
    "temporal.smoothness" = "your_package.metrics:SmoothnessMetric"

The entry point should resolve to a Metric subclass (not an instance).

Calculator Pattern
------------------

Users create a Calculator object to manage metric computation::

    calculator = MetricCalculator()
    calculator.add_metric("completeness.nan_rate")
    results = calculator.calculate(landmarks_data)
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, TypedDict, TYPE_CHECKING

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from .base import Metric, MetricResult
else:
    Metric = type
    MetricResult = dict[str, Any]


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


def load_metrics() -> dict[str, MetricInfo]:
    """Load all registered metric plugins via importlib.metadata entry points.

    Discovers metrics registered under the "cslrtools2.sldataset.metrics" group.

    Returns:
        Dictionary mapping metric names to their information.

    Example:
        >>> metrics = load_metrics()
        >>> print(list(metrics.keys()))
        ['completeness.nan_rate', 'temporal.smoothness']
    """
    metrics: dict[str, MetricInfo] = {}

    eps = entry_points()
    # Python 3.10+ returns EntryPoints, Python 3.9 returns dict
    if hasattr(eps, "select"):
        group_eps = eps.select(group="cslrtools2.sldataset.metrics")
    else:
        # type: ignore[attr-defined]
        group_eps = eps.get("cslrtools2.sldataset.metrics", [])

    for ep in group_eps:
        full_name = ep.name
        metric_class = ep.load()

        # Parse name: "category.metric_name"
        if "." in full_name:
            category, metric_name = full_name.rsplit(".", 1)
        else:
            category = "general"
            metric_name = full_name

        # Default params can be empty or defined in class
        default_params = {}

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
        raise KeyError(f"Metric '{name}' not found. Available: {list(metrics.keys())}")
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


class MetricCalculator:
    """Calculator for managing and executing multiple metrics.

    This class implements the Calculator pattern, allowing users to:
    1. Register multiple metrics
    2. Calculate all metrics at once
    3. Collect results in a unified format

    Example:
        >>> calculator = MetricCalculator()
        >>> calculator.add_metric("completeness.nan_rate")
        >>> calculator.add_metric("temporal.smoothness", window=5)
        >>>
        >>> results = calculator.calculate(landmarks_data)
        >>> print(results["completeness.nan_rate"]["value"])
        0.05
    """

    def __init__(self) -> None:
        """Initialize an empty calculator."""
        self._metrics: dict[str, Metric] = {}

    def add_metric(self, name: str, **kwargs: Any) -> None:
        """Add a metric to the calculator.

        Args:
            name: Full metric name, e.g., "completeness.nan_rate".
            **kwargs: Parameters to pass to the metric constructor.

        Raises:
            KeyError: If the metric name is not found.

        Example:
            >>> calculator = MetricCalculator()
            >>> calculator.add_metric("completeness.nan_rate")
        """
        metric = create_metric(name, **kwargs)
        self._metrics[name] = metric

    def remove_metric(self, name: str) -> None:
        """Remove a metric from the calculator.

        Args:
            name: Full metric name.

        Raises:
            KeyError: If the metric is not registered.

        Example:
            >>> calculator.remove_metric("completeness.nan_rate")
        """
        del self._metrics[name]

    def list_metrics(self) -> list[str]:
        """List all registered metric names in the calculator.

        Returns:
            Sorted list of metric names.

        Example:
            >>> calculator.list_metrics()
            ['completeness.nan_rate', 'temporal.smoothness']
        """
        return sorted(self._metrics.keys())

    def calculate(self, data: Any, **kwargs: Any) -> dict[str, MetricResult]:
        """Calculate all registered metrics on the provided data.

        Args:
            data: Input data (type depends on metric requirements).
            **kwargs: Additional parameters passed to each metric's calculate().

        Returns:
            Dictionary mapping metric names to their results.

        Example:
            >>> results = calculator.calculate(landmarks_data)
            >>> print(results["completeness.nan_rate"]["value"])
            0.05
        """
        results: dict[str, MetricResult] = {}
        for name, metric in self._metrics.items():
            results[name] = metric.calculate(data, **kwargs)
        return results
