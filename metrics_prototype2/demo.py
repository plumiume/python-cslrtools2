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

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# Reason: SLDataset generic type parameters are inferred dynamically from zarr.
# Mapping.keys() returns KeysView with complex type inference.

"""Demo script for metrics prototype v2 using SLDataset integration.

**STATUS**: PROTOTYPE v2 - Demonstrates SLDataset integration patterns

This script shows improved integration with cslrtools2.sldataset:
    1. Using SLDataset.from_zarr() for data loading
    2. Leveraging zarr lazy loading (no explicit iteration needed)
    3. Using get_group()/get_array() utilities from sldataset
    4. Multi-part metrics calculation

Usage
-----

Calculate metrics for specific sample::

    $ uv run python -m metrics_prototype2.demo --dataset \\
        C:\\Users\\ikeda\\Downloads\\fs50-lmpipe-v5.2.1.zarr --sample-idx 0

Calculate metrics for all samples::

    $ uv run python -m metrics_prototype2.demo --dataset \\
        C:\\Users\\ikeda\\Downloads\\fs50-lmpipe-v5.2.1.zarr --all
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# Import sldataset
from cslrtools2.sldataset.dataset.core import SLDataset

# Import prototype modules
from metrics_prototype2 import create_metric
from metrics_prototype2.plugins.anatomical import MEDIAPIPE_POSE_BONES
from metrics_prototype2.utils import (
    categorize_landmarks,
    combine_landmarks,
)


def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def calculate_metrics_for_sample(
    item: Any,
    sample_idx: int,
) -> dict[str, Any]:
    """Calculate all metrics for a single sample.

    Args:
        item: SLDatasetItem (with zarr.Array references)
        sample_idx: Sample index for display

    Returns:
        Dictionary containing all metric results
    """
    print(f"\n[Sample {sample_idx}] Calculating metrics...")

    # Convert zarr.Array references to NumPy arrays using __array__ protocol
    landmarks_np = {
        k: np.asarray(v, dtype=np.float32) for k, v in item.landmarks.items()
    }

    # Categorize landmarks
    categories = categorize_landmarks(landmarks_np.keys())
    print("  Found categories:", list(categories.keys()))

    results: dict[str, Any] = {
        "sample_idx": sample_idx,
        "categories": categories,
        "metrics": {},
    }

    # Calculate NaN rate for each part
    nan_rate_metric = create_metric("completeness.nan_rate")

    for category, keys in categories.items():
        if not keys:
            continue

        # Use first key for single-part categories
        if len(keys) == 1:
            array = landmarks_np[keys[0]]
            result = nan_rate_metric.calculate(array)
            results["metrics"][category] = {
                "nan_rate": result["values"]["nan_rate"],
                "frames_with_nan": result["metadata"]["frames_with_nan"],
                "total_frames": result["metadata"]["total_frames"],
            }

    # Calculate combined "Hands" and "All" metrics
    if "Left Hand" in categories and "Right Hand" in categories:
        hands_keys = categories["Left Hand"] + categories["Right Hand"]
        hands_combined = combine_landmarks(landmarks_np, hands_keys, axis=1)
        result = nan_rate_metric.calculate(hands_combined)
        results["metrics"]["Hands"] = {
            "nan_rate": result["values"]["nan_rate"],
            "frames_with_nan": result["metadata"]["frames_with_nan"],
            "total_frames": result["metadata"]["total_frames"],
        }

    # Calculate "All" parts combined
    all_keys = list(landmarks_np.keys())
    all_combined = combine_landmarks(landmarks_np, all_keys, axis=1)
    result = nan_rate_metric.calculate(all_combined)
    results["metrics"]["All"] = {
        "nan_rate": result["values"]["nan_rate"],
        "frames_with_nan": result["metadata"]["frames_with_nan"],
        "total_frames": result["metadata"]["total_frames"],
    }

    # Calculate temporal consistency (using "All" combined)
    temporal_metric = create_metric("temporal.consistency")
    try:
        result = temporal_metric.calculate(all_combined)
        results["metrics"]["temporal_consistency"] = {
            "mean_velocity": result["values"]["mean_velocity"],
            "smoothness": result["values"]["smoothness"],
        }
    except ValueError as e:
        results["metrics"]["temporal_consistency"] = {"error": str(e)}

    # Calculate anatomical constraint (using Pose if available)
    if "Pose" in categories:
        pose_key = categories["Pose"][0]
        pose_array = landmarks_np[pose_key]
        anatomical_metric = create_metric("anatomical.bone_length")
        try:
            result = anatomical_metric.calculate(
                pose_array, bone_pairs=MEDIAPIPE_POSE_BONES
            )
            results["metrics"]["anatomical_constraint"] = {
                "mean_variation": result["values"]["mean_variation"],
                "std_variation": result["values"]["std_variation"],
            }
        except ValueError as e:
            results["metrics"]["anatomical_constraint"] = {"error": str(e)}

    # Display results
    print("\n  Completeness (NaN Rate):")
    for part, metrics in results["metrics"].items():
        if "nan_rate" in metrics:
            print(
                f"    {part}: {metrics['nan_rate']:.4f} "
                f"({metrics['frames_with_nan']}/{metrics['total_frames']} frames)"
            )

    if "temporal_consistency" in results["metrics"]:
        tc = results["metrics"]["temporal_consistency"]
        if "error" not in tc:
            print("\n  Temporal Consistency:")
            print(f"    Mean velocity: {tc['mean_velocity']:.6f}")
            print(f"    Smoothness: {tc['smoothness']:.6f}")

    if "anatomical_constraint" in results["metrics"]:
        ac = results["metrics"]["anatomical_constraint"]
        if "error" not in ac:
            print("\n  Anatomical Constraint:")
            print(f"    Mean variation: {ac['mean_variation']:.6f}")
            print(f"    Std variation: {ac['std_variation']:.6f}")

    return results


def calculate_all_samples(
    dataset: SLDataset[Any, Any, Any, Any, Any],
) -> list[dict[str, Any]]:
    """Calculate metrics for all samples in dataset.

    Args:
        dataset: SLDataset instance

    Returns:
        List of result dictionaries
    """
    print(f"\n[Batch Mode] Calculating metrics for {len(dataset)} samples...")

    all_results: list[dict[str, Any]] = []

    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(dataset)} samples...")

        item = dataset[i]

        results = calculate_metrics_for_sample(item, i)
        all_results.append(results)

    return all_results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Metrics Prototype v2 Demo (SLDataset Integration)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to zarr dataset",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help=(
            "Sample index to calculate metrics for "
            "(default: 0 if --all not specified)"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Calculate metrics for all samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (optional)",
    )

    args = parser.parse_args()

    print_header("Metrics Prototype v2 Demo")

    # Load dataset
    print(f"\n[2] Loading dataset: {args.dataset}")

    # Performance optimization: Check zarr.json existence for zarr detection
    dataset_path = Path(args.dataset)
    zarr_json_path = dataset_path / ".zgroup"
    if not zarr_json_path.exists():
        # Try .zarray as fallback
        zarr_json_path = dataset_path / ".zarray"

    if zarr_json_path.exists():
        # Confirmed zarr format
        root = zarr.open_group(str(dataset_path), mode="r")
    else:
        raise ValueError(
            f"Not a valid zarr dataset: {dataset_path}\n"
            f"Expected .zgroup or .zarray file in dataset root."
        )

    # Handle nested zarr structure (*.zarr/*.zarr)
    root_keys = list(root.keys())
    if len(root_keys) == 1 and "metadata" not in root_keys:
        inner_key = root_keys[0]
        print(f"  Nested zarr detected, using inner group: {inner_key}")
        inner = root[inner_key]
        if not isinstance(inner, zarr.Group):
            raise ValueError(f"Expected Group, got {type(inner)}")
        root = inner

    dataset = SLDataset.from_zarr(root)
    print(f"  ✓ Loaded {len(dataset)} samples")

    # Calculate metrics
    if args.all:
        results = calculate_all_samples(dataset)
        print(f"\n✓ Calculated metrics for {len(results)} samples")
    else:
        sample_idx = args.sample_idx if args.sample_idx is not None else 0
        if sample_idx >= len(dataset):
            print(f"  ✗ Sample index {sample_idx} out of range (0-{len(dataset)-1})")
            return 1

        results = [calculate_metrics_for_sample(dataset[sample_idx], sample_idx)]

    # Save results if output specified
    if args.output:
        print(f"\n[3] Saving results to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved {len(results)} result(s)")

    print_header("Demo completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
