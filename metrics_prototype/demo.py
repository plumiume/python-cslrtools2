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

"""Demo script for metrics prototype plugin system.

**STATUS**: PROTOTYPE demonstration

This script shows how the metrics plugin system would work in practice,
demonstrating:
    1. Plugin registration (simulating Entry Points)
    2. Metric discovery and listing
    3. Metric instantiation and calculation
    4. Result interpretation

Run this script to see the plugin system in action.

Usage
-----

From the project root::

    $ uv run python metrics_prototype/demo.py

    # With real dataset (sldataset2 format):
    $ uv run python metrics_prototype/demo.py --dataset \\
        H:\\SLRDataset\\fs50-mp-holistic-v5\\090\\P2_S090_00.mp4

Expected Output::

    ============================================================
    Metrics Prototype Demo
    ============================================================

    [1] Registering metrics plugins...
    ✓ Registered: completeness.nan_rate

    [2] Discovering available metrics...
    Found 1 metric(s):
      - completeness.nan_rate (category: completeness)

    [3] Creating test data...
    Generated landmarks: shape=(100, 33, 3), dtype=float32
    Injected 10% NaN values (frames 10-19)

    [4] Calculating NaN rate metric...
    ✓ Metric: nan_rate
    ✓ NaN rate: 10.00%
    ✓ Total values: 9900
    ✓ Missing values: 990

    [5] Interpreting results...
    Status: ✓ PASS - Data quality is acceptable (< 20% missing)

    ============================================================
    Demo completed successfully!
    ============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

# Import prototype modules
from metrics_prototype import register_metric, list_metric_names, create_metric
from metrics_prototype.plugins.completeness import NaNRateMetric
from metrics_prototype.plugins.temporal import TemporalConsistencyMetric
from metrics_prototype.plugins.anatomical import (
    AnatomicalConstraintMetric,
    MEDIAPIPE_POSE_BONES,
)


def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)


def print_step(step_num: int, description: str) -> None:
    """Print step header."""
    print(f"\n[{step_num}] {description}")


def find_item_dirs(root: Path) -> list[Path]:
    """Find all item directories under the root that contain landmarks/.

    Searches for pattern: root/**/{item_dir}/landmarks/

    Args:
        root: Root directory to search from

    Returns:
        List of item directories (parent of landmarks/)
    """
    item_dirs: list[Path] = []
    for landmarks_dir in root.rglob("landmarks"):
        if landmarks_dir.is_dir():
            item_dirs.append(landmarks_dir.parent)
    return item_dirs


def load_landmark_data(
    sample_dir: Path, part: str = "pose", ext: str = ".npy"
) -> np.ndarray | None:
    """Load landmark data from sldataset2 format directory.

    Expected structure:
        (root)/**/
            {item_dir}/
                landmarks/
                    *.{part}{ext}  # e.g., mediapipe.pose.npy

    Examples:
        H:\\SLRDataset\\fs50-mp-holistic-v5\\090\\P2_S090_00.mp4\\landmarks\\
            mediapipe.pose.npy
            mediapipe.left_hand.npy
            mediapipe.right_hand.npy

    Args:
        sample_dir: Path to item directory (parent of landmarks/)
        part: Landmark part name (e.g., "pose", "left_hand", "right_hand")
        ext: File extension (default: ".npy")

    Returns:
        Landmark array with shape (T, K, D) or None if not found
    """
    landmarks_dir = sample_dir / "landmarks"
    if not landmarks_dir.exists():
        print(f"  ✗ Landmarks directory not found: {landmarks_dir}")
        return None

    # Search for files matching pattern: *.{part}{ext}
    pattern = f"*.{part}{ext}"
    matching_files = list(landmarks_dir.glob(pattern))

    if not matching_files:
        print(f"  ✗ No files matching pattern '{pattern}' in {landmarks_dir}")
        return None

    # Use the first matching file
    landmark_file = matching_files[0]

    try:
        landmarks = np.load(landmark_file)
        print(f"  ✓ Loaded landmarks: {landmark_file.name}")
        print(f"    Path: {landmark_file}")
        print(f"    Shape: {landmarks.shape}, dtype: {landmarks.dtype}")
        return landmarks.astype(np.float32)
    except Exception as e:
        print(f"  ✗ Failed to load landmarks: {e}")
        return None


def load_all_landmarks_from_zarr(
    zarr_path: Path, item_index: int = 0
) -> dict[str, np.ndarray] | None:
    """Load all available landmark parts from sldataset2 zarr format.

    Expected structure:
        dataset.zarr/
            metadata/           # Dataset-level information
            connections/        # Landmark connectivity graphs
            items/              # Individual samples
                [0]/
                    landmarks/
                        {landmark_key}  # zarr array
                [1]/
                ...

    Args:
        zarr_path: Path to zarr dataset
        item_index: Index of item to load (default: 0)

    Returns:
        Dictionary of landmark arrays {part_name: array} or None if failed
    """
    try:
        # Handle nested zarr structure (*.zarr/*.zarr)
        actual_zarr_path = zarr_path
        if zarr_path.is_dir():
            # Check for nested .zarr file
            nested_zarr = list(zarr_path.glob("*.zarr"))
            if nested_zarr:
                actual_zarr_path = nested_zarr[0]
                print(f"  → Found nested zarr: {actual_zarr_path.name}")

        root = zarr.open_group(str(actual_zarr_path), mode="r")
        print(f"  ✓ Opened zarr dataset: {actual_zarr_path}")

        # Navigate to items group
        if "items" not in root:
            print("  ✗ 'items' group not found in zarr root")
            return None

        items_group = root["items"]  # pyright: ignore[reportIndexIssue]
        if not isinstance(items_group, zarr.Group):
            print("  ✗ 'items' is not a zarr Group")
            return None

        print(f"  ✓ Found {len(list(items_group.keys()))} items")

        # Get specific item
        item_key = str(item_index)
        if item_key not in items_group:
            print(f"  ✗ Item [{item_index}] not found")
            return None

        item_group = items_group[item_key]  # pyright: ignore[reportIndexIssue]
        if not isinstance(item_group, zarr.Group):
            print(f"  ✗ Item [{item_index}] is not a zarr Group")
            return None

        # Get landmarks group
        if "landmarks" not in item_group:
            print(f"  ✗ 'landmarks' group not found in item [{item_index}]")
            return None

        landmarks_group = item_group["landmarks"]  # pyright: ignore[reportIndexIssue]
        if not isinstance(landmarks_group, zarr.Group):
            print("  ✗ 'landmarks' is not a zarr Group")
            return None

        available_keys = list(landmarks_group.keys())
        print(f"  ✓ Available landmark keys: {available_keys}")

        # Load all landmarks
        landmarks_dict: dict[str, np.ndarray] = {}
        for key in available_keys:
            # pyright: ignore[reportIndexIssue]
            landmark_array = landmarks_group[key]
            landmarks = np.array(landmark_array).astype(np.float32)
            landmarks_dict[key] = landmarks
            print(
                f"  ✓ Loaded {key}: "
                f"shape={landmarks.shape}, dtype={landmarks.dtype}"
            )

        return landmarks_dict

    except Exception as e:
        print(f"  ✗ Failed to load from zarr: {e}")
        return None


def load_landmark_from_zarr(
    zarr_path: Path, item_index: int = 0, landmark_key: str = "pose"
) -> np.ndarray | None:
    """Load landmark data from sldataset2 zarr format.

    Expected structure:
        dataset.zarr/
            metadata/           # Dataset-level information
            connections/        # Landmark connectivity graphs
            items/              # Individual samples
                [0]/
                    landmarks/
                        {landmark_key}  # zarr array
                [1]/
                ...

    Args:
        zarr_path: Path to zarr dataset
        item_index: Index of item to load (default: 0)
        landmark_key: Landmark key (e.g., "pose", "left_hand", "right_hand")

    Returns:
        Landmark array with shape (T, K, D) or None if not found
    """
    try:
        # Handle nested zarr structure (*.zarr/*.zarr)
        actual_zarr_path = zarr_path
        if zarr_path.is_dir():
            # Check for nested .zarr file
            nested_zarr = list(zarr_path.glob("*.zarr"))
            if nested_zarr:
                actual_zarr_path = nested_zarr[0]
                print(f"  → Found nested zarr: {actual_zarr_path.name}")

        root = zarr.open_group(str(actual_zarr_path), mode="r")
        print(f"  ✓ Opened zarr dataset: {actual_zarr_path}")

        if "items" not in root:
            print("  ✗ 'items' group not found in zarr root")
            return None

        items_group = root["items"]  # pyright: ignore[reportIndexIssue]
        if not isinstance(items_group, zarr.Group):
            print("  ✗ 'items' is not a zarr Group")
            return None

        print(f"  ✓ Found {len(list(items_group.keys()))} items")

        # Get specific item
        item_key = str(item_index)
        if item_key not in items_group:
            print(f"  ✗ Item [{item_index}] not found")
            return None

        item_group = items_group[item_key]  # pyright: ignore[reportIndexIssue]
        if not isinstance(item_group, zarr.Group):
            print(f"  ✗ Item [{item_index}] is not a zarr Group")
            return None

        # Get landmarks group
        if "landmarks" not in item_group:
            print(f"  ✗ 'landmarks' group not found in item [{item_index}]")
            return None

        landmarks_group = item_group["landmarks"]  # pyright: ignore[reportIndexIssue]
        if not isinstance(landmarks_group, zarr.Group):
            print("  ✗ 'landmarks' is not a zarr Group")
            return None

        print(f"  ✓ Available landmark keys: {list(landmarks_group.keys())}")

        # Get specific landmark
        if landmark_key not in landmarks_group:
            print(f"  ✗ Landmark key '{landmark_key}' not found")
            return None

        # pyright: ignore[reportIndexIssue]
        landmark_array = landmarks_group[landmark_key]
        landmarks = np.array(landmark_array)
        print(f"  ✓ Loaded landmarks: {landmark_key}")
        print(f"    Item: [{item_index}]")
        print(f"    Shape: {landmarks.shape}, dtype: {landmarks.dtype}")
        return landmarks.astype(np.float32)

    except Exception as e:
        print(f"  ✗ Failed to load from zarr: {e}")
        return None


def calculate_all_samples_metrics(zarr_path: Path) -> None:
    """Calculate NaN rate metrics for all samples in a zarr dataset.

    Args:
        zarr_path: Path to zarr dataset
    """
    # Handle nested zarr structure
    actual_zarr_path = zarr_path
    if zarr_path.is_dir():
        nested_zarr = list(zarr_path.glob("*.zarr"))
        if nested_zarr:
            actual_zarr_path = nested_zarr[0]

    try:
        root = zarr.open_group(str(actual_zarr_path), mode="r")
        if "items" not in root:
            print("  ✗ 'items' group not found")
            return

        items_group = root["items"]  # pyright: ignore[reportIndexIssue]
        if not isinstance(items_group, zarr.Group):
            print("  ✗ 'items' is not a zarr Group")
            return

        num_items = len(list(items_group.keys()))
        print(f"  ✓ Found {num_items} items in dataset\n")

        # Register metrics
        from metrics_prototype import register_metric, create_metric
        from metrics_prototype.plugins.completeness import NaNRateMetric

        register_metric("completeness.nan_rate", NaNRateMetric, {})
        metric_nan = create_metric("completeness.nan_rate")

        # Calculate for all samples
        all_results: dict[str, list[float]] = {
            "Pose": [],
            "L-Hand": [],
            "R-Hand": [],
            "Hands": [],
            "All": [],
        }

        print("Calculating NaN rates for all samples...\n")
        failed_count = 0
        for i in range(num_items):
            if i % 1000 == 0:
                print(f"  Processing sample {i}/{num_items}...")

            try:
                landmarks_dict = load_all_landmarks_from_zarr(
                    zarr_path, item_index=i
                )
                if not landmarks_dict:
                    continue

                # Categorize parts
                part_categories: dict[str, list[str]] = {
                    "Pose": [],
                    "L-Hand": [],
                    "R-Hand": [],
                }

                for key in landmarks_dict.keys():
                    key_lower = key.lower()
                    if "pose" in key_lower:
                        part_categories["Pose"].append(key)
                    elif "left" in key_lower or "l_hand" in key_lower:
                        part_categories["L-Hand"].append(key)
                    elif "right" in key_lower or "r_hand" in key_lower:
                        part_categories["R-Hand"].append(key)

                # Calculate for each category
                for category, keys in part_categories.items():
                    if keys:
                        key = keys[0]
                        result = metric_nan.calculate(landmarks_dict[key])
                        all_results[category].append(
                            result["values"]["nan_rate"]
                        )

                # Calculate Hands (combined)
                if "L-Hand" in part_categories and "R-Hand" in part_categories:
                    lhand_keys = part_categories["L-Hand"]
                    rhand_keys = part_categories["R-Hand"]
                    if lhand_keys and rhand_keys:
                        hands_combined = np.concatenate(
                            [
                                landmarks_dict[lhand_keys[0]],
                                landmarks_dict[rhand_keys[0]],
                            ],
                            axis=1,
                        )
                        result_hands = metric_nan.calculate(hands_combined)
                        all_results["Hands"].append(
                            result_hands["values"]["nan_rate"]
                        )

                # Calculate All (combined)
                all_landmarks: list[np.ndarray] = []
                for key in landmarks_dict.keys():
                    all_landmarks.append(landmarks_dict[key])
                if all_landmarks:
                    all_combined = np.concatenate(all_landmarks, axis=1)
                    result_all = metric_nan.calculate(all_combined)
                    all_results["All"].append(result_all["values"]["nan_rate"])

            except Exception as e:
                # Skip samples with errors (e.g., all NaN frames)
                failed_count += 1
                if i % 1000 == 0 or failed_count <= 10:
                    print(f"    ⚠️ Sample {i} skipped: {e}")

        # Display summary statistics
        print(
            f"\n✓ Processed {num_items} samples "
            f"(skipped {failed_count} with errors)\n"
        )
        print_header("NaN Rate Statistics (All Samples)")

        for category, rates in all_results.items():
            if rates:
                mean_rate = float(np.mean(rates))
                std_rate = float(np.std(rates))
                min_rate = float(np.min(rates))
                max_rate = float(np.max(rates))
                median_rate = float(np.median(rates))

                print(f"\n[{category}]")
                print(f"  Mean:   {mean_rate:.2%}")
                print(f"  Median: {median_rate:.2%}")
                print(f"  Std:    {std_rate:.2%}")
                print(f"  Min:    {min_rate:.2%}")
                print(f"  Max:    {max_rate:.2%}")

                # Evaluation
                if mean_rate < 0.10:
                    eval_str = "✅ Excellent"
                elif mean_rate < 0.20:
                    eval_str = "⚠️ Acceptable"
                else:
                    eval_str = "❌ Poor"
                print(f"  Evaluation: {eval_str}")

    except Exception as e:
        print(f"  ✗ Failed to calculate metrics: {e}")


def main(
    dataset_path: str | None = None,
    sample_idx: int = 0,
    calculate_all: bool = False,
) -> None:
    """Run the metrics prototype demo.

    Args:
        dataset_path: Optional path to real dataset sample directory
        sample_idx: Index of sample to load from zarr dataset (default: 0)
        calculate_all: Calculate metrics for all samples in dataset (default: False)
    """
    print_header("Metrics Prototype Demo")

    # Step 1: Register plugins
    print_step(1, "Registering metrics plugins...")
    register_metric("completeness.nan_rate", NaNRateMetric, {})
    register_metric("temporal.consistency", TemporalConsistencyMetric, {})
    register_metric("anatomical.constraint", AnatomicalConstraintMetric, {})
    print("✓ Registered: completeness.nan_rate")
    print("✓ Registered: temporal.consistency")
    print("✓ Registered: anatomical.constraint")

    # Step 2: Discover metrics
    print_step(2, "Discovering available metrics...")
    metric_names = list_metric_names()
    print(f"Found {len(metric_names)} metric(s):")
    for name in metric_names:
        # Parse category
        category = name.split(".")[0] if "." in name else "general"
        print(f"  - {name} (category: {category})")

    # Step 3: Load or create test data
    print_step(3, "Loading test data...")

    landmarks: np.ndarray | None = None
    landmarks_dict: dict[str, np.ndarray] | None = None

    # Try to load real data if path provided
    if dataset_path:
        dataset_root = Path(dataset_path)
        if dataset_root.exists():
            # Check if it's a zarr dataset
            if dataset_root.suffix == ".zarr" or (
                dataset_root / ".zgroup"
            ).exists():
                # Zarr dataset format
                if calculate_all:
                    print(f"Loading from zarr dataset: {dataset_root}")
                    print("Calculating metrics for ALL samples...\n")
                    calculate_all_samples_metrics(dataset_root)
                    return
                else:
                    print(f"Loading from zarr dataset: {dataset_root}")
                    print(f"Using sample index: {sample_idx}\n")
                    landmarks_dict = load_all_landmarks_from_zarr(
                        dataset_root, item_index=sample_idx
                    )
                if landmarks_dict:
                    # Use pose for other metrics
                    pose_key = next(
                        (k for k in landmarks_dict if "pose" in k.lower()),
                        None,
                    )
                    if pose_key:
                        landmarks = landmarks_dict[pose_key]
            elif (dataset_root / "landmarks").exists():
                # Direct path to item directory
                print(f"Loading from item directory: {dataset_root}")
                landmarks = load_landmark_data(dataset_root, part="pose")
            else:
                # Search for item directories under root
                print(f"Searching for item directories under: {dataset_root}")
                item_dirs = find_item_dirs(dataset_root)
                if item_dirs:
                    print(f"  Found {len(item_dirs)} item directory(ies)")
                    print(f"  Using first item: {item_dirs[0]}")
                    landmarks = load_landmark_data(item_dirs[0], part="pose")
                else:
                    print("  ✗ No item directories with landmarks/ found")
        else:
            print(f"  ✗ Dataset path not found: {dataset_root}")

    # Fall back to synthetic data
    if landmarks is None:
        print("Using synthetic test data...")
        np.random.seed(42)
        landmarks = np.random.rand(100, 33, 3).astype(np.float32)

        # Inject NaN values (10% missing data)
        landmarks[10:20, :, :] = np.nan

        print(
            f"  Generated landmarks: shape={landmarks.shape}, "
            f"dtype={landmarks.dtype}"
        )
        print("  Injected 10% NaN values (frames 10-19)")
    else:
        print(
            f"  Loaded real landmarks: shape={landmarks.shape}, "
            f"dtype={landmarks.dtype}"
        )

    # Step 4: Calculate NaN rate metric (per part if available)
    print_step(4, "Calculating NaN rate metric...")
    metric_nan = create_metric("completeness.nan_rate")

    if landmarks_dict:
        # Calculate NaN rate for each part
        print("Calculating NaN rate for each landmark part:\n")

        # Categorize parts
        part_categories: dict[str, list[str]] = {
            "Pose": [],
            "L-Hand": [],
            "R-Hand": [],
        }

        for key in landmarks_dict.keys():
            key_lower = key.lower()
            if "pose" in key_lower:
                part_categories["Pose"].append(key)
            elif "left" in key_lower or "l_hand" in key_lower:
                part_categories["L-Hand"].append(key)
            elif "right" in key_lower or "r_hand" in key_lower:
                part_categories["R-Hand"].append(key)

        # Calculate for each category
        results_by_category = {}
        for category, keys in part_categories.items():
            if keys:
                # Use first key in category
                key = keys[0]
                result = metric_nan.calculate(landmarks_dict[key])
                results_by_category[category] = result
                print(f"  [{category}] {key}")
                print(f"    NaN rate: {result['values']['nan_rate']:.2%}")
                print(
                    f"    Frames with NaN: "
                    f"{result['metadata']['frames_with_nan']}/"
                    f"{result['metadata']['total_frames']}"
                )

        # Calculate Hands (combined)
        if "L-Hand" in results_by_category and "R-Hand" in results_by_category:
            lhand_key = part_categories["L-Hand"][0]
            rhand_key = part_categories["R-Hand"][0]
            # Stack hands: (T, K_left + K_right, D)
            hands_combined = np.concatenate(
                [landmarks_dict[lhand_key], landmarks_dict[rhand_key]],
                axis=1,
            )
            result_hands = metric_nan.calculate(hands_combined)
            print("  [Hands] Combined L-Hand + R-Hand")
            print(f"    NaN rate: {result_hands['values']['nan_rate']:.2%}")
            print(
                f"    Frames with NaN: "
                f"{result_hands['metadata']['frames_with_nan']}/"
                f"{result_hands['metadata']['total_frames']}"
            )
            results_by_category["Hands"] = result_hands

        # Calculate All (combined all parts)
        all_landmarks: list[np.ndarray] = []
        for key in landmarks_dict.keys():
            all_landmarks.append(landmarks_dict[key])
        if all_landmarks:
            # Stack all: (T, K_total, D)
            all_combined = np.concatenate(all_landmarks, axis=1)
            result_all = metric_nan.calculate(all_combined)
            print("  [All] All parts combined")
            print(f"    NaN rate: {result_all['values']['nan_rate']:.2%}")
            print(
                f"    Frames with NaN: "
                f"{result_all['metadata']['frames_with_nan']}/"
                f"{result_all['metadata']['total_frames']}"
            )
            results_by_category["All"] = result_all

        # Store first result for backward compatibility
        result_nan = next(iter(results_by_category.values()))
    else:
        # Single landmark calculation
        result_nan = metric_nan.calculate(landmarks)
        print(f"✓ Metric: {result_nan['metric_name']}")
        print(f"✓ NaN rate: {result_nan['values']['nan_rate']:.2%}")
        print(f"✓ Total frames: {result_nan['metadata']['total_frames']}")
        print(f"✓ Frames with NaN: {result_nan['metadata']['frames_with_nan']}")

    # Step 5: Calculate temporal consistency metric
    print_step(5, "Calculating temporal consistency metric...")

    # For temporal analysis, use real data if available
    # (metric handles NaN with nanmean/nanstd)
    if dataset_path:
        landmarks_temporal = landmarks
        print("Using real landmark data for temporal analysis")
    else:
        # Create clean synthetic data without NaN
        landmarks_temporal = np.random.rand(100, 33, 3).astype(np.float32)
        # Add some smooth motion pattern
        for t in range(1, 100):
            landmarks_temporal[t] = (
                landmarks_temporal[t - 1] * 0.9 + landmarks_temporal[t] * 0.1
            )
        print("Using synthetic smooth motion data")

    metric_temporal = create_metric("temporal.consistency")
    result_temporal = metric_temporal.calculate(landmarks_temporal)

    print(f"✓ Metric: {result_temporal['metric_name']}")
    print(f"✓ Mean velocity: {result_temporal['values']['mean_velocity']:.6f}")
    print(f"✓ Std velocity: {result_temporal['values']['std_velocity']:.6f}")
    print(f"✓ Mean acceleration: {result_temporal['values']['mean_acceleration']:.6f}")
    print(f"✓ Smoothness (std accel): {result_temporal['values']['smoothness']:.6f}")

    # Step 6: Calculate anatomical constraint metric
    print_step(6, "Calculating anatomical constraint metric...")

    # For anatomical analysis, use real data if available
    if dataset_path:
        landmarks_anatomical = landmarks
        bone_pairs = MEDIAPIPE_POSE_BONES
        print("Using real landmark data with MediaPipe Pose bones (12 bones)")
    else:
        # Create synthetic data with consistent bone lengths
        landmarks_anatomical = np.random.rand(100, 33, 3).astype(np.float32)
        bone_pairs = MEDIAPIPE_POSE_BONES
        print("Using synthetic data with MediaPipe Pose bones")

    metric_anatomical = create_metric("anatomical.constraint")
    result_anatomical = metric_anatomical.calculate(
        landmarks_anatomical, bone_pairs=bone_pairs
    )

    print(f"✓ Metric: {result_anatomical['metric_name']}")
    print(
        f"✓ Mean variation coef: {result_anatomical['values']['mean_variation']:.6f}"
    )
    print(
        f"✓ Std variation coef: {result_anatomical['values']['std_variation']:.6f}"
    )
    print(
        f"✓ Min variation coef: {result_anatomical['values']['min_variation']:.6f}"
    )
    print(
        f"✓ Max variation coef: {result_anatomical['values']['max_variation']:.6f}"
    )

    # Step 7: Interpret results
    print_step(7, "Interpreting results...")
    nan_rate = result_nan['values']['nan_rate']
    threshold = 0.2  # 20% recommended threshold

    if nan_rate < threshold:
        status = "✓ PASS"
        message = "Data quality is acceptable"
    else:
        status = "✗ FAIL"
        message = "Data quality is poor"

    print(
        "Completeness: "
        f"{status} - {message} "
        f"({'<' if nan_rate < threshold else '>='} {threshold:.0%} missing)"
    )
    print(
        f"Temporal: Smoothness = {result_temporal['values']['smoothness']:.6f} "
        "(lower is better)"
    )
    print(
        "Anatomical: Mean variation = "
        f"{result_anatomical['values']['mean_variation']:.6f} "
        "(lower is better)"
    )

    # Completion
    print_header("Demo completed successfully!")
    print("\nNext steps:")
    print("  1. Review the plugin architecture in metrics_prototype/")
    print("  2. Implement more metrics (temporal, anatomical)")
    print("  3. Integrate into src/cslrtools2/sldataset/metrics/")
    print("  4. Add Entry Points to pyproject.toml")
    print("  5. Create CLI commands (sldataset metrics)")


if __name__ == "__main__":
    # Parse command line arguments
    dataset_path = None
    sample_idx = 0
    calculate_all = False

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--help", "-h"):
            print("Usage: uv run python demo.py [OPTIONS]")
            print("")
            print("Options:")
            print("  --dataset PATH      Path to dataset root or item directory")
            print("  --sample-idx N      Sample index to load (default: 0)")
            print("  --all               Calculate metrics for all samples")
            print("  -h, --help          Show this help message")
            print("")
            print("Supported directory structures:")
            print("  1. Item directory (direct):")
            print("     PATH/{item_dir}/landmarks/*.{part}.npy")
            print(
                "     Example: "
                "H:\\SLRDataset\\fs50-mp-holistic-v5\\090\\P2_S090_00.mp4"
            )
            print("")
            print("  2. Root directory (searches recursively):")
            print("     PATH/**/{item_dir}/landmarks/*.{part}.npy")
            print("     Example: H:\\SLRDataset\\fs50-mp-holistic-v5\\090")
            sys.exit(0)

        # Parse arguments
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--dataset" and i + 1 < len(sys.argv):
                dataset_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--sample-idx" and i + 1 < len(sys.argv):
                sample_idx = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--all":
                calculate_all = True
                i += 1
            else:
                print(f"Unknown argument: {sys.argv[i]}", file=sys.stderr)
                sys.exit(1)

    main(dataset_path, sample_idx, calculate_all)
