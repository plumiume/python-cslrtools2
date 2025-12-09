"""Calculate metrics for real dataset.

Dataset: fs50-lmpipe-v5.2.1.zarr
Indices: idx=0 and idx="all"
Time limit: < 10 minutes
"""

import time
from pathlib import Path
import numpy as np
import zarr

from cslrtools2.sldataset.metrics.calculator import MetricCalculator
from cslrtools2.sldataset.metrics.methods import (
    anatomical,
    completeness,
    temporal,
)

# Dataset configuration
DATASET_ROOT = Path(r"C:\Users\ikeda\Downloads\fs50-lmpipe-v5.2.1.zarr\fs50-lmpipe-v5.2.1.zarr")

# Create calculator with all metrics
calculator = MetricCalculator([
    completeness.NaNRateMetric(),
    temporal.TemporalConsistencyMetric(),
    {
        "method": anatomical.BoneLengthMetric,
        "bone_pairs": anatomical.MEDIAPIPE_POSE_BONES,
    },
])

print("="*70)
print("Metric Calculator - Real Dataset Evaluation")
print("="*70)
print(f"Dataset: {DATASET_ROOT.name}")
print(f"Registered metrics: {calculator.list_metrics()}")
print()

# Open zarr dataset
store = zarr.DirectoryStore(str(DATASET_ROOT))
root = zarr.open(store, mode='r')

print(f"Zarr groups: {list(root.keys())}")
print()

# Get pose landmarks array
if 'landmarks' in root and 'pose' in root['landmarks']:
    pose_array = root['landmarks/pose']
    print(f"Pose landmarks shape: {pose_array.shape}")
    print(f"Pose landmarks dtype: {pose_array.dtype}")
    print()
else:
    print("ERROR: landmarks/pose not found in dataset")
    exit(1)

# Test 1: Calculate for idx=0
print("="*70)
print("Test 1: Calculate metrics for idx=0")
print("="*70)

start_time = time.time()

# Load data for idx=0
pose_data_0 = pose_array[0]  # Shape: (frames, keypoints, coords)
print(f"Data shape for idx=0: {pose_data_0.shape}")
print(f"Data type: {pose_data_0.dtype}")

# Convert to float32 if needed
if pose_data_0.dtype != np.float32:
    pose_data_0 = pose_data_0.astype(np.float32)

# Calculate metrics
results_0 = calculator.calculate(pose_data_0)

elapsed_0 = time.time() - start_time

print(f"\nResults for idx=0 (computed in {elapsed_0:.2f}s):")
print("-" * 70)

for metric_name, result in results_0.items():
    print(f"\n{metric_name}:")
    print(f"  Metric: {result['name']}")
    print(f"  Values:")
    for key, value in result['values'].items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    print(f"  Metadata:")
    for key, value in result['metadata'].items():
        if key == 'bone_pairs':
            print(f"    {key}: {len(value)} pairs")
        else:
            print(f"    {key}: {value}")

# Test 2: Calculate for all indices
print("\n" + "="*70)
print("Test 2: Calculate metrics for all indices (aggregated)")
print("="*70)

start_time = time.time()

# Calculate for all indices
num_samples = pose_array.shape[0]
print(f"Total samples: {num_samples}")

all_results = {
    'NaNRateMetric': {'nan_rate': []},
    'TemporalConsistencyMetric': {
        'smoothness': [],
        'mean_velocity': [],
        'mean_acceleration': []
    },
    'BoneLengthMetric': {
        'mean_variation': [],
        'std_variation': [],
        'min_variation': [],
        'max_variation': []
    }
}

print("\nProcessing samples...")
for idx in range(num_samples):
    if idx % 10 == 0:
        print(f"  Processing sample {idx}/{num_samples}...", end='\r')
    
    # Load data
    pose_data = pose_array[idx]
    if pose_data.dtype != np.float32:
        pose_data = pose_data.astype(np.float32)
    
    # Skip if not enough frames
    if pose_data.shape[0] < 3:
        continue
    
    # Calculate metrics
    try:
        results = calculator.calculate(pose_data)
        
        # Collect results
        for metric_name, result in results.items():
            for key, value in result['values'].items():
                if key in all_results[metric_name]:
                    all_results[metric_name][key].append(value)
    except Exception as e:
        print(f"\nWarning: Failed to process idx={idx}: {e}")
        continue

print(f"\nProcessing complete!")

elapsed_all = time.time() - start_time

# Calculate statistics
print(f"\nAggregated Results (computed in {elapsed_all:.2f}s):")
print("-" * 70)

for metric_name, values_dict in all_results.items():
    print(f"\n{metric_name}:")
    for key, values in values_dict.items():
        if len(values) > 0:
            values_array = np.array(values)
            print(f"  {key}:")
            print(f"    mean: {np.mean(values_array):.6f}")
            print(f"    std:  {np.std(values_array):.6f}")
            print(f"    min:  {np.min(values_array):.6f}")
            print(f"    max:  {np.max(values_array):.6f}")
            print(f"    samples: {len(values)}")
        else:
            print(f"  {key}: No valid samples")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"Total time: {elapsed_0 + elapsed_all:.2f}s")
print(f"  idx=0: {elapsed_0:.2f}s")
print(f"  all indices: {elapsed_all:.2f}s")
print(f"Average time per sample: {elapsed_all / num_samples:.3f}s")
print(f"Successful samples: {len(all_results['NaNRateMetric']['nan_rate'])}/{num_samples}")
print()
print("✅ Evaluation complete!")
print("="*70)
