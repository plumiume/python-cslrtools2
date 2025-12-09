"""Calculate metrics for real dataset using prototype v2.

Dataset: fs50-lmpipe-v5.2.1.zarr
Indices: idx=0 and idx="all"
Time limit: < 10 minutes
"""

import time
from pathlib import Path
import numpy as np
import zarr

from metrics_prototype2.plugins.completeness import NaNRateMetric
from metrics_prototype2.plugins.temporal import TemporalConsistencyMetric
from metrics_prototype2.plugins.anatomical import AnatomicalConstraintMetric

# Dataset configuration
DATASET_ROOT = Path(r"C:\Users\ikeda\Downloads\fs50-lmpipe-v5.2.1.zarr")

# Define categories to process
CATEGORIES = [
    "pose",
    "left_hand",
    "right_hand",
    {"hands": ["left_hand", "right_hand"]},
    {"all": ["pose", "left_hand", "right_hand"]}
]

# Create metrics
metrics = {
    "completeness.nan_rate": NaNRateMetric(),
    "temporal.temporal_consistency": TemporalConsistencyMetric(),
    "anatomical.bone_length": AnatomicalConstraintMetric(),
}

print("="*70)
print("Metric Calculator Prototype v2 - Real Dataset Evaluation")
print("="*70)
print(f"Dataset: {DATASET_ROOT.name}")
print(f"Registered metrics: {list(metrics.keys())}")
print()

# Open zarr dataset
root = zarr.open(str(DATASET_ROOT), mode='r')

print(f"Zarr groups: {list(root.keys())}")

# Navigate to items
dataset_group = root['fs50-lmpipe-v5.2.zarr']
items_group = dataset_group['items']
num_items = len(list(items_group.keys()))
print(f"Number of items: {num_items}")

# Get first item to check structure
item0 = items_group['0']
landmarks_group = item0['landmarks']
landmark_types = list(landmarks_group.keys())
print(f"Landmark types: {landmark_types}")
print()


def get_category_keys(category):
    """Get landmark keys for a category.
    
    Args:
        category: str, list, or dict defining the category
        
    Returns:
        tuple: (category_name, list of landmark keys)
    """
    if isinstance(category, str):
        # Single landmark type
        return category, [f"mediapipe.{category}"]
    elif isinstance(category, dict):
        # Combined category
        name = list(category.keys())[0]
        keys_list = category[name]
        return name, [f"mediapipe.{k}" for k in keys_list]
    else:
        raise ValueError(f"Invalid category format: {category}")


def load_category_data(landmarks, category_keys):
    """Load and concatenate data for multiple landmark types.
    
    Args:
        landmarks: zarr landmarks group
        category_keys: list of landmark keys to load
        
    Returns:
        numpy array with concatenated data or None if no data available
    """
    data_list = []
    for key in category_keys:
        try:
            if key in landmarks:
                data = landmarks[key][()]
                data_list.append(data)
        except (KeyError, FileNotFoundError):
            continue
    
    if not data_list:
        return None
    
    if len(data_list) == 1:
        return data_list[0]
    
    # Concatenate along keypoint dimension (axis=1)
    return np.concatenate(data_list, axis=1)

# Process each category
for category in CATEGORIES:
    category_name, category_keys = get_category_keys(category)
    
    print("="*70)
    print(f"Category: {category_name}")
    print(f"Landmark keys: {category_keys}")
    print("="*70)
    
    # Test 1: Calculate for idx=0
    print(f"\nTest 1: Calculate metrics for idx=0")
    print("-" * 70)
    
    start_time = time.time()
    
    # Load data for idx=0
    item_0 = items_group['0']
    landmarks_0 = item_0['landmarks']
    data_0 = load_category_data(landmarks_0, category_keys)
    
    if data_0 is None:
        print(f"⚠️  No data available for category '{category_name}' in idx=0")
        print()
        continue
    
    print(f"Data shape for idx=0: {data_0.shape}")
    print(f"Data type: {data_0.dtype}")
    
    # Convert to float32 if needed
    if data_0.dtype != np.float32:
        data_0 = data_0.astype(np.float32)
    
    # Calculate metrics
    results_0 = {}
    for name, metric in metrics.items():
        try:
            results_0[name] = metric.calculate(data_0)
        except Exception as e:
            print(f"⚠️  Failed to calculate {name}: {e}")
            continue
    
    elapsed_0 = time.time() - start_time
    
    print(f"\nResults for idx=0 (computed in {elapsed_0:.2f}s):")
    
    for metric_name, result in results_0.items():
        print(f"\n{metric_name}:")
        print(f"  Metric: {result['metric_name']}")
        print("  Values:")
        for key, value in result['values'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        print("  Metadata:")
        for key, value in result['metadata'].items():
            if key == 'bone_pairs':
                print(f"    {key}: {len(value)} pairs")
            else:
                print(f"    {key}: {value}")

    # Test 2: Calculate for all indices
    print("\n" + "-"*70)
    print("Test 2: Calculate metrics for all indices (aggregated)")
    print("-"*70)
    
    start_time_all = time.time()
    
    # Calculate for all indices
    item_keys = list(items_group.keys())
    num_samples = len(item_keys)
    print(f"Total samples: {num_samples}")
    
    all_results = {
        'completeness.nan_rate': {'nan_rate': []},
        'temporal.temporal_consistency': {
            'smoothness': [],
            'mean_velocity': [],
            'mean_acceleration': []
        },
        'anatomical.bone_length': {
            'mean_variation': [],
            'std_variation': [],
            'min_variation': [],
            'max_variation': []
        }
    }
    
    print("\nProcessing samples...")
    for idx, item_key in enumerate(item_keys):
        if idx % 100 == 0:
            elapsed = time.time() - start_time_all
            print(
                f"  Processing sample {idx}/{num_samples} "
                f"({elapsed:.1f}s)...",
                end='\r'
            )
        
        # Load data
        item = items_group[item_key]
        
        # Skip if landmarks doesn't exist
        try:
            if 'landmarks' not in item:
                continue
            landmarks = item['landmarks']
            
            # Load category data
            data = load_category_data(landmarks, category_keys)
            if data is None:
                continue
                
        except (KeyError, FileNotFoundError):
            continue
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Skip if not enough frames
        if data.shape[0] < 3:
            continue
        
        # Calculate metrics
        try:
            for metric_name, metric in metrics.items():
                result = metric.calculate(data)
                
                # Collect results
                for key, value in result['values'].items():
                    if key in all_results[metric_name]:
                        all_results[metric_name][key].append(value)
        except Exception as e:
            # Silently skip errors
            continue
    
    print(f"\nProcessing complete!                    ")
    
    elapsed_all = time.time() - start_time_all
    
    # Calculate statistics
    print(f"\nAggregated Results (computed in {elapsed_all:.2f}s):")
    
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
    
    # Summary for this category
    nan_rate_samples = len(all_results['completeness.nan_rate']['nan_rate'])
    print(f"\n✅ Category '{category_name}' complete!")
    print(f"   Successful samples: {nan_rate_samples}/{num_samples}")
    print(f"   Time: idx=0={elapsed_0:.2f}s, all={elapsed_all:.2f}s")
    print()

# Final summary
print("="*70)
print("✅ All categories evaluation complete!")
print("="*70)
