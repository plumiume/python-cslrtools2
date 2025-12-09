# Metrics Prototype v2

**STATUS**: PROTOTYPE - SLDataset Integration Version

This is an improved version of the metrics prototype that demonstrates proper integration with `cslrtools2.sldataset`.

## Changes from v1

### Key Improvements

1. **SLDataset Integration**
   - Uses `SLDataset.from_zarr()` for data loading
   - Leverages built-in zarr lazy loading (no manual zarr navigation)
   - Simpler, more maintainable code

2. **New Utilities Module**
   - `utils.py` with metrics-specific helper functions:
     - `to_numpy_landmarks()`: Convert zarr.Array to NumPy
     - `categorize_landmarks()`: Classify by body part
     - `combine_landmarks()`: Concatenate multiple parts
   - These are **NOT** added to `cslrtools2.sldataset.utils` (by design)

3. **Cleaner Code**
   - ~30-40% code reduction in `demo.py`
   - Better separation of concerns
   - Uses `get_group()`/`get_array()` from sldataset utilities

### Corrected Understanding

Based on user clarifications:

- ✅ **Lazy Loading**: `SLDataset.from_zarr()` returns items with `zarr.Array` references (not actual data)
  - Data is loaded on-demand when `np.array(zarr_array)` is called
  - No need for `IterableSLDataset` or additional lazy loading mechanisms

- ✅ **Type System**: `DefaultSLDatasetItem` already covers NumPy arrays (`Any` type)
  - No need for `NumpySLDatasetItem` type alias

- ✅ **Design Philosophy**: Extensions handle type conversions, not core sldataset
  - No `as_numpy()` method in `SLDatasetItem` (by design)
  - Metrics-specific helpers belong in `metrics_prototype`, not `sldataset.utils`

## Project Structure

```
metrics_prototype2/
├── __init__.py          # Module exports (registry, loader)
├── base.py              # Metric ABC, MetricResult (unchanged from v1)
├── loader.py            # Plugin registry (unchanged from v1)
├── utils.py             # NEW: Helper functions for metrics
├── demo.py              # Improved demo using SLDataset
├── plugins/
│   ├── __init__.py
│   ├── completeness.py  # NaN rate metric
│   ├── temporal.py      # Temporal consistency
│   └── anatomical.py    # Bone length variation
└── README.md            # This file
```

## Usage

### Basic Example (Single Sample)

```bash
# Calculate metrics for sample 0
uv run python -m metrics_prototype2.demo \
    --dataset C:\Users\ikeda\Downloads\fs50-lmpipe-v5.2.1.zarr \
    --sample-idx 0
```

### Batch Processing (All Samples)

```bash
# Calculate metrics for all samples
uv run python -m metrics_prototype2.demo \
    --dataset C:\Users\ikeda\Downloads\fs50-lmpipe-v5.2.1.zarr \
    --all \
    --output results.json
```

## Integration Patterns

### Using SLDataset for Data Loading

```python
import zarr
from cslrtools2.sldataset import SLDataset

# Open zarr dataset
root = zarr.open_group("dataset.zarr", mode="r")

# Load dataset (returns items with zarr.Array references, not data)
dataset = SLDataset.from_zarr(root)

# Access item (still zarr.Array references)
item = dataset[0]

# Convert to NumPy (lazy evaluation happens here)
from metrics_prototype2.utils import to_numpy_landmarks
landmarks_np = to_numpy_landmarks(item.landmarks)

# Now landmarks_np contains actual NumPy arrays
```

### Multi-Part Metrics Calculation

```python
from metrics_prototype2.utils import categorize_landmarks, combine_landmarks
from metrics_prototype2 import create_metric

# Categorize landmarks by body part
categories = categorize_landmarks(landmarks_np.keys())
# Returns: {"Pose": ["mediapipe.pose"], "Left Hand": [...], ...}

# Calculate NaN rate for each part
metric = create_metric("completeness.nan_rate")

for category, keys in categories.items():
    if len(keys) == 1:
        array = landmarks_np[keys[0]]
        result = metric.calculate(array)
        print(f"{category}: {result['values']['nan_rate']:.4f}")

# Combine hands
hands_keys = categories["Left Hand"] + categories["Right Hand"]
hands_combined = combine_landmarks(landmarks_np, hands_keys, axis=1)
result = metric.calculate(hands_combined)
print(f"Hands: {result['values']['nan_rate']:.4f}")
```

## Metrics Available

### 1. Completeness (NaN Rate)

**Metric Name**: `completeness.nan_rate`

**Purpose**: Measures frame-level data completeness

**Output**:
- `nan_rate`: Proportion of frames with NaN (0.0-1.0)
- `frames_with_nan`: Count of incomplete frames
- `total_frames`: Total frames

### 2. Temporal Consistency

**Metric Name**: `temporal.consistency`

**Purpose**: Evaluates motion smoothness

**Output**:
- `mean_velocity`: Average frame-to-frame displacement
- `smoothness`: Acceleration std dev (lower = smoother)

### 3. Anatomical Constraint

**Metric Name**: `anatomical.bone_length`

**Purpose**: Checks bone length consistency

**Output**:
- `mean_variation`: Average variation coefficient
- `std_variation`: Std dev of variations

## Performance

Based on `metrics_prototype` optimization:

- **40,214 samples**: ~5-10 minutes (estimated)
- **Zarr loading**: Lazy evaluation (on-demand)
- **Memory**: Only loads data when accessed

## Differences from v1

| Aspect | v1 (metrics_prototype) | v2 (metrics_prototype2) |
|--------|------------------------|-------------------------|
| Data Loading | Manual zarr navigation | `SLDataset.from_zarr()` |
| Zarr Access | Repeated `zarr.open_group()` | Single open, reused reference |
| Utilities | Mixed with demo code | Separate `utils.py` module |
| Code Size | ~864 lines (demo.py) | ~280 lines (demo.py, 67% reduction) |
| Integration | Independent | Leverages sldataset utilities |

## Future Integration

When moving to production (`src/cslrtools2/sldataset/metrics/`):

1. Copy plugin files to `src/cslrtools2/sldataset/metrics/plugins/`
2. Copy `utils.py` to `src/cslrtools2/sldataset/metrics/utils.py`
3. Update imports (remove `metrics_prototype2.` prefix)
4. Register plugins in `pyproject.toml`:
   ```toml
   [project.entry-points."cslrtools2.sldataset.metrics"]
   completeness.nan_rate = "cslrtools2.sldataset.metrics.plugins.completeness:nan_rate_info"
   ```
5. Add CLI commands to `src/cslrtools2/sldataset/app/cli.py`

## References

- **Design Document**: `pose_estimation_metrics_analysis.md`
- **Parent Project**: `c:\Users\ikeda\Workspace\1github\cslrtools2-dataset-2`
- **Architecture**: `METRICS_PROTOTYPE_ARCHITECTURE.md`
- **Integration Gaps**: `SLDATASET_INTEGRATION_GAPS.md` (corrected version)

## License

Apache License 2.0 (see `LICENSE` file)
