# Metrics Prototype

**STATUS**: 🚧 PROTOTYPE - Design reference for future implementation

## Overview

This directory contains a **prototype** of the metrics plugin system that will eventually be implemented in `src/cslrtools2/sldataset/metrics/`. The prototype demonstrates:

- ✅ Plugin architecture using simulated Entry Points
- ✅ **Phase 1**: NaN rate metric (completeness)
- ✅ **Phase 2**: Temporal consistency metric (velocity, acceleration, smoothness)
- ✅ **Phase 3**: Anatomical constraint metric (bone length variation)
- ✅ Type-safe metric interface with PEP 695 generics
- ✅ Engine-agnostic design (works with any landmark data)
- ✅ Ground Truth-free evaluation

## ⚠️ Important Notes

1. **Not Production Code**: This is a prototype and should NOT be imported in production code
2. **Design Reference**: Use this as a reference when implementing the real metrics system
3. **Entry Points Simulation**: Uses a simulated registry instead of real Entry Points
4. **Future Location**: Real implementation will be in `src/cslrtools2/sldataset/metrics/`

## Directory Structure

```
metrics_prototype/
├── base.py                    # Metric ABC and MetricResult type
├── loader.py                  # Plugin discovery system (simulated)
├── __init__.py                # Public API
├── demo.py                    # Demonstration script
├── plugins/
│   ├── completeness.py        # Phase 1: NaN rate metric
│   ├── temporal.py            # Phase 2: Temporal consistency
│   ├── anatomical.py          # Phase 3: Anatomical constraints
│   └── __init__.py            # Plugin module marker
└── README.md                  # This file
```

## Quick Start

### Run the Demo

```powershell
uv run python metrics_prototype/demo.py
```

Expected output:
```
============================================================
Metrics Prototype Demo
============================================================

[1] Registering metrics plugins...
✓ Registered: completeness.nan_rate
✓ Registered: temporal.consistency
✓ Registered: anatomical.constraint

[2] Discovering available metrics...
Found 3 metric(s):
  - anatomical.constraint (category: anatomical)
  - completeness.nan_rate (category: completeness)
  - temporal.consistency (category: temporal)

[3] Loading test data...
Generated landmarks: shape=(100, 33, 3), dtype=float32
Injected 10% NaN values (frames 10-19)

[4] Calculating NaN rate metric...
✓ Metric: nan_rate
✓ NaN rate: 10.00%
✓ Total frames: 100
✓ Frames with NaN: 10

[5] Calculating temporal consistency metric...
✓ Metric: temporal_consistency
✓ Mean velocity: 0.009192
✓ Std velocity: 0.019889
✓ Mean acceleration: 0.014494
✓ Smoothness (std accel): 0.031416

[6] Calculating anatomical constraint metric...
✓ Metric: anatomical_constraint
✓ Mean variation coef: 0.073624
✓ Std variation coef: 0.047481
✓ Min variation coef: 0.027523
✓ Max variation coef: 0.179017

[7] Interpreting results...
Completeness: ✓ PASS - Data quality is acceptable (< 20% missing)
Temporal: Smoothness = 0.031416 (lower is better)
Anatomical: Mean variation = 0.073624 (lower is better)
```

### Run with Real Data

```powershell
uv run python metrics_prototype/demo.py --dataset "H:\SLRDataset\fs50-mp-holistic-v5\090\P2_S090_00.mp4"
```

### Use Programmatically

```python
from metrics_prototype import register_metric, create_metric
from metrics_prototype.plugins.completeness import NaNRateMetric
from metrics_prototype.plugins.temporal import TemporalConsistencyMetric
from metrics_prototype.plugins.anatomical import (
    AnatomicalConstraintMetric,
    MEDIAPIPE_POSE_BONES,
)
import numpy as np

# Register plugins (simulates Entry Points)
register_metric("completeness.nan_rate", NaNRateMetric, {})
register_metric("temporal.consistency", TemporalConsistencyMetric, {})
register_metric("anatomical.constraint", AnatomicalConstraintMetric, {})

# Create metrics
metric_nan = create_metric("completeness.nan_rate")
metric_temporal = create_metric("temporal.consistency")
metric_anatomical = create_metric("anatomical.constraint")

# Calculate on data
data = np.random.rand(100, 33, 3).astype(np.float32)

# Completeness
result_nan = metric_nan.calculate(data)
print(f"NaN rate: {result_nan['values']['nan_rate']:.2%}")

# Temporal consistency
result_temporal = metric_temporal.calculate(data)
print(f"Smoothness: {result_temporal['values']['smoothness']:.6f}")

# Anatomical constraint
result_anatomical = metric_anatomical.calculate(
    data, bone_pairs=MEDIAPIPE_POSE_BONES
)
print(f"Mean variation: {result_anatomical['values']['mean_variation']:.6f}")
```

## Implemented Metrics

### Phase 1: Completeness (Basic) ✅

| Metric | Description | Status |
|--------|-------------|--------|
| `completeness.nan_rate` | Frame-level NaN detection using logical OR | ✅ Implemented |

**Formula**: `m = E_t [ ∨^K ∨^D isNaN(X_t,k,d) ]`

**Returns**:
- `nan_rate`: Proportion of frames with any NaN (0.0 to 1.0)
- `total_frames`: Total number of frames
- `frames_with_nan`: Number of frames containing at least one NaN

**Interpretation**:
- < 10%: Excellent
- 10-20%: Acceptable
- \> 20%: Poor quality

### Phase 2: Temporal Consistency ✅

| Metric | Description | Status |
|--------|-------------|--------|
| `temporal.consistency` | Velocity, acceleration, smoothness | ✅ Implemented |

**Calculations**:
- `velocity = landmarks[t+1] - landmarks[t]`
- `acceleration = velocity[t+1] - velocity[t]`
- `smoothness = std(acceleration)` (lower is better)

**Returns**:
- `mean_velocity`: Average movement speed across all keypoints
- `std_velocity`: Variance in velocity
- `mean_acceleration`: Average acceleration magnitude
- `smoothness`: Standard deviation of acceleration (jitter indicator)

**Interpretation**:
- Smoothness < 0.05: Excellent (stable tracking)
- Smoothness 0.05-0.10: Acceptable
- Smoothness > 0.10: Poor (excessive jitter)

### Phase 3: Anatomical Constraints ✅

| Metric | Description | Status |
|--------|-------------|--------|
| `anatomical.constraint` | Bone length consistency | ✅ Implemented |

**Formula**: `variation_coef = std(bone_length) / mean(bone_length)`

**MediaPipe Pose Bones** (12 bones):
- **Torso** (4): Left/Right Shoulder, Left/Right Hip, Shoulder-Hip connections
- **Arms** (4): Left/Right Upper Arm, Left/Right Forearm
- **Legs** (4): Left/Right Thigh, Left/Right Shin

**Returns**:
- `mean_variation`: Average variation coefficient across all bones
- `std_variation`: Variability in bone consistency
- `min_variation`: Most stable bone
- `max_variation`: Least stable bone

**Interpretation**:
- < 0.1 (10%): Excellent consistency
- 0.1-0.2: Acceptable
- \> 0.2 (20%): Poor (unstable bone lengths)

### Planned Metrics

| Phase | Metrics | Status |
|-------|---------|--------|
| Phase 4 | `geometric.multiview`, `anatomical.joint_angle` | 📋 Planned |

## Architecture

### Plugin System Design

```
Entry Points (pyproject.toml)
↓
loader.load_metrics() → dict[str, MetricInfo]
↓
create_metric(name) → Metric instance
↓
metric.calculate(data) → MetricResult
```

### Key Components

#### 1. **Metric ABC** (`base.py`)
```python
class Metric(ABC):
    @abstractmethod
    def calculate(data, **kwargs) -> MetricResult: ...
    
    @abstractmethod
    def get_description() -> str: ...
    
    def validate_inputs(data) -> bool: ...
```

#### 2. **MetricResult** (`base.py`)
```python
class MetricResult(TypedDict):
    metric_name: str
    values: Mapping[str, float]
    metadata: Mapping[str, Any]
```

#### 3. **Plugin Loader** (`loader.py`)
```python
def load_metrics() -> dict[str, MetricInfo]: ...
def create_metric(name, **kwargs) -> Metric: ...
```

## Migration Plan to Production

When implementing the real metrics system in `src/cslrtools2/sldataset/metrics/`:

### Step 1: Copy Architecture
```powershell
# Create directories
New-Item -Path "src/cslrtools2/sldataset/metrics" -ItemType Directory
New-Item -Path "src/cslrtools2/plugins/metrics" -ItemType Directory

# Copy base files (with modifications)
# - base.py
# - loader.py (replace simulated registry with real Entry Points)
# - __init__.py
```

### Step 2: Update Entry Points

In `pyproject.toml`:
```toml
[project.entry-points."cslrtools2.sldataset.metrics"]
"completeness.nan_rate" = "cslrtools2.plugins.metrics.completeness:nan_rate_info"
```

### Step 3: Replace Simulated Registry

In `loader.py`, replace:
```python
# OLD (prototype)
_SIMULATED_PLUGINS: dict[str, tuple[type[Metric], dict[str, Any]]] = {}

def register_metric(...): ...  # Remove this
```

With:
```python
# NEW (production)
import importlib.metadata

def load_metrics() -> dict[str, MetricInfo]:
    entry_points = importlib.metadata.entry_points(
        group="cslrtools2.sldataset.metrics"
    )
    # ... real Entry Points loading
```

### Step 4: Add CLI Integration

In `src/cslrtools2/sldataset/app/cli.py`:
```python
@sldataset_app.command()
def metrics(
    dataset: Path,
    metric: list[str],
    output: Path,
):
    """Calculate metrics on dataset."""
    # Use metrics_prototype architecture as reference
```

### Step 5: Comprehensive Testing

```python
# tests/unit/sldataset/metrics/test_completeness.py
def test_nan_rate_metric():
    metric = NaNRateMetric()
    data = np.zeros((100, 33, 3))
    data[0:10] = np.nan
    
    result = metric.calculate(data)
    assert result['values']['nan_rate'] == 0.1
```

### Step 6: Documentation

- Add to `docs/api/sldataset.md`
- Create usage examples in `docs/examples/metrics.md`
- Update README.md

## Design Decisions

### Why Plugin Architecture?

1. **Extensibility**: Third-party developers can add custom metrics
2. **Maintainability**: Each metric is isolated in its own module
3. **Consistency**: All metrics follow the same interface
4. **Discovery**: Automatic runtime discovery via Entry Points

### Why Simulated Registry in Prototype?

- Real Entry Points require package installation
- Simulated registry allows quick iteration
- Easy to demonstrate without modifying `pyproject.toml`

### Why Ground Truth Free?

- Production environments often lack Ground Truth
- Enables real-time quality monitoring
- Suitable for comparing different estimation engines

## References

- **Design Document**: `c:\Users\ikeda\Workspace\1github\cslrtools2-dataset-2\pose_estimation_metrics_analysis.md` (parent project)
- **LMPipe Plugin System**: `src/cslrtools2/lmpipe/app/plugins.py` (similar architecture)
- **Project Instructions**: `.github/copilot-instructions.md`

## Real Data Support

The prototype supports loading real landmark data from sldataset2 format:

### Directory Structure
```
H:\SLRDataset\fs50-mp-holistic-v5\
└── 090/
    └── P2_S090_00.mp4/
        └── landmarks/
            ├── mediapipe.pose.npy       # (T, K, D) = (160, 33, 4)
            ├── mediapipe.left_hand.npy
            └── mediapipe.right_hand.npy
```

### Pattern Matching
The `load_landmark_data()` function uses flexible pattern matching:
```python
# Pattern: *.{part}.npy or *_{part}.npy
load_landmark_data(item_dir, part="pose", ext=".npy")
# Matches: mediapipe.pose.npy, openpose_pose.npy, etc.
```

### Recursive Search
Supports both direct item paths and root-level search:
```python
# Direct item path
python demo.py --dataset "H:\...\090\P2_S090_00.mp4"

# Root search (finds all items recursively)
python demo.py --dataset "H:\...\090"
```

## Test Results with Real Data

**Dataset**: `H:\SLRDataset\fs50-mp-holistic-v5\090\P2_S090_00.mp4`
**Shape**: (160 frames, 33 keypoints, 4 dimensions)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| NaN Rate | 0.00% | ✅ Perfect completeness |
| Temporal Smoothness | 0.031416 | ✅ Excellent (stable tracking) |
| Anatomical Mean Variation | 0.073624 (7.4%) | ✅ Excellent (consistent bone lengths) |

## Contributing

When adding new metrics to this prototype:

1. Create metric class in `plugins/{category}.py`
2. Inherit from `Metric` ABC
3. Implement `calculate()` and `get_description()`
4. Create plugin info tuple: `metric_info = (MetricClass, {})`
5. Register in `demo.py` for testing
6. Update this README with metric description

Example:
```python
# metrics_prototype/plugins/anatomical.py
from metrics_prototype.base import Metric, MetricResult

class BoneLengthMetric(Metric):
    def calculate(
        self,
        data: np.ndarray,
        bone_pairs: list[tuple[int, int]],
        **kwargs,
    ) -> MetricResult:
        """Calculate bone length variation coefficients."""
        variation_coefficients: list[float] = []
        
        for i, j in bone_pairs:
            bone_lengths = np.linalg.norm(
                data[:, i, :] - data[:, j, :], axis=1
            )
            mean_length = float(np.nanmean(bone_lengths))
            std_length = float(np.nanstd(bone_lengths))
            variation_coef = std_length / mean_length
            variation_coefficients.append(variation_coef)
        
        return MetricResult(
            metric_name="bone_length_variation",
            values={
                "mean_variation": float(np.mean(variation_coefficients)),
            },
            metadata={"num_bones": len(bone_pairs)},
        )
    
    def get_description(self) -> str:
        return "Bone length consistency across frames"

bone_length_info = (BoneLengthMetric, {})
```

## License

Copyright 2025 cslrtools2 contributors. Licensed under Apache License 2.0.
