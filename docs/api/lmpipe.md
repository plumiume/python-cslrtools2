# LMPipe API Reference

## Overview

The `lmpipe` package provides a pipeline for extracting landmarks from sign language videos using various estimators (primarily MediaPipe).

## Core Components

### LMPipeInterface

High-level interface for running landmark extraction pipelines.

```python
from cslrtools2.lmpipe import LMPipeInterface
```

#### Constructor

```python
LMPipeInterface(
    estimator: Estimator,
    *,
    workers: int = 1,
    verbose: bool = True
)
```

**Parameters:**
- `estimator` (Estimator): Landmark estimator instance
- `workers` (int): Number of parallel workers (default: 1)
- `verbose` (bool): Enable progress display (default: True)

#### Methods

##### run_single_video

Process a single video file.

```python
interface.run_single_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    collector_type: str = "npz"
) -> None
```

**Parameters:**
- `video_path`: Path to input video
- `output_path`: Path to save landmarks
- `collector_type`: Output format ("npy", "npz", "zarr", "safetensors", "torch", "json", "csv")

**Example:**

```python
from cslrtools2.lmpipe import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

estimator = MediaPipeHolistic()
interface = LMPipeInterface(estimator)
interface.run_single_video("video.mp4", "landmarks.npz")
```

##### run_directory

Process all videos in a directory.

```python
interface.run_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    collector_type: str = "npz",
    pattern: str = "*.mp4"
) -> None
```

**Parameters:**
- `input_dir`: Directory containing videos
- `output_dir`: Directory to save landmarks
- `collector_type`: Output format
- `pattern`: Glob pattern for video files (default: "*.mp4")

---

## Estimator Base Class

### Estimator

Abstract base class for landmark estimators.

```python
from cslrtools2.lmpipe import Estimator
```

#### Methods

##### estimate

Process a single frame and return landmarks.

```python
def estimate(
    self,
    image: np.ndarray
) -> ProcessResult | None
```

**Parameters:**
- `image` (np.ndarray): Input image (H, W, C) in BGR format

**Returns:**
- `ProcessResult | None`: Landmark data or None if detection failed

---

## MediaPipe Estimators

### MediaPipeHolistic

Extract pose, hands, and face landmarks.

```python
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

estimator = MediaPipeHolistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Landmarks:**
- Pose: 33 landmarks
- Left hand: 21 landmarks
- Right hand: 21 landmarks
- Face: 468 landmarks

### MediaPipePose

Extract only pose landmarks.

```python
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipePose

estimator = MediaPipePose()
```

### MediaPipeHands

Extract hand landmarks (left, right, or both).

```python
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHands

# Both hands
estimator = MediaPipeHands(mode="both")

# Left hand only
estimator = MediaPipeHands(mode="left")

# Right hand only
estimator = MediaPipeHands(mode="right")
```

### MediaPipeFace

Extract face landmarks.

```python
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeFace

estimator = MediaPipeFace()
```

---

## Collectors

Collectors handle landmark output in various formats.

### Available Collectors

- **NPYCollector**: NumPy `.npy` format
- **NPZCollector**: Compressed NumPy `.npz` format
- **ZarrCollector**: Zarr array format
- **SafeTensorsCollector**: SafeTensors format
- **TorchCollector**: PyTorch `.pt` format
- **JSONCollector**: JSON format
- **CSVCollector**: CSV format

### Usage

Collectors are automatically selected based on `collector_type`:

```python
interface.run_single_video(
    "video.mp4",
    "output.npz",
    collector_type="npz"  # Selects NPZCollector
)
```

---

## CLI Usage

### Basic Command

```bash
lmpipe mediapipe.holistic input.mp4 -o output.npz
```

### Options

- `--workers N`: Number of parallel workers
- `--output-format FORMAT`: Output format (npz, npy, zarr, etc.)
- `--min-detection-confidence CONF`: Detection confidence threshold
- `--min-tracking-confidence CONF`: Tracking confidence threshold

### Examples

```bash
# Extract holistic landmarks with 4 workers
lmpipe mediapipe.holistic videos/ -o landmarks/ --workers 4

# Extract pose only, save as Zarr
lmpipe mediapipe.pose video.mp4 -o pose.zarr --output-format zarr

# Process directory with custom confidence
lmpipe mediapipe.holistic videos/ -o output/ \
    --min-detection-confidence 0.7 \
    --min-tracking-confidence 0.7
```

---

## Type Definitions

### ProcessResult

```python
from cslrtools2.lmpipe import ProcessResult

@dataclass
class ProcessResult:
    landmarks: dict[str, np.ndarray]  # Landmark arrays
    metadata: dict[str, Any]          # Additional metadata
```

---

## Navigation

- **[← API Reference](index.md)**
- **Next**: [SLDataset API](sldataset.md)
