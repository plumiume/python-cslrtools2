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


from __future__ import annotations

"""LMPipe: Landmark Extraction Pipeline Framework for Sign Language Videos.

**Software Type**: Processing Pipeline Framework / ETL (Extract-Transform-Load) System
**Pattern**: Pipeline Pattern, Plugin Architecture
**Dependencies**: MediaPipe, OpenCV, NumPy, Rich (for progress bars)

What This Module Does
----------------------
LMPipe (Landmark Pipeline) is a **video processing framework** that extracts skeletal
landmark data from sign language videos. It implements a complete ETL pipeline:

1. **Extract**: Read video frames from files, cameras, or image sequences
2. **Transform**: Detect landmarks (pose, hands, face) using MediaPipe
3. **Load**: Save landmarks to various formats (NPY, Zarr, SafeTensors, etc.)

Think of it as a **batch processing system** for converting raw videos into
machine-learning-ready data. It's designed to handle large-scale video datasets
with parallel processing and progress tracking.

Key Concept
-----------
**Pipeline Architecture**:
    LMPipe follows the Pipeline pattern where data flows through these
    stages::

        Video File(s) → Frame Extraction → Landmark Detection → Data Collection → Output File(s)  # noqa: E501
                      ↑                    ↑                     ↑
                      RunSpec              Estimator            Collector

    Each stage is modular and can be customized through plugins.

Core Components
---------------

1. **Estimator** (``estimator.py``)

   What it is:
       Abstract base class for landmark detection models. Defines the interface
       that all landmark detectors must implement.

   Software pattern: Strategy Pattern / Abstract Factory

   Example::

       from cslrtools2.lmpipe.estimator import Estimator

       class MyEstimator(Estimator):
           def process(self, frame):
               # Your landmark detection logic
               return ProcessResult(landmarks={...}, annotated_frame=frame)

2. **Collector** (``collector/``)

   What it is:
       Output handlers that save landmark data to various file formats.
       Implements the Collector pattern for flexible output strategies.

   Software pattern: Collector Pattern / Visitor Pattern

   Supported formats:
       - ``landmark_matrix/``: NPY, NPZ, Zarr, SafeTensors, PyTorch, JSON, CSV
       - ``annotated_frames/``: OpenCV, matplotlib, PIL, torchvision

   Example::

       from cslrtools2.lmpipe.collector import NpyLandmarkMatrixCollector
       collector = NpyLandmarkMatrixCollector(output_path="landmarks.npy")

3. **Interface** (``interface/``)

   What it is:
       Main user-facing API that orchestrates the entire pipeline.
       Provides both Python API and CLI functionality.

   Software pattern: Facade Pattern / Coordinator

   Key classes:
       - ``LMPipeInterface``: User-facing API for running pipelines
       - ``LMPipeRunner``: Internal execution coordinator (1,252 lines)

   Example::

       from cslrtools2.lmpipe.interface import LMPipeInterface

       interface = LMPipeInterface(
           estimator=my_estimator,
           collectors=[npy_collector],
           workers=4
       )
       interface.run("input_video.mp4", "output_dir/")

4. **RunSpec** (``runspec.py``)

   What it is:
       Specification object that defines a single processing job.
       Encapsulates source (video/camera) and destination (output directory).

   Software pattern: Value Object / Data Transfer Object (DTO)

   Example::

       from cslrtools2.lmpipe.runspec import RunSpec
       spec = RunSpec.from_pathlikes("input.mp4", "output/")

5. **Options** (``options.py``)

   What it is:
       Configuration management for pipeline execution (parallel vs sequential,
       executor type, etc.)

   Software pattern: Configuration Object

   Example::

       from cslrtools2.lmpipe.options import LMPipeOptions, ExecutorMode
       options = LMPipeOptions(
           mode=ExecutorMode.PARALLEL,
           workers=4,
           show_progress=True
       )

Architecture
------------

**Plugin System**:
    LMPipe supports plugins for custom estimators and dataset-specific logic.
    Plugins are registered via entry points in ``pyproject.toml``::

        [project.entry-points."cslrtools2.lmpipe.plugins"]
        "mediapipe.holistic" = "cslrtools2.plugins.mediapipe.lmpipe.holistic_args:holistic_info"  # noqa: E501
        "mediapipe.pose" = "cslrtools2.plugins.mediapipe.lmpipe.pose_args:pose_info"  # noqa: E501
        # ... more plugins

**Execution Modes**:
    - **Sequential**: Process videos one-by-one (``mode='sequential'``)
    - **Parallel**: Use multiprocessing pool (``mode='parallel'``, default)
    - **Thread-based**: For I/O-bound tasks (``mode='thread'``)

**Progress Tracking**:
    Built-in progress bars using Rich library for better user experience::

        Processing videos... ━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15

Use Cases
---------

1. **Research Dataset Preprocessing**:

   Extract landmarks from thousands of sign language videos::

       lmpipe mediapipe.holistic videos/ -o dataset.zarr --workers 8

2. **Real-time Landmark Extraction**:

   Process webcam feed::

       lmpipe mediapipe.pose 0 -o live_landmarks.npz

3. **Custom Estimator Development**:

   Build your own landmark detector::

       from cslrtools2.lmpipe import Estimator, LMPipeInterface

       class MyEstimator(Estimator):
           def process(self, frame):
               # Your logic
               return ProcessResult(...)

       interface = LMPipeInterface(MyEstimator())
       interface.run("video.mp4", "output/")

4. **Batch Video Processing**:

   Process entire directory with progress tracking::

       from cslrtools2.lmpipe.interface import LMPipeInterface
       interface = LMPipeInterface(estimator, collectors, workers=4)
       interface.run_batch("video_directory/", "output_directory/")

Command-Line Interface
----------------------

The ``lmpipe`` command provides a full-featured CLI::

    # Basic usage
    lmpipe mediapipe.holistic input.mp4 -o landmarks.npz

    # With options
    lmpipe mediapipe.pose videos/ -o output.zarr \\
        --workers 4 \\
        --format zarr \\
        --annotated-frames opencv

    # List available plugins
    lmpipe --list-plugins

For CLI details, see :mod:`cslrtools2.lmpipe.app.cli`.

Software Engineering Patterns Used
-----------------------------------

- **Pipeline Pattern**: Data flows through Extract → Transform → Load stages
- **Strategy Pattern**: Pluggable estimators (MediaPipe, custom models)
- **Collector Pattern**: Pluggable output formats
- **Facade Pattern**: LMPipeInterface simplifies complex subsystem
- **Factory Pattern**: Plugin system for creating estimators
- **Observer Pattern**: Event callbacks for monitoring pipeline progress
- **Command Pattern**: CLI commands map to pipeline operations

Performance
-----------

Typical processing speed (depends on hardware and estimator):
- **MediaPipe Holistic**: ~15-30 FPS on modern CPU
- **Parallel processing**: Linear speedup with CPU cores (up to ~8 workers)
- **Memory usage**: ~200-500MB per worker process

Optimization tips:
- Use parallel mode for multiple videos
- Choose appropriate output format (Zarr for large datasets)
- Reduce video resolution if landmark precision isn't critical

Dependencies
------------

Core:
    - opencv-python (video I/O)
    - numpy (array operations)
    - rich (progress bars, terminal UI)
    - loky (parallel processing)

Optional (for specific features):
    - mediapipe (landmark detection, separate install required)
    - torch (PyTorch format output)
    - zarr (Zarr format output)
    - safetensors (SafeTensors format output)
    - matplotlib, pillow, torchvision (annotated frame formats)

See Also
--------

- Estimator API: :mod:`cslrtools2.lmpipe.estimator`
- Collector API: :mod:`cslrtools2.lmpipe.collector`
- Interface API: :mod:`cslrtools2.lmpipe.interface`
- CLI Documentation: :mod:`cslrtools2.lmpipe.app.cli`
- MediaPipe Plugin: :mod:`cslrtools2.plugins.mediapipe.lmpipe`

Examples
--------

Basic usage with MediaPipe::

    from cslrtools2.lmpipe.interface import LMPipeInterface
    from cslrtools2.plugins.mediapipe.lmpipe import HolisticEstimator
    from cslrtools2.lmpipe.collector import NpzLandmarkMatrixCollector

    estimator = HolisticEstimator()
    collector = NpzLandmarkMatrixCollector("output.npz")

    interface = LMPipeInterface(estimator, [collector])
    interface.run("input_video.mp4", "output_directory/")

Custom estimator::

    from cslrtools2.lmpipe.estimator import Estimator, ProcessResult

    class MyEstimator(Estimator):
        def process(self, frame):
            # Your landmark detection code
            landmarks = {"keypoint_1": np.array([[x, y, z]])}
            return ProcessResult(
                landmarks=landmarks,
                annotated_frame=frame
            )

        def configure_estimator_name(self):
            return "my_estimator"

    interface = LMPipeInterface(MyEstimator(), collectors)
    interface.run_batch("videos/", "output/")

"""

# Note: This __init__.py intentionally does NOT import submodules to avoid
# loading heavy dependencies. Users should import explicitly:
#
#   from cslrtools2.lmpipe.interface import LMPipeInterface
#   from cslrtools2.lmpipe.estimator import Estimator
#   from cslrtools2.lmpipe.collector import NpyLandmarkMatrixCollector
#
# The CLI (lmpipe command) is defined as an entry point in pyproject.toml
# and routes to cslrtools2.lmpipe.app.cli.main()
