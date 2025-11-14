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

"""cslrtools2: Comprehensive toolkit for Continuous Sign Language Recognition.

**Software Type**: Research Library / Toolkit Package  
**Domain**: Computer Vision, Sign Language Recognition, Deep Learning  
**License**: Apache 2.0

What This Package Does
----------------------
cslrtools2 is a **research-oriented library** that provides end-to-end tools for 
analyzing sign language videos using computer vision and deep learning techniques. 
It bridges the gap between raw video data and machine learning models by offering:

1. **Data Preprocessing**: Extract skeletal landmarks from sign language videos
2. **Data Management**: Efficiently store and load large-scale sign language datasets
3. **Model Building**: Helper utilities for constructing PyTorch neural networks

The package is designed for researchers, data scientists, and developers working on:
- Continuous Sign Language Recognition (CSLR) systems
- Sign language video analysis and preprocessing
- Gesture recognition and human pose estimation research

Main Components
---------------

1. **lmpipe** - Landmark Extraction Pipeline (Framework/Processing Pipeline)
   
   What it does:
       Converts sign language videos into structured landmark data that machine 
       learning models can process. Think of it as a preprocessing pipeline that 
       transforms raw videos into numerical representations.
   
   Key features:
       - Parallel video processing with progress tracking
       - MediaPipe integration for pose, hand, and face detection
       - Multiple output formats: NPY, NPZ, Zarr, SafeTensors, PyTorch, JSON, CSV
       - Batch processing for large video datasets
       - Annotated frame visualization
   
   Command-line tool: ``lmpipe``
   
   Example::
   
       # Extract landmarks from a video
       lmpipe mediapipe.holistic input_video.mp4 -o landmarks.npz
       
       # Process entire directory with 4 workers
       lmpipe mediapipe.pose videos/ -o output.zarr --workers 4

2. **sldataset** - Sign Language Dataset Management (Data Layer/Utility Module)
   
   What it does:
       Provides a unified interface for managing sign language datasets with 
       efficient storage and PyTorch integration. It's a data abstraction layer 
       that handles the complexity of storing videos, landmarks, and labels together.
   
   Key features:
       - Zarr-based storage for large arrays (efficient, compressed, chunked)
       - PyTorch ``Dataset`` and ``IterableDataset`` compatibility
       - Flexible schema supporting videos, landmarks, and arbitrary targets
       - Plugin architecture for specific datasets (e.g., FluentSigners50)
       - Array loader system for various file formats
   
   Command-line tool: ``sldataset2``
   
   Example::
   
       # Load a dataset in Python
       from cslrtools2.sldataset import SLDataset
       dataset = SLDataset.from_zarr(zarr_group)
       
       # Use with PyTorch DataLoader
       from torch.utils.data import DataLoader
       loader = DataLoader(dataset, batch_size=32, shuffle=True)

3. **convsize** - PyTorch Convolution Utilities (Helper/Utility Module)
   
   What it does:
       Calculates output tensor dimensions for PyTorch convolutional layers. 
       Useful when designing neural network architectures to avoid shape mismatch 
       errors and understand how tensor sizes change through the network.
   
   Key features:
       - Output size calculation for Conv1d, Conv2d, Conv3d
       - Pooling layer size calculation
       - Transpose convolution (deconvolution) support
       - Layer-by-layer shape tracking
   
   Example::
   
       from cslrtools2.convsize import conv_size
       output_h, output_w = conv_size(
           input_h=224, input_w=224,
           kernel=3, stride=2, padding=1
       )

Software Architecture
---------------------

**Package Type**: Namespace Package (Single Top-Level)

    Python packaging best practices require a single top-level package. This 
    project follows a modular architecture with clear separation of concerns:
    
    - ``cslrtools2/`` (root package - minimal imports)
      - ``lmpipe/`` (processing framework)
      - ``sldataset/`` (data management)
      - ``convsize.py`` (utilities)
      - ``plugins/`` (extensible components)

**Design Philosophy**: Lazy Loading

    This ``__init__.py`` intentionally keeps imports minimal to avoid loading 
    heavy dependencies (torch, mediapipe, opencv, zarr) at package import time. 
    Each submodule is designed to be imported explicitly when needed.

Usage Patterns
--------------

Recommended import style to avoid unnecessary dependency loading::

    # ✅ Lightweight - only loads version metadata (~37ms)
    import cslrtools2
    print(cslrtools2.__version__)
    
    # ✅ Explicit submodule import - loads only what's needed
    from cslrtools2.convsize import conv_size
    from cslrtools2.sldataset import SLDataset
    from cslrtools2.lmpipe.interface import LMPipeInterface
    
    # ❌ Avoid wildcard imports - loads everything
    from cslrtools2 import *  # Don't do this

Command-Line Interface
----------------------

Two CLI applications are provided as console scripts (defined in ``pyproject.toml``):

1. ``lmpipe`` → :func:`cslrtools2.lmpipe.app.cli.main`
2. ``sldataset2`` → :func:`cslrtools2.sldataset.app.cli.main`

These are separate entry points that load their respective heavy dependencies 
only when the command is executed.

Target Audience
---------------

- **Researchers**: Working on sign language recognition, gesture analysis
- **Data Scientists**: Preprocessing sign language video datasets
- **ML Engineers**: Building CSLR models with PyTorch
- **Students**: Learning computer vision and deep learning

Quick Links
-----------

- **Documentation**: https://ikegami-yukino.github.io/python-cslrtools2/
- **Repository**: https://github.com/ikegami-yukino/python-cslrtools2
- **Issues**: https://github.com/ikegami-yukino/python-cslrtools2/issues
- **PyPI**: https://pypi.org/project/cslrtools2/

Attributes:
    __version__ (str): Package version string (e.g., "0.1.0")

See Also:
    - LMPipe API: :mod:`cslrtools2.lmpipe`
    - SLDataset API: :mod:`cslrtools2.sldataset`
    - CLI Documentation: :mod:`cslrtools2.lmpipe.app.cli`, :mod:`cslrtools2.sldataset.app.cli`

"""

# Get version from package metadata
try:
    from ._version import __version__
except ImportError:
    # Fallback during development or when _version.py doesn't exist yet
    try:
        from importlib.metadata import version
        __version__ = version("cslrtools2")
    except Exception:
        __version__ = "0.0.0+unknown"

# Import exceptions for re-export (lightweight, no heavy dependencies)
from .exceptions import (
    CSLRToolsError,
    ConfigurationError,
    ValidationError,
    LMPipeError,
    SLDatasetError,
)

# Public API: Only expose lightweight, commonly-used symbols
# Heavy imports (torch, cv2, mediapipe, etc.) are NOT imported here
__all__ = [
    "__version__",
    # Exceptions (re-exported for convenience)
    "CSLRToolsError",
    "ConfigurationError",
    "ValidationError",
    "LMPipeError",
    "SLDatasetError",
    # Subpackages are accessible via explicit import:
    # - from cslrtools2 import lmpipe
    # - from cslrtools2 import sldataset
    # - from cslrtools2 import convsize
]

# Note: We do NOT eagerly import submodules here to keep import time fast.
# Users should explicitly import what they need:
#   from cslrtools2.convsize import conv_size  # Lightweight
#   from cslrtools2.sldataset import SLDataset # Medium weight (torch, zarr)
#   from cslrtools2.lmpipe import ...          # Heavy (mediapipe, cv2)
