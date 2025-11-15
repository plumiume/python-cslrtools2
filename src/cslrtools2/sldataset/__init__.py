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

"""SLDataset: Sign Language Dataset Management and Loading Utilities.

**Software Type**: Data Access Layer / Data Management Module
**Pattern**: Repository Pattern, Data Transfer Object (DTO)
**Dependencies**: PyTorch, Zarr, NumPy

What This Module Does
----------------------
SLDataset (Sign Language Dataset) is a **data abstraction layer** that provides
a unified interface for managing and loading sign language datasets. It solves
the problem of efficiently storing and accessing large-scale video datasets with
associated landmark annotations and labels.

Think of it as a **database system** specifically designed for sign language data,
where:
- **Tables** = Zarr groups (metadata, connections, items)
- **Rows** = Individual sign language samples
- **Columns** = Videos, landmarks, targets (labels/glosses)

Key Concept
-----------
**Unified Schema**:
    All sign language datasets, regardless of their original format, are converted
    to a standardized schema::

        Dataset Root (Zarr Group)
        ├── metadata/           # Dataset-level information (name, version, etc.)
        ├── connections/        # Landmark connectivity graphs
        └── items/              # Individual samples
            ├── [0]/
            │   ├── videos/     # Video data for sample 0
            │   │   └── {video_key}: video data
            │   ├── landmarks/  # Landmark data
            │   │   └── {landmark_key}: landmark arrays
            │   └── targets/    # Labels/glosses
            │       └── {target_key}: target data
            ├── [1]/
            └── ...

This design enables:
- **Efficient storage**: Compressed, chunked arrays (Zarr)
- **Random access**: Load individual samples without reading entire dataset
- **PyTorch compatibility**: Direct integration with DataLoader
- **Flexibility**: Support custom metadata keys for different datasets

Core Components
---------------

1. **SLDatasetItem** (``dataset.py``)

   What it is:
       Data Transfer Object (DTO) representing a single sign language sample.
       Contains videos, landmarks, and target labels for one sign instance.

   Software pattern: Value Object / Data Transfer Object (DTO)

   Generic type parameters:
       - ``Kvid``: Video key type (e.g., "rgb", "depth")
       - ``Vvid``: Video value type (e.g., Tensor, Array)
       - ``Klm``: Landmark key type (e.g., "pose", "left_hand")
       - ``Vlm``: Landmark value type (e.g., Tensor, NDArray)
       - ``Ktgt``: Target key type (e.g., "gloss", "label")
       - ``Vtgt``: Target value type (e.g., Tensor, int)

   Example::

       from cslrtools2.sldataset import SLDatasetItem

       item = SLDatasetItem(
           videos={"rgb": video_tensor},
           landmarks={"pose": pose_array, "left_hand": hand_array},
           targets={"gloss": gloss_id}
       )

2. **SLDataset** (``dataset.py``)

   What it is:
       PyTorch-compatible Dataset class for sign language data. Implements
       the Repository pattern for data access.

   Software pattern: Repository Pattern / PyTorch Dataset

   Key features:
       - Random access: ``dataset[index]`` → ``SLDatasetItem``
       - Metadata storage: Dataset-level information
       - Connection graphs: Landmark connectivity for visualization
       - Zarr backend: Efficient storage and loading

   Example::

       from cslrtools2.sldataset import SLDataset
       from torch.utils.data import DataLoader

       # Load from Zarr
       import zarr
       root = zarr.open("dataset.zarr", mode="r")
       dataset = SLDataset.from_zarr(root)

       # Use with PyTorch
       loader = DataLoader(dataset, batch_size=32, shuffle=True)
       for batch in loader:
           videos = batch.videos
           landmarks = batch.landmarks
           targets = batch.targets

3. **IterableSLDataset** (``dataset.py``)

   What it is:
       PyTorch IterableDataset variant for streaming large datasets that don't
       fit in memory. Useful for very large sign language corpora.

   Software pattern: Iterator Pattern / Stream Processing

   Example::

       from cslrtools2.sldataset import IterableSLDataset

       dataset = IterableSLDataset.from_zarr(zarr_root)
       for item in dataset:
           process(item)

4. **Array Loaders** (``array_loader.py``)

   What it is:
       Factory system for loading various array formats (NPY, NPZ, Zarr,
       SafeTensors, PyTorch). Implements the Strategy pattern.

   Software pattern: Factory Pattern / Strategy Pattern

   Supported formats:
       - NPY: NumPy single array (``.npy``)
       - NPZ: NumPy compressed archive (``.npz``)
       - Zarr: Chunked, compressed arrays (``.zarr``)
       - SafeTensors: Fast tensor serialization (``.safetensors``)
       - PyTorch: PyTorch tensor files (``.pt``, ``.pth``)

   Example::

       from cslrtools2.sldataset.array_loader import load_array

       # Automatically detects format
       data = load_array("landmarks.npy")
       data = load_array("features.safetensors")

5. **Plugins** (``../plugins/``)

   What it is:
       Dataset-specific adapters for popular sign language datasets.
       Converts original formats to the unified SLDataset schema.

   Software pattern: Adapter Pattern / Plugin Architecture

   Available plugins:
       - FluentSigners50: Adapter for FluentSigners-50 dataset
       - (More can be added via plugin system)

   Example::

       # Plugin automatically handles dataset-specific structure
       from cslrtools2.plugins.fluentsigners50.sldataset import load_fluentsigners50
       dataset = load_fluentsigners50("path/to/fs50")

Use Cases
---------

1. **Loading Preprocessed Datasets**:

   After using LMPipe to extract landmarks, load them for training::

       from cslrtools2.sldataset import SLDataset
       import zarr

       root = zarr.open("preprocessed_dataset.zarr", mode="r")
       dataset = SLDataset.from_zarr(root)
       print(f"Dataset size: {len(dataset)} samples")

2. **PyTorch Training Loop**:

   Integrate with PyTorch for model training::

       from torch.utils.data import DataLoader
       from cslrtools2.sldataset import SLDataset

       dataset = SLDataset.from_zarr(zarr.open("data.zarr"))
       loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

       for epoch in range(num_epochs):
           for batch in loader:
               landmarks = batch.landmarks["pose"]  # Shape: [B, T, N, 3]
               targets = batch.targets["gloss"]
               # Training logic...

3. **Dataset Conversion**:

   Convert from file system to Zarr format::

       from cslrtools2.sldataset import SLDatasetItem
       import zarr

       root = zarr.open("output.zarr", mode="w")
       items = []

       for sample_path in dataset_paths:
           item = SLDatasetItem.from_file_system(sample_path)
           items.append(item)

       # Save to Zarr
       from cslrtools2.sldataset import SLDataset
       dataset = SLDataset(metadata={...}, connections={...}, items=items)
       dataset.to_zarr(root)

4. **Custom Collate Function**:

   Handle variable-length sequences::

       from cslrtools2.sldataset import collate_to_padded_batch

       loader = DataLoader(
           dataset,
           batch_size=32,
           collate_fn=collate_to_padded_batch
       )

Architecture
------------

**Storage Backend**:
    Uses Zarr for efficient array storage:
    - **Chunked**: Only load needed portions of data
    - **Compressed**: Reduce disk space (typically 5-10x compression)
    - **Cloud-ready**: Can store on S3, GCS, or local filesystem

**Type Safety**:
    Fully typed with generic type parameters for compile-time safety::

        SLDataset[
            Kmeta: str,        # Metadata keys
            Kvid: str,         # Video keys
            Klm: str,          # Landmark keys
            Ktgt: str,         # Target keys
            V: Tensor | Array  # Value type
        ]

**Plugin System**:
    Extensible architecture for adding new dataset formats::

        [project.entry-points."cslrtools2.sldataset.plugins"]
        "fluentsigners50" = "cslrtools2.plugins.fluentsigners50.sldataset:plugin_info"

Performance
-----------

Storage efficiency:
- **Zarr compression**: 5-10x smaller than uncompressed NPY
- **Random access**: <10ms per sample (SSD)
- **Parallel loading**: PyTorch DataLoader num_workers support

Memory usage:
- **Lazy loading**: Only load requested samples
- **Chunked arrays**: Partial array loading for large samples

Command-Line Interface
----------------------

The ``sldataset2`` command provides dataset management tools::

    # List dataset contents
    sldataset2 info dataset.zarr

    # Convert format
    sldataset2 convert --from npy --to zarr input/ output.zarr

    # Validate dataset
    sldataset2 validate dataset.zarr

For CLI details, see :mod:`cslrtools2.sldataset.app.cli`.

Software Engineering Patterns Used
-----------------------------------

- **Repository Pattern**: SLDataset abstracts data access
- **Data Transfer Object (DTO)**: SLDatasetItem encapsulates sample data
- **Factory Pattern**: Array loaders for different formats
- **Strategy Pattern**: Pluggable storage backends
- **Adapter Pattern**: Dataset-specific plugins
- **Iterator Pattern**: IterableSLDataset for streaming
- **Template Method**: Base classes for custom datasets

Dependencies
------------

Core:
    - torch (PyTorch Dataset compatibility)
    - zarr (efficient array storage)
    - numpy (array operations)

Optional:
    - safetensors (SafeTensors format)
    - Various dataset-specific dependencies (via plugins)

See Also
--------

- Dataset API: :mod:`cslrtools2.sldataset.dataset`
- Array Loaders: :mod:`cslrtools2.sldataset.array_loader`
- Utilities: :mod:`cslrtools2.sldataset.utils`
- CLI: :mod:`cslrtools2.sldataset.app.cli`
- FluentSigners50 Plugin: :mod:`cslrtools2.plugins.fluentsigners50.sldataset`

Examples
--------

Basic dataset loading::

    import zarr
    from cslrtools2.sldataset import SLDataset

    root = zarr.open("my_dataset.zarr", mode="r")
    dataset = SLDataset.from_zarr(root)

    print(f"Dataset size: {len(dataset)}")
    print(f"Metadata: {dataset.metadata}")

    # Access a sample
    item = dataset[0]
    print(f"Videos: {item.videos.keys()}")
    print(f"Landmarks: {item.landmarks.keys()}")
    print(f"Targets: {item.targets.keys()}")

PyTorch integration::

    from torch.utils.data import DataLoader
    from cslrtools2.sldataset import SLDataset

    dataset = SLDataset.from_zarr(zarr.open("data.zarr"))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in loader:
        # Process batch
        pass

Creating a custom dataset::

    from cslrtools2.sldataset import SLDataset, SLDatasetItem

    items = [
        SLDatasetItem(
            videos={"rgb": video_data},
            landmarks={"pose": landmarks},
            targets={"label": label}
        )
        for video_data, landmarks, label in my_data
    ]

    dataset = SLDataset(
        metadata={"name": "MyDataset", "version": "1.0"},
        connections={},
        items=items
    )

    # Save to Zarr
    import zarr
    root = zarr.open("my_dataset.zarr", mode="w")
    dataset.to_zarr(root)

"""

# Note: This __init__.py intentionally does NOT import submodules to avoid
# loading heavy dependencies (torch, zarr). Users should import explicitly:
#
#   from cslrtools2.sldataset.dataset import SLDataset, SLDatasetItem
#   from cslrtools2.sldataset.array_loader import load_array
#
# The CLI (sldataset2 command) is defined as an entry point in pyproject.toml
# and routes to cslrtools2.sldataset.app.cli.main()
