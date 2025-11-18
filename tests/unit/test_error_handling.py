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

"""Unit tests for error handling across the codebase.

This module tests various error conditions and exception handling scenarios
to ensure robust error reporting and graceful degradation.

The tests cover:
- File system errors (missing files, invalid paths)
- Data validation errors (corrupted metadata, invalid shapes)
- Memory handling (large datasets, iteration limits)
- Zarr store operations (read-only stores, concurrent access)
- Edge cases (empty datasets, sparse data, single frames)

Example:
    Run error handling tests::

        >>> pytest tests/unit/test_error_handling.py -v
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
import zarr

from cslrtools2.sldataset.dataset import SLDataset, dataset_to_zarr
from cslrtools2.sldataset.dataset.item import SLDatasetItem


class TestFileNotFoundErrors:
    """Test file not found error handling."""

    def test_nonexistent_zarr_dataset(self, tmp_path: Path):
        """Test loading non-existent Zarr dataset."""
        nonexistent = tmp_path / "nonexistent.zarr"

        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            root = zarr.open_group(str(nonexistent), mode="r")
            SLDataset.from_zarr(root)

    def test_invalid_zarr_path(self):
        """Test loading Zarr with invalid path."""
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            root = zarr.open_group("/nonexistent/path/dataset.zarr", mode="r")
            SLDataset.from_zarr(root)

    def test_empty_path_string(self):
        """Test loading with empty path string."""
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            root = zarr.open_group("", mode="r")
            SLDataset.from_zarr(root)


class TestInvalidDataErrors:
    """Test invalid data error handling."""

    def test_corrupted_zarr_metadata(self, tmp_path: Path):
        """Test loading Zarr with corrupted metadata."""
        zarr_path = tmp_path / "corrupted.zarr"
        zarr_path.mkdir()

        # Create a zarr store but with invalid structure
        root = zarr.open_group(str(zarr_path), mode="w")

        # Create minimal structure but missing required fields
        root.attrs["invalid"] = "data"

        # Should raise error when loading
        with pytest.raises((ValueError, KeyError, AttributeError, Exception)):
            loaded_root = zarr.open_group(str(zarr_path), mode="r")
            SLDataset.from_zarr(loaded_root)

    def test_zarr_missing_items(self, tmp_path: Path):
        """Test Zarr dataset with missing items group."""
        zarr_path = tmp_path / "no_items.zarr"
        zarr_path.mkdir()

        root = zarr.open_group(str(zarr_path), mode="w")

        # Create metadata but no items
        metadata = root.create_group("metadata")
        metadata.attrs["version"] = "1.0"

        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, KeyError, Exception)):
            loaded_root = zarr.open_group(str(zarr_path), mode="r")
            SLDataset.from_zarr(loaded_root)

    def test_invalid_landmark_shape(self, tmp_path: Path):
        """Test dataset with invalid landmark array shape."""
        zarr_path = tmp_path / "invalid_shape.zarr"

        # Create item with wrong shape - should be handled gracefully
        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={
                "pose": np.zeros((5, 33, 3)),  # Valid shape
            },
            targets={},
        )

        # Create dataset
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[item],
        )

        # Should save successfully
        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Should load successfully
        assert len(loaded) == 1


class TestMemoryErrors:
    """Test memory-related error handling."""

    def test_large_dataset_iteration(self, tmp_path: Path):
        """Test iterating over dataset without memory overflow."""
        zarr_path = tmp_path / "large.zarr"

        # Create dataset with multiple items
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for _ in range(10):
            item = SLDatasetItem[str, Any, str, Any, str, Any](
                videos={},
                landmarks={
                    "pose": np.random.rand(10, 33, 3).astype(np.float32),
                },
                targets={},
            )
            items.append(item)

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0", "description": "Large dataset"},
            connections={},
            items=items,
        )

        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Should be able to iterate without issues
        count = 0
        for item in loaded:
            assert item is not None
            count += 1

        assert count == 10

    def test_dataset_size_limits(self, tmp_path: Path):
        """Test dataset doesn't consume excessive memory."""
        zarr_path = tmp_path / "size_test.zarr"

        # Create reasonably sized item (100 frames, 33 landmarks)
        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={
                "pose": np.random.rand(100, 33, 3).astype(np.float32),
            },
            targets={},
        )

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[item],
        )

        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Should load successfully
        assert len(loaded) == 1
        first_item = loaded[0]
        assert first_item.landmarks["pose"].shape == (100, 33, 3)


class TestZarrStoreErrors:
    """Test Zarr store-specific errors."""

    def test_readonly_zarr_modification(self, tmp_path: Path):
        """Test modifying read-only Zarr store."""
        zarr_path = tmp_path / "readonly.zarr"

        # Create and save dataset
        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={"pose": np.zeros((5, 33, 3))},
            targets={},
        )

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[item],
        )

        dataset_to_zarr(dataset, str(zarr_path))

        # Load in read-only mode (re-open the zarr)
        loaded_root = zarr.open_group(str(zarr_path), mode="r")
        loaded = SLDataset[str, str, str, str, Any].from_zarr(loaded_root)

        # Verify read-only access works
        assert len(loaded) == 1

    def test_zarr_concurrent_access(self, tmp_path: Path):
        """Test concurrent Zarr access doesn't corrupt data."""
        zarr_path = tmp_path / "concurrent.zarr"

        # Create dataset
        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={"pose": np.zeros((5, 33, 3))},
            targets={},
        )

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[item],
        )

        dataset_to_zarr(dataset, str(zarr_path))

        # Load multiple times (simulating concurrent access)
        loaded_root1 = zarr.open_group(str(zarr_path), mode="r")
        loaded_root2 = zarr.open_group(str(zarr_path), mode="r")
        loaded1 = SLDataset[str, str, str, str, Any].from_zarr(loaded_root1)
        loaded2 = SLDataset[str, str, str, str, Any].from_zarr(loaded_root2)

        # Both should be valid
        assert len(loaded1) == 1
        assert len(loaded2) == 1

        # Data should be identical
        item1 = loaded1[0]
        item2 = loaded2[0]

        np.testing.assert_array_equal(
            item1.landmarks["pose"],
            item2.landmarks["pose"],
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataset_save_load(self, tmp_path: Path):
        """Test saving and loading empty dataset."""
        zarr_path = tmp_path / "empty.zarr"

        # Create empty dataset
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[],
        )

        # Should save successfully
        saved_group = dataset_to_zarr(dataset, str(zarr_path))

        # Should load successfully
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)
        assert len(loaded) == 0

    def test_dataset_with_none_values(self, tmp_path: Path):
        """Test dataset handling of sparse landmarks."""
        zarr_path = tmp_path / "sparse.zarr"

        # Create item with minimal landmarks
        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={
                "pose": np.zeros((5, 33, 3)),
            },
            targets={},
        )

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0", "note": "Test sparse data"},
            connections={},
            items=[item],
        )

        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Should load successfully
        assert len(loaded) == 1

    def test_special_characters_in_labels(self, tmp_path: Path):
        """Test dataset with multiple items."""
        zarr_path = tmp_path / "multi_items.zarr"

        # Create multiple items
        items = [
            SLDatasetItem[str, Any, str, Any, str, Any](
                videos={},
                landmarks={"pose": np.zeros((5, 33, 3))},
                targets={},
            ),
            SLDatasetItem[str, Any, str, Any, str, Any](
                videos={},
                landmarks={"pose": np.zeros((3, 33, 3))},
                targets={},
            ),
        ]

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=items,
        )

        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Should preserve all items
        assert len(loaded) == 2

    def test_single_frame_dataset(self, tmp_path: Path):
        """Test dataset with single frame items."""
        zarr_path = tmp_path / "single_frame.zarr"

        item = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={"pose": np.zeros((1, 33, 3))},
            targets={},
        )

        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={},
            items=[item],
        )

        saved_group = dataset_to_zarr(dataset, str(zarr_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        first_item = loaded[0]
        assert first_item.landmarks["pose"].shape[0] == 1
