"""Integration tests for SLDataset workflow.

Tests complete workflows including:
- Dataset creation from scratch
- PyTorch DataLoader integration
- Zarr storage round-trip
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]

from cslrtools2.sldataset.dataset import SLDataset, dataset_to_zarr
from cslrtools2.sldataset.dataset.item import SLDatasetItem


def identity_collate(
    batch: list[SLDatasetItem[str, Any, str, Any, str, Any]]
):
    return batch


class TestSLDatasetCreation:
    """Test SLDataset creation from scratch."""

    def test_create_empty_dataset(self, tmp_path: Path):
        """Test creating an empty SLDataset."""
        # Create empty dataset
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"name": "Empty Dataset"},
            connections={},
            items=[],
        )

        # Verify dataset structure
        assert len(dataset) == 0
        assert "name" in dataset.metadata

        # Save to Zarr
        dataset_path = tmp_path / "empty_dataset.zarr"
        saved_group = dataset_to_zarr(dataset, str(dataset_path))

        # Verify Zarr structure
        assert dataset_path.exists()
        assert dataset_path.is_dir()
        assert "metadata" in saved_group  # type: ignore[operator]
        assert "items" in saved_group  # type: ignore[operator]

    def test_create_dataset_with_metadata(self, tmp_path: Path):
        """Test creating a dataset with custom metadata."""
        metadata = {
            "name": "Test Dataset",
            "version": "1.0",
            "description": "Integration test dataset",
            "language": "JSL",
        }

        dataset = SLDataset[str, str, str, str, Any](
            metadata=metadata,
            connections={},
            items=[],
        )

        # Verify metadata
        assert dataset.metadata["name"] == "Test Dataset"
        assert dataset.metadata["version"] == "1.0"
        assert dataset.metadata["language"] == "JSL"

    def test_add_items_to_dataset(self, tmp_path: Path):
        """Test adding landmark items to dataset."""
        # Create items
        item1 = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={
                "pose": np.random.randn(10, 33, 3).astype(np.float32),
                "hand": np.random.randn(10, 21, 2).astype(np.float32),
            },
            targets={"label": np.array([[0]], dtype=np.int64)},
        )

        item2 = SLDatasetItem[str, Any, str, Any, str, Any](
            videos={},
            landmarks={
                "pose": np.random.randn(15, 33, 3).astype(np.float32),
                "hand": np.random.randn(15, 21, 2).astype(np.float32),
            },
            targets={"label": np.array([[1]], dtype=np.int64)},
        )

        # Create dataset with items
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"name": "Multi-item Dataset"},
            connections={},
            items=[item1, item2],
        )

        # Verify dataset size
        assert len(dataset) == 2

        # Verify first item
        retrieved_item1 = dataset[0]
        assert retrieved_item1.landmarks["pose"].shape == (10, 33, 3)
        assert retrieved_item1.landmarks["hand"].shape == (10, 21, 2)
        assert retrieved_item1.targets["label"][0][0] == 0

        # Verify second item
        retrieved_item2 = dataset[1]
        assert retrieved_item2.landmarks["pose"].shape == (15, 33, 3)
        assert retrieved_item2.targets["label"][0][0] == 1


class TestSLDatasetZarrRoundTrip:
    """Test Zarr storage round-trip for SLDataset."""

    def test_save_and_load_dataset(self, tmp_path: Path):
        """Test saving and loading a complete dataset."""
        dataset_path = tmp_path / "roundtrip_dataset.zarr"

        # Create items with varying frame counts
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(5):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={
                    "pose": np.random.randn(10 + i * 2, 33, 3).astype(np.float32),
                },
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        # Create and save dataset
        original_dataset = SLDataset(
            metadata={"test": "round-trip", "items_count": 5},
            connections={("pose", "pose"): np.array([[0, 1], [1, 2]])},
            items=items,
        )

        saved_group = dataset_to_zarr(original_dataset, str(dataset_path))

        # Load dataset
        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Verify dataset properties
        assert len(loaded_dataset) == 5
        assert loaded_dataset.metadata["test"] == "round-trip"
        assert ("pose", "pose") in loaded_dataset.connections

        # Verify items - check labels to identify items
        labels_found: set[int] = set()
        for i in range(5):
            item = loaded_dataset[i]
            label = item.targets["label"][0][0]
            labels_found.add(label)

            # Verify correct frame count for each label
            expected_frames = 10 + label * 2
            assert item.landmarks["pose"].shape[0] == expected_frames

        # Verify all labels present
        assert labels_found == {0, 1, 2, 3, 4}

    def test_dataset_persistence_with_connections(self, tmp_path: Path) -> None:
        """Test that connections and metadata persist after save/load."""
        dataset_path = tmp_path / "persistent_dataset.zarr"

        # Define connection graph
        connections = {
            ("pose", "left_hand"): np.array([[0, 1], [1, 2]], dtype=np.int32),
            ("pose", "right_hand"): np.array([[0, 3], [3, 4]], dtype=np.int32),
        }

        # Create item
        item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
            videos={},
            landmarks={
                "pose": np.ones((5, 33, 3), dtype=np.float32),
                "left_hand": np.ones((5, 21, 2), dtype=np.float32),
                "right_hand": np.ones((5, 21, 2), dtype=np.float32),
            },
            targets={"label": np.array([[0]], dtype=np.int64)},
        )

        # Save dataset
        dataset = SLDataset(
            metadata={"version": "1.0", "fps": 30},
            connections=connections,
            items=[item],
        )
        saved_group = dataset_to_zarr(dataset, str(dataset_path))

        # Load and verify
        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(saved_group)
        assert len(loaded_dataset) == 1
        assert loaded_dataset.metadata["version"] == "1.0"
        assert loaded_dataset.metadata["fps"] == 30
        assert ("pose", "left_hand") in loaded_dataset.connections
        assert ("pose", "right_hand") in loaded_dataset.connections

        # Verify connection data
        pose_left = loaded_dataset.connections[("pose", "left_hand")]
        np.testing.assert_array_equal(pose_left, [[0, 1], [1, 2]])


class TestPyTorchDataLoaderIntegration:
    """Test PyTorch DataLoader integration with SLDataset."""

    def test_dataloader_basic_iteration(self, tmp_path: Path) -> None:
        """Test basic DataLoader iteration over dataset."""
        pytest.importorskip("torch", reason="PyTorch not installed")
        from torch.utils.data import DataLoader

        # Create dataset with 10 items
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(10):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"pose": np.random.randn(5, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "DataLoader Test"},
            connections={},
            items=items,
        )

        # Create DataLoader with custom collate_fn that returns list
        dataloader = DataLoader(
            dataset, batch_size=3, shuffle=False, collate_fn=identity_collate
        )

        # Verify iteration
        batch_count = 0
        total_items = 0
        for batch in dataloader:
            batch_count += 1
            # Each batch should be a list of SLDatasetItem objects
            total_items += len(batch)
            assert len(batch) <= 3  # batch_size

            # Verify item structure
            for item in batch:
                assert hasattr(item, "landmarks")
                assert hasattr(item, "targets")
                assert "pose" in item.landmarks

        assert total_items == 10
        assert batch_count == 4  # 3 + 3 + 3 + 1

    def test_dataloader_with_shuffle(self, tmp_path: Path) -> None:
        """Test DataLoader with shuffling."""
        pytest.importorskip("torch", reason="PyTorch not installed")
        from torch.utils.data import DataLoader

        # Create dataset with labeled items
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(20):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"hand": np.ones((3, 21, 2), dtype=np.float32) * i},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Shuffle Test"},
            connections={},
            items=items,
        )

        # Create two DataLoaders with different random seeds
        dataloader1 = DataLoader(
            dataset, batch_size=5, shuffle=True, collate_fn=identity_collate
        )
        dataloader2 = DataLoader(
            dataset, batch_size=5, shuffle=True, collate_fn=identity_collate
        )

        # Get labels from both iterations
        labels1: list[int] = []
        for batch in dataloader1:
            for item in batch:
                labels1.append(item.targets["label"][0][0])

        labels2: list[int] = []
        for batch in dataloader2:
            for item in batch:
                labels2.append(item.targets["label"][0][0])

        # Both should have all items
        assert len(labels1) == 20
        assert len(labels2) == 20
        assert set(labels1) == set(labels2)  # Same items

        # Verify all expected labels are present
        expected_labels = set(range(20))
        assert set(labels1) == expected_labels

    def test_dataloader_batch_size_one(self, tmp_path: Path) -> None:
        """Test DataLoader with batch size of 1."""
        pytest.importorskip("torch", reason="PyTorch not installed")
        from torch.utils.data import DataLoader

        # Create dataset with 5 items
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(5):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"pose": np.random.randn(3, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Batch Size 1"},
            connections={},
            items=items,
        )

        # DataLoader with batch_size=1
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=identity_collate
        )

        batches = list(dataloader)
        assert len(batches) == 5

        for i, batch in enumerate(batches):
            assert len(batch) == 1
            assert batch[0].targets["label"][0][0] == i


class TestDatasetIndexingAndSlicing:
    """Test dataset indexing and slicing operations."""

    def test_positive_indexing(self, tmp_path: Path) -> None:
        """Test positive index access."""
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(10):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"hand": np.ones((2, 21, 2), dtype=np.float32) * i},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Indexing Test"},
            connections={},
            items=items,
        )

        # Test individual access
        assert dataset[0].targets["label"][0][0] == 0
        assert dataset[5].targets["label"][0][0] == 5
        assert dataset[9].targets["label"][0][0] == 9

    def test_negative_indexing(self, tmp_path: Path) -> None:
        """Test negative index access."""
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(10):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"pose": np.ones((2, 33, 3), dtype=np.float32) * i},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Negative Index"},
            connections={},
            items=items,
        )

        # Negative indices
        assert dataset[-1].targets["label"][0][0] == 9
        assert dataset[-5].targets["label"][0][0] == 5
        assert dataset[-10].targets["label"][0][0] == 0

    def test_out_of_bounds_indexing(self, tmp_path: Path) -> None:
        """Test that out-of-bounds indices raise appropriate errors."""
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(5):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={"hand": np.ones((2, 21, 2), dtype=np.float32)},
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Bounds Test"},
            connections={},
            items=items,
        )

        # Test out of bounds
        with pytest.raises(IndexError):
            _ = dataset[10]

        with pytest.raises(IndexError):
            _ = dataset[-11]


class TestDatasetEdgeCases:
    """Test edge cases in dataset operations."""

    def test_dataset_with_single_item(self, tmp_path: Path) -> None:
        """Test dataset with only one item."""
        item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
            videos={},
            landmarks={"pose": np.random.randn(5, 33, 3).astype(np.float32)},
            targets={"label": np.array([[42]], dtype=np.int64)},
        )

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Single Item"},
            connections={},
            items=[item],
        )

        assert len(dataset) == 1
        retrieved_item = dataset[0]
        assert retrieved_item.landmarks["pose"].shape == (5, 33, 3)
        assert retrieved_item.targets["label"][0][0] == 42

    def test_dataset_with_variable_frame_counts(self, tmp_path: Path) -> None:
        """Test dataset where items have different frame counts."""
        frame_counts = [5, 10, 15, 20, 25]
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []

        for i, frame_count in enumerate(frame_counts):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={
                    "hand": np.random.randn(frame_count, 21, 2).astype(np.float32)
                },
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Variable Frames"},
            connections={},
            items=items,
        )

        # Verify each item has correct frame count
        for i, expected_frames in enumerate(frame_counts):
            item = dataset[i]
            assert item.landmarks["hand"].shape[0] == expected_frames

    def test_dataset_with_multiple_landmark_keys(self, tmp_path: Path) -> None:
        """Test dataset with multiple landmark types."""
        item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
            videos={},
            landmarks={
                "pose": np.random.randn(8, 33, 3).astype(np.float32),
                "left_hand": np.random.randn(8, 21, 2).astype(np.float32),
                "right_hand": np.random.randn(8, 21, 2).astype(np.float32),
                "face": np.random.randn(8, 468, 3).astype(np.float32),
            },
            targets={"label": np.array([[0]], dtype=np.int64)},
        )

        dataset: SLDataset[str, str, str, str, Any] = SLDataset(
            metadata={"name": "Multi-key Dataset"},
            connections={},
            items=[item],
        )

        # Verify all keys are present
        retrieved_item = dataset[0]
        landmark_keys = {"pose", "left_hand", "right_hand", "face"}
        assert set(retrieved_item.landmarks.keys()) == landmark_keys
        assert retrieved_item.landmarks["pose"].shape == (8, 33, 3)
        assert retrieved_item.landmarks["face"].shape == (8, 468, 3)

    def test_dataset_save_load_preserves_structure(self, tmp_path: Path) -> None:
        """Test that save/load cycle preserves complete dataset structure."""
        dataset_path = tmp_path / "complete_structure.zarr"

        # Create complex dataset
        items: list[SLDatasetItem[str, Any, str, Any, str, Any]] = []
        for i in range(3):
            item: SLDatasetItem[str, Any, str, Any, str, Any] = SLDatasetItem(
                videos={},
                landmarks={
                    "pose": np.random.randn(10, 33, 3).astype(np.float32),
                    "hand": np.random.randn(10, 21, 2).astype(np.float32),
                },
                targets={"label": np.array([[i]], dtype=np.int64)},
            )
            items.append(item)

        original = SLDataset(
            metadata={"name": "Complex", "version": "2.0"},
            connections={("pose", "hand"): np.array([[0, 1]])},
            items=items,
        )

        # Save and load
        saved_group = dataset_to_zarr(original, str(dataset_path))
        loaded = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        # Verify everything matches
        assert len(loaded) == len(original)
        assert loaded.metadata == original.metadata
        assert len(loaded.connections) == len(original.connections)

        for i in range(len(original)):
            orig_item = original[i]
            load_item = loaded[i]
            assert set(orig_item.landmarks.keys()) == set(load_item.landmarks.keys())
            assert set(orig_item.targets.keys()) == set(load_item.targets.keys())
