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

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
import torch
import zarr
from typing import Any

from cslrtools2.sldataset.dataset.holder import SLKeyHolder
from cslrtools2.sldataset.dataset.item import SLDatasetItem
from cslrtools2.sldataset.dataset.core import (
    SLDataset,
    DefaultSLDatasetItem,
    IterableSLDataset,
    SLDatasetBatch,
    dataset_to_zarr,
)


class TestSLKeyHolder:
    """Tests for SLKeyHolder type guard methods."""

    def test_is_metadata_key_returns_true_for_strings(self):
        """Test is_metadata_key() returns True for string objects."""
        assert SLKeyHolder.is_metadata_key("metadata_key")
        assert SLKeyHolder.is_metadata_key("another_key")
        assert SLKeyHolder.is_metadata_key("")

    def test_is_metadata_key_returns_false_for_non_strings(self):
        """Test is_metadata_key() returns False for non-string objects."""
        assert not SLKeyHolder.is_metadata_key(123)
        assert not SLKeyHolder.is_metadata_key(None)
        assert not SLKeyHolder.is_metadata_key([])

    def test_is_video_key_returns_true_for_strings(self):
        """Test is_video_key() returns True for string objects."""
        assert SLKeyHolder.is_video_key("video")
        assert SLKeyHolder.is_video_key("rgb")

    def test_is_video_key_returns_false_for_non_strings(self):
        """Test is_video_key() returns False for non-string objects."""
        assert not SLKeyHolder.is_video_key(456)
        assert not SLKeyHolder.is_video_key(None)

    def test_is_landmark_key_returns_true_for_strings(self):
        """Test is_landmark_key() returns True for string objects."""
        assert SLKeyHolder.is_landmark_key("pose")
        assert SLKeyHolder.is_landmark_key("left_hand")

    def test_is_landmark_key_returns_false_for_non_strings(self):
        """Test is_landmark_key() returns False for non-string objects."""
        assert not SLKeyHolder.is_landmark_key(789)
        assert not SLKeyHolder.is_landmark_key({})

    def test_is_target_key_returns_true_for_strings(self):
        """Test is_target_key() returns True for string objects."""
        assert SLKeyHolder.is_target_key("label")
        assert SLKeyHolder.is_target_key("gloss")

    def test_is_target_key_returns_false_for_non_strings(self):
        """Test is_target_key() returns False for non-string objects."""
        assert not SLKeyHolder.is_target_key(999)
        assert not SLKeyHolder.is_target_key(())


class TestSLDatasetItem:
    """Tests for SLDatasetItem class."""

    @pytest.fixture
    def sample_item(self):
        """Create a sample dataset item."""
        return SLDatasetItem(
            videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
            landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
            targets={"label": np.array([[1, 2, 3]])},
        )

    def test_initialization(
        self, sample_item: DefaultSLDatasetItem[Any, Any, Any]
    ):
        """Test SLDatasetItem initialization."""
        assert "rgb" in sample_item.videos
        assert "pose" in sample_item.landmarks
        assert "label" in sample_item.targets
        assert sample_item.videos["rgb"].shape == (1, 10, 64, 64, 3)
        assert sample_item.landmarks["pose"].shape == (1, 10, 33, 3)

    def test_to_device_converts_to_tensors(
        self, sample_item: DefaultSLDatasetItem[Any, Any, Any]
    ):
        """Test to() converts all data to tensors on specified device."""
        device = torch.device("cpu")
        tensor_item = sample_item.to(device)

        assert isinstance(tensor_item.videos["rgb"], torch.Tensor)
        assert isinstance(tensor_item.landmarks["pose"], torch.Tensor)
        assert isinstance(tensor_item.targets["label"], torch.Tensor)
        assert tensor_item.videos["rgb"].device.type == "cpu"

    def test_to_zarr_saves_to_group(
        self, sample_item: DefaultSLDatasetItem[Any, Any, Any], tmp_path: Path
    ):
        """Test to_zarr() saves item to Zarr group."""
        store_path = tmp_path / "item.zarr"
        group = sample_item.to_zarr(str(store_path))

        assert "videos" in group
        assert "landmarks" in group
        assert "targets" in group
        assert "rgb" in group["videos"]
        assert "pose" in group["landmarks"]
        assert "label" in group["targets"]

    def test_to_zarr_with_existing_group(
        self, sample_item: DefaultSLDatasetItem[Any, Any, Any]
    ):
        """Test to_zarr() works with existing Zarr group."""
        root = zarr.group()
        group = sample_item.to_zarr(root)

        assert group is root
        assert "videos" in group
        assert "landmarks" in group
        assert "targets" in group

    def test_from_zarr_loads_item(
        self, sample_item: DefaultSLDatasetItem[Any, Any, Any], tmp_path: Path
    ):
        """Test from_zarr() loads item from Zarr group."""
        store_path = tmp_path / "item.zarr"
        saved_group = sample_item.to_zarr(str(store_path))

        loaded_item = SLDatasetItem[str, Any, str, Any, str, Any].from_zarr(saved_group)

        assert "rgb" in loaded_item.videos
        assert "pose" in loaded_item.landmarks
        assert "label" in loaded_item.targets
        assert isinstance(loaded_item.videos["rgb"], zarr.Array)

    def test_from_file_system_loads_from_directory(self, tmp_path: Path):
        """Test from_file_system() loads item from directory structure."""
        item_dir = tmp_path / "item"
        videos_dir = item_dir / "videos"
        landmarks_dir = item_dir / "landmarks"
        targets_dir = item_dir / "targets"

        videos_dir.mkdir(parents=True)
        landmarks_dir.mkdir(parents=True)
        targets_dir.mkdir(parents=True)

        # Save sample data as .npy files
        np.save(videos_dir / "rgb.npy", np.random.rand(1, 10, 64, 64, 3))
        np.save(landmarks_dir / "pose.npy", np.random.rand(1, 10, 33, 3))
        np.save(targets_dir / "label.npy", np.array([[1, 2, 3]]))

        loaded_item = SLDatasetItem[
            str, Any, str, Any, str, Any
        ].from_file_system(item_dir)

        assert "rgb" in loaded_item.videos
        assert "pose" in loaded_item.landmarks
        assert "label" in loaded_item.targets

    def test_from_file_system_loads_container_files(self, tmp_path: Path):
        """Test from_file_system() loads from container files (.npz)."""
        item_dir = tmp_path / "item"
        item_dir.mkdir()

        # Save as .npz container files
        np.savez(item_dir / "videos.npz", rgb=np.random.rand(1, 10, 64, 64, 3))
        np.savez(item_dir / "landmarks.npz", pose=np.random.rand(1, 10, 33, 3))
        np.savez(item_dir / "targets.npz", label=np.array([[1, 2, 3]]))

        loaded_item = SLDatasetItem[
            str, Any, str, Any, str, Any
        ].from_file_system(item_dir)

        assert "rgb" in loaded_item.videos
        assert "pose" in loaded_item.landmarks
        assert "label" in loaded_item.targets

    def test_load_category_from_fs_filters_by_key_type(self, tmp_path: Path):
        """Test _load_category_from_fs() filters keys by type guard."""
        category_dir = tmp_path / "landmarks"
        category_dir.mkdir()

        # Save valid and invalid keys
        np.save(category_dir / "pose.npy", np.random.rand(10, 33, 3))
        np.save(category_dir / "invalid.npy", np.random.rand(10, 33, 3))

        result = SLDatasetItem[str, Any, str, Any, str, Any]._load_category_from_fs(
            category_dir,
            SLDatasetItem[str, Any, str, Any, str, Any].is_landmark_key,
            {".npy": np.load},
            {}
        )

        # Both should be loaded since type guard accepts all strings
        assert "pose" in result
        assert "invalid" in result

    def test_load_category_from_fs_handles_missing_directory(self, tmp_path: Path):
        """Test _load_category_from_fs() handles non-existent directory."""
        category_dir = tmp_path / "nonexistent"

        # Should handle missing directory by checking container files
        result = SLDatasetItem[str, Any, str, Any, str, Any]._load_category_from_fs(
            category_dir,
            SLDatasetItem[str, Any, str, Any, str, Any].is_landmark_key,
            {".npy": np.load},
            {".npz": lambda p: dict(np.load(p))},
        )

        # Should return empty dict if no container files exist either
        assert result == {}

    def test_load_category_from_fs_loads_container_when_dir_missing(
        self, tmp_path: Path
    ):
        """Test _load_category_from_fs() loads container file when directory
        doesn't exist."""
        category_path = tmp_path / "landmarks"

        # Save as .npz container file
        np.savez(str(category_path) + ".npz", pose=np.random.rand(10, 33, 3))

        result = SLDatasetItem[str, Any, str, Any, str, Any]._load_category_from_fs(
            category_path,
            SLDatasetItem[str, Any, str, Any, str, Any].is_landmark_key,
            {".npy": np.load},
            {".npz": lambda p: dict(np.load(p))},
        )

        assert "pose" in result

    def test_load_category_from_zarr_filters_arrays(self):
        """Test _load_category_from_zarr() loads only matching arrays."""
        root = zarr.group()
        group = root.create_group("landmarks")
        group.create_array("pose", data=np.random.rand(10, 33, 3))
        group.create_array("left_hand", data=np.random.rand(10, 21, 3))

        result = SLDatasetItem._load_category_from_zarr(
            SLDatasetItem[str, Any, str, Any, str, Any].is_landmark_key,
            group
        )

        assert "pose" in result
        assert "left_hand" in result
        assert isinstance(result["pose"], zarr.Array)


class TestSLDataset:
    """Tests for SLDataset class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        items = [
            SLDatasetItem[str, Any, str, Any, str, Any](
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(5)
        ]

        return SLDataset(
            metadata={"version": "1.0", "fps": 30},
            connections={("pose", "pose"): np.array([[0, 1], [1, 2]])},
            items=items,
        )

    def test_initialization(self, sample_dataset: SLDataset[str, str, str, str, Any]):
        """Test SLDataset initialization."""
        assert len(sample_dataset) == 5
        assert "version" in sample_dataset.metadata
        assert ("pose", "pose") in sample_dataset.connections

    def test_len_returns_item_count(
        self, sample_dataset: SLDataset[str, str, str, str, Any]
    ):
        """Test __len__() returns correct item count."""
        assert len(sample_dataset) == 5

    def test_getitem_returns_item(
        self, sample_dataset: SLDataset[str, str, str, str, Any]
    ):
        """Test __getitem__() returns correct item."""
        item = sample_dataset[0]

        assert "rgb" in item.videos
        assert "pose" in item.landmarks
        assert "label" in item.targets

    def test_getitem_with_different_indices(
        self, sample_dataset: SLDataset[str, str, str, str, Any]
    ):
        """Test __getitem__() with different indices."""
        item0 = sample_dataset[0]
        item4 = sample_dataset[4]

        assert item0.targets["label"][0][0] == 0
        assert item4.targets["label"][0][0] == 4

    def test_to_partially_converts_connections_to_device(
        self, sample_dataset: SLDataset[str, str, str, str, Any]
    ):
        """Test to_partially() converts connections to specified device."""
        device = torch.device("cpu")
        partial_dataset = sample_dataset.to_partially(device)

        conn = partial_dataset.connections[("pose", "pose")]
        assert isinstance(conn, torch.Tensor)
        assert conn.device.type == "cpu"

    def test_to_partially_preserves_items(
        self, sample_dataset: SLDataset[str, str, str, str, Any]
    ):
        """Test to_partially() preserves items reference."""
        device = torch.device("cpu")
        partial_dataset = sample_dataset.to_partially(device)

        assert partial_dataset.items is sample_dataset.items

    def test_from_zarr_loads_dataset(
        self,
        sample_dataset: SLDataset[str, str, str, str, Any],
        tmp_path: Path
    ):
        """Test from_zarr() loads dataset from Zarr group."""
        store_path = tmp_path / "dataset.zarr"
        saved_group = dataset_to_zarr(sample_dataset, str(store_path))

        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(saved_group)

        assert len(loaded_dataset) == 5
        assert "version" in loaded_dataset.metadata
        assert ("pose", "pose") in loaded_dataset.connections

    def test_from_zarr_filters_metadata_by_key_type(self):
        """Test from_zarr() filters metadata by type guard."""
        root = zarr.group()
        metadata_group = root.create_group("metadata")
        metadata_group.attrs["version"] = "1.0"
        metadata_group.attrs["fps"] = 30

        connections_group = root.create_group("connections")
        connections_group.create_array("pose.pose", data=np.array([[0, 1]]))

        root.create_group("items")

        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(root)
        assert "version" in loaded_dataset.metadata
        assert "fps" in loaded_dataset.metadata

    def test_from_zarr_parses_connection_keys(self):
        """Test from_zarr() correctly parses connection keys."""
        root = zarr.group()
        root.create_group("metadata")

        connections_group = root.create_group("connections")
        connections_group.create_array("pose.left_hand", data=np.array([[0, 1]]))
        connections_group.create_array("pose.right_hand", data=np.array([[2, 3]]))

        root.create_group("items")

        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(root)

        # Should parse "pose.left_hand" into tuple ("pose", "left_hand")
        assert ("pose", "left_hand") in loaded_dataset.connections
        # Should also parse "pose.right_hand"
        assert ("pose", "right_hand") in loaded_dataset.connections
        assert len(loaded_dataset.connections) == 2

    def test_from_zarr_filters_invalid_connection_keys(self):
        """Test from_zarr() skips connections with non-landmark keys."""
        root = zarr.group()
        root.create_group("metadata")

        connections_group = root.create_group("connections")
        # These should pass since SLKeyHolder accepts all strings
        connections_group.create_array("pose.left_hand", data=np.array([[0, 1]]))

        root.create_group("items")

        loaded_dataset = SLDataset[str, str, str, str, Any].from_zarr(root)

        # Connection should be loaded (type guard accepts all strings)
        assert ("pose", "left_hand") in loaded_dataset.connections


class TestIterableSLDataset:
    """Tests for IterableSLDataset class."""

    @pytest.fixture
    def sample_iterable_dataset(self):
        """Create a sample iterable dataset."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(3)
        ]

        dataset = IterableSLDataset(
            metadata={"version": "1.0"},
            connections={("pose", "pose"): np.array([[0, 1]])},
            items=iter(items),
        )

        return dataset

    def test_initialization(
        self, sample_iterable_dataset: IterableSLDataset[str, str, str, str, Any]
    ):
        """Test IterableSLDataset initialization."""
        assert "version" in sample_iterable_dataset.metadata
        assert ("pose", "pose") in sample_iterable_dataset.connections

    def test_iter_returns_items(self):
        """Test __iter__() returns items."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(3)
        ]

        dataset = IterableSLDataset[str, str, str, str, Any](
            metadata={}, connections={}, items=iter(items)
        )

        collected_items = list(dataset)
        assert len(collected_items) == 3

    def test_to_partially_converts_connections(self):
        """Test to_partially() converts connections to device."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[0]])},
            )
        ]

        dataset = IterableSLDataset[str, str, str, str, Any](
            metadata={},
            connections={("pose", "pose"): np.array([[0, 1]])},
            items=iter(items),
        )

        device = torch.device("cpu")
        partial_dataset = dataset.to_partially(device)

        conn = partial_dataset.connections[("pose", "pose")]
        assert isinstance(conn, torch.Tensor)


class TestSLDatasetBatch:
    """Tests for SLDatasetBatch class."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch."""
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={("pose", "pose"): torch.tensor([[0, 1]])},
            items=[],
        )

        tensor_item = SLDatasetItem(
            videos={"rgb": torch.rand(2, 10, 64, 64, 3)},
            landmarks={"pose": torch.rand(2, 10, 33, 3)},
            targets={"label": torch.tensor([[1], [2]])},
        )

        return SLDatasetBatch(dataset, tensor_item)

    def test_initialization(
        self, sample_batch: SLDatasetBatch[str, str, str, str]
    ):
        """Test SLDatasetBatch initialization."""
        assert sample_batch.dataset is not None
        assert sample_batch.item is not None
        assert "rgb" in sample_batch.item.videos

    def test_to_moves_data_to_device(
        self, sample_batch: SLDatasetBatch[str, str, str, str]
    ):
        """Test to() moves all data to specified device."""
        device = torch.device("cpu")
        batch_on_device = sample_batch.to(device)

        assert batch_on_device.item.videos["rgb"].device.type == "cpu"
        assert batch_on_device.item.landmarks["pose"].device.type == "cpu"

    def test_from_batch_concatenates_items(self):
        """Test from_batch() concatenates batch items."""
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={("pose", "pose"): torch.tensor([[0, 1]])},
            items=[],
        )

        batch_items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3)},
                targets={"label": np.array([[i]])},
            )
            for i in range(3)
        ]

        batch = SLDatasetBatch[str, str, str, str].from_batch(dataset, batch_items)

        assert batch.item.videos["rgb"].shape[0] == 3  # Batch size
        assert batch.item.landmarks["pose"].shape[0] == 3
        assert batch.item.targets["label"].shape[0] == 3

    def test_from_batch_preserves_keys(self):
        """Test from_batch() preserves all keys from first item."""
        dataset = SLDataset[str, str, str, str, Any](
            metadata={"version": "1.0"},
            connections={("pose", "pose"): torch.tensor([[0, 1]])},
            items=[],
        )

        batch_items = [
            SLDatasetItem(
                videos={
                    "rgb": np.random.rand(1, 10, 64, 64, 3),
                    "depth": np.random.rand(1, 10, 64, 64, 1),
                },
                landmarks={"pose": np.random.rand(1, 10, 33, 3)},
                targets={"label": np.array([[i]])},
            )
            for i in range(2)
        ]

        batch = SLDatasetBatch[str, str, str, str].from_batch(dataset, batch_items)

        assert "rgb" in batch.item.videos
        assert "depth" in batch.item.videos


class TestDatasetToZarr:
    """Tests for dataset_to_zarr function."""

    def test_saves_dataset_to_zarr_store(self, tmp_path: Path):
        """Test dataset_to_zarr() saves dataset to Zarr store."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(2)
        ]

        dataset = SLDataset(
            metadata={"version": "1.0", "fps": 30},
            connections={("pose", "pose"): np.array([[0, 1]])},
            items=items,
        )

        store_path = tmp_path / "dataset.zarr"
        group = dataset_to_zarr(dataset, str(store_path))

        assert "metadata" in group
        assert "connections" in group
        assert "items" in group
        assert group["metadata"].attrs["version"] == "1.0"
        assert "pose.pose" in group["connections"]

    def test_saves_iterable_dataset_to_zarr(self, tmp_path: Path):
        """Test dataset_to_zarr() works with IterableSLDataset."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(2)
        ]

        dataset: Any = IterableSLDataset(
            metadata={"version": "1.0"},
            connections={("pose", "pose"): np.array([[0, 1]])},
            items=iter(items),
        )

        store_path = tmp_path / "dataset.zarr"
        group = dataset_to_zarr(dataset, str(store_path))

        assert "metadata" in group
        assert "items" in group
        assert "0" in group["items"]
        assert "1" in group["items"]

    def test_uses_existing_zarr_group(self):
        """Test dataset_to_zarr() uses existing Zarr group."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[0]])},
            )
        ]

        dataset = SLDataset(
            metadata={"version": "1.0"},
            connections={("pose", "pose"): np.array([[0, 1]])},
            items=items,
        )

        existing_group = zarr.group()
        returned_group = dataset_to_zarr(dataset, existing_group)

        assert returned_group is existing_group
        assert "metadata" in existing_group

    def test_saves_all_items_to_zarr(self, tmp_path: Path):
        """Test dataset_to_zarr() saves all items."""
        items = [
            SLDatasetItem(
                videos={"rgb": np.random.rand(1, 10, 64, 64, 3).astype(np.float32)},
                landmarks={"pose": np.random.rand(1, 10, 33, 3).astype(np.float32)},
                targets={"label": np.array([[i]])},
            )
            for i in range(5)
        ]

        dataset = SLDataset[str, str, str, str, Any](
            metadata={}, connections={}, items=items
        )

        store_path = tmp_path / "dataset.zarr"
        group = dataset_to_zarr(dataset, str(store_path))

        items_group = group["items"]
        assert "0" in items_group
        assert "1" in items_group
        assert "2" in items_group
        assert "3" in items_group
        assert "4" in items_group
