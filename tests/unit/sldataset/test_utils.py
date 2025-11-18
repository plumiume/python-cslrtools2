"""Unit tests for sldataset/utils.py

Tests for Zarr and tensor utility functions.
Coverage target: 48% → 90%+
"""

from __future__ import annotations

import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
import torch
import zarr
from pathlib import Path

from cslrtools2.sldataset.utils import get_array, get_group, as_tensor
from cslrtools2.exceptions import DataLoadError


@pytest.fixture
def temp_zarr_store(tmp_path: Path) -> zarr.Group:
    """Create a temporary Zarr store with test data."""
    store_path = tmp_path / "test.zarr"
    root = zarr.open_group(store_path, mode="w")

    # Add array (dtype inferred from data)
    root.create_array("data", data=np.arange(10, dtype=np.float32))

    # Add nested group with array
    subgroup = root.create_group("subgroup")
    subgroup.create_array("nested_data", data=np.arange(5, dtype=np.int32))

    return root


class TestGetArray:
    """Tests for get_array function."""

    def test_get_existing_array(self, temp_zarr_store: zarr.Group):
        """Test retrieving an existing array."""
        array = get_array(temp_zarr_store, "data")
        assert isinstance(array, zarr.Array)
        assert array.shape == (10,)
        assert array.dtype == np.float32

    def test_get_nested_array(self, temp_zarr_store: zarr.Group):
        """Test retrieving a nested array."""
        subgroup = temp_zarr_store["subgroup"]
        assert isinstance(subgroup, zarr.Group)
        array = get_array(subgroup, "nested_data")
        assert isinstance(array, zarr.Array)
        assert array.shape == (5,)

    def test_get_nonexistent_array_raises_error(self, temp_zarr_store: zarr.Group):
        """Test that retrieving non-existent array raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Array not found at path: nonexistent"):
            get_array(temp_zarr_store, "nonexistent")

    def test_get_group_as_array_raises_error(self, temp_zarr_store: zarr.Group):
        """Test that attempting to get a group as array raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Array not found at path: subgroup"):
            get_array(temp_zarr_store, "subgroup")

    def test_error_message_includes_available_keys(self, temp_zarr_store: zarr.Group):
        """Test that error message includes available keys."""
        with pytest.raises(DataLoadError) as exc_info:
            get_array(temp_zarr_store, "missing")
        assert "Available keys:" in str(exc_info.value)
        assert "data" in str(exc_info.value)


class TestGetGroup:
    """Tests for get_group function."""

    def test_get_existing_group(self, temp_zarr_store: zarr.Group):
        """Test retrieving an existing group."""
        subgroup = get_group(temp_zarr_store, "subgroup")
        assert isinstance(subgroup, zarr.Group)
        assert "nested_data" in subgroup

    def test_get_nonexistent_group_raises_error(self, temp_zarr_store: zarr.Group):
        """Test that retrieving non-existent group raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Group not found at path: nonexistent"):
            get_group(temp_zarr_store, "nonexistent")

    def test_get_array_as_group_raises_error(self, temp_zarr_store: zarr.Group):
        """Test that attempting to get an array as group raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Group not found at path: data"):
            get_group(temp_zarr_store, "data")

    def test_error_message_includes_available_keys(self, temp_zarr_store: zarr.Group):
        """Test that error message includes available keys."""
        with pytest.raises(DataLoadError) as exc_info:
            get_group(temp_zarr_store, "missing")
        assert "Available keys:" in str(exc_info.value)
        assert "subgroup" in str(exc_info.value)


class TestAsTensor:
    """Tests for as_tensor function."""

    def test_tensor_input_returns_same_tensor(self):
        """Test that tensor input returns the same tensor object."""
        original = torch.tensor([1, 2, 3])
        result = as_tensor(original)
        assert result is original
        assert torch.equal(result, original)

    def test_numpy_array_conversion(self):
        """Test conversion from numpy array."""
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = as_tensor(np_array)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_list_conversion(self):
        """Test conversion from Python list."""
        list_data = [1, 2, 3, 4, 5]
        result = as_tensor(list_data)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1, 2, 3, 4, 5]))

    def test_nested_list_conversion(self):
        """Test conversion from nested list."""
        nested_list = [[1, 2], [3, 4], [5, 6]]
        result = as_tensor(nested_list)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 2)
        expected = torch.tensor([[1, 2], [3, 4], [5, 6]])
        assert torch.equal(result, expected)

    def test_scalar_conversion(self):
        """Test conversion from scalar."""
        scalar = 42
        result = as_tensor(scalar)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 42

    def test_multidimensional_numpy_array(self):
        """Test conversion from multidimensional numpy array."""
        np_array = np.random.rand(3, 4, 5).astype(np.float32)
        result = as_tensor(np_array)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 4, 5)
        assert torch.allclose(
            result, torch.tensor(np_array)
        )

    def test_preserves_data_type(self):
        """Test that data type is preserved during conversion."""
        np_int = np.array([1, 2, 3], dtype=np.int64)
        result_int = as_tensor(np_int)
        assert result_int.dtype == torch.int64

        np_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result_float = as_tensor(np_float)
        assert result_float.dtype == torch.float64
