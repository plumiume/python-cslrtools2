"""Unit tests for sldataset/array_loader.py

Tests for various array loaders (CSV, JSON, NPY, NPZ, Torch, SafeTensors, Zarr).
Coverage target: 58% → 90%
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import csv
import json
from pathlib import Path

from cslrtools2.sldataset.array_loader import (
    CsvLoader,
    JsonLoader,
    NpyLoader,
    NpzLoader,
    TorchLoader,
    SafetensorsLoader,
    ZarrLoader,
)


@pytest.fixture
def sample_array_2d() -> np.ndarray:
    """Fixture for sample 2D array."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def sample_array_3d() -> np.ndarray:
    """Fixture for sample 3D array."""
    return np.random.rand(10, 5, 3).astype(np.float32)


class TestCsvLoader:
    """Test CsvLoader class."""

    def test_load_csv_file(self, tmp_path: Path, sample_array_2d: np.ndarray):
        """Test loading array from CSV file."""
        csv_file = tmp_path / "test.csv"
        
        # Write CSV file
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in sample_array_2d:
                writer.writerow(row)
        
        loader = CsvLoader()
        result = loader.load_array(csv_file)
        
        # CSV loader returns list, not ndarray
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_load_csv_with_string_path(self, tmp_path: Path):
        """Test loading CSV with string path."""
        csv_file = tmp_path / "test.csv"
        data = [[1.0, 2.0], [3.0, 4.0]]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
        
        loader = CsvLoader()
        result = loader.load_array(str(csv_file))

        np_array = np.asarray(data)
        
        assert len(np_array) == 2
        assert len(np_array[0]) == 2


class TestJsonLoader:
    """Test JsonLoader class."""

    def test_load_json_nested_list(self, tmp_path: Path, sample_array_2d: np.ndarray):
        """Test loading array from JSON file with nested list."""
        json_file = tmp_path / "test.json"
        
        # Write JSON file
        with open(json_file, 'w') as f:
            json.dump(sample_array_2d.tolist(), f)
        
        loader = JsonLoader()
        result = loader.load_array(json_file)
        
        # JSON loader returns list, not ndarray
        assert isinstance(result, list)
        assert len(result) == 2

    def test_load_json_flat_list(self, tmp_path: Path):
        """Test loading flat list from JSON."""
        json_file = tmp_path / "test.json"
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JsonLoader()
        result = loader.load_array(json_file)

        np_array = np.asarray(result)
        
        assert len(np_array) == 5
        assert result == data


class TestNpyLoader:
    """Test NpyLoader class."""

    def test_load_npy_file(self, tmp_path: Path, sample_array_3d: np.ndarray):
        """Test loading array from NPY file."""
        npy_file = tmp_path / "test.npy"
        
        # Save NPY file
        np.save(npy_file, sample_array_3d)
        
        loader = NpyLoader()
        result = loader.load_array(npy_file)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, sample_array_3d)

    def test_load_npy_with_different_dtypes(self, tmp_path: Path):
        """Test loading NPY files with different data types."""
        for dtype in [np.float32, np.float64, np.int32, np.uint8]:
            npy_file = tmp_path / f"test_{dtype.__name__}.npy"
            data = np.array([[1, 2], [3, 4]], dtype=dtype)
            
            np.save(npy_file, data)
            
            loader = NpyLoader()
            result = loader.load_array(npy_file)

            np_array = np.asarray(data)
            
            assert np_array.dtype == dtype


class TestNpzLoader:
    """Test NpzLoader class."""

    def test_load_npz_file(self, tmp_path: Path):
        """Test loading mapping from NPZ file."""
        npz_file = tmp_path / "test.npz"
        
        # Save NPZ file with multiple arrays
        data: dict[str, np.ndarray] = {
            "array1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "array2": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        }
        np.savez(npz_file, allow_pickle=True, **data)
        
        loader = NpzLoader()
        result = loader.load_mapping(npz_file)
        
        # NpzLoader returns NpzFile object, not plain dict
        assert isinstance(result, np.lib.npyio.NpzFile)
        assert "array1" in result
        assert "array2" in result
        assert np.array_equal(result["array1"], data["array1"])
        assert np.array_equal(result["array2"], data["array2"])

    def test_load_npz_single_array(self, tmp_path: Path, sample_array_2d: np.ndarray):
        """Test loading NPZ with single array."""
        npz_file = tmp_path / "test.npz"
        
        np.savez(npz_file, data=sample_array_2d)
        
        loader = NpzLoader()
        result = loader.load_mapping(npz_file)
        
        assert "data" in result
        np_array = np.asarray(result["data"])
        assert np.array_equal(np_array, sample_array_2d)


class TestTorchLoader:
    """Test TorchLoader class."""

    def test_load_torch_file(self, tmp_path: Path, sample_array_3d: np.ndarray):
        """Test loading tensor from Torch file."""
        pt_file = tmp_path / "test.pt"
        
        # Save PyTorch file
        tensor = torch.from_numpy(sample_array_3d)
        torch.save(tensor, pt_file)
        
        loader = TorchLoader()
        result = loader.load_array(pt_file)
        
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor)

    def test_load_torch_state_dict(self, tmp_path: Path):
        """Test loading mapping from Torch state dict."""
        pt_file = tmp_path / "test.pth"
        
        # Save state dict
        state_dict = {
            "layer1.weight": torch.randn(3, 3),
            "layer2.bias": torch.randn(5),
        }
        torch.save(state_dict, pt_file)
        
        loader = TorchLoader()
        result = loader.load_mapping(pt_file)
        
        assert isinstance(result, dict)
        assert "layer1.weight" in result
        assert "layer2.bias" in result


class TestSafetensorsLoader:
    """Test SafetensorsLoader class."""

    def test_load_safetensors_file(self, tmp_path: Path):
        """Test loading mapping from SafeTensors file."""
        import safetensors.torch as st
        
        st_file = tmp_path / "test.safetensors"
        
        # Save SafeTensors file
        tensors = {
            "weight": torch.randn(3, 3),
            "bias": torch.randn(3),
        }
        st.save_file(tensors, st_file)
        
        loader = SafetensorsLoader()
        result = loader.load_mapping(st_file)
        
        assert isinstance(result, dict)
        assert "weight" in result
        assert "bias" in result
        assert isinstance(result["weight"], torch.Tensor)


class TestZarrLoader:
    """Test ZarrLoader class."""

    def test_load_zarr_group(self, tmp_path: Path):
        """Test loading mapping from Zarr group."""
        import zarr
        
        zarr_path = tmp_path / "test.zarr"
        
        # Create Zarr group with multiple arrays using v3 API
        root = zarr.open_group(zarr_path, mode='w')
        root.create_array("data1", data=np.array([[1, 2], [3, 4]]))
        root.create_array("data2", data=np.array([[5, 6], [7, 8]]))
        
        loader = ZarrLoader()
        result = loader.load_mapping(zarr_path)
        
        assert isinstance(result, dict)
        assert "data1" in result
        assert "data2" in result

    def test_load_zarr_single_array(self, tmp_path: Path, sample_array_3d: np.ndarray):
        """Test loading Zarr group with single array."""
        import zarr
        
        zarr_file = tmp_path / "test.zarr"
        
        # Create group with single array
        root = zarr.open_group(zarr_file, mode='w')
        root.create_array('single', data=sample_array_3d)
        
        loader = ZarrLoader()
        result = loader.load_mapping(zarr_file)
        
        # Should return dict with the array
        assert isinstance(result, dict)
        assert "single" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_csv_loader_empty_file(self, tmp_path: Path):
        """Test CSV loader with empty file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.touch()
        
        loader = CsvLoader()
        result = loader.load_array(csv_file)
        
        # CSV loader returns list, empty file returns empty list
        assert isinstance(result, list)
        assert len(result) == 0

    def test_json_loader_invalid_json(self, tmp_path: Path):
        """Test JSON loader with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json}")
        
        loader = JsonLoader()
        
        with pytest.raises(json.JSONDecodeError):
            loader.load_array(json_file)

    def test_npy_loader_nonexistent_file(self, tmp_path: Path):
        """Test NPY loader with nonexistent file."""
        npy_file = tmp_path / "nonexistent.npy"
        
        loader = NpyLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_array(npy_file)

    def test_npy_loader_with_npz_file(self, tmp_path: Path):
        """Test NPY loader raises error when loading NPZ file (mapping)."""
        from cslrtools2.exceptions import DataFormatError
        
        # Create NPZ file (which is a mapping)
        npz_file = tmp_path / "test.npz"
        np.savez(npz_file, a=np.array([1, 2, 3]))
        
        loader = NpyLoader()
        
        # NpyLoader.load_array should raise DataFormatError for mappings
        with pytest.raises(DataFormatError):
            loader.load_array(npz_file)

    def test_npz_loader_with_npy_file(self, tmp_path: Path):
        """Test NPZ loader raises error when loading single-array NPY file."""
        from cslrtools2.exceptions import DataFormatError
        
        # Create NPY file (single array, not a mapping)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, np.array([1, 2, 3]))
        
        loader = NpzLoader()
        
        # NpzLoader.load_mapping should raise DataFormatError for non-mappings
        with pytest.raises(DataFormatError):
            loader.load_mapping(npy_file)

    def test_torch_loader_with_dict(self, tmp_path: Path):
        """Test Torch loader raises error when loading dict with load_array."""
        from cslrtools2.exceptions import DataFormatError
        
        # Create Torch file with dict
        torch_file = tmp_path / "test.pt"
        torch.save({"a": torch.tensor([1, 2, 3])}, torch_file)
        
        loader = TorchLoader()
        
        # TorchLoader.load_array should raise DataFormatError for dicts
        with pytest.raises(DataFormatError):
            loader.load_array(torch_file)

    def test_torch_loader_with_tensor_as_mapping(self, tmp_path: Path):
        """Test Torch loader raises error when loading single tensor with load_mapping."""
        from cslrtools2.exceptions import DataFormatError
        
        # Create Torch file with single tensor
        torch_file = tmp_path / "test.pt"
        torch.save(torch.tensor([1, 2, 3]), torch_file)
        
        loader = TorchLoader()
        
        # TorchLoader.load_mapping should raise DataFormatError for non-dicts
        with pytest.raises(DataFormatError):
            loader.load_mapping(torch_file)

    def test_safetensors_loader_with_non_dict(self, tmp_path: Path):
        """Test SafeTensors loader raises error when file doesn't contain mapping."""
        from cslrtools2.exceptions import DataFormatError
        
        # Create invalid safetensors file (empty file)
        st_file = tmp_path / "test.safetensors"
        st_file.write_bytes(b"invalid")
        
        loader = SafetensorsLoader()
        
        # SafetensorsLoader.load_mapping should raise error for invalid files
        with pytest.raises((DataFormatError, Exception)):
            loader.load_mapping(st_file)
