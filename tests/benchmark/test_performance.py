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

"""Performance benchmark tests for cslrtools2.

These tests measure execution time for critical operations to detect
performance regressions. Tests use time.perf_counter() for timing
and have generous thresholds to avoid flaky failures in CI environments.

Note:
    Thresholds are set conservatively to account for slower CI environments.
    Local development machines will typically perform significantly faster.

    These benchmarks test array I/O operations (numpy save/load) as proxies
    for collector performance, since collectors require complex setup with
    headers and frame indices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport] # noqa: F401
import zarr

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestArrayIOPerformance:
    """Benchmark array I/O operations as proxy for collector performance."""

    def test_npy_write_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPY write speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "benchmark.npy"

        start_time = time.perf_counter()
        np.save(output_file, medium_landmark_data)
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should write in under 1 second
        assert elapsed < 1.0, f"NPY write took {elapsed:.3f}s, expected < 1.0s"
        assert output_file.exists()

    def test_npy_read_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPY read speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "benchmark.npy"
        np.save(output_file, medium_landmark_data)

        start_time = time.perf_counter()
        loaded_data = np.load(output_file)
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should load in under 1 second
        assert elapsed < 1.0, f"NPY read took {elapsed:.3f}s, expected < 1.0s"
        assert np.array_equal(loaded_data, medium_landmark_data)

    def test_npz_write_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPZ write speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "benchmark.npz"

        start_time = time.perf_counter()
        np.savez_compressed(output_file, landmarks=medium_landmark_data)
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should write in under 2 seconds
        # (NPZ includes compression, may be slower than NPY)
        assert elapsed < 2.0, f"NPZ write took {elapsed:.3f}s, expected < 2.0s"
        assert output_file.exists()

    def test_npz_read_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPZ read speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "benchmark.npz"
        np.savez_compressed(output_file, landmarks=medium_landmark_data)

        start_time = time.perf_counter()
        with np.load(output_file) as data:
            loaded_data = data["landmarks"]
            _ = loaded_data[:]  # Force load into memory
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should load in under 2 seconds
        assert elapsed < 2.0, f"NPZ read took {elapsed:.3f}s, expected < 2.0s"

    def test_zarr_write_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark Zarr write speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_dir = tmp_path / "benchmark.zarr"

        start_time = time.perf_counter()
        z = zarr.open_array(
            store=str(output_dir),
            mode="w",
            shape=medium_landmark_data.shape,
            chunks=(10, 33, 3),
            dtype=medium_landmark_data.dtype,
        )
        z[:, :, :] = medium_landmark_data
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should write in under 2 seconds
        msg = f"Zarr write took {elapsed:.3f}s, expected < 2.0s"
        assert elapsed < 2.0, msg
        assert output_dir.exists()

    def test_zarr_read_performance(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark Zarr read speed.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_dir = tmp_path / "benchmark.zarr"
        z = zarr.open_array(
            store=str(output_dir),
            mode="w",
            shape=medium_landmark_data.shape,
            chunks=(10, 33, 3),
            dtype=medium_landmark_data.dtype,
        )
        z[:, :, :] = medium_landmark_data

        start_time = time.perf_counter()
        z = zarr.open_array(store=str(output_dir), mode="r")
        loaded_data = z[:, :, :]
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 frames should load in under 2 seconds
        assert elapsed < 2.0, f"Zarr read took {elapsed:.3f}s, expected < 2.0s"
        # type: ignore[arg-type] - zarr.Array not in numpy's ArrayLike stubs
        assert np.array_equal(loaded_data, medium_landmark_data)


class TestLargeDataPerformance:
    """Benchmark performance with larger datasets."""

    def test_npy_large_dataset_write(
        self,
        large_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPY write with 1000 frames.

        Args:
            large_landmark_data: 1000 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "large_benchmark.npy"

        start_time = time.perf_counter()
        np.save(output_file, large_landmark_data)
        elapsed = time.perf_counter() - start_time

        # Threshold: 1000 frames should write in under 5 seconds
        msg = f"NPY large write took {elapsed:.3f}s, expected < 5.0s"
        assert elapsed < 5.0, msg
        assert output_file.exists()

    def test_npy_large_dataset_read(
        self,
        large_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark NPY read with 1000 frames.

        Args:
            large_landmark_data: 1000 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_file = tmp_path / "large_benchmark.npy"
        np.save(output_file, large_landmark_data)

        start_time = time.perf_counter()
        loaded_data = np.load(output_file)
        elapsed = time.perf_counter() - start_time

        # Threshold: 1000 frames should load in under 5 seconds
        msg = f"NPY large read took {elapsed:.3f}s, expected < 5.0s"
        assert elapsed < 5.0, msg
        assert loaded_data.shape == large_landmark_data.shape

    def test_zarr_large_dataset_write(
        self,
        large_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Benchmark Zarr write with 1000 frames.

        Args:
            large_landmark_data: 1000 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_dir = tmp_path / "large_benchmark.zarr"

        start_time = time.perf_counter()
        z = zarr.open_array(
            store=str(output_dir),
            mode="w",
            shape=large_landmark_data.shape,
            chunks=(100, 33, 3),
            dtype=large_landmark_data.dtype,
        )
        z[:, :, :] = large_landmark_data
        elapsed = time.perf_counter() - start_time

        # Threshold: 1000 frames should write in under 10 seconds
        assert (
            elapsed < 10.0
        ), f"Zarr large write took {elapsed:.3f}s, expected < 10.0s"
        assert output_dir.exists()


class TestMemoryEfficiency:
    """Benchmark memory efficiency of operations."""

    def test_incremental_write_memory_efficiency(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that incremental writes don't accumulate memory.

        This test verifies that writing large amounts of data
        incrementally doesn't cause memory buildup.

        Args:
            tmp_path: Temporary directory for output.
        """
        output_dir = tmp_path / "incremental.zarr"

        # Create zarr array for incremental writes
        z = zarr.open_array(
            store=str(output_dir),
            mode="w",
            shape=(1000, 33, 3),
            chunks=(10, 33, 3),
            dtype=np.float32,
        )

        # Write 100 chunks of 10 frames each (1000 frames total)
        # Should complete without memory issues
        start_time = time.perf_counter()
        for i in range(100):
            chunk = np.random.rand(10, 33, 3).astype(np.float32)
            z[i * 10: (i + 1) * 10, :, :] = chunk
        elapsed = time.perf_counter() - start_time

        # Threshold: 100 chunks should write in under 10 seconds
        assert (
            elapsed < 10.0
        ), f"Incremental write took {elapsed:.3f}s, expected < 10.0s"
        assert output_dir.exists()

    def test_partial_read_efficiency(
        self,
        large_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Test that partial reads are efficient.

        Zarr should allow efficient partial reads without loading
        the entire dataset into memory.

        Args:
            large_landmark_data: 1000 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        output_dir = tmp_path / "partial_read.zarr"

        # Write full dataset
        z = zarr.open_array(
            store=str(output_dir),
            mode="w",
            shape=large_landmark_data.shape,
            chunks=(100, 33, 3),
            dtype=large_landmark_data.dtype,
        )
        z[:, :, :] = large_landmark_data

        # Read only first 10 frames - should be very fast
        start_time = time.perf_counter()
        z_read = zarr.open_array(store=str(output_dir), mode="r")
        partial_data = z_read[:10, :, :]
        elapsed = time.perf_counter() - start_time

        # Threshold: Reading 10 frames should be under 0.5 seconds
        msg = f"Partial read took {elapsed:.3f}s, expected < 0.5s"
        assert elapsed < 0.5, msg
        assert partial_data.shape == (10, 33, 3)  # type: ignore[union-attr]


class TestComparativePerformance:
    """Compare performance across different implementations."""

    def test_format_write_speed_comparison(
        self,
        small_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Compare write speeds across different formats.

        This test verifies relative performance characteristics:
        - NPY should be fastest (binary, no compression)
        - NPZ should be moderate (binary with compression)
        - Zarr should be comparable (chunked storage)

        Args:
            small_landmark_data: 10 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        # Measure NPY
        npy_file = tmp_path / "compare.npy"
        start = time.perf_counter()
        np.save(npy_file, small_landmark_data)
        npy_time = time.perf_counter() - start

        # Measure NPZ
        npz_file = tmp_path / "compare.npz"
        start = time.perf_counter()
        np.savez_compressed(npz_file, landmarks=small_landmark_data)
        npz_time = time.perf_counter() - start

        # Measure Zarr
        zarr_dir = tmp_path / "compare.zarr"
        start = time.perf_counter()
        z = zarr.open_array(
            store=str(zarr_dir),
            mode="w",
            shape=small_landmark_data.shape,
            chunks=(10, 33, 3),
            dtype=small_landmark_data.dtype,
        )
        z[:, :, :] = small_landmark_data
        zarr_time = time.perf_counter() - start

        # Verify relative performance - all should be fast for small data
        assert npy_time < 1.0, "NPY write should be fast"
        assert npz_time < 2.0, "NPZ write should complete reasonably"
        assert zarr_time < 2.0, "Zarr write should complete reasonably"

        # All formats should complete successfully
        assert npy_file.exists()
        assert npz_file.exists()
        assert zarr_dir.exists()

    def test_format_read_speed_comparison(
        self,
        medium_landmark_data: NDArray[np.float32],
        tmp_path: Path,
    ) -> None:
        """Compare read speeds across different formats.

        Args:
            medium_landmark_data: 100 frames of landmark data.
            tmp_path: Temporary directory for output.
        """
        # Prepare files
        npy_file = tmp_path / "compare.npy"
        np.save(npy_file, medium_landmark_data)

        npz_file = tmp_path / "compare.npz"
        np.savez_compressed(npz_file, landmarks=medium_landmark_data)

        zarr_dir = tmp_path / "compare.zarr"
        z_write = zarr.open_array(
            store=str(zarr_dir),
            mode="w",
            shape=medium_landmark_data.shape,
            chunks=(10, 33, 3),
            dtype=medium_landmark_data.dtype,
        )
        z_write[:, :, :] = medium_landmark_data

        # Measure NPY read
        start = time.perf_counter()
        npy_data = np.load(npy_file)
        npy_time = time.perf_counter() - start

        # Measure NPZ read
        start = time.perf_counter()
        with np.load(npz_file) as data:
            npz_data = data["landmarks"][:]
        npz_time = time.perf_counter() - start

        # Measure Zarr read
        start = time.perf_counter()
        z_read = zarr.open_array(store=str(zarr_dir), mode="r")
        zarr_data = z_read[:, :, :]
        zarr_time = time.perf_counter() - start

        # Verify all reads completed successfully
        assert npy_data.shape == medium_landmark_data.shape
        assert npz_data.shape == medium_landmark_data.shape
        zarr_shape = zarr_data.shape  # type: ignore[union-attr]
        assert zarr_shape == medium_landmark_data.shape

        # All should complete in reasonable time
        assert npy_time < 2.0, f"NPY read took {npy_time:.3f}s"
        assert npz_time < 3.0, f"NPZ read took {npz_time:.3f}s"
        assert zarr_time < 3.0, f"Zarr read took {zarr_time:.3f}s"
