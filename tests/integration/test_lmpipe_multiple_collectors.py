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

"""Integration tests for LMPipe with multiple collectors.

Scenario 1.2: Multiple format simultaneous saving
Tests that multiple collectors can be used simultaneously to save landmarks
in different formats from a single video processing run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pytest

from cslrtools2.lmpipe.collector.landmark_matrix import (
    CsvLandmarkMatrixSaveCollector,
    NpyLandmarkMatrixSaveCollector,
    NpzLandmarkMatrixSaveCollector,
)
from cslrtools2.lmpipe.estimator import Estimator
from cslrtools2.lmpipe.interface import LMPipeInterface
from cslrtools2.typings import NDArrayFloat, NDArrayStr, MatLike

from .helpers import verify_npz_structure


class SimpleEstimator(Estimator[Literal["test"]]):
    """Simple estimator for testing multiple collectors."""

    def setup(self) -> None:
        pass

    @property
    def shape(self) -> Mapping[Literal["test"], tuple[int, int]]:
        return {"test": (3, 3)}

    def configure_estimator_name(self) -> Literal["test"]:
        return "test"

    @property
    def headers(self) -> Mapping[Literal["test"], NDArrayStr]:
        return {"test": np.array(["x", "y", "z"], dtype=str)}

    def estimate(
        self, frame_src: MatLike | None, frame_idx: int
    ) -> Mapping[Literal["test"], NDArrayFloat]:
        # Generate deterministic data based on frame index
        return {
            "test": np.array(
                [
                    [float(frame_idx), 1.0, 2.0],
                    [3.0, float(frame_idx) + 1.0, 5.0],
                    [6.0, 7.0, float(frame_idx) + 2.0],
                ]
            )
        }

    def annotate(
        self,
        frame_src: MatLike | None,
        frame_idx: int,
        landmarks: Mapping[Literal["test"], NDArrayFloat]
    ) -> MatLike:
        return (
            frame_src if frame_src is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        )


@pytest.mark.integration
@pytest.mark.requires_video
class TestLMPipeMultipleCollectors:
    """Test LMPipe with multiple collectors simultaneously."""

    def test_multiple_collectors_csv_npy_npz(
        self, sample_video_stop: Path, integration_tmp_path: Path
    ):
        """
        Scenario: Single execution saves to multiple formats

        Given: Test video and 3 collectors (CSV, NPY, NPZ)
        When: LMPipe.run() with all collectors
        Then: All 3 format files are generated with identical data
        """
        # Arrange
        estimator = SimpleEstimator()
        csv_collector = CsvLandmarkMatrixSaveCollector[Literal["test"]]()
        npy_collector = NpyLandmarkMatrixSaveCollector[Literal["test"]]()
        npz_collector = NpzLandmarkMatrixSaveCollector[Literal["test"]]()

        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[csv_collector, npy_collector, npz_collector],
        )

        output_dir = integration_tmp_path / "multi_output"

        # Act
        interface.run(sample_video_stop, output_dir)

        # Assert - All files exist
        # Per-key collectors create "landmarks" subdirectory
        landmarks_dir = output_dir / "landmarks"
        csv_file = landmarks_dir / "test.csv"
        npy_file = landmarks_dir / "test.npy"
        # Container collector creates file directly in output_dir
        npz_file = output_dir / "landmarks.npz"

        assert csv_file.exists(), f"CSV file should be created at {csv_file}"
        assert npy_file.exists(), f"NPY file should be created at {npy_file}"
        assert npz_file.exists(), f"NPZ file should be created at {npz_file}"

        # Assert - NPZ structure is valid
        verify_npz_structure(npz_file, expected_keys=["test"])

        # Assert - Data consistency across formats
        npy_data = np.load(npy_file)
        npz_data = np.load(npz_file)["test"]

        assert npy_data.shape == npz_data.shape, "NPY and NPZ should have same shape"
        np.testing.assert_array_equal(
            npy_data, npz_data, err_msg="NPY and NPZ data should be identical"
        )

        # Assert - CSV has correct structure (at least check it's readable)
        csv_content = csv_file.read_text()
        assert (
            "x,y,z" in csv_content
        ), (
            "CSV should contain header with estimator-defined column names "
            "(no extra metadata)"
        )
        lines = csv_content.strip().split("\n")
        assert len(lines) > 1, "CSV should contain data rows"
        # Verify first data line has 3 values (x,y,z)
        first_data = lines[1].split(",")
        assert (
            len(first_data) == 3
        ), "Each row should have exactly 3 values matching headers"

    def test_multiple_collectors_selective_combination(
        self, sample_video_stop: Path, integration_tmp_path: Path
    ):
        """
        Scenario: Use only NPY and NPZ collectors (skip CSV)

        Given: Test video and 2 collectors (NPY, NPZ only)
        When: LMPipe.run() with selective collectors
        Then: Only selected format files are generated
        """
        # Arrange
        estimator = SimpleEstimator()
        npy_collector = NpyLandmarkMatrixSaveCollector[Literal["test"]]()
        npz_collector = NpzLandmarkMatrixSaveCollector[Literal["test"]]()

        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[npy_collector, npz_collector],
        )

        output_dir = integration_tmp_path / "selective_output"

        # Act
        interface.run(sample_video_stop, output_dir)

        # Assert - Only selected files exist
        landmarks_dir = output_dir / "landmarks"
        csv_file = landmarks_dir / "test.csv"
        npy_file = landmarks_dir / "test.npy"
        npz_file = output_dir / "landmarks.npz"

        assert not csv_file.exists(), "CSV file should NOT be created"
        assert npy_file.exists(), f"NPY file should be created at {npy_file}"
        assert npz_file.exists(), f"NPZ file should be created at {npz_file}"

    def test_multiple_collectors_with_empty_list(
        self, sample_video_stop: Path, integration_tmp_path: Path
    ):
        """
        Scenario: Run with empty collectors list

        Given: Test video and no collectors
        When: LMPipe.run() with collectors=[]
        Then: Processing completes but no output files are created
        """
        # Arrange
        estimator = SimpleEstimator()
        interface = LMPipeInterface(estimator=estimator, collectors=[])

        output_dir = integration_tmp_path / "no_output"

        # Act - should complete without error (but log warning)
        interface.run(sample_video_stop, output_dir)

        # Assert - Output directory may not exist or is empty
        if output_dir.exists():
            # Check that no landmark files were created
            csv_files = list(output_dir.rglob("*.csv"))
            npy_files = list(output_dir.rglob("*.npy"))
            npz_files = list(output_dir.rglob("*.npz"))

            assert len(csv_files) == 0, "No CSV files should be created"
            assert len(npy_files) == 0, "No NPY files should be created"
            assert len(npz_files) == 0, "No NPZ files should be created"

    def test_multiple_collectors_data_integrity(
        self, sample_video_stop: Path, integration_tmp_path: Path
    ):
        """
        Scenario: Verify all collectors receive identical landmark data

        Given: Test video with deterministic estimator output
        When: Multiple collectors save landmarks
        Then: All formats contain exactly the same numerical values
        """
        # Arrange
        estimator = SimpleEstimator()
        npy_collector = NpyLandmarkMatrixSaveCollector[Literal["test"]]()
        npz_collector = NpzLandmarkMatrixSaveCollector[Literal["test"]]()

        interface = LMPipeInterface(
            estimator=estimator,
            collectors=[npy_collector, npz_collector],
        )

        output_dir = integration_tmp_path / "integrity_output"

        # Act
        interface.run(sample_video_stop, output_dir)

        # Assert - Load and compare data
        npy_file = output_dir / "landmarks" / "test.npy"
        npz_file = output_dir / "landmarks.npz"

        npy_data = np.load(npy_file)
        npz_data = np.load(npz_file)["test"]

        # Check exact equality (not just approximate)
        assert npy_data.dtype == npz_data.dtype, "Data types should match"
        assert npy_data.shape == npz_data.shape, "Shapes should match"
        np.testing.assert_array_equal(
            npy_data,
            npz_data,
            err_msg="All collectors should receive identical data",
        )

        # Verify shape matches estimator specification
        expected_num_frames = npy_data.shape[0]
        expected_shape = (expected_num_frames, 3, 3)  # (frames, points, coords)
        assert npy_data.shape == expected_shape, f"Expected shape {expected_shape}"
