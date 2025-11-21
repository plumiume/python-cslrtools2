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

"""Unit tests for lmpipe/collector/landmark_matrix/csv_lmsc.py

Tests for CsvLandmarkMatrixSaveCollector.
Coverage target: 29% → 85%
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import csv
import pytest  # pyright: ignore[reportUnusedImport]
import numpy as np
from pathlib import Path
from typing import Literal, Mapping

from cslrtools2.lmpipe.collector.landmark_matrix.csv_lmsc import (
    CsvLandmarkMatrixSaveCollector,
    csv_lmsc_creator,
    tsv_lmsc_creator,
    ssv_lmsc_creator,
)
from cslrtools2.typings import NDArrayFloat, NDArrayStr


# Helper function to create headers for testing
def _make_headers[K: str](landmarks: Mapping[K, NDArrayFloat]) -> dict[K, NDArrayStr]:
    """Create header mappings for landmarks based on their shape."""
    headers: dict[K, NDArrayStr] = {}
    for key, array in landmarks.items():
        arr = np.asarray(array)
        # Calculate flattened width: prod(shape[1:])
        if arr.ndim > 1:
            width = int(np.prod(arr.shape[1:]))
        else:
            width = 1
        # Generate column names: value_0, value_1, ..., value_{width-1}
        headers[key] = np.array([f"value_{i}" for i in range(width)], dtype=str)
    return headers


# Fixtures
@pytest.fixture
def sample_landmark_matrix() -> NDArrayFloat:
    """Sample landmark matrix (frames, landmarks, dims)."""
    np.random.seed(42)
    return np.random.rand(10, 33, 3).astype(np.float32)


@pytest.fixture
def sample_single_key_result(
    sample_landmark_matrix: NDArrayFloat,
) -> dict[Literal["pose"], NDArrayFloat]:
    """Single key result mapping."""
    return {"pose": sample_landmark_matrix}


@pytest.fixture
def sample_multi_key_result(
    sample_landmark_matrix: NDArrayFloat,
) -> dict[str, NDArrayFloat]:
    """Multiple key result mapping."""
    np.random.seed(42)
    return {
        "pose": sample_landmark_matrix,
        "left_hand": np.random.rand(10, 21, 3).astype(np.float32),
        "right_hand": np.random.rand(10, 21, 3).astype(np.float32),
    }


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Test class: Initialization
class TestCSVLMSCInitialization:
    """Tests for CsvLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization(self):
        """Test default initialization with comma delimiter."""
        collector = CsvLandmarkMatrixSaveCollector[str]()
        assert collector.delimiter == ","
        assert collector.encoding == "utf-8"
        assert collector.extension == ".csv"
        assert collector.is_perkey is True
        assert collector.is_container is False

    def test_custom_delimiter_csv(self):
        """Test initialization with custom comma delimiter."""
        collector = CsvLandmarkMatrixSaveCollector[str](delimiter=",")
        assert collector.delimiter == ","
        assert collector.extension == ".csv"

    def test_tab_delimiter_tsv(self):
        """Test initialization with tab delimiter."""
        collector = CsvLandmarkMatrixSaveCollector[str](delimiter="\t")
        assert collector.delimiter == "\t"
        assert collector.extension == ".tsv"

    def test_semicolon_delimiter_ssv(self):
        """Test initialization with semicolon delimiter."""
        collector = CsvLandmarkMatrixSaveCollector[str](delimiter=";")
        assert collector.delimiter == ";"
        assert collector.extension == ".ssv"

    def test_custom_extension_override(self):
        """Test custom extension override."""
        collector = CsvLandmarkMatrixSaveCollector[str](delimiter=",", extension=".txt")
        assert collector.extension == ".txt"

    def test_custom_encoding(self):
        """Test custom encoding."""
        collector = CsvLandmarkMatrixSaveCollector[str](encoding="utf-16")
        assert collector.encoding == "utf-16"


# Test class: File operations
class TestCSVLMSCFileOperations:
    """Tests for CSV file save and read operations."""

    def test_save_single_key_csv(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key to CSV file."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]]()

        # Use the correct API: _open_file, _append_result, _close_file
        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_headers(sample_single_key_result), sample_single_key_result
            )
        finally:
            collector._close_file()

        # Verify file exists
        csv_file = temp_output_dir / "landmarks" / "pose.csv"
        assert csv_file.exists()

        # Verify content
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # 10 frames (flattened from shape[0])
            assert len(rows) == 10

            # Each row should have 99 values (33*3) - no key/sample_index metadata
            assert all(len(row) == 99 for row in rows)

            # Check first row structure - should only have value columns
            first_row = rows[0]
            # 3 dimensions flattened
            assert "value_0" in first_row
            assert "value_1" in first_row
            assert "value_2" in first_row
            # Verify no metadata columns
            assert "key" not in first_row
            assert "sample_index" not in first_row

    def test_save_multiple_keys_csv(
        self, temp_output_dir: Path, sample_multi_key_result: dict[str, NDArrayFloat]
    ):
        """Test saving multiple keys to separate CSV files."""
        collector = CsvLandmarkMatrixSaveCollector[str]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify all files exist
        landmarks_dir = temp_output_dir / "landmarks"
        assert (landmarks_dir / "pose.csv").exists()
        assert (landmarks_dir / "left_hand.csv").exists()
        assert (landmarks_dir / "right_hand.csv").exists()

    def test_save_tsv_format(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving with tab delimiter (TSV)."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]](delimiter="\t")

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_headers(sample_single_key_result), sample_single_key_result
            )
        finally:
            collector._close_file()

        tsv_file = temp_output_dir / "landmarks" / "pose.tsv"
        assert tsv_file.exists()

        # Verify tab delimiter
        with tsv_file.open("r") as f:
            first_line = f.readline()
            assert "\t" in first_line
            assert "," not in first_line

    def test_save_ssv_format(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving with semicolon delimiter (SSV)."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]](delimiter=";")

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_headers(sample_single_key_result), sample_single_key_result
            )
        finally:
            collector._close_file()

        ssv_file = temp_output_dir / "landmarks" / "pose.ssv"
        assert ssv_file.exists()

        # Verify semicolon delimiter
        with ssv_file.open("r") as f:
            first_line = f.readline()
            assert ";" in first_line

    def test_multiple_append_calls(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple append calls accumulate data."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_headers(sample_single_key_result), sample_single_key_result
            )
            collector._append_result(
                1, _make_headers(sample_single_key_result), sample_single_key_result
            )  # Append again
        finally:
            collector._close_file()

        csv_file = temp_output_dir / "landmarks" / "pose.csv"
        with csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Should have double the rows (2 * 10 frames)
            assert len(rows) == 2 * 10


# Test class: Error handling
class TestCSVLMSCErrorHandling:
    """Tests for error handling."""

    def test_inconsistent_sample_width_raises_error(self, temp_output_dir: Path):
        """Test that inconsistent sample widths raise CollectorError."""
        from cslrtools2.exceptions import CollectorError

        collector = CsvLandmarkMatrixSaveCollector[str]()

        first_result = {"pose": np.random.rand(5, 33, 3).astype(np.float32)}
        # Different shape: 21 landmarks instead of 33
        second_result = {"pose": np.random.rand(5, 21, 3).astype(np.float32)}

        with pytest.raises(CollectorError, match="Inconsistent landmark sample width"):
            collector._open_file(temp_output_dir)
            try:
                collector._append_result(0, _make_headers(first_result), first_result)
                collector._append_result(0, _make_headers(second_result), second_result)
            finally:
                collector._close_file()

    def test_writer_not_prepared_raises_error(self):
        """Test accessing writer before opening file raises CollectorError."""
        from cslrtools2.exceptions import CollectorError

        collector = CsvLandmarkMatrixSaveCollector[str]()

        # Directly calling _ensure_writer without _open_file
        with pytest.raises(CollectorError, match="CSV landmark directory not prepared"):
            collector._ensure_writer("pose", 99)


# Test class: Creator functions
class TestCSVLMSCCreators:
    """Tests for creator helper functions."""

    def test_csv_lmsc_creator(self):
        """Test csv_lmsc_creator function."""
        collector = csv_lmsc_creator(str)
        assert isinstance(collector, CsvLandmarkMatrixSaveCollector)
        assert collector.delimiter == ","
        assert collector.extension == ".csv"

    def test_tsv_lmsc_creator(self):
        """Test tsv_lmsc_creator function."""
        collector = tsv_lmsc_creator(str)
        assert isinstance(collector, CsvLandmarkMatrixSaveCollector)
        assert collector.delimiter == "\t"
        assert collector.extension == ".tsv"

    def test_ssv_lmsc_creator(self):
        """Test ssv_lmsc_creator function."""
        collector = ssv_lmsc_creator(str)
        assert isinstance(collector, CsvLandmarkMatrixSaveCollector)
        assert collector.delimiter == ";"
        assert collector.extension == ".ssv"


# Test class: Edge cases
class TestCSVLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_landmark_array(self, temp_output_dir: Path):
        """Test handling empty landmark arrays."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]]()

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        csv_file = temp_output_dir / "landmarks" / "pose.csv"
        assert csv_file.exists()

        with csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Only header, no data rows
            assert len(rows) == 0

    def test_single_frame_landmark(self, temp_output_dir: Path):
        """Test handling single frame."""
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]]()

        single_frame: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(1, 33, 3).astype(np.float32)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_headers(single_frame), single_frame)
        finally:
            collector._close_file()

        csv_file = temp_output_dir / "landmarks" / "pose.csv"
        with csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # 1 frame (not 1 * 33 landmarks)
            assert len(rows) == 1

    def test_nested_output_directory_creation(self, temp_output_dir: Path):
        """Test creating nested output directories."""
        nested_path = temp_output_dir / "a" / "b" / "c" / "test.csv"
        collector = CsvLandmarkMatrixSaveCollector[Literal["pose"]]()

        result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32)
        }

        collector._open_file(nested_path.parent)
        try:
            collector._append_result(0, _make_headers(result), result)
        finally:
            collector._close_file()

        csv_file = temp_output_dir / "a" / "b" / "c" / "landmarks" / "pose.csv"
        assert csv_file.exists()
