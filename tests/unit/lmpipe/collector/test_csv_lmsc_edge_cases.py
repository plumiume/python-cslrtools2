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

"""Additional edge case tests for CSV LMSC to improve coverage.

This module focuses on uncovered lines in :mod:`csv_lmsc` module,
particularly testing delimiter detection, extension guessing, and
error handling in CSV landmark matrix save collectors.

Coverage Focus:
    - Line 68: Tab delimiter → .tsv extension
    - Line 133: Delimiter detection logic
    - Extension guessing for various delimiters
    - Error conditions and edge cases

Example:
    Run CSV edge case tests::

        >>> pytest tests/unit/lmpipe/collector/test_csv_lmsc_edge_cases.py -v
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]

from cslrtools2.lmpipe.collector.landmark_matrix.csv_lmsc import (
    CsvLandmarkMatrixSaveCollector,
)
from cslrtools2.exceptions import CollectorError


class TestCsvLMSCDelimiterDetection:
    """Test delimiter detection and extension guessing."""

    def test_guess_extension_for_tab_delimiter(self):
        """Test that tab delimiter results in .tsv extension.

        Covers csv_lmsc.py line 68 (part of _guess_extension).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](delimiter="\t")
        # When delimiter is tab, extension should be .tsv
        assert csv_lmsc.extension == ".tsv"

    def test_guess_extension_for_semicolon_delimiter(self):
        """Test that semicolon delimiter results in .ssv extension."""
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](delimiter=";")
        assert csv_lmsc.extension == ".ssv"

    def test_guess_extension_for_custom_delimiter(self):
        """Test that custom delimiter defaults to .csv extension."""
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](delimiter="|")
        # Unknown delimiters should default to .csv
        assert csv_lmsc.extension == ".csv"

    def test_explicit_extension_overrides_guessing(self):
        """Test that explicit extension parameter overrides auto-detection.

        Covers csv_lmsc.py line 133 (extension parameter usage).
        """
        # Even with tab delimiter, explicit extension should be used
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](
            delimiter="\t", extension=".custom"
        )
        assert csv_lmsc.extension == ".custom"


class TestCsvLMSCWriterInitialization:
    """Test CSV writer initialization and error conditions."""

    def test_ensure_writer_without_open_file_raises_error(self, tmp_path: Path):
        """Test that ensuring writer without opening file raises error.

        Covers csv_lmsc.py line 203-206 (error when base_dir is None).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str]()

        # Try to create writer without calling _open_file first
        with pytest.raises(CollectorError, match="CSV landmark directory not prepared"):
            csv_lmsc._ensure_writer("test_key", sample_width=3)

    def test_writer_creation_sets_proper_attributes(self, tmp_path: Path):
        """Test that writer creation initializes all tracking attributes.

        Covers csv_lmsc.py lines 207-223 (writer creation and attribute setup).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "output"

        # Open file to set base_dir
        csv_lmsc._open_file(output_dir)

        try:
            # Ensure writer for a new key
            writer = csv_lmsc._ensure_writer("pose", sample_width=99)

            # Verify writer was created
            assert writer is not None
            assert "pose" in csv_lmsc._writers
            assert "pose" in csv_lmsc._file_handles
            assert "pose" in csv_lmsc._sample_width
            assert "pose" in csv_lmsc._row_index

            # Verify tracking values
            assert csv_lmsc._sample_width["pose"] == 99
            assert csv_lmsc._row_index["pose"] == 0

            # Verify file was created (in landmarks subdirectory)
            landmarks_dir = output_dir / "landmarks"
            expected_file = landmarks_dir / "pose.csv"
            assert expected_file.exists()
        finally:
            csv_lmsc._close_file()

    def test_inconsistent_sample_width_raises_error(self, tmp_path: Path):
        """Test that inconsistent sample width for same key raises error.

        Covers csv_lmsc.py lines 224-229 (sample width validation).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "landmarks"

        csv_lmsc._open_file(output_dir)

        try:
            # First call with sample_width=3
            csv_lmsc._ensure_writer("hand", sample_width=3)

            # Second call with different sample_width should raise error
            with pytest.raises(
                CollectorError,
                match="Inconsistent landmark sample width for key 'hand'",
            ):
                csv_lmsc._ensure_writer("hand", sample_width=6)
        finally:
            csv_lmsc._close_file()


class TestCsvLMSCRowWriting:
    """Test CSV row writing with edge cases."""

    def test_write_rows_with_frame_indexing(self, tmp_path: Path):
        """Test that row index is properly tracked across multiple writes.

        Covers csv_lmsc.py line 180 (row index tracking in _append_with_headers).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "output"

        csv_lmsc._open_file(output_dir)

        try:
            # Use _append_result with proper headers
            landmarks_data = {"test": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
            headers_data = {"test": np.array(["x", "y"])}  # Provide proper headers

            # Initial row index should not exist yet
            assert "test" not in csv_lmsc._row_index

            # Append data (3 landmarks)
            csv_lmsc._append_result(0, headers_data, landmarks_data)

            # Verify 3 rows were written
            assert csv_lmsc._row_index["test"] == 3
        finally:
            csv_lmsc._close_file()

    def test_write_row_creates_proper_dict(self, tmp_path: Path):
        """Test that _append_result creates proper rows in CSV.

        Covers csv_lmsc.py line 177
        (dictionary creation for row in _append_with_headers).
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str]()
        output_dir = tmp_path / "output"

        csv_lmsc._open_file(output_dir)

        try:
            # Append landmarks with proper headers
            landmarks_data = {"landmarks": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
            headers_data = {
                "landmarks": np.array(["x", "y", "z"])
            }  # Provide proper headers
            csv_lmsc._append_result(0, headers_data, landmarks_data)

            # Flush file to disk
            for handle in csv_lmsc._file_handles.values():
                handle.flush()

            # Verify file contains the data
            landmarks_dir = output_dir / "landmarks"
            csv_file = landmarks_dir / "landmarks.csv"
            content = csv_file.read_text()

            # Should have header and 2 data rows
            lines = content.strip().split("\n")
            assert len(lines) == 3  # header + 2 data rows
            assert "x,y,z" in lines[0]  # header
            assert "1.0,2.0,3.0" in lines[1]  # first row
            assert "4.0,5.0,6.0" in lines[2]  # second row
        finally:
            csv_lmsc._close_file()


class TestCsvLMSCCompleteWorkflow:
    """Test complete CSV LMSC workflow with real landmarks."""

    def test_complete_csv_writing_workflow(self, tmp_path: Path):
        """Test complete workflow from open to close with multiple keys.

        Ensures all code paths in CSV writing are exercised.
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](delimiter=",")
        output_dir = tmp_path / "output"

        # Open file
        csv_lmsc._open_file(output_dir)

        try:
            # Append landmarks for multiple keys (frame 1) with proper headers
            landmarks_frame1 = {
                "pose": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "hand": np.array([[7.0, 8.0], [9.0, 10.0]]),
            }
            headers = {"pose": np.array(["x", "y", "z"]), "hand": np.array(["x", "y"])}

            # Append first frame
            csv_lmsc._append_result(0, headers, landmarks_frame1)

            # Append second frame (same keys)
            landmarks_frame2 = {
                "pose": np.array([[10.0, 11.0, 12.0]]),
                "hand": np.array([[13.0, 14.0]]),
            }
            csv_lmsc._append_result(1, headers, landmarks_frame2)

            # Flush files to disk
            for handle in csv_lmsc._file_handles.values():
                handle.flush()

            # Verify files exist
            landmarks_dir = output_dir / "landmarks"
            assert (landmarks_dir / "pose.csv").exists()
            assert (landmarks_dir / "hand.csv").exists()

            # Verify row counts (pose: 2+1=3 rows, hand: 2+1=3 rows)
            assert csv_lmsc._row_index["pose"] == 3
            assert csv_lmsc._row_index["hand"] == 3

            # Verify content
            pose_content = (landmarks_dir / "pose.csv").read_text()
            assert "x,y,z" in pose_content
            assert "1.0,2.0,3.0" in pose_content
            assert "4.0,5.0,6.0" in pose_content
            assert "10.0,11.0,12.0" in pose_content

            hand_content = (landmarks_dir / "hand.csv").read_text()
            assert "x,y" in hand_content
            assert "7.0,8.0" in hand_content
            assert "9.0,10.0" in hand_content
            assert "13.0,14.0" in hand_content
        finally:
            csv_lmsc._close_file()

    def test_tsv_format_with_tab_delimiter(self, tmp_path: Path):
        """Test TSV format with tab delimiter.

        Ensures tab delimiter results in proper TSV files.
        """
        csv_lmsc = CsvLandmarkMatrixSaveCollector[str](delimiter="\t")  # Tab delimiter
        output_dir = tmp_path / "output"

        csv_lmsc._open_file(output_dir)

        try:
            landmarks_data = {"test": np.array([[1.0, 2.0]])}
            headers_data = {
                "test": np.array(["col1", "col2"])
            }  # Provide proper headers
            csv_lmsc._append_result(0, headers_data, landmarks_data)

            # Flush file to disk
            for handle in csv_lmsc._file_handles.values():
                handle.flush()

            # Should create .tsv file (auto-detected from delimiter)
            landmarks_dir = output_dir / "landmarks"
            tsv_file = landmarks_dir / "test.tsv"
            assert tsv_file.exists()

            # Content should use tabs
            content = tsv_file.read_text()
            assert "\t" in content  # Tab separator
            # Data row should have tabs, not commas
            lines = content.strip().split("\n")
            assert "\t" in lines[1]  # Data row has tabs
        finally:
            csv_lmsc._close_file()
