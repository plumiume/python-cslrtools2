"""Unit tests for lmpipe/collector/landmark_matrix/json_lmsc.py

Tests for JsonLandmarkMatrixSaveCollector.
Coverage target: 43% → 85%+
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]

from typing import Literal, Mapping

from cslrtools2.lmpipe.collector.landmark_matrix.json_lmsc import (
    JsonLandmarkMatrixSaveCollector,
    json_lmsc_creator,
    jsonc_lmsc_creator,
)
from cslrtools2.typings import NDArrayFloat, NDArrayStr


def _make_empty_headers[K: str](
    landmarks: Mapping[K, NDArrayFloat],
) -> dict[K, NDArrayStr]:
    """Create empty header mappings for landmarks (used when headers are not needed)."""
    return {key: np.array([], dtype=str) for key in landmarks.keys()}


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_single_key_result() -> dict[Literal["pose"], NDArrayFloat]:
    """Create a sample result with a single key."""
    np.random.seed(42)
    return {"pose": np.random.rand(10, 33, 3).astype(np.float32)}


@pytest.fixture
def sample_multi_key_result() -> dict[Literal["pose", "left_hand"], NDArrayFloat]:
    """Create a sample result with multiple keys."""
    np.random.seed(42)
    return {
        "pose": np.random.rand(5, 33, 3).astype(np.float32),
        "left_hand": np.random.rand(5, 21, 3).astype(np.float32),
    }


class TestJSONLMSCInitialization:
    """Tests for JsonLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization(self):
        """Test default initialization with .json extension."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.indent == 2
        assert collector.encoding == "utf-8"
        assert collector.extension == ".json"
        assert collector.is_perkey is True
        assert collector.is_container is False
        assert collector.file_ext == ".json"

    def test_custom_indent(self):
        """Test initialization with custom indent."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](indent=4)
        assert collector.indent == 4

    def test_no_indent(self):
        """Test initialization with no indent (compact JSON)."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](indent=None)
        assert collector.indent is None

    def test_jsonc_extension(self):
        """Test initialization with .jsonc extension."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](extension=".jsonc")
        assert collector.extension == ".jsonc"
        assert collector.file_ext == ".jsonc"

    def test_invalid_extension_raises_error(self):
        """Test that invalid extension raises ValueError."""
        with pytest.raises(
            ValueError, match="Invalid extension.*Must be '.json' or '.jsonc'"
        ):
            JsonLandmarkMatrixSaveCollector[Literal["pose"]](extension=".txt")

    def test_custom_encoding(self):
        """Test initialization with custom encoding."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](encoding="utf-16")
        assert collector.encoding == "utf-16"


class TestJSONLMSCFileOperations:
    """Tests for JSON file operations."""

    def test_save_single_key_json(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key to JSON file."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify file exists
        json_file = temp_output_dir / "landmarks" / "pose.json"
        assert json_file.exists()

        # Verify content
        with json_file.open("r", encoding="utf-8") as f:
            data: list[Any] = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1  # One append call
            assert len(data[0]) == 10  # 10 frames
            assert len(data[0][0]) == 33  # 33 landmarks
            assert len(data[0][0][0]) == 3  # x, y, z

    def test_save_multiple_keys_json(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys to separate JSON files."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify both files exist
        pose_file = temp_output_dir / "landmarks" / "pose.json"
        hand_file = temp_output_dir / "landmarks" / "left_hand.json"
        assert pose_file.exists()
        assert hand_file.exists()

        # Verify pose content
        with pose_file.open("r") as f:
            pose_data = json.load(f)
            assert len(pose_data) == 1
            assert len(pose_data[0]) == 5
            assert len(pose_data[0][0]) == 33

        # Verify hand content
        with hand_file.open("r") as f:
            hand_data = json.load(f)
            assert len(hand_data) == 1
            assert len(hand_data[0]) == 5
            assert len(hand_data[0][0]) == 21

    def test_save_jsonc_format(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving with .jsonc extension."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](extension=".jsonc")

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        jsonc_file = temp_output_dir / "landmarks" / "pose.jsonc"
        assert jsonc_file.exists()

        with jsonc_file.open("r") as f:
            data = json.load(f)
            assert isinstance(data, list)

    def test_compact_json_no_indent(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving compact JSON without indentation."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]](indent=None)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        json_file = temp_output_dir / "landmarks" / "pose.json"
        content = json_file.read_text()
        # Compact JSON should not have newlines within the data
        assert content.count("\n") < 5  # Minimal newlines

    def test_multiple_append_calls(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple append calls accumulate data."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
            collector._append_result(
                1,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )  # Append again
        finally:
            collector._close_file()

        json_file = temp_output_dir / "landmarks" / "pose.json"
        with json_file.open("r") as f:
            data = json.load(f)
            # Should have two entries in the list
            assert len(data) == 2
            assert len(data[0]) == 10
            assert len(data[1]) == 10


class TestJSONLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_landmark_array(self, temp_output_dir: Path):
        """Test handling empty landmark arrays."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        json_file = temp_output_dir / "landmarks" / "pose.json"
        assert json_file.exists()

        with json_file.open("r") as f:
            data = json.load(f)
            assert len(data) == 1  # One append call
            assert len(data[0]) == 0  # Empty array

    def test_single_frame_landmark(self, temp_output_dir: Path):
        """Test handling single frame."""
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()

        single_frame: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(1, 33, 3).astype(np.float32)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(single_frame), single_frame)
        finally:
            collector._close_file()

        json_file = temp_output_dir / "landmarks" / "pose.json"
        with json_file.open("r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert len(data[0]) == 1  # 1 frame

    def test_nested_output_directory_creation(self, temp_output_dir: Path):
        """Test creating nested output directories."""
        nested_path = temp_output_dir / "a" / "b" / "c" / "test.json"
        collector = JsonLandmarkMatrixSaveCollector[Literal["pose"]]()

        result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.random.rand(2, 33, 3).astype(np.float32)
        }

        collector._open_file(nested_path.parent)
        try:
            collector._append_result(0, _make_empty_headers(result), result)
        finally:
            collector._close_file()

        json_file = temp_output_dir / "a" / "b" / "c" / "landmarks" / "pose.json"
        assert json_file.exists()


class TestJSONLMSCCreators:
    """Tests for creator functions."""

    def test_json_lmsc_creator(self):
        """Test json_lmsc_creator function."""
        collector = json_lmsc_creator(str)
        assert isinstance(collector, JsonLandmarkMatrixSaveCollector)
        assert collector.extension == ".json"
        assert collector.indent == 2

    def test_jsonc_lmsc_creator(self):
        """Test jsonc_lmsc_creator function."""
        collector = jsonc_lmsc_creator(str)
        assert isinstance(collector, JsonLandmarkMatrixSaveCollector)
        assert collector.extension == ".jsonc"
        assert collector.indent == 2
