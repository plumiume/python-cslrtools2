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

"""Unit tests for lmpipe/collector/landmark_matrix/safetensors_lmsc.py

Tests for SafetensorsLandmarkMatrixSaveCollector (container mode only).
Coverage target: 51% → 85%+
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
from pathlib import Path
from typing import Literal, Mapping

from cslrtools2.lmpipe.collector.landmark_matrix.safetensors_lmsc import (
    SafetensorsLandmarkMatrixSaveCollector,
    safetensors_lmsc_creator,
)
from cslrtools2.typings import NDArrayFloat, NDArrayStr


def _make_empty_headers[K: str](
    landmarks: Mapping[K, NDArrayFloat],
) -> dict[K, NDArrayStr]:
    """Create empty header mappings for landmarks (used when headers are not needed)."""
    return {key: np.array([], dtype=str) for key in landmarks.keys()}


# Import safetensors for reading
try:
    from safetensors.numpy import load_file  # type: ignore[import-not-found]

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False  # pyright: ignore[reportConstantRedefinition]

    def load_file(path: str) -> dict[str, np.ndarray]:
        """Dummy function for type checking."""
        raise ImportError("safetensors not available")


pytestmark = pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not available")


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


class TestSafetensorsLMSCInitialization:
    """Tests for SafetensorsLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization(self):
        """Test default initialization (container mode only)."""
        collector = SafetensorsLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.is_perkey is False
        assert collector.is_container is True
        assert collector.file_ext == ".safetensors"
        assert collector.USE_LANDMARK_DIR is False


class TestSafetensorsLMSCContainerMode:
    """Tests for Safetensors container mode (always container)."""

    def test_save_single_key(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key."""
        collector = SafetensorsLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify safetensors file exists
        safetensors_file = temp_output_dir / "landmarks.safetensors"
        assert safetensors_file.exists()
        assert safetensors_file.is_file()

        # Verify content
        data = load_file(str(safetensors_file))
        assert "pose" in data
        assert data["pose"].shape == (1, 10, 33, 3)

    def test_save_multiple_keys(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys."""
        collector = SafetensorsLandmarkMatrixSaveCollector[
            Literal["pose", "left_hand"]
        ]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify single safetensors file with both keys
        safetensors_file = temp_output_dir / "landmarks.safetensors"
        assert safetensors_file.exists()

        data = load_file(str(safetensors_file))
        assert "pose" in data
        assert "left_hand" in data
        assert data["pose"].shape == (1, 5, 33, 3)
        assert data["left_hand"].shape == (1, 5, 21, 3)

    def test_multiple_appends(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple appends (stacking behavior)."""
        collector = SafetensorsLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        safetensors_file = temp_output_dir / "landmarks.safetensors"
        data = load_file(str(safetensors_file))
        assert data["pose"].shape == (2, 10, 33, 3)


class TestSafetensorsLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self, temp_output_dir: Path):
        """Test saving empty array."""
        collector = SafetensorsLandmarkMatrixSaveCollector[Literal["pose"]]()

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        safetensors_file = temp_output_dir / "landmarks.safetensors"
        data = load_file(str(safetensors_file))
        assert data["pose"].shape == (1, 0, 33, 3)

    def test_no_appends(self, temp_output_dir: Path):
        """Test no appends (creates empty file)."""
        collector = SafetensorsLandmarkMatrixSaveCollector[Literal["pose"]]()

        collector._open_file(temp_output_dir)
        # Don't append anything
        collector._close_file()

        safetensors_file = temp_output_dir / "landmarks.safetensors"
        assert safetensors_file.exists()

        data = load_file(str(safetensors_file))
        # Empty safetensors file has no keys
        assert len(data) == 0


class TestSafetensorsLMSCCreators:
    """Tests for creator functions."""

    def test_safetensors_lmsc_creator(self):
        """Test safetensors_lmsc_creator function."""
        collector = safetensors_lmsc_creator(str)
        assert isinstance(collector, SafetensorsLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".safetensors"
        assert collector.is_container is True
        assert collector.is_perkey is False
