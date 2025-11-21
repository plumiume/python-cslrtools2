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

"""Unit tests for lmpipe/collector/landmark_matrix/torch_lmsc.py

Tests for TorchLandmarkMatrixSaveCollector (dual mode: per-key and container).
Coverage target: 33% → 85%+
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Literal, Mapping
from pathlib import Path

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]
import torch

from cslrtools2.lmpipe.collector.landmark_matrix.torch_lmsc import (
    TorchLandmarkMatrixSaveCollector,
    pt_lmsc_creator,
    pth_lmsc_creator,
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


class TestTorchLMSCInitialization:
    """Tests for TorchLandmarkMatrixSaveCollector initialization."""

    def test_default_initialization_pt(self):
        """Test default initialization with .pt extension."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]]()
        assert collector.is_perkey is False
        assert collector.is_container is True
        assert collector.file_ext == ".pt"
        assert collector.USE_LANDMARK_DIR is False

    def test_initialization_pth_extension(self):
        """Test initialization with .pth extension."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](extension=".pth")
        assert collector.file_ext == ".pth"

    def test_per_key_mode_initialization(self):
        """Test initialization with per_key=True."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)
        assert collector.is_perkey is True
        assert collector.is_container is False
        assert collector.USE_LANDMARK_DIR is True

    def test_invalid_extension(self):
        """Test that invalid extension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid extension"):
            TorchLandmarkMatrixSaveCollector[Literal["pose"]](extension=".txt")


class TestTorchLMSCContainerMode:
    """Tests for Torch container mode (per_key=False)."""

    def test_save_single_key_container_pt(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key in container mode with .pt extension."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](extension=".pt")

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify pt file exists
        pt_file = temp_output_dir / "landmarks.pt"
        assert pt_file.exists()
        assert pt_file.is_file()

        # Verify content
        data = torch.load(pt_file, weights_only=True)
        assert "pose" in data
        assert data["pose"].shape == torch.Size([1, 10, 33, 3])

    def test_save_single_key_container_pth(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key in container mode with .pth extension."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](extension=".pth")

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify pth file exists
        pth_file = temp_output_dir / "landmarks.pth"
        assert pth_file.exists()

        # Verify content
        data = torch.load(pth_file, weights_only=True)
        assert data["pose"].shape == torch.Size([1, 10, 33, 3])

    def test_save_multiple_keys_container(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys in container mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]](
            per_key=False
        )

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify single pt file with both keys
        pt_file = temp_output_dir / "landmarks.pt"
        assert pt_file.exists()

        data = torch.load(pt_file, weights_only=True)
        assert "pose" in data
        assert "left_hand" in data
        assert data["pose"].shape == torch.Size([1, 5, 33, 3])
        assert data["left_hand"].shape == torch.Size([1, 5, 21, 3])

    def test_multiple_appends_container(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple appends in container mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

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

        pt_file = temp_output_dir / "landmarks.pt"
        data = torch.load(pt_file, weights_only=True)
        assert data["pose"].shape == torch.Size([2, 10, 33, 3])


class TestTorchLMSCPerKeyMode:
    """Tests for Torch per-key mode (per_key=True)."""

    def test_save_single_key_perkey(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test saving single key in per-key mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0,
                _make_empty_headers(sample_single_key_result),
                sample_single_key_result,
            )
        finally:
            collector._close_file()

        # Verify landmarks directory and key-specific pt file
        landmarks_dir = temp_output_dir / "landmarks"
        assert landmarks_dir.exists()

        pt_file = landmarks_dir / "pose.pt"
        assert pt_file.exists()
        assert pt_file.is_file()

        # Verify content
        data = torch.load(pt_file, weights_only=True)
        assert data.shape == torch.Size([1, 10, 33, 3])

    def test_save_multiple_keys_perkey(
        self,
        temp_output_dir: Path,
        sample_multi_key_result: dict[Literal["pose", "left_hand"], NDArrayFloat],
    ):
        """Test saving multiple keys in per-key mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose", "left_hand"]](
            per_key=True
        )

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(
                0, _make_empty_headers(sample_multi_key_result), sample_multi_key_result
            )
        finally:
            collector._close_file()

        # Verify separate pt files for each key
        landmarks_dir = temp_output_dir / "landmarks"
        pose_pt = landmarks_dir / "pose.pt"
        hand_pt = landmarks_dir / "left_hand.pt"

        assert pose_pt.exists()
        assert hand_pt.exists()

        # Verify pose content
        pose_data = torch.load(pose_pt, weights_only=True)
        assert pose_data.shape == torch.Size([1, 5, 33, 3])

        # Verify hand content
        hand_data = torch.load(hand_pt, weights_only=True)
        assert hand_data.shape == torch.Size([1, 5, 21, 3])

    def test_multiple_appends_perkey(
        self,
        temp_output_dir: Path,
        sample_single_key_result: dict[Literal["pose"], NDArrayFloat],
    ):
        """Test multiple appends in per-key mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

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

        pt_file = temp_output_dir / "landmarks" / "pose.pt"
        data = torch.load(pt_file, weights_only=True)
        assert data.shape == torch.Size([2, 10, 33, 3])


class TestTorchLMSCEdgeCases:
    """Tests for edge cases."""

    def test_empty_array_container_mode(self, temp_output_dir: Path):
        """Test empty array in container mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        pt_file = temp_output_dir / "landmarks.pt"
        data = torch.load(pt_file, weights_only=True)
        assert data["pose"].shape == torch.Size([1, 0, 33, 3])

    def test_empty_array_perkey_mode(self, temp_output_dir: Path):
        """Test empty array in per-key mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        empty_result: dict[Literal["pose"], NDArrayFloat] = {
            "pose": np.array([], dtype=np.float32).reshape(0, 33, 3)
        }

        collector._open_file(temp_output_dir)
        try:
            collector._append_result(0, _make_empty_headers(empty_result), empty_result)
        finally:
            collector._close_file()

        pt_file = temp_output_dir / "landmarks" / "pose.pt"
        data = torch.load(pt_file, weights_only=True)
        assert data.shape == torch.Size([1, 0, 33, 3])

    def test_no_appends_container_mode(self, temp_output_dir: Path):
        """Test no appends in container mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=False)

        collector._open_file(temp_output_dir)
        # Don't append anything
        collector._close_file()

        pt_file = temp_output_dir / "landmarks.pt"
        assert pt_file.exists()

        data = torch.load(pt_file, weights_only=True)
        # Empty torch file has empty dict
        assert len(data) == 0

    def test_no_appends_perkey_mode(self, temp_output_dir: Path):
        """Test no appends in per-key mode."""
        collector = TorchLandmarkMatrixSaveCollector[Literal["pose"]](per_key=True)

        collector._open_file(temp_output_dir)
        # Manually add empty buffer to trigger file creation
        collector._buffer["pose"] = []
        collector._close_file()

        pt_file = temp_output_dir / "landmarks" / "pose.pt"
        assert pt_file.exists()

        data = torch.load(pt_file, weights_only=True)
        assert data.shape == torch.Size([0])


class TestTorchLMSCCreators:
    """Tests for creator functions."""

    def test_pt_lmsc_creator(self):
        """Test pt_lmsc_creator function (default .pt extension)."""
        collector = pt_lmsc_creator(str)
        assert isinstance(collector, TorchLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".pt"
        assert collector.is_container is True
        assert collector.is_perkey is False

    def test_pth_lmsc_creator(self):
        """Test pth_lmsc_creator function (.pth extension)."""
        collector = pth_lmsc_creator(str)
        assert isinstance(collector, TorchLandmarkMatrixSaveCollector)
        assert collector.file_ext == ".pth"
        assert collector.is_container is True
        assert collector.is_perkey is False


class TestTorchLMSCEmptyBuffer:
    """Tests for empty buffer edge cases."""

    def test_close_file_with_empty_buffer_per_key_mode(self, temp_output_dir: Path):
        """Test _close_file in per-key mode with empty buffer."""
        from cslrtools2.lmpipe.runspec import RunSpec

        collector = TorchLandmarkMatrixSaveCollector[str](per_key=True, extension=".pt")

        video_file = temp_output_dir / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, temp_output_dir)

        # Collect with empty results
        collector.collect_results(runspec, [])

        # Verify landmarks directory was created but empty
        landmarks_dir = temp_output_dir / "landmarks"
        assert landmarks_dir.exists()
        assert landmarks_dir.is_dir()

        # No files should be created for empty buffer
        assert len(list(landmarks_dir.glob("*.pt"))) == 0

    def test_close_file_with_empty_buffer_container_mode(self, temp_output_dir: Path):
        """Test _close_file in container mode with empty buffer."""
        from cslrtools2.lmpipe.runspec import RunSpec

        collector = TorchLandmarkMatrixSaveCollector[str](
            per_key=False, extension=".pt"
        )

        video_file = temp_output_dir / "test.mp4"
        video_file.touch()
        runspec = RunSpec(video_file, temp_output_dir)

        # Collect with empty results
        collector.collect_results(runspec, [])

        # Verify empty pt file was created
        pt_file = temp_output_dir / "landmarks.pt"
        assert pt_file.exists()

        # Verify it's a valid empty dict
        data: dict[Any, Any] = torch.load(pt_file, weights_only=False)
        assert isinstance(data, dict)
        assert len(data) == 0
