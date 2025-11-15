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

"""Tests for the lmpipe.estimator module."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pytest

from cslrtools2.lmpipe.estimator import Estimator, ProcessResult
from cslrtools2.typings import MatLike, NDArrayFloat


class DummyEstimator(Estimator[Literal["test"]]):
    """A dummy estimator for testing purposes."""

    @property
    def shape(self) -> Mapping[Literal["test"], tuple[int, int]]:
        """Return the shape of the estimation result.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`Literal["test"]`, :obj:`tuple`\\[:obj:`int`, :obj:`int`\\]\\]:
                Shape mapped by key.
        """
        return {"test": (1, 3)}  # 1 landmark point with 3 coordinates (x, y, z)

    def estimate(
        self, frame_src: MatLike | None, frame_idx: int
    ) -> Mapping[Literal["test"], NDArrayFloat]:
        """Estimate landmarks from a frame.

        Args:
            frame_src (`MatLike` | :obj:`None`): Source frame.
            frame_idx (`int`): Frame index.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`Literal["test"]`, :class:`NDArrayFloat`\\]:
                Estimated landmarks mapped by key.
        """
        return {"test": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}

    def process(self, frame: MatLike) -> ProcessResult[Literal["test"]]:
        """Process a dummy frame.

        Args:
            frame (`MatLike`): Input frame (not actually used).

        Returns:
            :class:`ProcessResult`: A minimal valid result.
        """
        headers: Mapping[Literal["test"], np.ndarray] = {
            "test": np.array(["x", "y", "z"], dtype=str)
        }
        landmarks: Mapping[Literal["test"], np.ndarray] = {
            "test": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        }
        return ProcessResult(
            frame_id=0,
            headers=headers,
            landmarks=landmarks,
            annotated_frame=frame,
        )

    def configure_estimator_name(self) -> Literal["test"]:
        """Return the estimator name.

        Returns:
            :obj:`Literal["test"]`: The fixed estimator name.
        """
        return "test"


class TestEstimatorABC:
    """Test the Estimator abstract base class."""

    def test_estimator_abstract(self) -> None:
        """Test that Estimator is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Estimator()  # type: ignore[reportAbstractUsage]

    def test_dummy_estimator_instantiation(self) -> None:
        """Test that a concrete Estimator subclass can be instantiated."""
        estimator = DummyEstimator()
        assert estimator is not None
        assert isinstance(estimator, Estimator)

    def test_configure_estimator_name(self) -> None:
        """Test that configure_estimator_name returns the correct name."""
        estimator = DummyEstimator()
        name = estimator.configure_estimator_name()
        assert name == "test"

    def test_process_returns_valid_result(self) -> None:
        """Test that process returns a valid ProcessResult."""
        estimator = DummyEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = estimator.process(frame)
        
        assert isinstance(result, ProcessResult)
        assert result.frame_id == 0
        assert "test" in result.headers
        assert "test" in result.landmarks
        assert np.array_equal(result.annotated_frame, frame)


class TestProcessResult:
    """Test the ProcessResult dataclass."""

    def test_processresult_creation(self) -> None:
        """Test creating a ProcessResult instance."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        headers = {"test": np.array(["x", "y"], dtype=str)}
        landmarks = {"test": np.array([[1.0, 2.0]], dtype=np.float32)}
        
        result = ProcessResult(
            frame_id=42,
            headers=headers,
            landmarks=landmarks,
            annotated_frame=frame,
        )
        
        assert result.frame_id == 42
        assert "test" in result.headers
        assert "test" in result.landmarks
        assert np.array_equal(result.headers["test"], np.array(["x", "y"]))
        assert np.array_equal(result.landmarks["test"], np.array([[1.0, 2.0]]))

    def test_processresult_with_multiple_keys(self) -> None:
        """Test ProcessResult with multiple landmark keys."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        headers = {
            "pose": np.array(["x", "y", "z"], dtype=str),
            "hand": np.array(["x", "y"], dtype=str),
        }
        landmarks = {
            "pose": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "hand": np.array([[4.0, 5.0]], dtype=np.float32),
        }
        
        result = ProcessResult(
            frame_id=0,
            headers=headers,
            landmarks=landmarks,
            annotated_frame=frame,
        )
        
        assert len(result.headers) == 2
        assert len(result.landmarks) == 2
        assert "pose" in result.headers
        assert "hand" in result.headers
