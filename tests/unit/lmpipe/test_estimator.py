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

from typing import Any, Literal, Mapping

import numpy as np
import pytest

from cslrtools2.lmpipe.estimator import (
    Estimator,
    ProcessResult,
    shape,
    headers,
    estimate,
    annotate,
)
from cslrtools2.typings import MatLike, NDArrayFloat


class DummyEstimator(Estimator[Literal["test"]]):
    """A dummy estimator for testing purposes."""

    def __init__(self):
        """Initialize with optional setup tracking."""
        super().__init__()
        self.setup_called = False

    @property
    def shape(self) -> Mapping[Literal["test"], tuple[int, int]]:
        """Return the shape of the estimation result.

        Returns:
            :class:`~typing.Mapping`\\[:obj:`Literal["test"]`,
            :obj:`tuple`\\[:obj:`int`, :obj:`int`\\]\\]: Shape mapped by key.
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

    def setup(self):
        """Track setup calls."""
        self.setup_called = True


class MultiKeyEstimator(Estimator[Literal["pose", "hand"]]):
    """An estimator with multiple output keys."""

    @property
    def shape(self) -> Mapping[Literal["pose", "hand"], tuple[int, int]]:
        """Return shapes for multiple keys."""
        return {
            "pose": (33, 3),  # 33 pose landmarks with x,y,z
            "hand": (21, 2),  # 21 hand landmarks with x,y
        }

    def estimate(
        self, frame_src: MatLike | None, frame_idx: int
    ) -> Mapping[Literal["pose", "hand"], NDArrayFloat]:
        """Estimate multiple landmark sets."""
        return {
            "pose": np.random.rand(33, 3).astype(np.float32),
            "hand": np.random.rand(21, 2).astype(np.float32),
        }

    def configure_estimator_name(self) -> Literal["pose"]:
        """Return primary estimator name."""
        return "pose"


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


class TestEstimatorMethods:
    """Test Estimator class methods."""

    def test_setup_method(self) -> None:
        """Test that setup method can be called and overridden."""
        estimator = DummyEstimator()
        assert not estimator.setup_called
        estimator.setup()
        assert estimator.setup_called

    def test_configure_missing_array(self) -> None:
        """Test configure_missing_array returns correct shape and values."""
        estimator = DummyEstimator()
        missing = estimator.configure_missing_array("test")

        missing_array = np.asarray(missing)
        assert missing_array.shape == (1, 3)
        assert np.all(np.isnan(missing_array))

    def test_missing_value_property(self) -> None:
        """Test that missing_value can be accessed and modified."""
        estimator = DummyEstimator()
        assert np.isnan(estimator.missing_value)

        # Test custom missing value
        estimator.missing_value = -1.0
        missing = estimator.configure_missing_array("test")
        missing_array = np.asarray(missing)
        assert np.all(missing_array == -1.0)

    def test_shape_property_multiple_keys(self) -> None:
        """Test shape property with multiple keys."""
        estimator = MultiKeyEstimator()
        shapes = estimator.shape

        assert len(shapes) == 2
        assert shapes["pose"] == (33, 3)
        assert shapes["hand"] == (21, 2)

    def test_estimate_multiple_keys(self) -> None:
        """Test estimate with multiple output keys."""
        estimator = MultiKeyEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = estimator.estimate(frame, 0)

        assert "pose" in result
        assert "hand" in result
        assert result["pose"].shape == (33, 3)
        assert result["hand"].shape == (21, 2)

    def test_headers_default_implementation(self) -> None:
        """Test that default headers implementation returns None."""
        estimator = DummyEstimator()
        # The decorator converts None to a mapping, so check the result
        headers_result = estimator.headers
        assert isinstance(headers_result, Mapping)

    def test_annotate_default_implementation(self) -> None:
        """Test that default annotate implementation returns None."""
        estimator = DummyEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks: Mapping[Literal["test"], NDArrayFloat] = {
            "test": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        }

        result = estimator.annotate(frame, 0, landmarks)
        # The decorator converts None to original frame
        assert result is not None


class TestDecoratorUsage:
    """Test decorator usage patterns."""

    def test_shape_decorator_with_single_value(self) -> None:
        """Test shape decorator when method returns single tuple."""

        class SingleShapeEstimator(Estimator[Literal["data"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (10, 2)

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> NDArrayFloat | None:
                return np.zeros((10, 2), dtype=np.float32)

            def configure_estimator_name(self) -> Literal["data"]:
                return "data"

        estimator = SingleShapeEstimator()
        shapes = estimator.shape
        assert isinstance(shapes, Mapping)
        assert "data" in shapes
        assert shapes["data"] == (10, 2)

    def test_estimate_decorator_with_none(self) -> None:
        """Test estimate decorator when method returns None."""

        class NoneEstimator(Estimator[Literal["test"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (5, 3)

            @estimate
            def estimate(self, frame_src: MatLike | None, frame_idx: int) -> None:
                return None

            def configure_estimator_name(self) -> Literal["test"]:
                return "test"

        estimator = NoneEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame, 0)

        assert isinstance(result, Mapping)
        assert "test" in result
        # None should be converted to missing array
        assert result["test"].shape == (5, 3)

    def test_headers_decorator_with_array(self) -> None:
        """Test headers decorator when method returns array."""

        class HeadersEstimator(Estimator[Literal["coords"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (3, 2)

            @property
            @headers
            def headers(self) -> list[str]:
                return ["x", "y"]

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> NDArrayFloat:
                return np.zeros((3, 2), dtype=np.float32)

            def configure_estimator_name(self) -> Literal["coords"]:
                return "coords"

        estimator = HeadersEstimator()
        headers_result = estimator.headers

        assert isinstance(headers_result, Mapping)
        assert "coords" in headers_result
        assert len(headers_result["coords"]) == 2


class TestDecoratorWithKeyOptions:
    """Test decorators with KeyOptions TypedDict pattern.

    These tests target the _typeddict_typeis branches in decorators.
    """

    def test_shape_decorator_with_keyoptions(self) -> None:
        """Test shape decorator with KeyOptions pattern (lines 181, 188, 196)."""

        class KeyOptionsShapeEstimator(Estimator[Literal["part"]]):
            @property
            @shape({"key": "part"})  # type: ignore[reportArgumentType]
            def shape(self) -> tuple[int, int]:
                return (8, 4)

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["part"], NDArrayFloat]:
                return {"part": np.zeros((8, 4), dtype=np.float32)}

            def configure_estimator_name(self) -> Literal["part"]:
                return "part"

        estimator = KeyOptionsShapeEstimator()
        shapes = estimator.shape

        assert isinstance(shapes, Mapping)
        assert "part" in shapes
        assert shapes["part"] == (8, 4)

    def test_shape_decorator_returns_mapping(self) -> None:
        """Test shape decorator when method returns Mapping (line 196)."""

        class MappingShapeEstimator(Estimator[Literal["a", "b"]]):
            @property
            @shape
            def shape(self) -> Mapping[Literal["a", "b"], tuple[int, int]]:
                # Directly return a Mapping to hit line 196
                return {"a": (3, 2), "b": (5, 4)}

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["a", "b"], NDArrayFloat]:
                return {
                    "a": np.zeros((3, 2), dtype=np.float32),
                    "b": np.zeros((5, 4), dtype=np.float32),
                }

            def configure_estimator_name(self) -> Literal["a"]:
                return "a"

        estimator = MappingShapeEstimator()
        shapes = estimator.shape

        assert len(shapes) == 2
        assert shapes["a"] == (3, 2)
        assert shapes["b"] == (5, 4)

    def test_shape_decorator_with_invalid_argument(self) -> None:
        """Test shape decorator with invalid non-callable argument (line 188)."""
        with pytest.raises(
            TypeError, match="First argument must be a callable or KeyOptions TypedDict"
        ):

            @shape(123)  # type: ignore[reportArgumentType]
            def invalid_shape(self: Any) -> tuple[int, int]:  # pyright: ignore[reportUnusedFunction] # noqa: E501
                return (1, 1)

    def test_headers_decorator_with_keyoptions(self) -> None:
        """Test headers decorator with KeyOptions pattern (lines 287, 294, 302)."""

        class KeyOptionsHeadersEstimator(Estimator[Literal["feature"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (4, 3)

            @property
            @headers({"key": "feature"})  # type: ignore[reportArgumentType]
            def headers(self) -> list[str]:
                return ["a", "b", "c"]

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["feature"], NDArrayFloat]:
                return {"feature": np.zeros((4, 3), dtype=np.float32)}

            def configure_estimator_name(self) -> Literal["feature"]:
                return "feature"

        estimator = KeyOptionsHeadersEstimator()
        headers_result = estimator.headers

        assert isinstance(headers_result, Mapping)
        assert "feature" in headers_result
        assert len(headers_result["feature"]) == 3

    def test_headers_decorator_returns_mapping(self) -> None:
        """Test headers decorator when method returns Mapping (line 302)."""

        class MappingHeadersEstimator(Estimator[Literal["x", "y"]]):
            @property
            @shape
            def shape(self) -> Mapping[Literal["x", "y"], tuple[int, int]]:
                return {"x": (2, 3), "y": (4, 2)}

            @property
            @headers
            def headers(self) -> Mapping[Literal["x", "y"], list[str]]:
                # Directly return a Mapping to hit line 302
                return {"x": ["a", "b", "c"], "y": ["p", "q"]}

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["x", "y"], NDArrayFloat]:
                return {
                    "x": np.zeros((2, 3), dtype=np.float32),
                    "y": np.zeros((4, 2), dtype=np.float32),
                }

            def configure_estimator_name(self) -> Literal["x"]:
                return "x"

        estimator = MappingHeadersEstimator()
        headers_result = estimator.headers

        assert isinstance(headers_result, Mapping)
        assert len(headers_result) == 2
        # Headers are converted to numpy arrays
        assert np.array_equal(headers_result["x"], np.array(["a", "b", "c"]))
        assert np.array_equal(headers_result["y"], np.array(["p", "q"]))

    def test_headers_decorator_with_invalid_argument(self) -> None:
        """Test headers decorator with invalid non-callable argument (line 294)."""
        with pytest.raises(
            TypeError, match="First argument must be a callable or KeyOptions TypedDict"
        ):

            @headers("invalid")  # type: ignore[reportArgumentType]
            def invalid_headers(self: Any) -> list[str]:  # pyright: ignore[reportUnusedFunction] # noqa: E501
                return ["x"]

    def test_estimate_decorator_with_keyoptions(self) -> None:
        """Test estimate decorator with KeyOptions pattern (lines 412, 419,
        428)."""
        # pyright: ignore[reportUnusedImport]
        from cslrtools2.lmpipe.estimator import KeyOptions

        class KeyOptionsEstimateEstimator(Estimator[Literal["output"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (6, 2)

            @estimate(KeyOptions[Literal["output"]]({"key": "output"}))
            def estimate(
                self, frame_src: MatLike, frame_idx: int
            ) -> Mapping[Literal["output"], NDArrayFloat]:
                return {"output": np.zeros((6, 2), dtype=np.float32)}

            def configure_estimator_name(self) -> Literal["output"]:
                return "output"

        estimator = KeyOptionsEstimateEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame, 0)

        assert isinstance(result, Mapping)
        assert "output" in result
        assert result["output"].shape == (6, 2)

    def test_estimate_decorator_with_none_frame(self) -> None:
        """Test estimate decorator with None frame_src (line 428)."""

        class NoneFrameEstimator(Estimator[Literal["data"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (7, 5)

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["data"], NDArrayFloat]:
                if frame_src is None:
                    # This should never happen as decorator handles it
                    raise AssertionError("frame_src should not be None")
                return {"data": np.zeros((7, 5), dtype=np.float32)}

            def configure_estimator_name(self) -> Literal["data"]:
                return "data"

        estimator = NoneFrameEstimator()
        # Pass None to trigger line 428 in the decorator
        result = estimator.estimate(None, 0)

        assert isinstance(result, Mapping)
        assert "data" in result
        # Should return missing array, not call the actual estimate method
        assert result["data"].shape == (7, 5)
        assert np.all(np.isnan(result["data"]))

    def test_estimate_decorator_with_invalid_argument(self) -> None:
        """Test estimate decorator with invalid non-callable argument (line 419)."""
        with pytest.raises(
            TypeError, match="First argument must be a callable or KeyOptions TypedDict"
        ):

            @estimate([1, 2, 3])  # type: ignore[reportArgumentType]
            def invalid_estimate(  # pyright: ignore[reportUnusedFunction] # noqa: E501
                self: Any, frame_src: MatLike | None, frame_idx: int
            ) -> NDArrayFloat:
                return np.zeros((1, 1), dtype=np.float32)

    def test_annotate_decorator_with_keyoptions(self) -> None:
        """Test annotate decorator with KeyOptions pattern (lines 557, 566,
        580)."""
        # pyright: ignore[reportUnusedImport]
        from cslrtools2.lmpipe.estimator import KeyOptions

        class KeyOptionsAnnotateEstimator(Estimator[Literal["anno"]]):
            @property
            @shape
            def shape(self) -> tuple[int, int]:
                return (3, 2)

            @estimate
            def estimate(
                self, frame_src: MatLike | None, frame_idx: int
            ) -> Mapping[Literal["anno"], NDArrayFloat]:
                return {"anno": np.zeros((3, 2), dtype=np.float32)}

            @annotate(KeyOptions[Literal["anno"]]({"key": "anno"}))
            def annotate(
                self,
                frame_src: MatLike,
                frame_idx: int,
                landmarks: Mapping[Literal["anno"], NDArrayFloat],
            ) -> MatLike:
                return frame_src.copy()

            def configure_estimator_name(self) -> Literal["anno"]:
                return "anno"

        estimator = KeyOptionsAnnotateEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks: Mapping[Literal["anno"], NDArrayFloat] = {
            "anno": np.zeros((3, 2), dtype=np.float32)
        }
        result = estimator.annotate(frame, 0, landmarks)

        assert result is not None
        assert result.shape == frame.shape

    def test_annotate_decorator_with_invalid_argument(self) -> None:
        """Test annotate decorator with invalid non-callable argument (line 566)."""
        with pytest.raises(
            TypeError, match="First argument must be a callable or KeyOptions TypedDict"
        ):

            @annotate(42.0)  # type: ignore[reportArgumentType]
            def invalid_annotate(  # pyright: ignore[reportUnusedFunction] # noqa: E501
                self: Any,
                frame_src: MatLike,
                frame_idx: int,
                landmarks: Mapping[Literal["test"], NDArrayFloat],
            ) -> MatLike:
                return frame_src
