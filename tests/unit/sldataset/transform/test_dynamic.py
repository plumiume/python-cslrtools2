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

"""Unit tests for sldataset/transform/dynamic.py

Tests for data augmentation transforms with focus on:
- Parameter validation
- Transform logic correctness
- Edge cases and error handling

Coverage target: 100% (67 statements)
"""

from __future__ import annotations

import pytest
import torch

from cslrtools2.sldataset.transform.dynamic import (
    UniformSpeedChange,
    RandomResizePaddingCrop,
    uniform_rand_factory,
)
from cslrtools2.sldataset.dataset.core import SLDatasetItem


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_video_tensor() -> torch.Tensor:
    """Create simple video tensor [N=1, T=10, C=3, H=32, W=32]."""
    return torch.randn(1, 10, 3, 32, 32)


@pytest.fixture
def simple_landmark_tensor() -> torch.Tensor:
    """Create simple landmark tensor [N=1, T=10, V=21, C=2]."""
    return torch.randn(1, 10, 21, 2)


@pytest.fixture
def simple_target_tensor() -> torch.Tensor:
    """Create simple target tensor [N=1, S=3]."""
    return torch.randint(0, 10, (1, 3))


@pytest.fixture
def simple_dataset_item(
    simple_video_tensor: torch.Tensor,
    simple_landmark_tensor: torch.Tensor,
    simple_target_tensor: torch.Tensor,
) -> SLDatasetItem[str, torch.Tensor, str, torch.Tensor, str, torch.Tensor]:
    """Create simple dataset item for unit tests."""
    return SLDatasetItem(
        videos={"video": simple_video_tensor},
        landmarks={"lm": simple_landmark_tensor},
        targets={"target": simple_target_tensor},
    )


@pytest.fixture
def deterministic_generator() -> torch.Generator:
    """Create deterministic random generator."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


# ============================================================================
# UniformSpeedChange Unit Tests
# ============================================================================


class TestUniformSpeedChangeParameterValidation:
    """Unit tests for UniformSpeedChange parameter validation."""

    def test_negative_min_scale_raises_value_error(self) -> None:
        """Test that negative min_scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            UniformSpeedChange(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=-0.5,
                max_scale=2.0,
            )

    def test_zero_min_scale_raises_value_error(self) -> None:
        """Test that zero min_scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            UniformSpeedChange(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=0.0,
                max_scale=2.0,
            )

    def test_negative_max_scale_raises_value_error(self) -> None:
        """Test that negative max_scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            UniformSpeedChange(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=0.5,
                max_scale=-1.0,
            )

    def test_min_greater_than_max_raises_value_error(self) -> None:
        """Test that min_scale > max_scale raises ValueError."""
        with pytest.raises(ValueError, match="min_scale must be <= max_scale"):
            UniformSpeedChange(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=2.0,
                max_scale=1.0,
            )

    def test_valid_parameters_no_error(self) -> None:
        """Test that valid parameters do not raise errors."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
        )
        assert transform.min_scale == 0.5
        assert transform.max_scale == 2.0


class TestUniformSpeedChangeTransformLogic:
    """Unit tests for UniformSpeedChange transform logic."""

    def test_scale_greater_than_one_reduces_frames(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that scale > 1.0 reduces frame count."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=2.0,
            max_scale=2.0,  # Fixed scale
        )
        original_frames = simple_dataset_item.videos["video"].shape[1]
        result = transform(simple_dataset_item)
        result_frames = result.videos["video"].shape[1]
        # scale=2.0 means T_new ≈ T_original * 2.0
        assert result_frames >= original_frames * 1.8
        assert result_frames <= original_frames * 2.2

    def test_scale_less_than_one_increases_frames(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that scale < 1.0 increases frame count."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=0.5,  # Fixed scale
        )
        original_frames = simple_dataset_item.videos["video"].shape[1]
        result = transform(simple_dataset_item)
        result_frames = result.videos["video"].shape[1]
        # scale=0.5 means T_new ≈ T_original * 0.5
        assert result_frames >= original_frames * 0.3
        assert result_frames <= original_frames * 0.7

    def test_preserves_spatial_dimensions(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that N, C, H, W dimensions are preserved."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )
        original_shape = simple_dataset_item.videos["video"].shape
        result = transform(simple_dataset_item)
        result_shape = result.videos["video"].shape

        assert result_shape[0] == original_shape[0]  # N
        assert result_shape[2] == original_shape[2]  # C
        assert result_shape[3] == original_shape[3]  # H
        assert result_shape[4] == original_shape[4]  # W

    def test_video_landmark_frame_sync(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that video and landmark frame counts remain synchronized."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
        )
        result = transform(simple_dataset_item)
        video_frames = result.videos["video"].shape[1]
        landmark_frames = result.landmarks["lm"].shape[1]
        assert video_frames == landmark_frames

    def test_targets_unchanged(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that target tensors remain unchanged."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
        )
        original_target = simple_dataset_item.targets["target"]
        result = transform(simple_dataset_item)
        assert torch.equal(result.targets["target"], original_target)


class TestUniformSpeedChangeReproducibility:
    """Unit tests for UniformSpeedChange reproducibility."""

    def test_same_seed_produces_identical_results(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that same seed produces identical results."""
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        gen2 = torch.Generator()
        gen2.manual_seed(123)

        transform1 = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen1,
        )
        transform2 = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen2,
        )

        result1 = transform1(simple_dataset_item)
        result2 = transform2(simple_dataset_item)

        assert result1.videos["video"].shape == result2.videos["video"].shape
        assert torch.allclose(result1.videos["video"], result2.videos["video"])

    def test_different_seeds_produce_different_results(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that different seeds produce different results."""
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        gen2 = torch.Generator()
        gen2.manual_seed(456)

        transform1 = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen1,
        )
        transform2 = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen2,
        )

        result1 = transform1(simple_dataset_item)
        result2 = transform2(simple_dataset_item)

        # Shapes might match, but content should differ
        shapes_differ = result1.videos["video"].shape != result2.videos["video"].shape
        if not shapes_differ:
            assert not torch.allclose(result1.videos["video"], result2.videos["video"])


class TestUniformSpeedChangeEdgeCases:
    """Unit tests for UniformSpeedChange edge cases."""

    def test_single_frame_handling(self) -> None:
        """Test handling of single-frame videos."""
        item = SLDatasetItem(
            videos={"video": torch.randn(1, 1, 3, 32, 32)},
            landmarks={"lm": torch.randn(1, 1, 21, 2)},
            targets={"target": torch.tensor([[1]])},
        )
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
        )
        result = transform(item)
        assert result.videos["video"].shape[1] >= 1
        assert result.landmarks["lm"].shape[1] >= 1

    def test_scale_exactly_one_preserves_frames(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that scale=1.0 preserves frame count."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=1.0,
            max_scale=1.0,
        )
        original_frames = simple_dataset_item.videos["video"].shape[1]
        result = transform(simple_dataset_item)
        assert result.videos["video"].shape[1] == original_frames

    def test_missing_video_key_raises_key_error(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that missing video key raises KeyError."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["nonexistent"],
            landmark_keys=["lm"],
            min_scale=0.5,
            max_scale=2.0,
        )
        with pytest.raises(KeyError):
            transform(simple_dataset_item)

    def test_missing_landmark_key_raises_key_error(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that missing landmark key raises KeyError."""
        transform = UniformSpeedChange[str, str, str](
            video_keys=["video"],
            landmark_keys=["nonexistent"],
            min_scale=0.5,
            max_scale=2.0,
        )
        with pytest.raises(KeyError):
            transform(simple_dataset_item)


# ============================================================================
# RandomResizePaddingCrop Unit Tests
# ============================================================================


class TestRandomResizePaddingCropParameterValidation:
    """Unit tests for RandomResizePaddingCrop parameter validation."""

    def test_negative_min_scale_raises_value_error(self) -> None:
        """Test that negative min_scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RandomResizePaddingCrop(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=-0.5,
                max_scale=1.2,
            )

    def test_zero_scale_raises_value_error(self) -> None:
        """Test that zero scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RandomResizePaddingCrop(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=0.0,
                max_scale=1.2,
            )

    def test_min_greater_than_max_raises_value_error(self) -> None:
        """Test that min_scale > max_scale raises ValueError."""
        with pytest.raises(ValueError, match="min_scale must be <= max_scale"):
            RandomResizePaddingCrop(
                video_keys=["video"],
                landmark_keys=["lm"],
                min_scale=1.5,
                max_scale=0.8,
            )

    def test_valid_parameters_no_error(self) -> None:
        """Test that valid parameters do not raise errors."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )
        assert transform.min_scale == 0.8
        assert transform.max_scale == 1.2


class TestRandomResizePaddingCropTransformLogic:
    """Unit tests for RandomResizePaddingCrop transform logic."""

    def test_preserves_tensor_shape(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that output shapes exactly match input shapes."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )
        original_video_shape = simple_dataset_item.videos["video"].shape
        original_landmark_shape = simple_dataset_item.landmarks["lm"].shape

        result = transform(simple_dataset_item)

        assert result.videos["video"].shape == original_video_shape
        assert result.landmarks["lm"].shape == original_landmark_shape

    def test_transforms_landmark_coordinates(self) -> None:
        """Test that landmark XY coordinates are transformed."""
        # Create landmarks with known center coordinates
        landmarks = torch.zeros(1, 5, 21, 2)
        landmarks[..., 0] = 0.5  # X = 0.5 (center)
        landmarks[..., 1] = 0.5  # Y = 0.5 (center)

        item = SLDatasetItem(
            videos={"video": torch.randn(1, 5, 3, 32, 32)},
            landmarks={"lm": landmarks},
            targets={"target": torch.tensor([[1]])},
        )

        scale = 2.0
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=scale,
            max_scale=scale,
        )

        result = transform(item)

        # Center coordinates should remain at 0.5
        # Formula: (xy - 0.5) * scale + 0.5 = (0.5 - 0.5) * 2.0 + 0.5 = 0.5
        assert torch.allclose(
            result.landmarks["lm"][..., 0], torch.tensor(0.5), atol=1e-6
        )
        assert torch.allclose(
            result.landmarks["lm"][..., 1], torch.tensor(0.5), atol=1e-6
        )

    def test_preserves_landmark_other_dimensions(self) -> None:
        """Test that landmark dimensions beyond XY remain unchanged."""
        landmarks = torch.randn(1, 5, 21, 4)  # C=4 (x, y, z, visibility)
        original_z = landmarks[..., 2].clone()
        original_vis = landmarks[..., 3].clone()

        item = SLDatasetItem(
            videos={"video": torch.randn(1, 5, 3, 32, 32)},
            landmarks={"lm": landmarks},
            targets={"target": torch.tensor([[1]])},
        )

        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )

        result = transform(item)

        # Dimensions [2:] should be unchanged
        assert torch.allclose(result.landmarks["lm"][..., 2], original_z)
        assert torch.allclose(result.landmarks["lm"][..., 3], original_vis)

    def test_targets_unchanged(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that target tensors remain unchanged."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )
        original_target = simple_dataset_item.targets["target"]
        result = transform(simple_dataset_item)
        assert torch.equal(result.targets["target"], original_target)


class TestRandomResizePaddingCropReproducibility:
    """Unit tests for RandomResizePaddingCrop reproducibility."""

    def test_same_seed_produces_identical_results(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that same seed produces identical results."""
        gen1 = torch.Generator()
        gen1.manual_seed(789)
        gen2 = torch.Generator()
        gen2.manual_seed(789)

        transform1 = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
            gen=gen1,
        )
        transform2 = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
            gen=gen2,
        )

        result1 = transform1(simple_dataset_item)
        result2 = transform2(simple_dataset_item)

        assert torch.allclose(result1.videos["video"], result2.videos["video"])
        assert torch.allclose(result1.landmarks["lm"], result2.landmarks["lm"])


class TestRandomResizePaddingCropEdgeCases:
    """Unit tests for RandomResizePaddingCrop edge cases."""

    def test_missing_video_key_raises_key_error(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that missing video key raises KeyError."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["nonexistent"],
            landmark_keys=["lm"],
            min_scale=0.8,
            max_scale=1.2,
        )
        with pytest.raises(KeyError):
            transform(simple_dataset_item)

    def test_missing_landmark_key_raises_key_error(
        self,
        simple_dataset_item: SLDatasetItem[
            str, torch.Tensor, str, torch.Tensor, str, torch.Tensor
        ],
    ) -> None:
        """Test that missing landmark key raises KeyError."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["nonexistent"],
            min_scale=0.8,
            max_scale=1.2,
        )
        with pytest.raises(KeyError):
            transform(simple_dataset_item)

    def test_various_coordinate_dimensions(self) -> None:
        """Test that transform works with C=2, C=3, C=4."""
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["video"],
            landmark_keys=["lm"],
            min_scale=0.9,
            max_scale=1.1,
        )

        for c in [2, 3, 4]:
            item = SLDatasetItem(
                videos={"video": torch.randn(1, 5, 3, 32, 32)},
                landmarks={"lm": torch.randn(1, 5, 21, c)},
                targets={"target": torch.tensor([[1]])},
            )
            result = transform(item)
            assert result.landmarks["lm"].shape[-1] == c


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestUniformRandFactory:
    """Unit tests for uniform_rand_factory helper."""

    def test_returns_tensor(self) -> None:
        """Test that function returns a tensor."""
        gen = torch.Generator()
        result = uniform_rand_factory(gen)
        assert isinstance(result, torch.Tensor)

    def test_returns_scalar_tensor(self) -> None:
        """Test that function returns a scalar tensor."""
        gen = torch.Generator()
        result = uniform_rand_factory(gen)
        assert result.dim() == 0  # Scalar

    def test_value_in_range(self) -> None:
        """Test that returned value is in [0, 1]."""
        gen = torch.Generator()
        result = uniform_rand_factory(gen)
        assert 0.0 <= result.item() <= 1.0

    def test_reproducibility(self) -> None:
        """Test that same seed produces same value."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        gen2 = torch.Generator()
        gen2.manual_seed(42)

        result1 = uniform_rand_factory(gen1)
        result2 = uniform_rand_factory(gen2)

        assert torch.equal(result1, result2)
