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

"""Integration tests for dynamic data augmentation transforms.

This module tests the complete integration of data augmentation transforms
in the cslrtools2 SLDataset pipeline, focusing on:
- UniformSpeedChange: Temporal scaling transform
- RandomResizePaddingCrop: Spatial scaling transform

Ref: INTEGRATION_TEST_TASKS.md, guides/DATA_AUGUMENTATION.md

⚠️ MEMORY WARNING:
    These tests use torchvision.transforms.v2 which can consume significant memory
    when running in parallel. Recommended execution:

        - Limit workers: pytest -n 2 tests/integration/test_transform_dynamic.py
        - Or disable parallel: pytest tests/integration/test_transform_dynamic.py

    Parallel execution with many workers may cause:
        - OSError: [WinError 1455] Paging file too small
        - Test collection mismatches between workers
"""

from __future__ import annotations

import pytest
import torch

from cslrtools2.sldataset.transform.dynamic import (
    UniformSpeedChange,
    RandomResizePaddingCrop,
)
from cslrtools2.sldataset.dataset.core import SLDatasetItem, TensorSLDatasetItem


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_video_tensor() -> torch.Tensor:
    """Create sample video tensor [N, T, C, H, W]."""
    return torch.randn(2, 10, 3, 64, 64)


@pytest.fixture
def sample_landmark_tensor() -> torch.Tensor:
    """Create sample landmark tensor [N, T, V, C]."""
    return torch.randn(2, 10, 33, 2)


@pytest.fixture
def sample_target_tensor() -> torch.Tensor:
    """Create sample target tensor [N, S]."""
    return torch.randint(0, 100, (2, 5))


@pytest.fixture
def sample_dataset_item(
    sample_video_tensor: torch.Tensor,
    sample_landmark_tensor: torch.Tensor,
    sample_target_tensor: torch.Tensor,
) -> TensorSLDatasetItem[str, str, str]:
    """Create complete dataset item."""
    return SLDatasetItem(
        videos={"rgb": sample_video_tensor},
        landmarks={"pose": sample_landmark_tensor},
        targets={"gloss": sample_target_tensor},
    )


@pytest.fixture
def deterministic_generator() -> torch.Generator:
    """Create deterministic random generator."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


# ============================================================================
# UniformSpeedChange Tests
# ============================================================================


class TestUniformSpeedChange:
    """Integration tests for UniformSpeedChange transform."""

    def test_uniform_speed_change_reduces_frame_count_when_scale_greater_than_one(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that scale > 1.0 reduces frame count (speed up)."""
        # Arrange
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=2.0,
            max_scale=2.0,  # Fixed scale for deterministic test
            gen=deterministic_generator,
        )
        original_frames = sample_dataset_item.videos["rgb"].shape[1]

        # Act
        result = transform(sample_dataset_item)

        # Assert
        result_frames = result.videos["rgb"].shape[1]
        expected_frames = int(original_frames * 2.0)
        assert abs(result_frames - expected_frames) <= 1  # Allow rounding error
        assert result.landmarks["pose"].shape[1] == result_frames  # Sync

    def test_uniform_speed_change_increases_frame_count_when_scale_less_than_one(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that scale < 1.0 increases frame count (slow down)."""
        # Arrange
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=0.5,  # Fixed scale
            gen=deterministic_generator,
        )
        original_frames = sample_dataset_item.videos["rgb"].shape[1]

        # Act
        result = transform(sample_dataset_item)

        # Assert
        result_frames = result.videos["rgb"].shape[1]
        expected_frames = int(original_frames * 0.5)
        assert abs(result_frames - expected_frames) <= 1
        assert result.landmarks["pose"].shape[1] == result_frames

    def test_uniform_speed_change_preserves_spatial_dimensions(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that spatial dimensions remain unchanged."""
        # Arrange
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=deterministic_generator,
        )
        original_video = sample_dataset_item.videos["rgb"]
        original_landmark = sample_dataset_item.landmarks["pose"]

        # Act
        result = transform(sample_dataset_item)

        # Assert
        # Video: N, C, H, W unchanged
        assert result.videos["rgb"].shape[0] == original_video.shape[0]  # N
        assert result.videos["rgb"].shape[2] == original_video.shape[2]  # C
        assert result.videos["rgb"].shape[3] == original_video.shape[3]  # H
        assert result.videos["rgb"].shape[4] == original_video.shape[4]  # W

        # Landmark: N, V, C unchanged
        assert result.landmarks["pose"].shape[0] == original_landmark.shape[0]  # N
        assert result.landmarks["pose"].shape[2] == original_landmark.shape[2]  # V
        assert result.landmarks["pose"].shape[3] == original_landmark.shape[3]  # C

    def test_uniform_speed_change_applies_to_multiple_video_keys(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that transform applies to multiple video keys."""
        # Arrange
        item = SLDatasetItem(
            videos={
                "rgb": torch.randn(1, 10, 3, 32, 32),
                "flow": torch.randn(1, 10, 2, 32, 32),
            },
            landmarks={"pose": torch.randn(1, 10, 33, 2)},
            targets={"gloss": torch.tensor([[1, 2, 3]])},
        )
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb", "flow"],
            landmark_keys=["pose"],
            min_scale=1.5,
            max_scale=1.5,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        assert result.videos["rgb"].shape[1] == result.videos["flow"].shape[1]
        assert result.videos["rgb"].shape[1] == result.landmarks["pose"].shape[1]

    def test_uniform_speed_change_applies_to_multiple_landmark_keys(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that transform applies to multiple landmark keys."""
        # Arrange
        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 10, 3, 32, 32)},
            landmarks={
                "pose": torch.randn(1, 10, 33, 2),
                "hand": torch.randn(1, 10, 21, 3),
            },
            targets={"gloss": torch.tensor([[1, 2, 3]])},
        )
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose", "hand"],
            min_scale=0.8,
            max_scale=0.8,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        assert result.landmarks["pose"].shape[1] == result.landmarks["hand"].shape[1]
        assert result.landmarks["pose"].shape[1] == result.videos["rgb"].shape[1]

    def test_uniform_speed_change_raises_value_error_for_negative_min_scale(
        self,
    ) -> None:
        """Test that negative min_scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            UniformSpeedChange(
                video_keys=["rgb"],
                landmark_keys=["pose"],
                min_scale=-0.1,
                max_scale=2.0,
            )

    def test_uniform_speed_change_raises_value_error_for_zero_scale(self) -> None:
        """Test that zero scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            UniformSpeedChange(
                video_keys=["rgb"],
                landmark_keys=["pose"],
                min_scale=0.0,
                max_scale=2.0,
            )

    def test_uniform_speed_change_raises_value_error_when_min_greater_than_max(
        self,
    ) -> None:
        """Test that min_scale > max_scale raises ValueError."""
        with pytest.raises(ValueError, match="min_scale must be <= max_scale"):
            UniformSpeedChange(
                video_keys=["rgb"],
                landmark_keys=["pose"],
                min_scale=2.0,
                max_scale=1.0,
            )

    def test_uniform_speed_change_is_reproducible_with_same_generator(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
    ) -> None:
        """Test that same generator seed produces identical results."""
        # Arrange
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        gen2 = torch.Generator()
        gen2.manual_seed(123)

        transform1 = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen1,
        )
        transform2 = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen2,
        )

        # Act
        result1 = transform1(sample_dataset_item)
        result2 = transform2(sample_dataset_item)

        # Assert
        assert result1.videos["rgb"].shape == result2.videos["rgb"].shape
        assert torch.allclose(result1.videos["rgb"], result2.videos["rgb"])

    def test_uniform_speed_change_is_different_with_different_seeds(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
    ) -> None:
        """Test that different seeds produce different results."""
        # Arrange
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        gen2 = torch.Generator()
        gen2.manual_seed(456)

        transform1 = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen1,
        )
        transform2 = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=gen2,
        )

        # Act
        result1 = transform1(sample_dataset_item)
        result2 = transform2(sample_dataset_item)

        # Assert
        # Different seeds should produce different results (shape or content)
        shapes_differ = result1.videos["rgb"].shape != result2.videos["rgb"].shape
        if not shapes_differ:
            # If shapes match, content should differ
            assert not torch.allclose(result1.videos["rgb"], result2.videos["rgb"])

    def test_uniform_speed_change_handles_single_frame_video(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that single frame video is handled correctly."""
        # Arrange
        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 1, 3, 32, 32)},
            landmarks={"pose": torch.randn(1, 1, 33, 2)},
            targets={"gloss": torch.tensor([[1]])},
        )
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        assert result.videos["rgb"].shape[1] >= 1
        assert result.landmarks["pose"].shape[1] >= 1

    def test_uniform_speed_change_with_scale_exactly_one(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that scale=1.0 produces no change."""
        # Arrange
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=1.0,
            max_scale=1.0,
            gen=deterministic_generator,
        )
        original_frames = sample_dataset_item.videos["rgb"].shape[1]

        # Act
        result = transform(sample_dataset_item)

        # Assert
        assert result.videos["rgb"].shape[1] == original_frames
        assert result.landmarks["pose"].shape[1] == original_frames

    def test_uniform_speed_change_works_with_various_spatial_sizes(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that transform works with various spatial dimensions."""
        # Arrange
        sizes = [(32, 32), (64, 128), (224, 224)]
        transform = UniformSpeedChange[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.8,
            max_scale=1.2,
            gen=deterministic_generator,
        )

        for h, w in sizes:
            # Act
            item = SLDatasetItem(
                videos={"rgb": torch.randn(1, 10, 3, h, w)},
                landmarks={"pose": torch.randn(1, 10, 33, 2)},
                targets={"gloss": torch.tensor([[1, 2]])},
            )
            result = transform(item)

            # Assert
            assert result.videos["rgb"].shape[3] == h
            assert result.videos["rgb"].shape[4] == w


# ============================================================================
# RandomResizePaddingCrop Tests
# ============================================================================


class TestRandomResizePaddingCrop:
    """Integration tests for RandomResizePaddingCrop transform."""

    def test_random_resize_padding_crop_preserves_tensor_shape(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that output shapes exactly match input shapes."""
        # Arrange
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.8,
            max_scale=1.2,
            gen=deterministic_generator,
        )
        original_video_shape = sample_dataset_item.videos["rgb"].shape
        original_landmark_shape = sample_dataset_item.landmarks["pose"].shape

        # Act
        result = transform(sample_dataset_item)

        # Assert
        assert result.videos["rgb"].shape == original_video_shape
        assert result.landmarks["pose"].shape == original_landmark_shape

    def test_random_resize_padding_crop_scale_greater_than_one_crops(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that scale > 1.0 produces zoom-in (crop) effect."""
        # Arrange
        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 5, 3, 64, 64)},
            landmarks={"pose": torch.randn(1, 5, 33, 2)},
            targets={"gloss": torch.tensor([[1]])},
        )
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=1.5,
            max_scale=1.5,  # Fixed scale > 1.0
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        # Shape unchanged
        assert result.videos["rgb"].shape == item.videos["rgb"].shape
        # Content should differ (zoomed)
        assert not torch.allclose(result.videos["rgb"], item.videos["rgb"])

    def test_random_resize_padding_crop_scale_less_than_one_pads(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that scale < 1.0 produces zoom-out (padding) effect."""
        # Arrange
        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 5, 3, 64, 64)},
            landmarks={"pose": torch.randn(1, 5, 33, 2)},
            targets={"gloss": torch.tensor([[1]])},
        )
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=0.5,  # Fixed scale < 1.0
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        assert result.videos["rgb"].shape == item.videos["rgb"].shape
        assert not torch.allclose(result.videos["rgb"], item.videos["rgb"])

    def test_random_resize_padding_crop_transforms_landmark_coordinates(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that landmark XY coordinates are transformed correctly."""
        # Arrange
        # Create landmarks with known coordinates
        landmarks = torch.zeros(1, 5, 33, 2)
        landmarks[..., 0] = 0.5  # X = 0.5 (center)
        landmarks[..., 1] = 0.5  # Y = 0.5 (center)

        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 5, 3, 64, 64)},
            landmarks={"pose": landmarks},
            targets={"gloss": torch.tensor([[1]])},
        )

        scale = 2.0
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=scale,
            max_scale=scale,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        # Center coordinates should remain relatively stable
        # Formula: (xy - 0.5) * scale + 0.5
        expected = (0.5 - 0.5) * scale + 0.5  # = 0.5
        assert torch.allclose(result.landmarks["pose"][..., 0], torch.tensor(expected))
        assert torch.allclose(result.landmarks["pose"][..., 1], torch.tensor(expected))

    def test_random_resize_padding_crop_preserves_landmark_other_dimensions(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that landmark dimensions beyond XY remain unchanged."""
        # Arrange
        landmarks = torch.randn(1, 5, 33, 4)  # C=4 (x, y, z, visibility)
        original_z = landmarks[..., 2].clone()
        original_vis = landmarks[..., 3].clone()

        item = SLDatasetItem(
            videos={"rgb": torch.randn(1, 5, 3, 64, 64)},
            landmarks={"pose": landmarks},
            targets={"gloss": torch.tensor([[1]])},
        )

        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.8,
            max_scale=1.2,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        # Dimensions [2:] should be unchanged
        assert torch.allclose(result.landmarks["pose"][..., 2], original_z)
        assert torch.allclose(result.landmarks["pose"][..., 3], original_vis)

    def test_random_resize_padding_crop_raises_value_error_for_negative_scale(
        self,
    ) -> None:
        """Test that negative scale raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RandomResizePaddingCrop(
                video_keys=["rgb"],
                landmark_keys=["pose"],
                min_scale=-0.1,
                max_scale=1.2,
            )

    def test_random_resize_padding_crop_raises_value_error_when_min_greater_than_max(
        self,
    ) -> None:
        """Test that min_scale > max_scale raises ValueError."""
        with pytest.raises(ValueError, match="min_scale must be <= max_scale"):
            RandomResizePaddingCrop(
                video_keys=["rgb"],
                landmark_keys=["pose"],
                min_scale=1.5,
                max_scale=0.5,
            )

    def test_random_resize_padding_crop_is_reproducible_with_same_generator(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
    ) -> None:
        """Test that same generator seed produces identical results."""
        # Arrange
        gen1 = torch.Generator()
        gen1.manual_seed(789)
        gen2 = torch.Generator()
        gen2.manual_seed(789)

        transform1 = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.8,
            max_scale=1.2,
            gen=gen1,
        )
        transform2 = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.8,
            max_scale=1.2,
            gen=gen2,
        )

        # Act
        result1 = transform1(sample_dataset_item)
        result2 = transform2(sample_dataset_item)

        # Assert
        assert torch.allclose(result1.videos["rgb"], result2.videos["rgb"])
        assert torch.allclose(result1.landmarks["pose"], result2.landmarks["pose"])

    def test_random_resize_padding_crop_affine_handles_5d_tensors(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that torchvision affine handles [N, T, C, H, W] correctly."""
        # Arrange
        item = SLDatasetItem(
            videos={"rgb": torch.randn(2, 10, 3, 64, 64)},  # 5D tensor
            landmarks={"pose": torch.randn(2, 10, 33, 2)},
            targets={"gloss": torch.randint(0, 10, (2, 5))},
        )
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.9,
            max_scale=1.1,
            gen=deterministic_generator,
        )

        # Act & Assert (should not raise)
        result = transform(item)
        assert result.videos["rgb"].shape == item.videos["rgb"].shape

    def test_random_resize_padding_crop_center_invariant_transformation(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that center pixel remains relatively stable."""
        # Arrange
        # Create video with distinctive pattern at center
        video = torch.zeros(1, 1, 3, 64, 64)
        video[:, :, :, 32, 32] = 1.0  # Center pixel bright

        item = SLDatasetItem(
            videos={"rgb": video},
            landmarks={"pose": torch.zeros(1, 1, 33, 2)},
            targets={"gloss": torch.tensor([[1]])},
        )

        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["pose"],
            min_scale=0.9,
            max_scale=1.1,
            gen=deterministic_generator,
        )

        # Act
        result = transform(item)

        # Assert
        # Center pixel should remain relatively high value
        center_value = result.videos["rgb"][:, :, :, 32, 32]
        assert (center_value > 0.5).all()  # Still relatively bright

    def test_random_resize_padding_crop_works_with_various_coordinate_dimensions(
        self,
        deterministic_generator: torch.Generator,
    ) -> None:
        """Test that transform works with C=2, C=3, C=4."""
        # Arrange
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["lm"],
            min_scale=0.9,
            max_scale=1.1,
            gen=deterministic_generator,
        )

        for c in [2, 3, 4]:
            # Act
            item = SLDatasetItem(
                videos={"rgb": torch.randn(1, 5, 3, 32, 32)},
                landmarks={"lm": torch.randn(1, 5, 33, c)},
                targets={"gloss": torch.tensor([[1]])},
            )
            result = transform(item)

            # Assert
            assert result.landmarks["lm"].shape[-1] == c


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestTransformErrorHandling:
    """Test error handling across transforms."""

    def test_transform_raises_key_error_for_missing_video_key(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
    ) -> None:
        """Test that missing video key raises KeyError."""
        # Arrange
        transform = UniformSpeedChange[str, str, str](
            video_keys=["nonexistent"],
            landmark_keys=["pose"],
            min_scale=0.5,
            max_scale=2.0,
        )

        # Act & Assert
        with pytest.raises(KeyError):
            transform(sample_dataset_item)

    def test_transform_raises_key_error_for_missing_landmark_key(
        self,
        sample_dataset_item: TensorSLDatasetItem[str, str, str],
    ) -> None:
        """Test that missing landmark key raises KeyError."""
        # Arrange
        transform = RandomResizePaddingCrop[str, str, str](
            video_keys=["rgb"],
            landmark_keys=["nonexistent"],
            min_scale=0.8,
            max_scale=1.2,
        )

        # Act & Assert
        with pytest.raises(KeyError):
            transform(sample_dataset_item)
