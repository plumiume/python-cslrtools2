# Integration Test Tasks for transform/dynamic.py

**Target Branch**: `tests-enhancement`  
**Source Module**: `src/cslrtools2/sldataset/transform/dynamic.py`  
**Related Guides**: 
- `guides/INTEGRATION_TEST_STRATEGY.md`
- `guides/DATA_AUGUMENTATION.md`
- `guides/CODING_STYLE_GUIDE.md`

---

## Overview

This document outlines integration test tasks for the newly implemented data augmentation transforms in `sldataset/transform/dynamic.py`. Two transforms are currently implemented and require comprehensive testing:

1. **UniformSpeedChange** - Temporal scaling transform
2. **RandomResizePaddingCrop** - Spatial scaling transform

---

## Implemented Transforms

### 1. UniformSpeedChange

**Purpose**: Apply uniform temporal scaling to simulate speed variations in sign language videos.

**Key Features**:
- Interpolation-based temporal scaling (changes frame count)
- Applies to both videos `[N, T, C, H, W]` and landmarks `[N, T, V, C]`
- Scale range: `min_scale` to `max_scale` (default: 0.5 to 2.0)
- Uses `torch.nn.functional.interpolate` for resizing

**Parameters**:
```python
UniformSpeedChange(
    video_keys: Sequence[Kvid],
    landmark_keys: Sequence[Klm],
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    rand_factory: Callable = uniform_rand_factory,
    mode: str = "nearest",
    gen: torch.Generator | None = None,
)
```

**Error Handling**:
- `ValueError`: If scale values are non-positive or `min_scale > max_scale`
- `KeyError`: If specified keys not found in item (natural error propagation)

---

### 2. RandomResizePaddingCrop

**Purpose**: Apply random spatial scaling with automatic padding or cropping.

**Key Features**:
- Center-based affine scaling (tensor shape unchanged)
- Scale > 1.0: zoom in (crop effect)
- Scale < 1.0: zoom out (padding effect)
- Uses `torchvision.transforms.v2.functional.affine` for videos
- Linear coordinate transformation for landmarks

**Parameters**:
```python
RandomResizePaddingCrop(
    video_keys: Sequence[Kvid],
    landmark_keys: Sequence[Klm],
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    gen: torch.Generator | None = None,
)
```

**Error Handling**:
- `ValueError`: If scale values are non-positive or `min_scale > max_scale`
- `KeyError`: If specified keys not found in item

---

## Required Test Cases

### Test File Location
`tests/integration/test_transform_dynamic.py`

### Test Class Structure
```python
import pytest
import torch
from cslrtools2.sldataset.transform.dynamic import (
    UniformSpeedChange,
    RandomResizePaddingCrop,
)
from cslrtools2.sldataset.dataset.core import SLDatasetItem, TensorSLDatasetItem


class TestUniformSpeedChange:
    """Integration tests for UniformSpeedChange transform."""
    
class TestRandomResizePaddingCrop:
    """Integration tests for RandomResizePaddingCrop transform."""
    
class TestTransformErrorHandling:
    """Test error handling across transforms."""
```

---

## Test Cases for UniformSpeedChange

### 1. Basic Functionality Tests

#### `test_uniform_speed_change_reduces_frame_count_when_scale_greater_than_one`
- **Setup**: Create item with T=100 frames, scale=2.0
- **Action**: Apply transform
- **Assert**: Output frame count T' ≈ 50 (speed up)
- **Verify**: Videos and landmarks both have same new T'

#### `test_uniform_speed_change_increases_frame_count_when_scale_less_than_one`
- **Setup**: Create item with T=100 frames, scale=0.5
- **Action**: Apply transform
- **Assert**: Output frame count T' ≈ 200 (slow down)

#### `test_uniform_speed_change_preserves_spatial_dimensions`
- **Setup**: Videos [N, T, C, H, W], landmarks [N, T, V, C]
- **Action**: Apply transform with various scales
- **Assert**: N, C, H, W (videos) and N, V, C (landmarks) unchanged

#### `test_uniform_speed_change_applies_to_multiple_video_keys`
- **Setup**: Item with multiple video keys (e.g., "rgb", "flow")
- **Action**: Apply transform specifying both keys
- **Assert**: All specified videos transformed with same scale

#### `test_uniform_speed_change_applies_to_multiple_landmark_keys`
- **Setup**: Item with multiple landmark keys (e.g., "pose", "hand")
- **Action**: Apply transform specifying both keys
- **Assert**: All specified landmarks transformed with same scale

### 2. Parameter Validation Tests

#### `test_uniform_speed_change_raises_value_error_for_negative_min_scale`
- **Setup**: Try to create transform with `min_scale=-0.1`
- **Assert**: `ValueError` raised

#### `test_uniform_speed_change_raises_value_error_for_zero_scale`
- **Setup**: Try to create transform with `min_scale=0.0`
- **Assert**: `ValueError` raised

#### `test_uniform_speed_change_raises_value_error_when_min_greater_than_max`
- **Setup**: Try to create transform with `min_scale=2.0, max_scale=1.0`
- **Assert**: `ValueError` raised with appropriate message

### 3. Reproducibility Tests

#### `test_uniform_speed_change_is_reproducible_with_same_generator`
- **Setup**: Create two transforms with same Generator seed
- **Action**: Apply both to same item
- **Assert**: Outputs are identical

#### `test_uniform_speed_change_is_different_with_different_seeds`
- **Setup**: Create two transforms with different Generator seeds
- **Action**: Apply both to same item
- **Assert**: Outputs differ (different scales sampled)

### 4. Edge Case Tests

#### `test_uniform_speed_change_handles_single_frame_video`
- **Setup**: Item with T=1 frame
- **Action**: Apply transform
- **Assert**: No crash, output has at least 1 frame

#### `test_uniform_speed_change_with_scale_exactly_one`
- **Setup**: Transform with `min_scale=1.0, max_scale=1.0`
- **Action**: Apply transform
- **Assert**: Frame count unchanged, content identical

---

## Test Cases for RandomResizePaddingCrop

### 1. Basic Functionality Tests

#### `test_random_resize_padding_crop_preserves_tensor_shape`
- **Setup**: Videos [2, 10, 3, 64, 64], landmarks [2, 10, 33, 2]
- **Action**: Apply transform with various scales
- **Assert**: Output shapes exactly match input shapes

#### `test_random_resize_padding_crop_scale_greater_than_one_crops`
- **Setup**: Fixed scale=1.5 (zoom in)
- **Action**: Apply transform
- **Assert**: Video content zoomed (outer regions lost)

#### `test_random_resize_padding_crop_scale_less_than_one_pads`
- **Setup**: Fixed scale=0.5 (zoom out)
- **Action**: Apply transform
- **Assert**: Video has padding regions (zeros at edges)

#### `test_random_resize_padding_crop_transforms_landmark_coordinates`
- **Setup**: Landmarks with known XY coordinates
- **Action**: Apply transform with fixed scale
- **Assert**: Coordinates transformed via `(xy - 0.5) * scale + 0.5`

#### `test_random_resize_padding_crop_preserves_landmark_other_dimensions`
- **Setup**: Landmarks [N, T, V, C] where C > 2 (e.g., z, visibility)
- **Action**: Apply transform
- **Assert**: Dimensions beyond [:2] remain unchanged

### 2. Parameter Validation Tests

#### `test_random_resize_padding_crop_raises_value_error_for_negative_scale`
- **Setup**: Try to create with `min_scale=-0.1`
- **Assert**: `ValueError` raised

#### `test_random_resize_padding_crop_raises_value_error_when_min_greater_than_max`
- **Setup**: `min_scale=1.5, max_scale=0.5`
- **Assert**: `ValueError` raised

### 3. Reproducibility Tests

#### `test_random_resize_padding_crop_is_reproducible_with_same_generator`
- **Setup**: Two transforms with same seed
- **Assert**: Identical outputs

### 4. Integration with torchvision Tests

#### `test_random_resize_padding_crop_affine_handles_5d_tensors`
- **Setup**: Video tensor [N, T, C, H, W]
- **Action**: Apply transform
- **Verify**: No errors from `transforms_v2.functional.affine`

#### `test_random_resize_padding_crop_center_invariant_transformation`
- **Setup**: Place distinctive pattern at image center
- **Action**: Apply transform
- **Assert**: Center pixel remains relatively stable

---

## Test Cases for Error Handling

### 1. Missing Key Tests

#### `test_transform_raises_key_error_for_missing_video_key`
- **Setup**: Transform specifies `video_keys=["nonexistent"]`
- **Action**: Apply to item without that key
- **Assert**: `KeyError` raised

#### `test_transform_raises_key_error_for_missing_landmark_key`
- **Setup**: Transform specifies `landmark_keys=["nonexistent"]`
- **Action**: Apply to item without that key
- **Assert**: `KeyError` raised

### 2. Shape Assumption Tests

#### `test_uniform_speed_change_works_with_various_spatial_sizes`
- **Setup**: Videos with H, W = (32, 32), (64, 128), (224, 224)
- **Action**: Apply transform
- **Assert**: All work correctly

#### `test_random_resize_padding_crop_works_with_various_coordinate_dimensions`
- **Setup**: Landmarks with C=2, C=3, C=4
- **Action**: Apply transform
- **Assert**: Only first 2 dimensions transformed

---

## Test Fixtures

### Required Fixtures

```python
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
    sample_video_tensor,
    sample_landmark_tensor,
    sample_target_tensor
) -> TensorSLDatasetItem:
    """Create complete dataset item."""
    return SLDatasetItem(
        videos={"rgb": sample_video_tensor},
        landmarks={"pose": sample_landmark_tensor},
        targets={"gloss": sample_target_tensor}
    )

@pytest.fixture
def deterministic_generator() -> torch.Generator:
    """Create deterministic random generator."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen
```

---

## Testing Guidelines

### 1. Follow Project Standards
- Reference: `guides/CODING_STYLE_GUIDE.md`
- Use type hints for all test functions
- Follow naming convention: `test_<component>_<behavior>_<condition>`

### 2. Assertion Patterns
```python
# Shape assertions
assert output.videos["rgb"].shape == expected_shape

# Approximate equality for floating point
assert torch.allclose(output_tensor, expected_tensor, atol=1e-5)

# Error assertions
with pytest.raises(ValueError, match="positive"):
    transform = UniformSpeedChange(min_scale=-0.1, ...)
```

### 3. Test Data Strategy
- Use small tensor sizes for speed (e.g., 64x64 images, 10 frames)
- Use deterministic generators for reproducibility
- Create fixtures for reusable test data

### 4. Coverage Goals
- Aim for 100% line coverage for new transforms
- Test all documented error conditions
- Include edge cases (T=1, scale=1.0, etc.)

---

## Success Criteria

✅ **All test cases implemented and passing**  
✅ **Coverage ≥ 95% for `transform/dynamic.py`**  
✅ **No Pyright errors in test code**  
✅ **Tests run in < 30 seconds**  
✅ **All error conditions verified**

---

## Reference Implementation Example

```python
def test_uniform_speed_change_reduces_frame_count_when_scale_greater_than_one(
    sample_dataset_item, deterministic_generator
):
    """Test that scale > 1.0 reduces frame count (speed up)."""
    # Arrange
    transform = UniformSpeedChange(
        video_keys=["rgb"],
        landmark_keys=["pose"],
        min_scale=2.0,
        max_scale=2.0,  # Fixed scale for deterministic test
        gen=deterministic_generator
    )
    original_frames = sample_dataset_item.videos["rgb"].shape[1]
    
    # Act
    result = transform(sample_dataset_item)
    
    # Assert
    result_frames = result.videos["rgb"].shape[1]
    expected_frames = int(original_frames * 2.0)
    assert abs(result_frames - expected_frames) <= 1  # Allow rounding error
    assert result.landmarks["pose"].shape[1] == result_frames  # Sync
```

---

## Next Steps After Testing

1. ✅ Verify all tests pass
2. ✅ Check coverage report
3. ✅ Review with `guides/INTEGRATION_TEST_STRATEGY.md`
4. 🔄 Merge to `integration-ai` branch
5. 🔄 Create PR to `main-ai` for human review

---

## Notes

- Tests should NOT modify the construction guard (`return_true()` check)
- Focus on integration behavior, not unit implementation details
- Use realistic tensor shapes matching actual CSLR datasets
- Document any discovered issues as GitHub issues

---

**Document Version**: 1.0  
**Created**: 2025-11-20  
**Target Completion**: TBD by tests-enhancement branch owner
