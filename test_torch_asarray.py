import numpy as np
import torch
import zarr

# Test 1: zarr.Array with torch.asarray
print("=== Test 1: zarr.Array ===")
z = zarr.zeros((2, 3))
print(f"zarr.Array type: {type(z)}")
print(f"Has __array__: {hasattr(z, '__array__')}")

try:
    t = torch.asarray(z)
    print(f"✅ torch.asarray(zarr.Array): {t}")
    print(f"   dtype: {t.dtype}, shape: {t.shape}")
except Exception as e:
    print(f"❌ torch.asarray(zarr.Array) failed: {type(e).__name__} - {e}")

# Test 2: Custom SupportsArray
print("\n=== Test 2: Custom SupportsArray ===")
class CustomArray:
    def __array__(self, dtype=None, copy=None):
        return np.array([[1, 2], [3, 4]])

c = CustomArray()
print(f"CustomArray has __array__: {hasattr(c, '__array__')}")

try:
    t = torch.asarray(c)
    print(f"✅ torch.asarray(CustomArray): {t}")
except Exception as e:
    print(f"❌ torch.asarray(CustomArray) failed: {type(e).__name__} - {e}")

# Test 3: Already a Tensor
print("\n=== Test 3: Already Tensor ===")
tensor_input = torch.tensor([[5, 6], [7, 8]])
print(f"Input is_tensor: {torch.is_tensor(tensor_input)}")
t = torch.asarray(tensor_input)
print(f"torch.asarray(Tensor): {t}")
print(f"Same object? {t is tensor_input}")

# Test 4: NumPy array
print("\n=== Test 4: NumPy array ===")
numpy_array = np.array([[9, 10], [11, 12]])
t = torch.asarray(numpy_array)
print(f"torch.asarray(np.ndarray): {t}")

# Test 5: torch.is_tensor for type checking
print("\n=== Test 5: Type checking ===")
print(f"torch.is_tensor(Tensor): {torch.is_tensor(tensor_input)}")
print(f"torch.is_tensor(np.ndarray): {torch.is_tensor(numpy_array)}")
print(f"torch.is_tensor(zarr.Array): {torch.is_tensor(z)}")
print(f"torch.is_tensor(CustomArray): {torch.is_tensor(c)}")
