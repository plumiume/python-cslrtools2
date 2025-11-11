import numpy as np
import torch
import zarr
import time

# Custom SupportsArray implementation
class CustomArray:
    def __init__(self, data):
        self._data = np.asarray(data)
    
    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            return self._data.astype(dtype, copy=copy if copy is not None else True)
        if copy:
            return self._data.copy()
        return self._data

# Conversion functions
def convert_with_check(obj):
    """torch.is_tensorとisinstanceでチェックしてから変換"""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj)
    return torch.as_tensor(np.asarray(obj))

def convert_always_np_asarray(obj):
    """常にnp.asarrayを経由"""
    return torch.as_tensor(np.asarray(obj))

def convert_direct_torch_asarray(obj):
    """torch.asarrayを直接使用（失敗する可能性あり）"""
    try:
        return torch.asarray(obj)
    except:
        return torch.as_tensor(np.asarray(obj))

# Benchmark function
def benchmark(name, obj, converter, iterations=1000):
    # Warm up
    for _ in range(10):
        try:
            result = converter(obj)
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            return None
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = converter(obj)
    elapsed = time.perf_counter() - start
    
    print(f"  {name:40s}: {elapsed*1000/iterations:.4f} ms/iter ({elapsed:.4f}s total)")
    return result

# Test objects
print("=== Creating test objects ===")
test_data = np.random.randn(100, 100)

# 1. Tensor
tensor_obj = torch.from_numpy(test_data)
print(f"1. Tensor: {type(tensor_obj)}, shape={tensor_obj.shape}")

# 2. NumPy array
numpy_obj = test_data.copy()
print(f"2. NumPy: {type(numpy_obj)}, shape={numpy_obj.shape}")

# 3. zarr.Array
import tempfile
import shutil
tmpdir = tempfile.mkdtemp()
zarr_obj = zarr.open_array(f"{tmpdir}/test.zarr", mode='w', shape=(100, 100), dtype='f8')
zarr_obj[:] = test_data
print(f"3. zarr.Array: {type(zarr_obj)}, shape={zarr_obj.shape}")

# 4. CustomArray
custom_obj = CustomArray(test_data)
print(f"4. CustomArray: {type(custom_obj)}, has __array__={hasattr(custom_obj, '__array__')}")

print("\n" + "="*80)
print("BENCHMARK: Tensor input (already converted)")
print("="*80)
benchmark("convert_with_check", tensor_obj, convert_with_check)
benchmark("convert_always_np_asarray", tensor_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_asarray", tensor_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: NumPy array input")
print("="*80)
benchmark("convert_with_check", numpy_obj, convert_with_check)
benchmark("convert_always_np_asarray", numpy_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_asarray", numpy_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: zarr.Array input (SupportsArray)")
print("="*80)
benchmark("convert_with_check", zarr_obj, convert_with_check)
benchmark("convert_always_np_asarray", zarr_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_asarray", zarr_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: CustomArray input (SupportsArray)")
print("="*80)
benchmark("convert_with_check", custom_obj, convert_with_check)
benchmark("convert_always_np_asarray", custom_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_asarray", custom_obj, convert_direct_torch_asarray)

# Verify correctness
print("\n" + "="*80)
print("VERIFICATION: Check all conversions produce same result")
print("="*80)

def verify_conversion(name, obj, converter):
    try:
        result = converter(obj)
        print(f"{name:25s}: shape={result.shape}, dtype={result.dtype}, is_tensor={torch.is_tensor(result)}")
        return result
    except Exception as e:
        print(f"{name:25s}: ❌ {type(e).__name__}: {e}")
        return None

for obj_name, obj in [("Tensor", tensor_obj), ("NumPy", numpy_obj), 
                       ("zarr.Array", zarr_obj), ("CustomArray", custom_obj)]:
    print(f"\n{obj_name}:")
    r1 = verify_conversion("  convert_with_check", obj, convert_with_check)
    r2 = verify_conversion("  convert_always_np_asarray", obj, convert_always_np_asarray)
    r3 = verify_conversion("  convert_direct_torch", obj, convert_direct_torch_asarray)
    
    if r1 is not None and r2 is not None:
        print(f"  Results match: {torch.allclose(r1.float(), r2.float())}")

# Cleanup
shutil.rmtree(tmpdir)
print("\n✅ Benchmark complete")
