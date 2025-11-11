import numpy as np
import torch
import time

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

def convert_direct_torch_tensor(obj):
    """torch.tensorを直接使用"""
    if torch.is_tensor(obj):
        return obj
    return torch.tensor(obj)

def convert_direct_torch_asarray(obj):
    """torch.asarrayを直接使用"""
    if torch.is_tensor(obj):
        return obj
    return torch.asarray(obj)

# Benchmark function
def benchmark(name, obj, converter, iterations=1000):
    # Warm up
    for _ in range(10):
        result = converter(obj)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = converter(obj)
    elapsed = time.perf_counter() - start
    
    avg_ms = elapsed * 1000 / iterations
    print(f"  {name:45s}: {avg_ms:7.4f} ms/iter")
    return result

# Test objects
print("=== Creating test objects ===")
test_data = [[float(i * 100 + j) for j in range(100)] for i in range(100)]

# 1. Tensor
tensor_obj = torch.tensor(test_data)
print(f"1. Tensor: {type(tensor_obj)}, shape={tensor_obj.shape}")

# 2. NumPy array
numpy_obj = np.array(test_data)
print(f"2. NumPy: {type(numpy_obj)}, shape={numpy_obj.shape}")

# 3. List (nested)
list_obj = test_data
print(f"3. List: {type(list_obj)}, len={len(list_obj)}, inner_len={len(list_obj[0])}")

# 4. Tuple (nested)
tuple_obj = tuple(tuple(row) for row in test_data)
print(f"4. Tuple: {type(tuple_obj)}, len={len(tuple_obj)}, inner_len={len(tuple_obj[0])}")

# 5. List (flat)
flat_list_obj = [float(i) for i in range(10000)]
print(f"5. Flat List: {type(flat_list_obj)}, len={len(flat_list_obj)}")

# 6. Tuple (flat)
flat_tuple_obj = tuple(float(i) for i in range(10000))
print(f"6. Flat Tuple: {type(flat_tuple_obj)}, len={len(flat_tuple_obj)}")

# 7. Range
range_obj = range(10000)
print(f"7. Range: {type(range_obj)}, len={len(range_obj)}")

# 8. List comprehension with int
int_list_obj = [i for i in range(10000)]
print(f"8. Int List: {type(int_list_obj)}, len={len(int_list_obj)}")

print("\n" + "="*80)
print("BENCHMARK: Tensor input (already converted)")
print("="*80)
benchmark("convert_with_check", tensor_obj, convert_with_check)
benchmark("convert_always_np_asarray", tensor_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_tensor", tensor_obj, convert_direct_torch_tensor)
benchmark("convert_direct_torch_asarray", tensor_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: NumPy array input")
print("="*80)
benchmark("convert_with_check", numpy_obj, convert_with_check)
benchmark("convert_always_np_asarray", numpy_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_tensor", numpy_obj, convert_direct_torch_tensor)
benchmark("convert_direct_torch_asarray", numpy_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: Nested List (100x100) input")
print("="*80)
benchmark("convert_with_check", list_obj, convert_with_check)
benchmark("convert_always_np_asarray", list_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_tensor", list_obj, convert_direct_torch_tensor)
benchmark("convert_direct_torch_asarray", list_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: Nested Tuple (100x100) input")
print("="*80)
benchmark("convert_with_check", tuple_obj, convert_with_check)
benchmark("convert_always_np_asarray", tuple_obj, convert_always_np_asarray)
benchmark("convert_direct_torch_tensor", tuple_obj, convert_direct_torch_tensor)
benchmark("convert_direct_torch_asarray", tuple_obj, convert_direct_torch_asarray)

print("\n" + "="*80)
print("BENCHMARK: Flat List (10000 floats) input")
print("="*80)
benchmark("convert_with_check", flat_list_obj, convert_with_check, iterations=1000)
benchmark("convert_always_np_asarray", flat_list_obj, convert_always_np_asarray, iterations=1000)
benchmark("convert_direct_torch_tensor", flat_list_obj, convert_direct_torch_tensor, iterations=1000)
benchmark("convert_direct_torch_asarray", flat_list_obj, convert_direct_torch_asarray, iterations=1000)

print("\n" + "="*80)
print("BENCHMARK: Flat Tuple (10000 floats) input")
print("="*80)
benchmark("convert_with_check", flat_tuple_obj, convert_with_check, iterations=1000)
benchmark("convert_always_np_asarray", flat_tuple_obj, convert_always_np_asarray, iterations=1000)
benchmark("convert_direct_torch_tensor", flat_tuple_obj, convert_direct_torch_tensor, iterations=1000)
benchmark("convert_direct_torch_asarray", flat_tuple_obj, convert_direct_torch_asarray, iterations=1000)

print("\n" + "="*80)
print("BENCHMARK: Range (10000 ints) input")
print("="*80)
benchmark("convert_with_check", range_obj, convert_with_check, iterations=1000)
benchmark("convert_always_np_asarray", range_obj, convert_always_np_asarray, iterations=1000)
benchmark("convert_direct_torch_tensor", range_obj, convert_direct_torch_tensor, iterations=1000)
benchmark("convert_direct_torch_asarray", range_obj, convert_direct_torch_asarray, iterations=1000)

print("\n" + "="*80)
print("BENCHMARK: Int List (10000 ints) input")
print("="*80)
benchmark("convert_with_check", int_list_obj, convert_with_check, iterations=1000)
benchmark("convert_always_np_asarray", int_list_obj, convert_always_np_asarray, iterations=1000)
benchmark("convert_direct_torch_tensor", int_list_obj, convert_direct_torch_tensor, iterations=1000)
benchmark("convert_direct_torch_asarray", int_list_obj, convert_direct_torch_asarray, iterations=1000)

# Verify correctness
print("\n" + "="*80)
print("VERIFICATION: Check all conversions produce correct shape")
print("="*80)

def verify_conversion(name, obj, converter):
    result = converter(obj)
    print(f"{name:30s}: shape={result.shape}, dtype={result.dtype}")
    return result

for obj_name, obj, expected_shape in [
    ("Tensor", tensor_obj, (100, 100)),
    ("NumPy", numpy_obj, (100, 100)),
    ("Nested List", list_obj, (100, 100)),
    ("Nested Tuple", tuple_obj, (100, 100)),
    ("Flat List", flat_list_obj, (10000,)),
    ("Flat Tuple", flat_tuple_obj, (10000,)),
    ("Range", range_obj, (10000,)),
    ("Int List", int_list_obj, (10000,))
]:
    print(f"\n{obj_name} (expected shape: {expected_shape}):")
    r1 = verify_conversion("  convert_with_check", obj, convert_with_check)
    r2 = verify_conversion("  convert_always_np_asarray", obj, convert_always_np_asarray)
    r3 = verify_conversion("  convert_direct_torch_tensor", obj, convert_direct_torch_tensor)
    r4 = verify_conversion("  convert_direct_torch_asarray", obj, convert_direct_torch_asarray)
    
    assert r1.shape == expected_shape, f"Shape mismatch: {r1.shape} != {expected_shape}"
    print(f"  ✅ All conversions match expected shape")

print("\n✅ Benchmark complete")
