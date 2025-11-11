import numpy as np
import torch

def safe_to_tensor(obj):
    """
    オブジェクトをTensorに変換する際、np.asarrayが必要かどうかを判断
    
    Returns:
        torch.Tensor
    """
    # 既にTensorなら何もしない
    if torch.is_tensor(obj):
        return obj
    
    # NumPy配列なら直接変換可能
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj)
    
    # それ以外は__array__を使う可能性があるので、np.asarrayを経由
    # (torch.asarrayは__array__プロトコルを完全にサポートしていない)
    return torch.as_tensor(np.asarray(obj))

# Test
print("=== isinstance checks for optimization ===")

# Tensor
t = torch.tensor([1, 2, 3])
print(f"torch.is_tensor(Tensor): {torch.is_tensor(t)}")
print(f"isinstance(Tensor, torch.Tensor): {isinstance(t, torch.Tensor)}")

# NumPy
arr = np.array([4, 5, 6])
print(f"isinstance(np.ndarray, np.ndarray): {isinstance(arr, np.ndarray)}")

# List (needs conversion)
lst = [7, 8, 9]
print(f"isinstance(list, (torch.Tensor, np.ndarray)): {isinstance(lst, (torch.Tensor, np.ndarray))}")

print("\n=== Conversion efficiency ===")
import time

def benchmark(name, obj, converter):
    start = time.perf_counter()
    for _ in range(10000):
        result = converter(obj)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")

tensor_obj = torch.randn(100)
numpy_obj = np.random.randn(100)
list_obj = list(range(100))

print("\nTensor input:")
benchmark("  torch.is_tensor check + skip", tensor_obj, 
          lambda x: x if torch.is_tensor(x) else torch.as_tensor(x))
benchmark("  Always torch.as_tensor", tensor_obj,
          lambda x: torch.as_tensor(x))

print("\nNumPy input:")
benchmark("  isinstance check + torch.as_tensor", numpy_obj,
          lambda x: torch.as_tensor(x) if isinstance(x, np.ndarray) else torch.as_tensor(np.asarray(x)))
benchmark("  Always np.asarray + torch.as_tensor", numpy_obj,
          lambda x: torch.as_tensor(np.asarray(x)))

print("\nList input:")
benchmark("  Smart conversion", list_obj,
          lambda x: torch.as_tensor(x) if isinstance(x, (torch.Tensor, np.ndarray)) else torch.as_tensor(np.asarray(x)))
