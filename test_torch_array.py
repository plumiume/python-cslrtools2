import numpy as np
import torch

class CustomArray:
    def __array__(self, dtype=None, copy=None):
        return np.array([[1, 2], [3, 4]])

c = CustomArray()
print('Has __array__:', hasattr(c, '__array__'))
print('__array__() result:', c.__array__())

try:
    t = torch.tensor(c)
    print('torch.tensor(c):', t)
except Exception as e:
    print('torch.tensor(c) failed:', type(e).__name__, '-', e)

try:
    t = torch.as_tensor(c)
    print('torch.as_tensor(c):', t)
except Exception as e:
    print('torch.as_tensor(c) failed:', type(e).__name__, '-', e)

# NumPy経由での変換
arr = np.asarray(c)
print('np.asarray(c):', arr)
t = torch.tensor(arr)
print('torch.tensor(np.asarray(c)):', t)
