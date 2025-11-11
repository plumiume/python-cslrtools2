import numpy as np
import torch
import zarr

from ..typings import ArrayLike

def get_array(group: zarr.Group, path: str) -> zarr.Array:
    array = group.get(path)
    if not isinstance(array, zarr.Array):
        raise KeyError(f"Array not found at path: {path}")
    return array

def get_group(group: zarr.Group, path: str) -> zarr.Group:
    subgroup = group.get(path)
    if not isinstance(subgroup, zarr.Group):
        raise KeyError(f"Group not found at path: {path}")
    return subgroup

def as_tensor(data: ArrayLike) -> torch.Tensor:
    if torch.is_tensor(data):
        return data
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data)
    return torch.as_tensor(np.asarray(data))

