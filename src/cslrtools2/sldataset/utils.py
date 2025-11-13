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

import numpy as np
import torch
import zarr

from ..typings import ArrayLike
from ..exceptions import DataLoadError

def get_array(group: zarr.Group, path: str) -> zarr.Array:
    """Retrieve a Zarr array from a group by path.
    
    Args:
        group (:class:`zarr.Group`): The Zarr group to search in.
        path (:obj:`str`): The path to the array within the group.
        
    Returns:
        :class:`zarr.Array`: The Zarr array at the specified path.
        
    Raises:
        :exc:`DataLoadError`: If no array is found at the specified path.
    """
    array = group.get(path)
    if not isinstance(array, zarr.Array):
        available_keys = list(group.keys()) if hasattr(group, 'keys') else []
        raise DataLoadError(
            f"Array not found at path: {path}. "
            f"Available keys: {available_keys}"
        )
    return array

def get_group(group: zarr.Group, path: str) -> zarr.Group:
    """Retrieve a Zarr subgroup from a group by path.
    
    Args:
        group (:class:`zarr.Group`): The parent Zarr group to search in.
        path (:obj:`str`): The path to the subgroup within the parent group.
        
    Returns:
        :class:`zarr.Group`: The Zarr subgroup at the specified path.
        
    Raises:
        :exc:`DataLoadError`: If no group is found at the specified path.
    """
    subgroup = group.get(path)
    if not isinstance(subgroup, zarr.Group):
        available_keys = list(group.keys()) if hasattr(group, 'keys') else []
        raise DataLoadError(
            f"Group not found at path: {path}. "
            f"Available keys: {available_keys}"
        )
    return subgroup

def as_tensor(data: ArrayLike) -> torch.Tensor:
    """Convert array-like data to a PyTorch tensor.
    
    Handles conversion from various array types including :class:`torch.Tensor`,
    :class:`numpy.ndarray`, and other array-like objects.
    
    Args:
        data (:class:`ArrayLike`): Array-like data to convert.
        
    Returns:
        :class:`torch.Tensor`: The converted tensor. If input is already a
            :class:`torch.Tensor`, returns it directly without copying.
    """
    if torch.is_tensor(data):
        return data
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data)
    return torch.as_tensor(np.asarray(data))


