import zarr

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
