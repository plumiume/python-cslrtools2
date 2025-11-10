from pathlib import Path

from fsspec.mapping import FSMap as _FSMap
from zarr.abc.store import Store
from zarr.storage import StorePath
from zarr.core.buffer import Buffer

type FSMap = _FSMap | None
type StoreLike = Store | StorePath | FSMap | Path | str | dict[str, Buffer]
type PathLike = Path | str
