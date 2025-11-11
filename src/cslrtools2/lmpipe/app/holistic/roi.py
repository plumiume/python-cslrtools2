from abc import ABC, abstractmethod
from typing import Mapping
from ....typings import MatLike, NDArrayFloat

class BaseROI(ABC):

    @abstractmethod
    def apply_roi(self, frame_src: MatLike) -> MatLike | None: ...

    @abstractmethod
    def apply_world_coords[K: str](
        self, local_coords: Mapping[K, NDArrayFloat]
        ) -> Mapping[K, NDArrayFloat | None]: ...

