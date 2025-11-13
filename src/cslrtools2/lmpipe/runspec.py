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

from __future__ import annotations

from pathlib import Path

from .logger import lmpipe_logger
from ..typings import PathLike
from ..exceptions import VideoProcessingError

__all__ = [
    "RunSpec",
]

class RunSpec[S: Path | int]:
    """Specification of a processing run, including source and destination.
    
    Encapsulates the source input (file path or camera index) and the
    destination output directory for a landmark processing pipeline run.
    
    Type Parameters:
        S: Source type, either :class:`pathlib.Path` for file inputs or
            :obj:`int` for camera device indices.
    
    Attributes:
        src (S): Source path or camera index.
        dst (:class:`pathlib.Path`): Destination output directory path.
    """
    
    def __init__(self, src: S, dst: Path):
        self.src: S = src
        "Source path or index."
        self.dst: Path = dst
        "Destination path."
        
    @classmethod
    def from_pathlikes(cls: type[RunSpec[Path]], src: PathLike, dst: PathLike) -> RunSpec[Path]:
        """Create a RunSpec from path-like source and destination.

        Args:
            src (:class:`PathLike`): Source file or directory path.
            dst (:class:`PathLike`): Destination output directory path.

        Returns:
            :class:`RunSpec`\\[:class:`pathlib.Path`\\]: The created :class:`RunSpec` instance.
            
        Raises:
            :exc:`VideoProcessingError`: If the source path does not exist.
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            lmpipe_logger.error(f"Source path does not exist: {src_path}")
            raise VideoProcessingError(
                f"Source path does not exist: {src_path}. "
                f"Ensure the path is correct and the file is accessible."
            )

        return cls(src_path, dst_path)
        
    @classmethod
    def from_index(cls: type[RunSpec[int]], src: int, dst: PathLike) -> RunSpec[int]:
        """Create a RunSpec from a camera index and a destination path.

        Args:
            src (:obj:`int`): Camera device index (e.g., ``0`` for default camera).
            dst (:class:`PathLike`): Destination output directory path.

        Returns:
            :class:`RunSpec`\\[:obj:`int`\\]: The created :class:`RunSpec` instance.
        """
        return cls(src, Path(dst))
