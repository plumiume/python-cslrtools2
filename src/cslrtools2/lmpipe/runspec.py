from pathlib import Path

from .logger import lmpipe_logger
from .typings import PathLike

__all__ = [
    "RunSpec",
]

# cp314 ready
type RunSpec[S: Path | int] = "RunSpec[S]" # pyright: ignore[reportRedeclaration]

class RunSpec[S: Path | int]:
    """Specification of a processing run, including source and destination."""
    
    def __init__(self, src: S, dst: Path):
        self.src: S = src
        "Source path or index."
        self.dst: Path = dst
        "Destination path."
    @classmethod
    def from_pathlikes(cls: type[RunSpec[Path]], src: PathLike, dst: PathLike) -> RunSpec[Path]:
        """Create a RunSpec from path-like source and destination.

        Args:
            src (`PathLike`): Source path.
            dst (`PathLike`): Destination path.

        Returns:
            :class:`RunSpec[Path]`: The created RunSpec instance.
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            lmpipe_logger.error(f"Video file does not exist: {src_path}")
            raise FileNotFoundError(f"Video file does not exist: {src_path}")

        return cls(src_path, dst_path)
    @classmethod
    def from_index(cls: type[RunSpec[int]], src: int, dst: PathLike) -> RunSpec[int]:
        """Create a RunSpec from an integer source index and a path-like destination.

        Args:
            src (`int`): Source index.
            dst (`PathLike`): Destination path.

        Returns:
            :class:`RunSpec[int]`: The created RunSpec instance.
        """
        return cls(src, Path(dst))
