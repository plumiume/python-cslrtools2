from pathlib import Path
from .typings import PathLike

# cp314 ready
type RunSpec[S: Path | int] = "RunSpec[S]" # pyright: ignore[reportRedeclaration]

class RunSpec[S: Path | int]:
    def __init__(self, src: S, dst: Path):
        self.src: S = src
        self.dst: Path = dst
    @classmethod
    def from_pathlikes(cls: type[RunSpec[Path]], src: PathLike, dst: PathLike) -> RunSpec[Path]:
        return cls(Path(src), Path(dst))
    @classmethod
    def from_index(cls: type[RunSpec[int]], src: int, dst: PathLike) -> RunSpec[int]:
        return cls(src, Path(dst))
