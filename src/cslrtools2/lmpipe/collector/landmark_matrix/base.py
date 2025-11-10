from abc import abstractmethod
from typing import Any, Iterable, Mapping, Protocol, final
from pathlib import Path

from ...typings import NDArrayFloat, ExistRule
from ...options import LMPipeOptions
from ...estimator import ProcessResult
from ...runspec import RunSpec

from ..base import Collector


class LandmarkMatrixSaveCollector[K: str](Collector[K]):
    """Base class for collectors that save landmark matrices to files."""

    LANDMARK_DIR_NAME = "landmarks"
    USE_LANDMARK_DIR = True  # True for per-key files, False for container files
    
    exist_rule: ExistRule
    """Existence rule for handling existing output files."""

    @property
    @abstractmethod
    def is_perkey(self) -> bool:
        """Whether this collector saves per-key files (True) or a container file (False)."""
        ...

    @property
    @abstractmethod
    def is_container(self) -> bool:
        """Whether this collector saves a container file (True) or per-key files (False)."""
        ...

    @property
    @abstractmethod
    def file_ext(self) -> str:
        """File extension for this collector's output format.
        
        For per-key collectors, this is the extension of individual files.
        For container collectors, this is the extension of the container file.
        """
        ...

    def configure_from_options(self, options: "LMPipeOptions") -> None:
        """Configure collector from LMPipe options.
        
        Args:
            options (`LMPipeOptions`): The pipeline options containing configuration.
        """
        self.exist_rule = options["landmark_matrix_save_exist_rule"]

    @final
    def _get_output_path(self, dst: Path) -> Path:
        """Get the output path for the landmark data.
        
        This method is final and uses the abstract properties (is_perkey, is_container, file_ext)
        to determine the correct output path.
        
        Args:
            dst: The destination directory path.
        
        Returns:
            :class:`Path`: The full output path.
                - For per-key collectors: dst/landmarks/ (directory)
                - For container collectors: dst/landmarks{file_ext}
        """
        if self.is_container:
            # Container mode: dst/landmarks{file_ext}
            return dst / f"{self.LANDMARK_DIR_NAME}{self.file_ext}"
        else:
            # Per-key mode: dst/landmarks/ directory
            return self._prepare_landmark_dir(dst)

    def apply_exist_rule(self, runspec: "RunSpec[Any]") -> bool:
        """Apply the existence rule to determine if processing should continue.
        
        This method can be overridden by subclasses for custom existence checking logic.
        
        Args:
            runspec: The RunSpec instance containing input/output path information.
        
        Returns:
            :class:`bool`: True if processing should continue, False if it should be skipped.
        
        Raises:
            FileExistsError: If exist_rule is "error" and the output already exists.
        """
        output_path = self._get_output_path(runspec.dst)
        
        if self.is_container:
            # Container mode: check if the container file exists
            exists = output_path.exists()
        else:
            # Per-key mode: check if the landmarks directory exists
            exists = output_path.exists() and output_path.is_dir()
        
        if not exists:
            return True  # No conflict, proceed
        
        if self.exist_rule == "skip":
            return False  # Skip processing
        elif self.exist_rule == "overwrite":
            return True  # Overwrite existing file/directory
        elif self.exist_rule == "suffix":
            # TODO: Implement suffix logic (e.g., landmarks_1.npz, landmarks_2.npz)
            return True  # For now, just overwrite
        elif self.exist_rule == "error":
            raise FileExistsError(f"Output already exists: {output_path}")
        
        return True  # Default: proceed

    # overridable method
    def _open_file(self, path: Path):
        """Prepare for writing landmark data.
        
        Args:
            path (`Path`): The destination directory path.
        """
        pass

    # overridable method
    @abstractmethod
    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        """Process and save landmark data from a single frame.
        
        Args:
            result (`Mapping[K, NDArrayFloat]`): A mapping of landmark keys to arrays.
        """
        ...

    # overridable method
    def _close_file(self):
        """Finalize and clean up after writing landmark data."""
        pass

    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect landmark results and save them.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of :class:`ProcessResult` objects.
        """
        self._open_file(runspec.dst)
        try:
            for result in results:
                self._append_result(result.landmarks)
        finally:
            self._close_file()

    def _prepare_landmark_dir(self, dst: Path) -> Path:
        """Ensure the landmark output directory exists for the collector.
        
        Args:
            dst (`Path`): The base destination directory.
            
        Returns:
            :class:`Path`: The landmark directory path (or dst itself for container types).
        """
        if self.USE_LANDMARK_DIR:
            landmark_dir = dst / self.LANDMARK_DIR_NAME
            landmark_dir.mkdir(parents=True, exist_ok=True)
            return landmark_dir
        else:
            dst.mkdir(parents=True, exist_ok=True)
            return dst

    def _get_landmark_file_path(self, dst: Path, filename: str) -> Path:
        """Get the path for a landmark file.
        
        Args:
            dst (`Path`): The base destination directory.
            filename (`str`): The filename (e.g., "landmarks.npz").
            
        Returns:
            :class:`Path`: The full file path.
        """
        if self.USE_LANDMARK_DIR:
            return dst / self.LANDMARK_DIR_NAME / filename
        else:
            return dst / filename



class _LMSCCreator(Protocol):
    def __call__[K: str](self, key_type: type[K]) -> LandmarkMatrixSaveCollector[K]: ...

lmsc_aliases: dict[str, _LMSCCreator] = {}
