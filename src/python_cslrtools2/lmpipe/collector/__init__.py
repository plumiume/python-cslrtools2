from abc import ABC, abstractmethod
from csv import DictWriter
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, TextIO

import numpy as np

from ..typings import NDArrayFloat
from ..estimator import ProcessResult
from ..runspec import RunSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    import zarr

class Collector[K: str](ABC):

    @abstractmethod
    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect and process the results from the estimator.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of :class:`ProcessResult` objects to be collected.
        """
        ...

    # overridable method
    def apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        """Determine whether to skip processing based on existing results.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.

        Raises:
            ValueError: If you want to stop processing due to existing results and the rule is set to 'error'.
        
        Returns:
            :class:`bool`: :code:`True` if processing should be skipped, :code:`False` otherwise.

        
        """
        return True

class LandmarkMatrixSaveCollector[K: str](Collector[K]):

    LANDMARK_DIR_NAME = "landmarks"

    # overridable method
    def _open_file(self, path: Path):
        pass

    # overridable method
    def _append_result(self, result: Mapping[K, NDArrayFloat]): ...

    # overridable method
    def _close_file(self): ...

    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        self._open_file(runspec.dst)
        try:
            for result in results:
                self._append_result(result.landmarks)
        finally:
            self._close_file()

    def _prepare_landmark_dir(self, dst: Path) -> Path:
        """Ensure the landmark output directory exists for the collector."""
        landmark_dir = dst / self.LANDMARK_DIR_NAME
        landmark_dir.mkdir(parents=True, exist_ok=True)
        return landmark_dir

class CsvLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist streamed landmarks into per-key delimited files under ``landmarks``."""

    def __init__(
        self,
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        extension: str | None = None,
    ) -> None:
        self.delimiter = delimiter
        self.encoding = encoding
        self.extension = extension or self._guess_extension(delimiter)
        self._base_dir: Path | None = None
        self._writers: dict[str, DictWriter[str]] = {}
        self._file_handles: dict[str, TextIO] = {}
        self._sample_width: dict[str, int] = {}
        self._row_index: dict[str, int] = {}

    @staticmethod
    def _guess_extension(delimiter: str) -> str:
        if delimiter == "\t":
            return ".tsv"
        if delimiter == ";":
            return ".ssv"
        return ".csv"

    def _open_file(self, path: Path):
        self._base_dir = self._prepare_landmark_dir(path)
        self._writers.clear()
        self._file_handles.clear()
        self._sample_width.clear()
        self._row_index.clear()

    def _ensure_writer(self, key: str, sample_width: int) -> DictWriter[str]:
        if self._base_dir is None:
            raise RuntimeError("CSV landmark directory not prepared.")
        writer = self._writers.get(key)
        if writer is None:
            file_path = self._base_dir / f"{key}{self.extension}"
            handle = file_path.open("w", newline="", encoding=self.encoding)
            fieldnames = [
                "key",
                "sample_index",
                *[f"value_{idx}" for idx in range(sample_width)],
            ]
            writer = DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                delimiter=self.delimiter,
            )
            writer.writeheader()
            self._writers[key] = writer
            self._file_handles[key] = handle
            self._sample_width[key] = sample_width
            self._row_index[key] = 0
            return writer
        if self._sample_width[key] != sample_width:
            raise ValueError(
                "Inconsistent landmark sample width encountered while writing CSV output."
            )
        return writer

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, array in result.items():
            key = str(raw_key)
            sample_matrix = np.atleast_2d(np.asarray(array, dtype=float))
            flattened = sample_matrix.reshape(sample_matrix.shape[0], -1)
            writer = self._ensure_writer(key, flattened.shape[1])
            for sample in flattened:
                row: dict[str, str | int | float] = {
                    "key": key,
                    "sample_index": self._row_index[key],
                }
                row.update(
                    {
                        f"value_{idx}": float(value)
                        for idx, value in enumerate(sample.tolist())
                    }
                )
                writer.writerow(row)
                self._row_index[key] += 1

    def _close_file(self):
        for handle in self._file_handles.values():
            handle.close()
        self._base_dir = None
        self._writers.clear()
        self._file_handles.clear()
        self._sample_width.clear()
        self._row_index.clear()

class JsonLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into per-key ``*.json`` files inside ``landmarks``."""

    def __init__(self, *, indent: int | None = 2, encoding: str = "utf-8") -> None:
        self.indent = indent
        self.encoding = encoding
        self._base_dir: Path | None = None
        self._buffers: dict[str, list[list[float]]] = {}

    def _open_file(self, path: Path):
        self._base_dir = self._prepare_landmark_dir(path)
        self._buffers = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, value in result.items():
            key = str(raw_key)
            buffer = self._buffers.setdefault(key, [])
            buffer.append(np.asarray(value).tolist())

    def _close_file(self):
        if self._base_dir is None:
            return
        for key, entries in self._buffers.items():
            file_path = self._base_dir / f"{key}.json"
            with file_path.open("w", encoding=self.encoding) as fh:
                json.dump(entries, fh, ensure_ascii=False, indent=self.indent)
        self._base_dir = None
        self._buffers = {}

class NpyLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist landmarks into per-key ``*.npy`` files."""

    def __init__(self) -> None:
        self._base_dir: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

    def _open_file(self, path: Path):
        self._base_dir = self._prepare_landmark_dir(path)
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, value in result.items():
            key = str(raw_key)
            bucket = self._buffer.setdefault(key, [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._base_dir is None:
            return
        for key, arrays in self._buffer.items():
            file_path = self._base_dir / f"{key}.npy"
            if arrays:
                np.save(file_path, np.stack(arrays))
            else:
                np.save(file_path, np.empty((0,), dtype=float))
        self._base_dir = None
        self._buffer = {}

class NpzLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Write landmarks into a ``landmarks.npz`` archive."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

    def _open_file(self, path: Path):
        landmark_dir = self._prepare_landmark_dir(path)
        self._path = landmark_dir / "landmarks.npz"
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            bucket = self._buffer.setdefault(str(key), [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._path is None:
            return
        if not self._buffer:
            np.savez(self._path)
        else:
            arrays = {key: np.stack(values) for key, values in self._buffer.items()}
            np.savez(self._path, **arrays)
        self._path = None
        self._buffer = {}

class TorchLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist landmarks using :mod:`torch` in ``landmarks.pt``."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}

    def _open_file(self, path: Path):
        landmark_dir = self._prepare_landmark_dir(path)
        self._path = landmark_dir / "landmarks.pt"
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            bucket = self._buffer.setdefault(str(key), [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._path is None:
            return
        if not self._buffer:
            self._path.touch()
            self._path = None
            self._buffer = {}
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TorchLandmarkMatrixSaveCollector requires the 'torch' package."
            ) from exc
        tensors = {
            key: torch.tensor(np.stack(values))
            for key, values in self._buffer.items()
        }
        if len(tensors) == 1:
            payload = next(iter(tensors.values()))
        else:
            payload = tensors
        torch.save(payload, self._path)
        self._path = None
        self._buffer = {}

class SafetensorsLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist landmarks using :mod:`safetensors` in ``landmarks.safetensors``."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._buffer: dict[str, list[NDArrayFloat]] = {}

    def _open_file(self, path: Path):
        landmark_dir = self._prepare_landmark_dir(path)
        self._path = landmark_dir / "landmarks.safetensors"
        self._buffer = {}

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for key, value in result.items():
            self._buffer.setdefault(str(key), []).append(np.asarray(value))

    def _close_file(self):
        if self._path is None:
            return
        if not self._buffer:
            self._path.touch()
            self._path = None
            self._buffer = {}
            return
        try:
            from safetensors.numpy import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SafetensorsLandmarkMatrixSaveCollector requires the 'safetensors' package."
            ) from exc
        tensors = {key: np.stack(values) for key, values in self._buffer.items()}
        save_file(tensors, str(self._path))
        self._path = None
        self._buffer = {}

class ZarrLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist landmarks as Zarr datasets in ``landmarks.zarr``."""

    def __init__(self) -> None:
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "ZarrLandmarkMatrixSaveCollector requires the 'zarr' package."
            ) from exc
        self._zarr = zarr
        self._store_path: Path | None = None
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._group: zarr.Group | None = None

    def _open_file(self, path: Path):
        landmark_dir = self._prepare_landmark_dir(path)
        self._store_path = landmark_dir / "landmarks.zarr"
        self._buffer = {}
        self._group = self._zarr.open_group(str(self._store_path), mode="w")

    def _append_result(self, result: Mapping[K, NDArrayFloat]):
        for raw_key, value in result.items():
            key = str(raw_key)
            bucket = self._buffer.setdefault(key, [])
            bucket.append(np.asarray(value))

    def _close_file(self):
        if self._store_path is None:
            return
        group = self._group or self._zarr.open_group(str(self._store_path), mode="w")
        for key, values in self._buffer.items():
            subgroup = group.create_group(str(key), overwrite=True)
            data = np.stack(values) if values else np.empty((0,), dtype=float)
            subgroup.create_array("values", data=data)
        self._store_path = None
        self._buffer = {}
        self._group = None

# 以下のクラスはフレームワーク依存です
class AnnotatedFramesSaveCollector[K: str](Collector[K]):
    """Base collector for saving annotated frames to files."""

    # overridable hook
    def _open_file(self, path: Path):
        """Prepare for writing annotated frames.
        
        Args:
            path (`Path`): The destination directory path.
        """
        pass

    # overridable hook
    def _append_result(self, result: ProcessResult[K]):
        """Process and save a single annotated frame result.
        
        Args:
            result (`ProcessResult[K]`): The result containing the annotated frame.
        """
        pass

    # overridable hook
    def _close_file(self):
        """Finalize and clean up after writing annotated frames."""
        pass

    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect annotated frame results and save them.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of :class:`ProcessResult` objects.
        """
        self._open_file(runspec.dst)
        try:
            for result in results:
                self._append_result(result)
        finally:
            self._close_file()

class Cv2AnnotatedFramesSaveCollector[K: str](AnnotatedFramesSaveCollector[K]):
    """Save annotated frames using OpenCV (cv2) backend.
    
    Saves frames as image files (PNG, JPG, etc.) in an ``annotated_frames`` directory.
    """

    def __init__(self, *, extension: str = ".png") -> None:
        """Initialize the OpenCV annotated frames saver.
        
        Args:
            extension (:class:`str`, optional): File extension for saved images. Defaults to ".png".
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Cv2AnnotatedFramesSaveCollector requires the 'opencv-python' package."
            ) from exc
        self._cv2 = cv2
        self.extension = extension
        self._frames_dir: Path | None = None

    def _open_file(self, path: Path):
        self._frames_dir = path / "annotated_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def _append_result(self, result: ProcessResult[K]):
        if self._frames_dir is None:
            return
        frame_path = self._frames_dir / f"frame_{result.frame_id:06d}{self.extension}"
        self._cv2.imwrite(str(frame_path), result.annotated_frame)

    def _close_file(self):
        self._frames_dir = None

class MatplotlibAnnotatedFramesSaveCollector[K: str](AnnotatedFramesSaveCollector[K]):
    """Save annotated frames using Matplotlib backend.
    
    Saves frames as image files using Matplotlib's image save functionality.
    """

    def __init__(self, *, extension: str = ".png", dpi: int = 100) -> None:
        """Initialize the Matplotlib annotated frames saver.
        
        Args:
            extension (:class:`str`, optional): File extension for saved images. Defaults to ".png".
            dpi (:class:`int`, optional): Dots per inch for saved images. Defaults to 100.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "MatplotlibAnnotatedFramesSaveCollector requires the 'matplotlib' package."
            ) from exc
        self._plt = plt
        self.extension = extension
        self.dpi = dpi
        self._frames_dir: Path | None = None

    def _open_file(self, path: Path):
        self._frames_dir = path / "annotated_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def _append_result(self, result: ProcessResult[K]):
        if self._frames_dir is None:
            return
        frame_path = self._frames_dir / f"frame_{result.frame_id:06d}{self.extension}"
        self._plt.imsave(str(frame_path), result.annotated_frame, dpi=self.dpi)

    def _close_file(self):
        self._frames_dir = None

class AnnotatedFramesShowCollector[K: str](AnnotatedFramesSaveCollector[K]):
    """Base collector for displaying annotated frames interactively."""

    # overridable hook
    def _setup(self):
        """Initialize display resources before showing frames."""
        pass

    # overridable hook
    def _update(self, result: ProcessResult[K]):
        """Update the display with a new annotated frame.
        
        Args:
            result (`ProcessResult[K]`): The result containing the annotated frame to display.
        """
        pass

    # overridable hook
    def _cleanup(self):
        """Clean up display resources after showing all frames."""
        pass

    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Display annotated frame results interactively.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of :class:`ProcessResult` objects.
        """
        self._setup()
        try:
            for result in results:
                self._update(result)
        finally:
            self._cleanup()

class Cv2AnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using OpenCV (cv2) highgui.
    
    Shows frames in an interactive window using cv2.imshow().
    """

    def __init__(self, *, window_name: str = "Annotated Frame", wait_key: int = 1) -> None:
        """Initialize the OpenCV frame viewer.
        
        Args:
            window_name (:class:`str`, optional): Name of the display window. Defaults to "Annotated Frame".
            wait_key (:class:`int`, optional): Milliseconds to wait between frames. Defaults to 1.
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Cv2AnnotatedFramesShowCollector requires the 'opencv-python' package."
            ) from exc
        self._cv2 = cv2
        self.window_name = window_name
        self.wait_key = wait_key

    def _setup(self):
        self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)

    def _update(self, result: ProcessResult[K]):
        self._cv2.imshow(self.window_name, result.annotated_frame)
        self._cv2.waitKey(self.wait_key)

    def _cleanup(self):
        self._cv2.destroyWindow(self.window_name)

class MatplotlibAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using Matplotlib interactive viewer.
    
    Shows frames in a Matplotlib figure window with interactive updates.
    """

    def __init__(self, *, figsize: tuple[int, int] = (10, 8)) -> None:
        """Initialize the Matplotlib frame viewer.
        
        Args:
            figsize (:class:`tuple[int, int]`, optional): Figure size in inches. Defaults to (10, 8).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "MatplotlibAnnotatedFramesShowCollector requires the 'matplotlib' package."
            ) from exc
        self._plt = plt
        self.figsize = figsize
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._im: AxesImage | None = None

    def _setup(self):
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=self.figsize)
        self._ax.axis('off')

    def _update(self, result: ProcessResult[K]):
        if self._im is None:
            self._im = self._ax.imshow(result.annotated_frame)
        else:
            self._im.set_data(result.annotated_frame)
        
        self._ax.set_title(f"Frame {result.frame_id}")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def _cleanup(self):
        self._plt.ioff()
        self._plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._im = None

class PilAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using PIL/Pillow image viewer.
    
    Shows frames using PIL's default image viewer.
    """

    def __init__(self) -> None:
        """Initialize the PIL frame viewer."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PilAnnotatedFramesShowCollector requires the 'Pillow' package."
            ) from exc
        self._Image = Image

    def _update(self, result: ProcessResult[K]):
        # Convert numpy array to PIL Image if needed
        if isinstance(result.annotated_frame, np.ndarray):
            img = self._Image.fromarray(result.annotated_frame)
        else:
            img = result.annotated_frame
        
        img.show(title=f"Frame {result.frame_id}")

class TorchVisionAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using TorchVision utilities.
    
    Shows frames using TorchVision's visualization tools.
    """

    def __init__(self, *, figsize: tuple[int, int] = (10, 8)) -> None:
        """Initialize the TorchVision frame viewer.
        
        Args:
            figsize (:class:`tuple[int, int]`, optional): Figure size in inches. Defaults to (10, 8).
        """
        try:
            import torch
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "TorchVisionAnnotatedFramesShowCollector requires 'torch' and 'matplotlib' packages."
            ) from exc
        self._torch = torch
        self._plt = plt
        self.figsize = figsize

    def _update(self, result: ProcessResult[K]):
        # Convert to tensor if needed
        if isinstance(result.annotated_frame, np.ndarray):
            frame_tensor = self._torch.from_numpy(result.annotated_frame)
        else:
            frame_tensor = result.annotated_frame
        
        self._plt.figure(figsize=self.figsize)
        self._plt.imshow(frame_tensor.permute(1, 2, 0) if frame_tensor.ndim == 3 and frame_tensor.shape[0] in [1, 3] else frame_tensor)
        self._plt.title(f"Frame {result.frame_id}")
        self._plt.axis('off')
        self._plt.show()