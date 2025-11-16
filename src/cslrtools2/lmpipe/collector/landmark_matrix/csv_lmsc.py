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

from csv import DictWriter
from pathlib import Path
from typing import Mapping, TextIO

import numpy as np

from ....exceptions import CollectorError
from ....typings import NDArrayFloat, NDArrayStr
from .base import LandmarkMatrixSaveCollector, lmsc_aliases


class CsvLandmarkMatrixSaveCollector[K: str](LandmarkMatrixSaveCollector[K]):
    """Persist streamed landmarks into per-key delimited files under ``landmarks``."""

    def __init__(
        self,
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        extension: str | None = None,
    ) -> None:
        """Initialize the CSV landmark matrix save collector.

        Args:
            delimiter (:class:`str`, optional): Field delimiter. Defaults to ",".
            encoding (:class:`str`, optional): File encoding.
                Defaults to "utf-8".
            extension (:class:`str | None`, optional): File extension.
                Auto-detected if None. Defaults to None.
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.extension = extension or self._guess_extension(delimiter)
        self._base_dir: Path | None = None
        self._writers: dict[str, DictWriter[str]] = {}
        self._file_handles: dict[str, TextIO] = {}
        self._sample_width: dict[str, int] = {}
        self._row_index: dict[str, int] = {}

    @property
    def is_perkey(self) -> bool:
        """Per-key mode: saves individual CSV/TSV/SSV files."""
        return True

    @property
    def is_container(self) -> bool:
        """Not a container collector."""
        return False

    @property
    def file_ext(self) -> str:
        """File extension for CSV/TSV/SSV files."""
        return self.extension

    @staticmethod
    def _guess_extension(delimiter: str) -> str:
        """Guess file extension based on delimiter.

        Args:
            delimiter (:class:`str`): The field delimiter.

        Returns:
            :class:`str`: The guessed file extension.
        """
        if delimiter == "\t":
            return ".tsv"
        if delimiter == ";":
            return ".ssv"
        return ".csv"

    def _open_file(self, path: Path):
        """Open the landmark collection session and prepare output directory.

        This method creates the "landmarks" subdirectory and initializes internal
        state for a new collection session. All previous writers and file handles
        are cleared.

        Args:
            path (:class:`Path`): The base output directory path.
        """
        self._base_dir = self._prepare_landmark_dir(path)
        self._writers.clear()
        self._file_handles.clear()
        self._sample_width.clear()
        self._row_index.clear()

    def _append_with_headers(self, key: str, array: NDArrayFloat, header: NDArrayStr):
        """Append landmark data using estimator-defined column headers.

        This method writes landmark data to a CSV file with headers extracted from
        the ProcessResult. The array is reshaped to 2D (V, C) where V is the number
        of vertices (landmarks/joints) and C is the flattened feature dimension
        per vertex.

        Args:
            key (:class:`str`): The landmark key (e.g., "pose", "left_hand").
            array (:class:`NDArrayFloat`): Landmark array with shape (V, ...).
                Will be flattened to (V, C) where C = prod(shape[1:]).
            header (:class:`NDArrayStr`): Column header names with shape (C,).
                Must match the flattened width of the array.

        Raises:
            CollectorError: If the CSV landmark directory is not prepared.
            CollectorError: If header length doesn't match flattened array width.
        """
        # Reshape array to 2D: (V, C) where V = num_vertices, C = flattened_width
        sample_matrix = np.atleast_2d(np.asarray(array, dtype=float))
        num_vertices = sample_matrix.shape[0]
        sample_width = int(np.prod(sample_matrix.shape[1:])) if num_vertices > 0 else 0
        flattened = sample_matrix.reshape(num_vertices, sample_width)

        # Reshape header to 1D: (C,)
        header_1d = np.atleast_1d(np.asarray(header, dtype=str))
        header_width = int(np.prod(header_1d.shape))
        header_flat = header_1d.reshape(header_width)

        # Validate header is provided when landmarks exist
        if sample_width > 0 and header_width == 0:
            raise CollectorError(
                f"Empty headers provided for key '{key}' with {sample_width} "
                f"columns. CSV format requires column headers. "
                f"Ensure ProcessResult.headers is populated."
            )

        # Validate header length matches data width
        if sample_width > 0 and header_width != sample_width:
            raise CollectorError(
                f"Header length mismatch for key '{key}': "
                f"expected {sample_width} columns, got {header_width} headers. "
                f"Ensure ProcessResult.headers matches landmark array dimensions."
            )

        # Create writer with headers
        if key not in self._writers:
            if self._base_dir is None:
                raise CollectorError(
                    "CSV landmark directory not prepared. "
                    "Call _open_file() before appending results."
                )
            file_path = self._base_dir / f"{key}{self.extension}"
            handle = file_path.open("w", newline="", encoding=self.encoding)
            fieldnames = header_flat.tolist()
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
        else:
            # Verify consistent sample width across frames
            if self._sample_width[key] != sample_width:
                raise CollectorError(
                    f"Inconsistent landmark sample width for key '{key}': "
                    f"expected {self._sample_width[key]}, got {sample_width}. "
                    f"All frames must have the same landmark dimensions."
                )

        writer = self._writers[key]
        header_list = header_flat.tolist()

        for vertex in flattened:
            row = {header_list[i]: float(v) for i, v in enumerate(vertex.tolist())}
            writer.writerow(row)
            self._row_index[key] += 1

    def _ensure_writer(self, key: str, sample_width: int) -> DictWriter[str]:
        """Ensure a CSV writer exists for the given key.

        Args:
            key (:class:`str`): The landmark key.
            sample_width (:class:`int`): The width of each sample.

        Returns:
            `DictWriter[str]`: The CSV writer for this key.

        Raises:
            CollectorError: If the landmark directory is not prepared.
            CollectorError: If sample width is inconsistent.
        """
        if self._base_dir is None:
            raise CollectorError(
                "CSV landmark directory not prepared. "
                "Call _open_file() before appending results."
            )
        writer = self._writers.get(key)
        if writer is None:
            file_path = self._base_dir / f"{key}{self.extension}"
            handle = file_path.open("w", newline="", encoding=self.encoding)
            fieldnames = [f"value_{idx}" for idx in range(sample_width)]
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
            raise CollectorError(
                f"Inconsistent landmark sample width for key '{key}': "
                f"expected {self._sample_width[key]} columns, got {sample_width}. "
                f"All landmark frames for the same key must have the same shape."
            )
        return writer

    def _append_result(
        self,
        frame_id: int,
        headers: Mapping[K, NDArrayStr],
        landmarks: Mapping[K, NDArrayFloat],
    ):
        """Append landmark data for a single frame to CSV files.

        This method processes landmark data for one frame. If headers are provided
        and match the landmark dimensions, it uses them. Otherwise, it falls back
        to generic column names "value_0", "value_1", etc.

        Args:
            frame_id (:class:`int`): The frame index (currently unused for CSV format).
            headers (:class:`Mapping[K, NDArrayStr]`): Mapping of keys to header arrays.
                Each header array should have shape (C,) matching the flattened
                landmark width.
            landmarks (:class:`Mapping[K, NDArrayFloat]`): Mapping of keys to
                landmark arrays.
                Each array has shape (V, ...) where V is the number of vertices
                (landmarks/joints).

        Raises:
            CollectorError: If the CSV landmark directory is not prepared.
            CollectorError: If sample width is inconsistent for a key.
        """
        for raw_key, array in landmarks.items():
            key = str(raw_key)
            # Get headers for this key if available
            key_headers = headers.get(raw_key, np.array([], dtype=str))
            self._append_with_headers(key, array, key_headers)

    def _close_file(self):
        """Close all open CSV file handles and reset internal state.

        This method should be called when landmark collection is complete.
        It closes all file handles and clears internal dictionaries to prepare
        for a new collection session.
        """
        for handle in self._file_handles.values():
            handle.close()
        self._base_dir = None
        self._writers.clear()
        self._file_handles.clear()
        self._sample_width.clear()
        self._row_index.clear()


def csv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create a CSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver
            with comma delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K]()


def tsv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create a TSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver
            with tab delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K](delimiter="\t")


def ssv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create an SSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver
            with semicolon delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K](delimiter=";")


lmsc_aliases.update(
    {
        ".csv": csv_lmsc_creator,
        ".tsv": tsv_lmsc_creator,
        ".ssv": ssv_lmsc_creator,
    }
)
