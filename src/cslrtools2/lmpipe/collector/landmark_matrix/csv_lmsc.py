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

from csv import DictWriter
from pathlib import Path
from typing import Mapping, TextIO

import numpy as np

from ....typings import NDArrayFloat
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
            encoding (:class:`str`, optional): File encoding. Defaults to "utf-8".
            extension (:class:`str | None`, optional): File extension. Auto-detected if None. Defaults to None.
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
        self._base_dir = self._prepare_landmark_dir(path)
        self._writers.clear()
        self._file_handles.clear()
        self._sample_width.clear()
        self._row_index.clear()

    def _ensure_writer(self, key: str, sample_width: int) -> DictWriter[str]:
        """Ensure a CSV writer exists for the given key.

        Args:
            key (:class:`str`): The landmark key.
            sample_width (:class:`int`): The width of each sample.

        Returns:
            `DictWriter[str]`: The CSV writer for this key.

        Raises:
            RuntimeError: If the landmark directory is not prepared.
            ValueError: If sample width is inconsistent.
        """
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


def csv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create a CSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver with comma delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K]()

def tsv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create a TSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver with tab delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K](delimiter="\t")

def ssv_lmsc_creator[K: str](key_type: type[K]) -> CsvLandmarkMatrixSaveCollector[K]:
    """Create an SSV landmark matrix save collector.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`CsvLandmarkMatrixSaveCollector[K]`: CSV landmark matrix saver with semicolon delimiter.
    """
    return CsvLandmarkMatrixSaveCollector[K](delimiter=";")

lmsc_aliases.update({
    ".csv": csv_lmsc_creator,
    ".tsv": tsv_lmsc_creator,
    ".ssv": ssv_lmsc_creator,
})
