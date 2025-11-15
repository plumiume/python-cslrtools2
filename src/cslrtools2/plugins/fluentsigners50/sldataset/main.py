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

from typing import Any, Literal, Mapping
from pathlib import Path
import re
import csv

from ....typings import PathLike, ArrayLike
from ....sldataset.array_loader import prekey_loaders, container_loaders
from ....sldataset.dataset import SLDatasetItem, SLDataset

type FS50MetaKeys = Literal[
    "signer_ids", "sentence_ids", "take_ids",
    "gloss_annotation", "russian_translation"
]

RE_PATTERN = re.compile(r"^P(?P<signer_id>\d{2})_S(?P<sentence_id>\d{3})_(?P<take_id>\d{2})$")
GLOB_PATTERN = "**/P??_S???_??*"

def load(
    origin: PathLike,
    # (origin) / 'KRSL_173_17_08' / <directory> / "P{:02d}_S{:03d}_{:02d}.mp4"
    # (origin) / 'gloss_annotation.csv'
    # (origin) / 'russian_translation.csv'
    processed: PathLike,
    # (processed) / <directory> / "P{:02d}_S{:03d}_{:02d}" / any files or dirs,
    # (processed) / connections (dir or file)
    ) -> SLDataset[FS50MetaKeys, Any, Any, Any, ArrayLike]:

    origin = Path(origin)
    processed = Path(processed)

    gloss_annotation: dict[int, list[str]] = {
        int(record["Id"]): record["Gloss"].split()
        for record in csv.DictReader(
            open(origin / "gloss_annotation.csv", encoding="utf-8")
        )
    }

    russian_translation: dict[int, str] = {
        int(record["Id"]): record["RussianTranslation"]
        for record in csv.DictReader(
            open(origin / "russian_translation.csv", encoding="utf-8")
        )
    }

    pathes = list(processed.glob(GLOB_PATTERN))

    items = [
        SLDatasetItem[Any, ArrayLike, Any, ArrayLike, Any, ArrayLike].from_file_system(p)
        for p in pathes
    ]

    matches: list[tuple[int, int, int]] = []
    for p in pathes:
        m = RE_PATTERN.match(p.stem)
        if m is None:
            raise ValueError(f"Invalid file or directory name: {p}")
        matches.append((
            int(m.group("signer_id")),
            int(m.group("sentence_id")),
            int(m.group("take_id")),
        ))

    conn_path = processed / "connections"
    connections: Mapping[tuple[str, str], Any] = {}
    if conn_path.exists() and conn_path.is_dir():
        for file in conn_path.iterdir():
            if not file.is_file():
                continue
            loader = prekey_loaders.get(file.suffix)
            if loader is None:
                continue
            a, b = file.stem.split('.', 1)
            connections[(a, b)] = loader(file)
    else:
        for ext, loader in container_loaders.items():
            file = conn_path.with_suffix(ext)
            if not file.exists() or not file.is_file():
                continue
            mapping = loader(file)
            for key, value in mapping.items():
                a, b = key.split('.', 1)
                connections[(a, b)] = value
            break

    return SLDataset(
        metadata={
            "signer_ids": [m[0] for m in matches],
            "sentence_ids": [m[1] for m in matches],
            "take_ids": [m[2] for m in matches],
            "gloss_annotation": gloss_annotation,
            "russian_translation": russian_translation,
        },
        connections=connections,
        items=items,
    )
