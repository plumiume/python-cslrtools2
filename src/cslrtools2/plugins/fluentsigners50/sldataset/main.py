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
from concurrent.futures import ThreadPoolExecutor

from ....sldataset.logger import sldataset_logger
from ....typings import PathLike, ArrayLike
from ....sldataset.array_loader import prekey_loaders, container_loaders
from ....sldataset.dataset import (
    SLDatasetItem,
    IterableSLDataset
)

type FS50MetaKeys = Literal[
    # item keys
    "signer_ids", "sentence_ids", "take_ids",
    # dataset keys
    "gloss_annotation", "russian_translation"
]

RE_PATTERN = re.compile(
    r"^P(?P<signer_id>\d+)_S(?P<sentence_id>\d+)_(?P<take_id>\d+)$"
)


def load(
    origin: PathLike,
    # (origin) / 'KRSL_173_17_08' / <directory> / "P{:02d}_S{:03d}_{:02d}.mp4"
    # (origin) / 'gloss_annotation.csv'
    # (origin) / 'russian_translation.csv'
    processed: PathLike,
    # (processed) / <directory> / "P{:02d}_S{:03d}_{:02d}" / any files or dirs,
    # (processed) / connections (dir or file)
) -> IterableSLDataset[FS50MetaKeys, Any, Any, Any, ArrayLike]:

    # Convert paths to Path objects for consistent handling
    origin = Path(origin)
    processed = Path(processed)

    # ========================================================================
    # Step 1: Load CSV annotations (gloss and Russian translation)
    # ========================================================================
    sldataset_logger.debug(
        f"Loading gloss annotations from: {origin / 'gloss_annotation.csv'}"
    )
    gloss_annotation: dict[int, list[str]] = {
        int(record["ID"]): record["Gloss"].split()
        for record in csv.DictReader(
            open(origin / "gloss_annotation.csv", encoding="utf-8")
        )
    }
    sldataset_logger.debug(f"Loaded {len(gloss_annotation)} gloss annotations")

    sldataset_logger.debug(
        f"Loading Russian translations from: {origin / 'russian_translation.csv'}"
    )
    russian_translation: dict[int, str] = {
        int(record["ID"]): record["Translation"]
        for record in csv.DictReader(
            open(origin / "russian_translation.csv", encoding="utf-8")
        )
    }
    sldataset_logger.debug(f"Loaded {len(russian_translation)} Russian translations")

    # ========================================================================
    # Step 2: Search filesystem for matching items and extract metadata
    # ========================================================================
    sldataset_logger.debug(f"Searching for items with pattern: {RE_PATTERN.pattern}")
    sldataset_logger.debug(f"Search path: {processed}")

    # Build path-match pairs by filtering during iteration
    # Each pair contains (path, (signer_id, sentence_id, take_id))
    path_matches_pairs: list[tuple[Path, tuple[int, int, int]]] = sorted(
        (
            (
                p,
                (
                    int(m.group("signer_id")),
                    int(m.group("sentence_id")),
                    int(m.group("take_id")),
                ),
            )
            for p in processed.glob("**/*")
            if (m := RE_PATTERN.match(p.stem)) is not None
        ),
        key=lambda x: (
            sldataset_logger.debug("Found matching path: %s", x[0]),
            x[1]
        )[1],
    )

    sldataset_logger.info(f"Found {len(path_matches_pairs)} items matching pattern")

    if path_matches_pairs:
        first_paths = [p.name for p, _ in path_matches_pairs[:3]]
        sldataset_logger.debug(f"First few paths: {first_paths}")

    # ========================================================================
    # Step 3: Load dataset items from filesystem
    # ========================================================================
    sldataset_logger.info("Loading items from file system...")
    with ThreadPoolExecutor(20) as executor:
        futures = [
            executor.submit(
                SLDatasetItem[
                    Any, ArrayLike,
                    Any, ArrayLike,
                    Any, ArrayLike,
                ].from_file_system,
                p
            )
            for p, _ in path_matches_pairs
        ]

    items = [future.result() for future in futures]

    sldataset_logger.info(f"Created item generator for {len(path_matches_pairs)} items")

    # ========================================================================
    # Step 4: Load landmark connection data
    # ========================================================================
    sldataset_logger.debug("Loading connection data...")
    conn_path = processed / "connections"
    connections: Mapping[tuple[Any, Any], ArrayLike] = {}

    # Try loading from directory first (per-key files)
    if conn_path.exists() and conn_path.is_dir():
        sldataset_logger.debug(f"Loading connections from directory: {conn_path}")
        for file in conn_path.iterdir():
            if not file.is_file():
                continue
            loader = prekey_loaders.get(file.suffix)
            if loader is None:
                continue
            a, b = file.stem.split(".", 1)
            connections[(a, b)] = loader(file)
            sldataset_logger.debug(f"Loaded connection: {a}.{b}")

    # Try loading from container file (all connections in one file)
    else:
        sldataset_logger.debug(f"Looking for container connection file at: {conn_path}")
        for ext, loader in container_loaders.items():
            file = conn_path.with_suffix(ext)
            if not file.exists() or not file.is_file():
                continue
            sldataset_logger.debug(f"Loading connections from container: {file}")
            mapping = loader(file)
            for key, value in mapping.items():
                a, b = key.split(".", 1)
                connections[(a, b)] = value
            sldataset_logger.debug(f"Loaded {len(mapping)} connections from container")
            break

    sldataset_logger.info(f"Loaded {len(connections)} connection entries")

    # ========================================================================
    # Step 5: Create and return SLDataset object
    # ========================================================================
    sldataset_logger.info("Creating SLDataset object...")
    dataset = IterableSLDataset[
        FS50MetaKeys, Any, Any, Any, ArrayLike
    ](
        metadata={
            "signer_ids": [m[0] for _, m in path_matches_pairs],
            "sentence_ids": [m[1] for _, m in path_matches_pairs],
            "take_ids": [m[2] for _, m in path_matches_pairs],
            "gloss_annotation": gloss_annotation,
            "russian_translation": russian_translation,
        },
        connections=connections,
        items=items,
    )
    sldataset_logger.info("SLDataset object created successfully")
    return dataset
