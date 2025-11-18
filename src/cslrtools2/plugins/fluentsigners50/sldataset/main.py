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

"""FluentSigners50 dataset loader and connection constructor.

This module provides functionality to load the FluentSigners50 (FS50) dataset,
a Russian Sign Language dataset containing video recordings with gloss annotations
and Russian translations. The module handles loading of landmark data, metadata,
and landmark connection graphs for MediaPipe-based sign language recognition.

The main components include:

- :func:`load`: Loads the complete FS50 dataset from filesystem
- :func:`construct_default_connections`: Builds default landmark connection matrices
- :func:`construct_pose_hand_connections`: Creates pose-to-hand connection matrices

Typical usage::

    >>> from pathlib import Path
    >>> dataset = load(
    ...     origin=Path("/data/fs50/origin"),
    ...     processed=Path("/data/fs50/processed")
    ... )
    >>> connections = construct_default_connections()
"""

from __future__ import annotations

from typing import Any, Literal, Mapping
from pathlib import Path
import re
import csv
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cslrtools2.plugins.mediapipe.lmpipe.mp_constants as mp_constants

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


type _PartKey = Literal["pose", "left_hand", "right_hand", "face"]


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
    """Load FluentSigners50 dataset from filesystem.

    Loads the FluentSigners50 dataset by reading CSV annotations (gloss and Russian
    translations) from the origin directory and loading processed landmark data from
    the processed directory. The function searches for items matching the pattern
    ``P{signer_id}_S{sentence_id}_{take_id}`` and loads corresponding landmark data
    and connection information.

    The dataset structure follows:

    - Origin directory contains:
        - ``gloss_annotation.csv``: Gloss annotations indexed by sentence ID
        - ``russian_translation.csv``: Russian translations indexed by sentence ID
        - ``KRSL_173_17_08/`` subdirectory with video files

    - Processed directory contains:
        - Item directories matching pattern ``P{signer_id}_S{sentence_id}_{take_id}``
        - ``connections/`` directory or file with landmark connectivity data

    Args:
        origin (`PathLike`): Path to the origin dataset directory containing CSV
            annotations and raw video files. Must contain ``gloss_annotation.csv``
            and ``russian_translation.csv``.
        processed (`PathLike`): Path to the processed dataset directory containing
            extracted landmarks and connection data. Items are loaded from
            subdirectories via :meth:`SLDatasetItem.from_file_system`.

    Returns:
        :class:`IterableSLDataset`: Dataset object with metadata keys including
            ``signer_ids``, ``sentence_ids``, ``take_ids``, ``gloss_annotation``,
            and ``russian_translation``. Items are loaded using
            :class:`ThreadPoolExecutor` with 20 workers for parallel processing.

    Note:
        Connection data can be stored either as per-key files in a directory or as
        a single container file (npz, zarr, etc.). The loader automatically detects
        the format using :data:`prekey_loaders` and :data:`container_loaders`.

        The function uses regular expression pattern :obj:`RE_PATTERN` to match
        item directories:
        ``^P(?P<signer_id>\\d+)_S(?P<sentence_id>\\d+)_(?P<take_id>\\d+)$``.

    Example::

        >>> from pathlib import Path
        >>> dataset = load(
        ...     origin=Path("/data/fs50/origin"),
        ...     processed=Path("/data/fs50/processed")
        ... )
        >>> print(len(dataset.metadata["signer_ids"]))
        250
        >>> print(dataset.metadata["gloss_annotation"][1])
        ['HELLO', 'WORLD']
    """

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


def construct_default_connections(
) -> dict[tuple[_PartKey, _PartKey], ArrayLike]:
    """Construct default landmark connection matrices for pose, hands, and face.

    Builds sparse adjacency matrices representing landmark connectivity within each
    body part (pose, left hand, right hand, face). Each connection matrix includes
    self-connections (identity matrix I), original connections (A), and reverse
    connections (A^-1) to create undirected graphs suitable for graph neural networks.

    The connections are derived from :data:`mp_constants.POSE_CONNECTIONS` and
    :data:`mp_constants.HAND_CONNECTIONS` defined in MediaPipe constants.

    Returns:
        :class:`dict`: Dictionary mapping body part tuples to connection matrices.
            Keys are tuples of ``(_PartKey, _PartKey)`` like ``("pose", "pose")``.
            Values are :class:`numpy.ndarray` with shape ``(2, N)`` representing
            sparse edge lists where each column ``[:, i]`` is a connection
            ``[source_idx, target_idx]``.

    Note:
        Face connections are not yet implemented and will be added in future versions.
        The function logs a TODO comment for this feature.

        The returned matrices follow the formula: I + A + A^-1 where:

        - I: Identity matrix (self-connections)
        - A: Original directed connections
        - A^-1: Reverse connections (transposed)

    Example::

        >>> connections = construct_default_connections()
        >>> pose_conn = connections[("pose", "pose")]
        >>> print(pose_conn.shape)
        (2, 231)  # 33 landmarks: 33 self + 99 forward + 99 reverse
        >>> left_hand_conn = connections[("left_hand", "left_hand")]
        >>> print(left_hand_conn.shape)
        (2, 81)  # 21 landmarks: 21 self + 30 forward + 30 reverse
    """

    connections = dict[tuple[_PartKey, _PartKey], ArrayLike]()

    # ============================================================
    # Pose Connections to Sparse Matrix
    # ============================================================

    # frozenset[tuple[int, int]]
    # I + A + A^-1
    pose_self_connections = np.array([
        (i, i) for i in range(len(mp_constants.PoseLandmark))
    ])
    pose_connections = np.asarray(mp_constants.POSE_CONNECTIONS)
    connections[("pose", "pose")] = np.concatenate([
        pose_self_connections,
        pose_connections,
        pose_connections[:, ::-1]
    ]).transpose()  # shape: (2, N)

    # ============================================================
    # Hands Connections to Sparse Matrix
    # ============================================================

    # frozenset[tuple[int, int]]
    # I + A + A^-1
    hand_self_connections = np.array([
        (i, i) for i in range(len(mp_constants.HandLandmark))
    ])
    hand_connections = np.asarray(mp_constants.HAND_CONNECTIONS)

    hand_catted = np.concatenate([
        hand_self_connections,
        hand_connections,
        hand_connections[:, ::-1]
    ]).transpose()  # shape: (2, N)

    connections[("left_hand", "left_hand")] = hand_catted
    connections[("right_hand", "right_hand")] = hand_catted

    # ============================================================
    # Face Connections to Sparse Matrix
    # ============================================================

    # TODO: Implement face connections to sparse matrix

    # ============================================================

    return connections


def construct_pose_hand_connections(
    pose_hand_key: list[str]
) -> dict[tuple[str, str], ArrayLike]:
    """Construct connection matrices between pose and hand landmarks.

    Creates sparse adjacency matrices connecting pose landmarks to hand landmarks
    based on anatomical correspondence. Supports connections at wrist, thumb, index,
    and pinky positions between pose body landmarks and hand landmarks from MediaPipe.

    The function generates bidirectional connections from pose to both left and right
    hands, using :class:`mp_constants.PoseLandmark` and
    :class:`mp_constants.HandLandmark` enumerations for landmark indexing.

    Args:
        pose_hand_key (`list[str]`): List of connection types to enable.
            Valid values are ``"wrist"``, ``"thumb"``, ``"index"``, and ``"pinky"``.
            Each key enables specific landmark connections:

            - ``"wrist"``: Connects :attr:`LEFT_WRIST`/:attr:`RIGHT_WRIST` to
              :attr:`HandLandmark.WRIST`
            - ``"thumb"``: Connects :attr:`LEFT_THUMB`/:attr:`RIGHT_THUMB` to
              :attr:`HandLandmark.THUMB_CMC`
            - ``"index"``: Connects :attr:`LEFT_INDEX`/:attr:`RIGHT_INDEX` to
              :attr:`HandLandmark.INDEX_FINGER_MCP`
            - ``"pinky"``: Connects :attr:`LEFT_PINKY`/:attr:`RIGHT_PINKY` to
              :attr:`HandLandmark.PINKY_MCP`

    Returns:
        :class:`dict`: Dictionary with two entries:

            - ``("pose", "left_hand")``: :class:`numpy.ndarray` of shape ``(2, N)``
              containing pose-to-left-hand connections
            - ``("pose", "right_hand")``: :class:`numpy.ndarray` of shape ``(2, N)``
              containing pose-to-right-hand connections

            where N is the number of enabled connection types. Each array is
            transposed to shape ``(2, N)`` with columns as edge pairs.

    Note:
        This function complements :func:`construct_default_connections` by adding
        inter-part connections. While default connections handle intra-part edges
        (e.g., within pose or within hands), this function creates edges between
        different body parts.

    Example::

        >>> connections = construct_pose_hand_connections(
        ...     pose_hand_key=["wrist", "index"]
        ... )
        >>> print(connections[("pose", "left_hand")].shape)
        (2, 2)  # Two connections: wrist and index
        >>> # First connection: pose wrist to hand wrist
        >>> # Second connection: pose index to hand index MCP
        >>>
        >>> # Using all connections
        >>> all_conns = construct_pose_hand_connections(
        ...     ["wrist", "thumb", "index", "pinky"]
        ... )
        >>> print(all_conns[("pose", "right_hand")].shape)
        (2, 4)  # Four connections enabled
    """

    # ============================================================
    # Pose-LeftHand Connections to Sparse Matrix
    # ============================================================

    pose_lefthand_conns = list[tuple[int, int]]()
    pose_righthand_conns = list[tuple[int, int]]()

    if "wrist" in pose_hand_key:
        pose_lefthand_conns.append((
            mp_constants.PoseLandmark.LEFT_WRIST,
            mp_constants.HandLandmark.WRIST
        ))
        pose_righthand_conns.append((
            mp_constants.PoseLandmark.RIGHT_WRIST,
            mp_constants.HandLandmark.WRIST
        ))

    if "thumb" in pose_hand_key:
        pose_lefthand_conns.append((
            mp_constants.PoseLandmark.LEFT_THUMB,
            mp_constants.HandLandmark.THUMB_CMC
        ))
        pose_righthand_conns.append((
            mp_constants.PoseLandmark.RIGHT_THUMB,
            mp_constants.HandLandmark.THUMB_CMC
        ))

    if "index" in pose_hand_key:
        pose_lefthand_conns.append((
            mp_constants.PoseLandmark.LEFT_INDEX,
            mp_constants.HandLandmark.INDEX_FINGER_MCP
        ))
        pose_righthand_conns.append((
            mp_constants.PoseLandmark.RIGHT_INDEX,
            mp_constants.HandLandmark.INDEX_FINGER_MCP
        ))

    if "pinky" in pose_hand_key:
        pose_lefthand_conns.append((
            mp_constants.PoseLandmark.LEFT_PINKY,
            mp_constants.HandLandmark.PINKY_MCP
        ))
        pose_righthand_conns.append((
            mp_constants.PoseLandmark.RIGHT_PINKY,
            mp_constants.HandLandmark.PINKY_MCP
        ))

    connections = dict[tuple[str, str], ArrayLike]()

    connections[("pose", "left_hand")] = np.array(
        pose_lefthand_conns
    ).transpose()  # shape: (2, N)

    connections[("pose", "right_hand")] = np.array(
        pose_righthand_conns
    ).transpose()  # shape: (2, N)

    return connections
