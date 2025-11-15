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

"""MediaPipe constants for landmarks and connections.

This module provides centralized access to MediaPipe's official landmark and
connection constants. By re-exporting these constants, we ensure consistency
with MediaPipe's definitions and reduce maintenance burden.

Module Structure
----------------
This module exports two types of constants:

1. **Landmark Enums**: Named indices for individual landmarks
2. **Connection Sets**: Pairs of landmark indices defining skeleton structure

Example:
    Using landmark enums for indexing::

        >>> from cslrtools2.plugins.mediapipe.lmpipe.constants import (
        ...     PoseLandmark,
        ...     HandLandmark
        ... )
        >>>
        >>> # Access specific landmarks by name
        >>> nose_idx = PoseLandmark.NOSE
        >>> wrist_idx = HandLandmark.WRIST
        >>>
        >>> # Use in landmark array indexing
        >>> nose_position = pose_landmarks[PoseLandmark.NOSE]
        >>> wrist_position = hand_landmarks[HandLandmark.WRIST]

    Using connections for skeleton visualization::

        >>> from cslrtools2.plugins.mediapipe.lmpipe.constants import (
        ...     POSE_CONNECTIONS,
        ...     HAND_CONNECTIONS
        ... )
        >>>
        >>> # Draw skeleton connections
        >>> for start_idx, end_idx in POSE_CONNECTIONS:
        ...     start_pt = landmarks[start_idx]
        ...     end_pt = landmarks[end_idx]
        ...     cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

Landmark Enums
--------------
**PoseLandmark** (IntEnum, 33 landmarks):
    Enum for pose landmark indices from MediaPipe Pose model.

    Key landmarks include:
    - ``NOSE`` (0): Nose tip
    - ``LEFT_SHOULDER`` (11), ``RIGHT_SHOULDER`` (12): Shoulders
    - ``LEFT_WRIST`` (15), ``RIGHT_WRIST`` (16): Wrists
    - ``LEFT_HIP`` (23), ``RIGHT_HIP`` (24): Hips
    - ``LEFT_ANKLE`` (27), ``RIGHT_ANKLE`` (28): Ankles

    See ``PoseLandmark`` documentation for complete list.

**HandLandmark** (IntEnum, 21 landmarks):
    Enum for hand landmark indices from MediaPipe Hands model.

    Key landmarks include:
    - ``WRIST`` (0): Wrist
    - ``THUMB_TIP`` (4): Thumb tip
    - ``INDEX_FINGER_TIP`` (8): Index finger tip
    - ``MIDDLE_FINGER_TIP`` (12): Middle finger tip
    - ``RING_FINGER_TIP`` (16): Ring finger tip
    - ``PINKY_TIP`` (20): Pinky tip

    See ``HandLandmark`` documentation for complete list.

Connection Sets
---------------
**Pose (33 landmarks, 35 connections)**:
    - ``POSE_CONNECTIONS``: Full body skeleton

**Hand (21 landmarks, 21 connections)**:
    - ``HAND_CONNECTIONS``: Hand skeleton for both left and right hands

**Face Mesh (468-478 landmarks)**:
    - ``FACEMESH_TESSELATION`` (2556): High-density triangular mesh
    - ``FACEMESH_CONTOURS`` (124): Outline only - recommended for performance
    - ``FACEMESH_IRISES`` (8): Iris circles (left + right)
    - ``FACEMESH_FACE_OVAL`` (36): Face boundary
    - ``FACEMESH_LEFT_EYE`` (16): Left eye outline
    - ``FACEMESH_RIGHT_EYE`` (16): Right eye outline
    - ``FACEMESH_LEFT_EYEBROW`` (8): Left eyebrow
    - ``FACEMESH_RIGHT_EYEBROW`` (8): Right eyebrow
    - ``FACEMESH_LEFT_IRIS`` (4): Left iris circle
    - ``FACEMESH_RIGHT_IRIS`` (4): Right iris circle
    - ``FACEMESH_LIPS`` (40): Lip outline
    - ``FACEMESH_NOSE`` (25): Nose outline

Type Safety
-----------
For type checking and IDE support, use the provided type aliases::

    >>> from cslrtools2.plugins.mediapipe.lmpipe.constants import ConnectionSet
    >>>
    >>> def draw_connections(
    ...     frame: np.ndarray,
    ...     landmarks: np.ndarray,
    ...     connections: ConnectionSet
    ... ) -> None:
    ...     for start_idx, end_idx in connections:
    ...         # Draw implementation
    ...         pass

Migration from Old Code
-----------------------
If you have existing code using ``MediaPipePoseNames`` or ``MediaPipeHandNames``,
you can simply replace them with the MediaPipe originals::

    # Old (cslrtools2 custom enum)
    from cslrtools2.plugins.mediapipe.lmpipe.pose import MediaPipePoseNames
    nose_idx = MediaPipePoseNames.NOSE

    # New (MediaPipe official enum)
    from cslrtools2.plugins.mediapipe.lmpipe.constants import PoseLandmark
    nose_idx = PoseLandmark.NOSE

The values are identical, ensuring backward compatibility.

See Also
--------
- `MediaPipe Pose <https://google.github.io/mediapipe/solutions/pose.html>`_
- `MediaPipe Hands <https://google.github.io/mediapipe/solutions/hands.html>`_
- `MediaPipe Face Mesh <https://google.github.io/mediapipe/solutions/face_mesh.html>`_

Notes
-----
All constants in this module are directly imported from MediaPipe's official
Python API. This ensures:

- **Consistency**: Definitions match MediaPipe's documentation
- **Maintenance**: Updates to MediaPipe automatically propagate
- **Correctness**: No risk of transcription errors in manual definitions

"""

from __future__ import annotations

# ============================================================================
# Landmark Enums
# ============================================================================

# Pose landmarks (33 landmarks)
from mediapipe.python.solutions.pose import (  # pyright: ignore[reportMissingTypeStubs]
    PoseLandmark as _PoseLandmark,
)

# Hand landmarks (21 landmarks)
from mediapipe.python.solutions.hands import (  # pyright: ignore[reportMissingTypeStubs]
    HandLandmark as _HandLandmark,
)

# Re-export landmark enums with proper types
PoseLandmark = _PoseLandmark
"""MediaPipe Pose landmark indices (IntEnum with 33 landmarks).

Complete list of pose landmarks:
    - NOSE (0)
    - LEFT_EYE_INNER (1), LEFT_EYE (2), LEFT_EYE_OUTER (3)
    - RIGHT_EYE_INNER (4), RIGHT_EYE (5), RIGHT_EYE_OUTER (6)
    - LEFT_EAR (7), RIGHT_EAR (8)
    - MOUTH_LEFT (9), MOUTH_RIGHT (10)
    - LEFT_SHOULDER (11), RIGHT_SHOULDER (12)
    - LEFT_ELBOW (13), RIGHT_ELBOW (14)
    - LEFT_WRIST (15), RIGHT_WRIST (16)
    - LEFT_PINKY (17), RIGHT_PINKY (18)
    - LEFT_INDEX (19), RIGHT_INDEX (20)
    - LEFT_THUMB (21), RIGHT_THUMB (22)
    - LEFT_HIP (23), RIGHT_HIP (24)
    - LEFT_KNEE (25), RIGHT_KNEE (26)
    - LEFT_ANKLE (27), RIGHT_ANKLE (28)
    - LEFT_HEEL (29), RIGHT_HEEL (30)
    - LEFT_FOOT_INDEX (31), RIGHT_FOOT_INDEX (32)
"""

HandLandmark = _HandLandmark
"""MediaPipe Hand landmark indices (IntEnum with 21 landmarks).

Complete list of hand landmarks:
    - WRIST (0)
    - THUMB_CMC (1), THUMB_MCP (2), THUMB_IP (3), THUMB_TIP (4)
    - INDEX_FINGER_MCP (5), INDEX_FINGER_PIP (6), INDEX_FINGER_DIP (7), INDEX_FINGER_TIP (8)
    - MIDDLE_FINGER_MCP (9), MIDDLE_FINGER_PIP (10), MIDDLE_FINGER_DIP (11), MIDDLE_FINGER_TIP (12)
    - RING_FINGER_MCP (13), RING_FINGER_PIP (14), RING_FINGER_DIP (15), RING_FINGER_TIP (16)
    - PINKY_MCP (17), PINKY_PIP (18), PINKY_DIP (19), PINKY_TIP (20)
"""


# ============================================================================
# Connection Sets
# ============================================================================

# Pose connections (33 landmarks, 35 connections)
from mediapipe.python.solutions.pose import (  # pyright: ignore[reportMissingTypeStubs]
    POSE_CONNECTIONS as _POSE_CONNECTIONS,
)

# Hand connections (21 landmarks, 21 connections)
from mediapipe.python.solutions.hands import (  # pyright: ignore[reportMissingTypeStubs]
    HAND_CONNECTIONS as _HAND_CONNECTIONS,  # pyright: ignore[reportUnknownVariableType]
)

# Face mesh connections (468 landmarks without irises, 478 with irises)
from mediapipe.python.solutions.face_mesh import (  # pyright: ignore[reportMissingTypeStubs]
    FACEMESH_TESSELATION as _FACEMESH_TESSELATION,
    FACEMESH_CONTOURS as _FACEMESH_CONTOURS,  # pyright: ignore[reportUnknownVariableType]
    FACEMESH_IRISES as _FACEMESH_IRISES,  # pyright: ignore[reportUnknownVariableType]
    FACEMESH_FACE_OVAL as _FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE as _FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE as _FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW as _FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW as _FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS as _FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS as _FACEMESH_RIGHT_IRIS,
    FACEMESH_LIPS as _FACEMESH_LIPS,
    FACEMESH_NOSE as _FACEMESH_NOSE,
)


# Type alias for connection sets
ConnectionSet = frozenset[tuple[int, int]]
"""Type alias for skeleton connection sets.

A connection set is a ``frozenset`` of ``(start_index, end_index)`` tuples,
where each tuple represents a line connecting two landmarks.
"""

# Re-export with proper type annotations
POSE_CONNECTIONS: ConnectionSet = _POSE_CONNECTIONS
"""Pose skeleton connections (35 connections between 33 landmarks).

Defines the full body skeleton structure for MediaPipe Pose model.
"""

HAND_CONNECTIONS: ConnectionSet = _HAND_CONNECTIONS  # pyright: ignore[reportUnknownVariableType]
"""Hand skeleton connections (21 connections between 21 landmarks).

Defines the hand skeleton structure for MediaPipe Hands model.
Applies to both left and right hands.
"""

FACEMESH_TESSELATION: ConnectionSet = _FACEMESH_TESSELATION
"""Face mesh tesselation (2556 connections).

High-density triangular mesh covering the entire face.
Recommended for high-quality offline processing only due to high overhead.
"""

FACEMESH_CONTOURS: ConnectionSet = _FACEMESH_CONTOURS  # pyright: ignore[reportUnknownVariableType]
"""Face mesh contours (124 connections).

Lightweight outline of facial features. Recommended for real-time applications.
"""

FACEMESH_IRISES: ConnectionSet = _FACEMESH_IRISES  # pyright: ignore[reportUnknownVariableType]
"""Iris connections (8 connections, 4 per iris).

Circles around left and right irises. Only available when using face mesh
with iris landmarks (478 landmarks total).
"""

FACEMESH_FACE_OVAL: ConnectionSet = _FACEMESH_FACE_OVAL
"""Face oval boundary (36 connections)."""

FACEMESH_LEFT_EYE: ConnectionSet = _FACEMESH_LEFT_EYE
"""Left eye outline (16 connections)."""

FACEMESH_RIGHT_EYE: ConnectionSet = _FACEMESH_RIGHT_EYE
"""Right eye outline (16 connections)."""

FACEMESH_LEFT_EYEBROW: ConnectionSet = _FACEMESH_LEFT_EYEBROW
"""Left eyebrow (8 connections)."""

FACEMESH_RIGHT_EYEBROW: ConnectionSet = _FACEMESH_RIGHT_EYEBROW
"""Right eyebrow (8 connections)."""

FACEMESH_LEFT_IRIS: ConnectionSet = _FACEMESH_LEFT_IRIS
"""Left iris circle (4 connections)."""

FACEMESH_RIGHT_IRIS: ConnectionSet = _FACEMESH_RIGHT_IRIS
"""Right iris circle (4 connections)."""

FACEMESH_LIPS: ConnectionSet = _FACEMESH_LIPS
"""Lip outline (40 connections)."""

FACEMESH_NOSE: ConnectionSet = _FACEMESH_NOSE
"""Nose outline (25 connections)."""


__all__ = [
    # Landmark enums
    "PoseLandmark",
    "HandLandmark",
    # Pose connections
    "POSE_CONNECTIONS",
    # Hand connections
    "HAND_CONNECTIONS",
    # Face mesh - Full
    "FACEMESH_TESSELATION",
    "FACEMESH_CONTOURS",
    "FACEMESH_IRISES",
    # Face mesh - Parts
    "FACEMESH_FACE_OVAL",
    "FACEMESH_LEFT_EYE",
    "FACEMESH_RIGHT_EYE",
    "FACEMESH_LEFT_EYEBROW",
    "FACEMESH_RIGHT_EYEBROW",
    "FACEMESH_LEFT_IRIS",
    "FACEMESH_RIGHT_IRIS",
    "FACEMESH_LIPS",
    "FACEMESH_NOSE",
    # Type alias
    "ConnectionSet",
]
