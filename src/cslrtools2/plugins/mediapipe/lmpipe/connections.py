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

"""Backward compatibility module for MediaPipe connections.

.. deprecated:: 0.1.0
    This module has been renamed to :mod:`~cslrtools2.plugins.mediapipe.lmpipe.mp_constants`.
    Use the new module instead::

        # Old (deprecated)
        from cslrtools2.plugins.mediapipe.lmpipe.connections import POSE_CONNECTIONS

        # New (recommended)
        from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import POSE_CONNECTIONS

    All exports from this module are re-exported from the mp_constants module.
"""

from __future__ import annotations

# Re-export everything from mp_constants for backward compatibility
from .mp_constants import (
    # Landmark enums
    PoseLandmark,
    HandLandmark,
    # Pose connections
    POSE_CONNECTIONS,
    # Hand connections
    HAND_CONNECTIONS,
    # Face mesh - Full
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES,
    # Face mesh - Parts
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    # Type alias
    ConnectionSet,
)

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

