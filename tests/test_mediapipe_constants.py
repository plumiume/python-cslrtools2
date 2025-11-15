# Copyright 2024 cslrtools2 contributors
# Licensed under the Apache License, Version 2.0

"""Test MediaPipe constants integration."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("mediapipe", reason="MediaPipe not installed") is None,
    reason="MediaPipe not installed"
)


def test_import_mediapipe_constants():
    """Test that MediaPipe constants can be imported."""
    from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import (
        PoseLandmark,
        HandLandmark,
        POSE_CONNECTIONS,
        HAND_CONNECTIONS,
    )

    assert PoseLandmark is not None
    assert HandLandmark is not None
    assert POSE_CONNECTIONS is not None
    assert HAND_CONNECTIONS is not None


@pytest.mark.mediapipe
def test_pose_landmark_enum():
    """Test PoseLandmark enum values."""
    from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import PoseLandmark

    assert hasattr(PoseLandmark, 'NOSE')
    assert hasattr(PoseLandmark, 'LEFT_WRIST')
    assert len(PoseLandmark) == 33


@pytest.mark.mediapipe
def test_hand_landmark_enum():
    """Test HandLandmark enum values."""
    from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import HandLandmark

    assert hasattr(HandLandmark, 'WRIST')
    assert hasattr(HandLandmark, 'THUMB_TIP')
    assert len(HandLandmark) == 21


@pytest.mark.mediapipe
def test_pose_connections():
    """Test POSE_CONNECTIONS structure."""
    from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import POSE_CONNECTIONS

    assert isinstance(POSE_CONNECTIONS, frozenset)
    assert len(POSE_CONNECTIONS) > 0
    # Each connection should be a tuple of two ints
    for conn in list(POSE_CONNECTIONS)[:5]:
        assert isinstance(conn, tuple)
        assert len(conn) == 2


@pytest.mark.mediapipe
def test_hand_connections():
    """Test HAND_CONNECTIONS structure."""
    from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import HAND_CONNECTIONS

    assert isinstance(HAND_CONNECTIONS, frozenset)
    assert len(HAND_CONNECTIONS) > 0
    # Each connection should be a tuple of two ints
    for conn in list(HAND_CONNECTIONS)[:5]:
        assert isinstance(conn, tuple)
        assert len(conn) == 2
