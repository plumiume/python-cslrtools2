# Copyright 2024 cslrtools2 contributors
# Licensed under the Apache License, Version 2.0

"""Pytest configuration and fixtures for cslrtools2 tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def skip_if_no_mediapipe():
    """Skip test if MediaPipe is not installed."""
    pytest.importorskip("mediapipe")
