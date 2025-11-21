# Copyright 2024 cslrtools2 contributors
# Licensed under the Apache License, Version 2.0

"""Pytest configuration and fixtures for cslrtools2 tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def initialize_torch_once():
    """Initialize PyTorch once at session start to avoid Triton namespace conflicts.

    This fixture ensures that PyTorch's Triton library is loaded only once
    per test session, preventing the "Only a single TORCH_LIBRARY can be used"
    error when running multiple tests that import torch.
    """
    try:
        import torch

        # Force torch initialization
        _ = torch.cuda.is_available()
    except ImportError:
        # torch not installed, skip initialization
        pass
    yield


@pytest.fixture
def skip_if_no_mediapipe():
    """Skip test if MediaPipe is not installed."""
    pytest.importorskip("mediapipe")
