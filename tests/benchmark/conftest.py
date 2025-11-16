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

"""Pytest configuration and fixtures for benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def benchmark_data_dir() -> Path:
    """Return path to benchmark test data directory.

    Returns:
        Path to tests/data directory.
    """
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def small_landmark_data() -> np.ndarray:
    """Generate small synthetic landmark data for benchmarking.

    Returns:
        Numpy array of shape (10, 33, 3) representing 10 frames
        of 33 landmarks with (x, y, z) coordinates.
    """
    np.random.seed(42)
    return np.random.rand(10, 33, 3).astype(np.float32)


@pytest.fixture
def medium_landmark_data() -> np.ndarray:
    """Generate medium synthetic landmark data for benchmarking.

    Returns:
        Numpy array of shape (100, 33, 3) representing 100 frames
        of 33 landmarks with (x, y, z) coordinates.
    """
    np.random.seed(42)
    return np.random.rand(100, 33, 3).astype(np.float32)


@pytest.fixture
def large_landmark_data() -> np.ndarray:
    """Generate large synthetic landmark data for benchmarking.

    Returns:
        Numpy array of shape (1000, 33, 3) representing 1000 frames
        of 33 landmarks with (x, y, z) coordinates.
    """
    np.random.seed(42)
    return np.random.rand(1000, 33, 3).astype(np.float32)
