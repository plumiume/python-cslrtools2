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

"""Pytest configuration and fixtures for integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for integration tests.

    Limits parallel execution for transform tests to avoid:
    - OSError: Paging file too small (torchvision memory usage)
    - Test collection mismatches between workers
    """
    # Check if running transform tests
    test_paths = config.args if config.args else []
    is_transform_test = any(
        "test_transform_dynamic" in str(p) for p in test_paths
    )

    if is_transform_test and hasattr(config.option, "numprocesses"):
        numprocs = config.option.numprocesses
        # Limit to 2 workers for transform tests (reduce memory pressure)
        if numprocs is not None and (numprocs == "auto" or numprocs > 2):
            config.option.numprocesses = 2


@pytest.fixture(scope="session")
def test_videos_dir() -> Path:
    """Return path to test videos directory.

    Returns:
        Path to tests/data/videos directory.

    Raises:
        pytest.skip: If test videos directory doesn't exist.
    """
    videos_dir = Path(__file__).parent.parent / "data" / "videos"
    if not videos_dir.exists():
        pytest.skip("Test videos not downloaded. Run setup_resources.py")
    return videos_dir


@pytest.fixture(scope="session")
def sample_video_stop(test_videos_dir: Path) -> Path:
    """Return path to hand_gesture_stop.mp4 test video.

    Args:
        test_videos_dir: Test videos directory fixture.

    Returns:
        Path to hand_gesture_stop.mp4.

    Raises:
        pytest.skip: If video file doesn't exist.
    """
    video = test_videos_dir / "hand_gesture_stop.mp4"
    if not video.exists():
        pytest.skip(f"Test video not found: {video}")
    return video


@pytest.fixture(scope="session")
def sample_video_man(test_videos_dir: Path) -> Path:
    """Return path to hand_gesture_man.mp4 test video.

    Args:
        test_videos_dir: Test videos directory fixture.

    Returns:
        Path to hand_gesture_man.mp4.

    Raises:
        pytest.skip: If video file doesn't exist.
    """
    video = test_videos_dir / "hand_gesture_man.mp4"
    if not video.exists():
        pytest.skip(f"Test video not found: {video}")
    return video


@pytest.fixture
def integration_tmp_path(tmp_path: Path) -> Path:
    """Return temporary directory for integration tests.

    Args:
        tmp_path: Pytest's tmp_path fixture.

    Returns:
        Path to integration test temporary directory.
    """
    integration_dir = tmp_path / "integration_test"
    integration_dir.mkdir(parents=True, exist_ok=True)
    return integration_dir


@pytest.fixture
def skip_if_no_mediapipe() -> None:
    """Skip test if MediaPipe is not installed.

    Raises:
        pytest.skip: If mediapipe package is not available.
    """
    pytest.importorskip("mediapipe")
