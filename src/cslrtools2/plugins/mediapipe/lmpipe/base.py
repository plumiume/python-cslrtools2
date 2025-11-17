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

import os
import warnings
import requests

try:
    from mediapipe.tasks.python.core.base_options import (
        BaseOptions,
    )  # pyright: ignore[reportMissingTypeStubs]
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )  # pyright: ignore[reportMissingTypeStubs]
    from mediapipe.tasks.python.components.containers.landmark import (
        NormalizedLandmark,
    )  # pyright: ignore[reportMissingTypeStubs]
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required to use this plugin. "
        "Install it with: pip install cslrtools2[mediapipe]"
    ) from exc

from ...._root import PACKAGE_ROOT
from ....lmpipe.estimator import Estimator
from ....exceptions import ValidationError, ModelDownloadError
from .base_args import MediaPipeBaseArgs

MODELS: dict[str, dict[str, str]] = {
    "pose": {
        "lite": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        ),
        "full": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        ),
        "heavy": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
    },
    "hand": {
        "full": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
    },
    "face": {
        "full": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
    },
}

ASSETS_PATH = PACKAGE_ROOT / "assets"
ASSETS_PATH.mkdir(parents=True, exist_ok=True)


def get_mediapipe_model(part: str, size: str) -> str:
    """Download or retrieve cached MediaPipe model file.

    Downloads the specified MediaPipe model from Google Cloud Storage
    if not already cached locally. Models are stored in the package
    assets directory.

    Args:
        part (:obj:`str`): Body part type. One of ``'pose'``, ``'hand'``, ``'face'``.
        size (:obj:`str`): Model size variant. Options vary by part:

            - pose: ``'lite'``, ``'full'``, ``'heavy'``
            - hand: ``'full'``
            - face: ``'full'``

    Returns:
        :obj:`str`: Path to the downloaded/cached model file.

    Raises:
        :exc:`ValidationError`: If invalid part or size is specified.
        :exc:`ModelDownloadError`: If model download fails.

    Example:
        Download pose landmarker model::

            >>> model_path = get_mediapipe_model('pose', 'lite')
            >>> print(model_path)
            '.../cslrtools2/assets/pose/lite.task'
    """

    part_map = MODELS.get(part)
    if part_map is None:
        raise ValidationError(
            f"Invalid model part: {part}. Available parts: {list(MODELS.keys())}"
        )

    model_url = part_map.get(size)
    if model_url is None:
        raise ValidationError(
            f"Invalid model size: {size} for part {part}. "
            f"Available sizes: {list(part_map.keys())}"
        )

    model_dir = ASSETS_PATH / part
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = (model_dir / size).with_suffix(".task")

    if model_file.exists():
        return str(model_file)

    response = requests.get(model_url)

    if response.status_code != 200:
        raise ModelDownloadError(
            f"Failed to download model from {model_url}. "
            f"Status code: {response.status_code}. "
            f"Reason: {response.reason}. "
            f"Ensure you have internet connectivity."
        )

    with model_file.open("wb") as f:
        f.write(response.content)

    return str(model_file)


class MediaPipeEstimator[K: str](Estimator[K]):
    """Base class for MediaPipe-based landmark estimators.

    Provides common functionality for MediaPipe vision tasks including
    model loading, landmark extraction, and coordinate normalization.

    Type Parameters:
        K: String type for landmark keys identifying different body parts.

    Attributes:
        channels (:obj:`int`): Number of channels per landmark. Defaults to ``4``
            (x, y, z, confidence).
        axis_names (:obj:`list`\\[:obj:`str`\\]): Names of the coordinate axes.
            Defaults to ``['x', 'y', 'z', 'c']``.
    """

    channels: int = 4

    axis_names: list[str] = ["x", "y", "z", "c"]

    _is_now_suppressing_stderr: bool = False

    def _get_array_from_landmarks(self, lm: NormalizedLandmark) -> list[float]:
        """Convert MediaPipe landmark to coordinate array.

        Args:
            lm (:class:`mediapipe.tasks.python.components.containers.landmark.
                NormalizedLandmark`):
                MediaPipe normalized landmark.

        Returns:
            :obj:`list`\\[:obj:`float`\\]: Coordinate array
            ``[x, y, z, confidence]``.
        """
        return [
            lm.x or self.missing_value,
            lm.y or self.missing_value,
            lm.z or self.missing_value,
            max(0.0, (lm.visibility or 0.0) * (lm.presence or 0.0)),
        ]

    def _enable_suppress_stderr(self):
        if self._is_now_suppressing_stderr:
            return

        self._is_now_suppressing_stderr = True

        self._original_stderr_fd = os.dup(2)
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 2)

    def _disable_suppress_stderr(self):
        if not self._is_now_suppressing_stderr:
            return

        os.dup2(self._original_stderr_fd, 2)
        os.close(self._devnull_fd)
        os.close(self._original_stderr_fd)

        self._is_now_suppressing_stderr = False

    def __init__(self, base_args: MediaPipeBaseArgs):
        # Suppress protobuf deprecation warning from mediapipe
        warnings.filterwarnings(
            "ignore",
            message="SymbolDatabase.GetPrototype\\(\\) is deprecated",
            category=UserWarning,
            module="google.protobuf.symbol_database",
        )

        self.base_args = base_args

        self.delegate = (
            BaseOptions.Delegate.GPU
            if base_args.delegate == "gpu"
            else BaseOptions.Delegate.CPU
        )

        self.running_mode = (
            VisionTaskRunningMode.VIDEO
            if base_args.running_mode == "video"
            else VisionTaskRunningMode.LIVE_STREAM
            if base_args.running_mode == "live_stream"
            else VisionTaskRunningMode.IMAGE
        )
