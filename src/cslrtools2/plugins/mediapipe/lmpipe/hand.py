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

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from typing import Mapping
from functools import cache
from itertools import product

import numpy as np

from mediapipe.tasks.python.core.base_options import (
    BaseOptions,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.components.containers.category import (
    Category,
)
from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
)
from mediapipe import Image, ImageFormat

from ....typings import MatLike, NDArrayFloat, NDArrayStr
from ....exceptions import ValidationError
from ....lmpipe.estimator import shape, headers, estimate, annotate
from ....lmpipe.app.holistic.estimator import HolisticPartEstimator
from .base import MediaPipeEstimator, get_mediapipe_model
from .hand_args import MediaPipeHandKey, MediaPipeHandCategory, MediaPipeHandArgs
from .mp_constants import HandLandmark


# Deprecated: Use HandLandmark from constants module instead
# This alias is kept for backward compatibility
MediaPipeHandNames = HandLandmark
"""Deprecated: Use ``HandLandmark`` from ``constants`` module instead.

This is an alias to MediaPipe's official ``HandLandmark`` enum for backward
compatibility. New code should import directly from the constants module::

    from cslrtools2.plugins.mediapipe.lmpipe.constants import HandLandmark

.. deprecated:: 0.1.0
    Use :class:`~cslrtools2.plugins.mediapipe.lmpipe.constants.HandLandmark` instead.
"""


class MediaPipeHandEstimator(
    MediaPipeEstimator[MediaPipeHandKey], HolisticPartEstimator[MediaPipeHandKey]
):
    _setuped: bool = False

    def __init__(
        self,
        hand_args: MediaPipeHandArgs.T = MediaPipeHandArgs.T(),
        category: MediaPipeHandCategory = "both",
    ):
        super().__init__(hand_args)
        self.hand_args = hand_args
        self.category = category

        self.model_asset_path = get_mediapipe_model("hand", hand_args.hand_model)

        self.landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path, delegate=self.delegate
            ),
            num_hands=2,
            running_mode=self.running_mode,
            min_hand_detection_confidence=hand_args.min_hand_detection_confidence,
            min_tracking_confidence=hand_args.min_hand_tracking_confidence,
            min_hand_presence_confidence=hand_args.min_hand_presence_confidence,
        )

    def setup(self):
        if self._setuped:
            return
        self._setuped = True

        self._enable_suppress_stderr()
        self.landmarker = HandLandmarker.create_from_options(self.landmarker_options)

    def configure_estimator_name(self) -> MediaPipeHandKey:
        if self.category == "left":
            return "mediapipe.left_hand"
        elif self.category == "right":
            return "mediapipe.right_hand"
        # bothの時、各メソッドはMapping[MediaPipeHandKey, NDArrayFloat | None]を返すようにする
        raise ValidationError(
            f"Invalid category for MediaPipeHandEstimator: {self.category}. "
            "Use 'left' or 'right'."
        )

    @property
    @shape
    @cache
    def shape(self) -> tuple[int, int] | Mapping[MediaPipeHandKey, tuple[int, int]]:
        if self.category == "both":
            return {
                "mediapipe.left_hand": (len(MediaPipeHandNames), self.channels),
                "mediapipe.right_hand": (len(MediaPipeHandNames), self.channels),
            }
        return (len(MediaPipeHandNames), self.channels)

    @property
    @headers
    @cache
    def headers(self) -> NDArrayStr | Mapping[MediaPipeHandKey, NDArrayStr]:
        # item: "{landmark_name}__{axis}"

        header_array = np.array(
            [
                f"{lm_name.name.lower()}__{axis}"
                for lm_name, axis in product(MediaPipeHandNames, self.axis_names)
            ]
        )

        if self.category == "both":
            return {
                "mediapipe.left_hand": header_array,
                "mediapipe.right_hand": header_array,
            }

        return header_array

    @estimate
    def estimate(
        self, frame_src: MatLike, frame_idx: int
    ) -> NDArrayFloat | None | Mapping[MediaPipeHandKey, NDArrayFloat | None]:
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_src)

        detection_result = self.landmarker.detect(  # pyright: ignore[reportUnknownMemberType] # noqa: E501
            mp_image
        )
        handedness: list[list[Category]] = (  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] # noqa: E501
            detection_result.handedness
        )
        landmarks: list[list[NormalizedLandmark]] = (  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] # noqa: E501
            detection_result.hand_landmarks
        )

        self._disable_suppress_stderr()

        if not landmarks:
            return None

        left_hand_landmarks: NDArrayFloat | None = None
        right_hand_landmarks: NDArrayFloat | None = None

        for ctgrs, lms in zip(handedness, landmarks):
            if not ctgrs:
                continue

            ctgr_name = ctgrs[0].category_name
            if ctgr_name and ctgr_name.lower() == "left":
                left_hand_landmarks = np.array(
                    [self._get_array_from_landmarks(lm) for lm in lms]
                )

            if ctgr_name and ctgr_name.lower() == "right":
                right_hand_landmarks = np.array(
                    [self._get_array_from_landmarks(lm) for lm in lms]
                )

        if self.category == "both":
            return {
                "mediapipe.left_hand": left_hand_landmarks,
                "mediapipe.right_hand": right_hand_landmarks,
            }

        if self.category == "left":
            return left_hand_landmarks

        if self.category == "right":
            return right_hand_landmarks

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[MediaPipeHandKey, NDArrayFloat],
    ) -> MatLike:
        # TODO: Implement hand annotation
        return frame_src
