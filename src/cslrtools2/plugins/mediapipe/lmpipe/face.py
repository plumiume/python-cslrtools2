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
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)
from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
)
from mediapipe import Image, ImageFormat

from ....typings import MatLike, NDArrayFloat, NDArrayStr
from ....lmpipe.estimator import shape, headers, estimate, annotate
from ....lmpipe.app.holistic.estimator import HolisticPartEstimator
from .base import MediaPipeEstimator, get_mediapipe_model
from .face_args import MediaPipeFaceKey, MediaPipeFaceArgs


MEDIA_PIPE_FACE_KEY = "mediapipe.face"

FACE_LANDMARKS_NUM = 468


class MediaPipeFaceEstimator(
    MediaPipeEstimator[MediaPipeFaceKey], HolisticPartEstimator[MediaPipeFaceKey]
):
    _setuped: bool = False

    def __init__(self, face_args: MediaPipeFaceArgs.T = MediaPipeFaceArgs.T()):
        super().__init__(face_args)
        self.face_args = face_args

        self.model_asset_path = get_mediapipe_model("face", face_args.face_model)

        self.landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path, delegate=self.delegate
            ),
            num_faces=1,
            running_mode=self.running_mode,
            min_face_detection_confidence=self.face_args.min_face_detection_confidence,
            min_tracking_confidence=self.face_args.min_face_tracking_confidence,
            min_face_presence_confidence=self.face_args.min_face_presence_confidence,
        )

    def setup(self):
        if self._setuped:
            return
        self._setuped = True

        self._enable_suppress_stderr()
        self.landmarker = FaceLandmarker.create_from_options(self.landmarker_options)

    def configure_estimator_name(self) -> MediaPipeFaceKey:
        return MEDIA_PIPE_FACE_KEY

    @property
    @shape
    @cache
    def shape(self) -> tuple[int, int]:
        return (FACE_LANDMARKS_NUM, self.channels)

    @property
    @headers
    @cache
    def headers(self) -> NDArrayStr:
        # item: "{landmark_name}__{axis}"
        return np.array(
            [
                f"{lm_idx:03d}__{axis}"
                for lm_idx, axis in product(range(FACE_LANDMARKS_NUM), self.axis_names)
            ]
        )

    @estimate
    def estimate(self, frame_src: MatLike, frame_idx: int) -> NDArrayFloat | None:

        mp_image = Image(
            image_format=ImageFormat.SRGB, data=np.ascontiguousarray(frame_src)
        )

        detection_result = self.landmarker.detect(  # pyright: ignore[reportUnknownMemberType] # noqa: E501
            mp_image
        )
        landmarks: list[list[NormalizedLandmark]] = (  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] # noqa: E501
            detection_result.face_landmarks
        )

        self._disable_suppress_stderr()

        if not landmarks:
            return None

        lm_list = landmarks[0]

        result = np.array([self._get_array_from_landmarks(lm) for lm in lm_list])

        return result

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[MediaPipeFaceKey, NDArrayFloat],
    ) -> MatLike:
        # TODO: Implement face annotation
        return frame_src
