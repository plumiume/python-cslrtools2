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

from typing import cast, Any, Mapping, NamedTuple
from functools import cache
from itertools import product

import numpy as np
import cv2

from mediapipe.python.solutions.holistic import ( # pyright: ignore[reportMissingTypeStubs]
    Holistic, PoseLandmark, HandLandmark
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import ( # pyright: ignore[reportMissingTypeStubs]
    Landmark, NormalizedLandmark
)

from ....typings import MatLike, NDArrayFloat, NDArrayStr
from ....exceptions import ValidationError
from ....lmpipe.estimator import shape, headers, estimate, annotate
from .base import MediaPipeEstimator
from .face import FACE_LANDMARKS_NUM
from .holistic_args import MediaPipeHolisticKey, MediaPipeHolisticArgs


class Landmarks[L: Landmark | NormalizedLandmark]:
    landmarks: list[L]


class DetectionResults(NamedTuple):
    pose_landmarks: Landmarks[NormalizedLandmark] | None
    pose_world_landmarks: Landmarks[Landmark] | None
    left_hand_landmarks: Landmarks[NormalizedLandmark] | None
    right_hand_landmarks: Landmarks[NormalizedLandmark] | None
    face_landmarks: Landmarks[NormalizedLandmark] | None
    segmentation_mask: Any

class MediaPipeHolisticEstimator(
    MediaPipeEstimator[MediaPipeHolisticKey]
    ):

    _setuped: bool = False

    def __init__(self, holistic_args: MediaPipeHolisticArgs.T = MediaPipeHolisticArgs.T()):

        super().__init__(base_args=holistic_args)
        self.holistic_args = holistic_args

    def setup(self):

        if self._setuped:
            return
        self._setuped = True

        self._enable_suppress_stderr()

        static_image_mode = self.running_mode == VisionTaskRunningMode.IMAGE

        self.landmarker = Holistic(
            static_image_mode=static_image_mode,
            model_complexity=self.holistic_args.model_complexity,
            smooth_landmarks=self.holistic_args.smooth_landmarks,
            enable_segmentation=self.holistic_args.enable_segmentation,
            smooth_segmentation=self.holistic_args.smooth_segmentation,
            refine_face_landmarks=self.holistic_args.refine_face_landmarks
        )

    def configure_estimator_name(self) -> MediaPipeHolisticKey:
        raise ValidationError(
            "Holistic estimator does not have a single key. "
            "Use estimator.pose_key, estimator.left_hand_key, etc. instead."
        )

    @property
    @shape
    @cache
    def shape(self) -> tuple[int, int]:
        num_landmarks = (
            len(PoseLandmark) # pose landmarks
            + 2 * len(HandLandmark) # left and right hand landmarks
            + FACE_LANDMARKS_NUM # face landmarks
        )
        return (num_landmarks, self.channels)


    @property
    @headers
    @cache
    def headers(self) -> Mapping[MediaPipeHolisticKey, NDArrayStr]:

        return {
            "mediapipe.pose": np.array([
                f"{lm_name.name.lower()}__{axis}"
                for lm_name, axis in product(
                    PoseLandmark, self.axis_names
                )
            ]),
            "mediapipe.left_hand": np.array([
                f"{lm_name.name.lower()}__{axis}"
                for lm_name, axis in product(
                    HandLandmark, self.axis_names
                )
            ]),
            "mediapipe.right_hand": np.array([
                f"{lm_name.name.lower()}__{axis}"
                for lm_name, axis in product(
                    HandLandmark, self.axis_names
                )
            ]),
            "mediapipe.face": np.array([
                f"{lm_idx:03d}__{axis}"
                for lm_idx, axis in product(
                    range(FACE_LANDMARKS_NUM),
                    self.axis_names
                )
            ])
        }


    @estimate
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> Mapping[MediaPipeHolisticKey, NDArrayFloat | None]:

        cv2_image_rgb = cv2.cvtColor(frame_src, cv2.COLOR_BGR2RGB)

        detection_results = cast(
            DetectionResults,
            self.landmarker.process(cv2_image_rgb)
        )

        self._disable_suppress_stderr()

        return {
            "mediapipe.pose": (
                None if detection_results.pose_landmarks is None else
                np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in detection_results.pose_landmarks.landmarks
                ])
            ),
            "mediapipe.left_hand": (
                None if detection_results.left_hand_landmarks is None else
                np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in detection_results.left_hand_landmarks.landmarks
                ])
            ),
            "mediapipe.right_hand": (
                None if detection_results.right_hand_landmarks is None else
                np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in detection_results.right_hand_landmarks.landmarks
                ])
            ),
            "mediapipe.face": (
                None if detection_results.face_landmarks is None else
                np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in detection_results.face_landmarks.landmarks
                ])
            )
        }

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[MediaPipeHolisticKey, NDArrayFloat]
        ) -> MatLike:

        # TODO: Implement holistic annotation
        return frame_src
