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

from abc import abstractmethod
from functools import cache
from typing import Any, Mapping

# import cv2

from ....typings import MatLike, NDArrayFloat, NDArrayStr
from ...estimator import (
    Estimator, shape, headers, estimate, annotate
)
from .roi import BaseROI


class HolisticPartEstimator[K: str](Estimator[K]):
    ...


class HolisticPoseEstimator[K: str](Estimator[K]):

    @abstractmethod
    def configure_left_hand_roi(
        self,
        landmarks: Mapping[K, NDArrayFloat],
        height: int, width: int
        ) -> BaseROI: ...

    @abstractmethod
    def configure_right_hand_roi(
        self,
        landmarks: Mapping[K, NDArrayFloat],
        height: int, width: int
        ) -> BaseROI: ...

    @abstractmethod
    def configure_both_hands_roi(
        self,
        landmarks: Mapping[K, NDArrayFloat],
        height: int, width: int
        ) -> BaseROI: ...

    @abstractmethod
    def configure_face_roi(
        self,
        landmarks: Mapping[K, NDArrayFloat],
        height: int, width: int
        ) -> BaseROI: ...


class HolisticEstimator(Estimator[str]):

    MIN_ROI_IMAGE_SHAPE = (32, 32)

    def configure_estimator_name(self) -> str:
        return "default_composite_holistic_estimator"

    def __init__(
        self,
        pose_estimator: HolisticPoseEstimator[Any],
        both_hands_estimator: HolisticPartEstimator[Any] | None = None,
        right_hand_estimator: HolisticPartEstimator[Any] | None = None,
        left_hand_estimator: HolisticPartEstimator[Any] | None = None,
        face_estimator: HolisticPartEstimator[Any] | None = None,
        ) -> None:
        self.pose_estimator = pose_estimator
        self.both_hands_estimator = both_hands_estimator
        self.right_hand_estimator = right_hand_estimator
        self.left_hand_estimator = left_hand_estimator
        self.face_estimator = face_estimator
        ...

    @property
    @shape
    def shape(self) -> Mapping[str, tuple[int, int]]:

        result: dict[str, tuple[int, int]] = {}

        result.update(self.pose_estimator.shape)

        if self.both_hands_estimator:
            result.update(self.both_hands_estimator.shape)

        if self.right_hand_estimator:
            result.update(self.right_hand_estimator.shape)

        if self.left_hand_estimator:
            result.update(self.left_hand_estimator.shape)

        if self.face_estimator:
            result.update(self.face_estimator.shape)

        return result

    @estimate
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> Mapping[str, NDArrayFloat | None]:

        result: dict[str, NDArrayFloat | None] = {}

        pose_landmarks = self.pose_estimator.estimate(frame_src, frame_idx)
        result.update({
            f'{klm}': vlm
            for klm, vlm in pose_landmarks.items()
        })

        if self.both_hands_estimator:
            both_hands_roi = self.pose_estimator.configure_both_hands_roi(
                pose_landmarks, frame_src.shape[0], frame_src.shape[1]
            )
            both_hands_frame_src = both_hands_roi.apply_roi(frame_src)
            if (
                both_hands_frame_src is None
                or both_hands_frame_src.shape[0] < self.MIN_ROI_IMAGE_SHAPE[0]
                or both_hands_frame_src.shape[1] < self.MIN_ROI_IMAGE_SHAPE[1]
                ):
                # ROIが計算できない、ROIが小さすぎる場合はスキップ
                result.update({
                    klm: None
                    for klm in self.both_hands_estimator.headers.keys()
                })
            else:
                # cv2.imshow("both_hands_frame_src", both_hands_frame_src)
                # cv2.imwrite(f"both_hands_frame/{frame_idx:04d}.png", both_hands_frame_src)
                result.update(
                    both_hands_roi.apply_world_coords(
                        self.both_hands_estimator.estimate(
                            both_hands_frame_src, frame_idx
                        )
                    )
                )

        if self.left_hand_estimator:
            left_hand_roi = self.pose_estimator.configure_left_hand_roi(
                pose_landmarks, frame_src.shape[0], frame_src.shape[1]
            )
            left_hand_frame_src = left_hand_roi.apply_roi(frame_src)
            if (
                left_hand_frame_src is None
                or left_hand_frame_src.shape[0] < self.MIN_ROI_IMAGE_SHAPE[0]
                or left_hand_frame_src.shape[1] < self.MIN_ROI_IMAGE_SHAPE[1]
                ):
                # ROIが計算できない、ROIが小さすぎる場合はスキップ
                result.update({
                    klm: None
                    for klm in self.left_hand_estimator.headers.keys()
                })
            else:
                # cv2.imshow("left_hand_frame_src", left_hand_frame_src)
                # cv2.imwrite(f"left_hand_frame/{frame_idx:04d}.png", left_hand_frame_src)
                result.update(
                    left_hand_roi.apply_world_coords(
                        self.left_hand_estimator.estimate(
                            left_hand_frame_src, frame_idx
                        )
                    )
                )

        if self.right_hand_estimator:
            right_hand_roi = self.pose_estimator.configure_right_hand_roi(
                pose_landmarks, frame_src.shape[0], frame_src.shape[1]
            )
            right_hand_frame_src = right_hand_roi.apply_roi(frame_src)
            if (
                right_hand_frame_src is None
                or right_hand_frame_src.shape[0] < self.MIN_ROI_IMAGE_SHAPE[0]
                or right_hand_frame_src.shape[1] < self.MIN_ROI_IMAGE_SHAPE[1]
                ):
                # ROIが計算できない、ROIが小さすぎる場合はスキップ
                result.update({
                    klm: None
                    for klm in self.right_hand_estimator.headers.keys()
                })
            else:
                # cv2.imshow("right_hand_frame_src", right_hand_frame_src)
                # cv2.imwrite(f"right_hand_frame/{frame_idx:04d}.png", right_hand_frame_src)
                result.update(
                    right_hand_roi.apply_world_coords(
                        self.right_hand_estimator.estimate(
                            right_hand_frame_src, frame_idx
                        )
                    )
                )

        if self.face_estimator:
            face_roi = self.pose_estimator.configure_face_roi(
                pose_landmarks, frame_src.shape[0], frame_src.shape[1]
            )
            face_frame_src = face_roi.apply_roi(frame_src)
            if (
                face_frame_src is None
                or face_frame_src.shape[0] < self.MIN_ROI_IMAGE_SHAPE[0]
                or face_frame_src.shape[1] < self.MIN_ROI_IMAGE_SHAPE[1]
                ):
                # ROIが計算できない、ROIが小さすぎる場合はスキップ
                result.update({
                    klm: None
                    for klm in self.face_estimator.headers.keys()
                })
            else:
                result.update(
                    face_roi.apply_world_coords(
                        self.face_estimator.estimate(
                            face_frame_src, frame_idx
                        )
                    )
                )

        # cv2.waitKey(1)

        return result

    @property
    @headers
    @cache
    def headers(self) -> Mapping[str, NDArrayStr]:

        result: dict[str, NDArrayStr] = {}

        result.update(self.pose_estimator.headers)

        if self.both_hands_estimator:
            result.update(self.both_hands_estimator.headers)

        if self.right_hand_estimator:
            result.update(self.right_hand_estimator.headers)

        if self.left_hand_estimator:
            result.update(self.left_hand_estimator.headers)

        if self.face_estimator:
            result.update(self.face_estimator.headers)

        return result

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[str, NDArrayFloat]
        ) -> MatLike:

        frame_src = self.pose_estimator.annotate(
            frame_src, frame_idx, landmarks
        )

        if self.both_hands_estimator:
            frame_src = self.both_hands_estimator.annotate(
                frame_src, frame_idx, landmarks
            )

        if self.right_hand_estimator:
            frame_src = self.right_hand_estimator.annotate(
                frame_src, frame_idx, landmarks
            )

        if self.left_hand_estimator:
            frame_src = self.left_hand_estimator.annotate(
                frame_src, frame_idx, landmarks
            )

        if self.face_estimator:
            frame_src = self.face_estimator.annotate(
                frame_src, frame_idx, landmarks
            )

        return frame_src

    def setup(self):
        self.pose_estimator.setup()
        if self.both_hands_estimator:
            self.both_hands_estimator.setup()
        if self.right_hand_estimator:
            self.right_hand_estimator.setup()
        if self.left_hand_estimator:
            self.left_hand_estimator.setup()
        if self.face_estimator:
            self.face_estimator.setup()
