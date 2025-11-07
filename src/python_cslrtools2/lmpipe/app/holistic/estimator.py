from abc import abstractmethod
from functools import cache
from typing import Any, Mapping
from ...typings import MatLike, NDArrayFloat, NDArrayStr
from ...estimator import (
    Estimator, shape, headers, estimate, annotate
)

class HolisticPartEstimator[K: str](Estimator[K]): ...

class HolisticPoseEstimator[K: str](Estimator[K]):

    @abstractmethod
    def clip_left_hand_frame(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]
        ) -> MatLike: ...

    @abstractmethod
    def clip_right_hand_frame(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]
        ) -> MatLike: ...

    @abstractmethod
    def clip_face_frame(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[K, NDArrayFloat]
        ) -> MatLike: ...

class HolisticEstimator(Estimator[str]):

    def configure_estimator_name(self) -> str:
        return "default_composite_holistic_estimator"

    def __init__(
        self,
        pose_estimator: HolisticPoseEstimator[Any],
        right_hand_estimator: HolisticPartEstimator[Any] | None = None,
        left_hand_estimator: HolisticPartEstimator[Any] | None = None,
        face_estimator: HolisticPartEstimator[Any] | None = None,
        ) -> None:
        self.pose_estimator = pose_estimator
        self.right_hand_estimator = right_hand_estimator
        self.left_hand_estimator = left_hand_estimator
        self.face_estimator = face_estimator
        ...

    @property
    @shape
    def shape(self) -> Mapping[str, tuple[int, int]]:

        result: dict[str, tuple[int, int]] = {}

        result.update(self.pose_estimator.shape)

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
        ) -> Mapping[str, NDArrayFloat]:

        result: dict[str, NDArrayFloat] = {}

        pose_landmarks = self.pose_estimator.estimate(frame_src, frame_idx)
        result.update({
            f'{klm}': vlm
            for klm, vlm in pose_landmarks.items()
        })

        if self.right_hand_estimator:
            right_hand_frame = self.pose_estimator.clip_right_hand_frame(
                frame_src, frame_idx, pose_landmarks
            )
            result.update(self.right_hand_estimator.estimate(
                right_hand_frame, frame_idx
            ))

        if self.left_hand_estimator:
            left_hand_frame = self.pose_estimator.clip_left_hand_frame(
                frame_src, frame_idx, pose_landmarks
            )
            result.update(self.left_hand_estimator.estimate(
                left_hand_frame, frame_idx
            ))

        if self.face_estimator:
            face_frame = self.pose_estimator.clip_face_frame(
                frame_src, frame_idx, pose_landmarks
            )
            result.update(self.face_estimator.estimate(
                face_frame, frame_idx
            ))

        return result

    @property
    @headers
    @cache
    def headers(self) -> Mapping[str, NDArrayStr]:

        result: dict[str, NDArrayStr] = {}

        result.update(self.pose_estimator.headers)

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
        if self.right_hand_estimator:
            self.right_hand_estimator.setup()
        if self.left_hand_estimator:
            self.left_hand_estimator.setup()
        if self.face_estimator:
            self.face_estimator.setup()

