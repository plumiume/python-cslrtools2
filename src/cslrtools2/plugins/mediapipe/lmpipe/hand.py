from typing import Mapping
from enum import IntEnum
from functools import cache
from itertools import product

import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.category import Category # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat

from ....lmpipe.typings import MatLike, NDArrayFloat, NDArrayStr
from ....lmpipe.estimator import shape, headers, estimate, annotate
from ....lmpipe.app.holistic.estimator import HolisticPartEstimator
from .base import MediaPipeEstimator, get_mediapipe_model
from .hand_args import MediaPipeHandKey, MediaPipeHandCategory, MediaPipeHandArgs


class MediaPipeHandNames(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class MediaPipeHandEstimator(
    MediaPipeEstimator[MediaPipeHandKey],
    HolisticPartEstimator[MediaPipeHandKey]
    ):

    _setuped: bool = False

    def __init__(
            self,
            hand_args: MediaPipeHandArgs.T = MediaPipeHandArgs.T(),
            category: MediaPipeHandCategory = "both"
            ):

        super().__init__(hand_args)
        self.hand_args = hand_args
        self.category = category

        self.model_asset_path = get_mediapipe_model("hand", hand_args.hand_model)

        self.landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=self.delegate
            ),
            num_hands=2,
            running_mode=self.running_mode,
            min_hand_detection_confidence=hand_args.min_hand_detection_confidence,
            min_tracking_confidence=hand_args.min_hand_tracking_confidence,
            min_hand_presence_confidence=hand_args.min_hand_presence_confidence
        )

    def setup(self):

        if self._setuped:
            return
        self._setuped = True

        self._enable_suppress_stderr()
        self.landmarker = HandLandmarker.create_from_options(
            self.landmarker_options
        )

    def configure_estimator_name(self) -> MediaPipeHandKey:
        if self.category == "left":
            return "mediapipe.left_hand"
        elif self.category == "right":
            return "mediapipe.right_hand"
        # bothの時、各メソッドはMapping[MediaPipeHandKey, NDArrayFloat | None]を返すようにする
        raise ValueError(
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
                "mediapipe.right_hand": (len(MediaPipeHandNames), self.channels)
            }
        return (len(MediaPipeHandNames), self.channels)

    @property
    @headers
    @cache
    def headers(self) -> NDArrayStr | Mapping[MediaPipeHandKey, NDArrayStr]:
        # item: "{landmark_name}__{axis}"

        header_array = np.array([
            f"{lm_name.name.lower()}__{axis}"
            for lm_name, axis in product(
                MediaPipeHandNames,
                self.axis_names
            )
        ])

        if self.category == "both":

            return {
                "mediapipe.left_hand": header_array,
                "mediapipe.right_hand": header_array
            }

        return header_array

    @estimate
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> NDArrayFloat | None | Mapping[MediaPipeHandKey, NDArrayFloat | None]:

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=frame_src
        )

        detection_result = self.landmarker.detect(mp_image) # pyright: ignore[reportUnknownMemberType]
        handedness: list[list[Category]] = detection_result.handedness # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        landmarks: list[list[NormalizedLandmark]] = detection_result.hand_landmarks # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

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
                left_hand_landmarks = np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in lms
                ])

            if ctgr_name and ctgr_name.lower() == "right":
                right_hand_landmarks = np.array([
                    self._get_array_from_landmarks(lm)
                    for lm in lms
                ])

        if self.category == "both":
            
            return {
                "mediapipe.left_hand": left_hand_landmarks,
                "mediapipe.right_hand": right_hand_landmarks
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
        landmarks: Mapping[MediaPipeHandKey, NDArrayFloat]
        ) -> MatLike:

        # TODO: Implement hand annotation
        return frame_src
