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

from typing import Literal, Mapping, cast
from abc import abstractmethod
from functools import cache
from itertools import product

import numpy as np
import cv2

from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat


from ....typings import MatLike, NDArrayFloat, NDArrayStr
from ....lmpipe.estimator import shape, headers, estimate, annotate
from ....lmpipe.app.holistic.estimator import HolisticPoseEstimator
from ....lmpipe.app.holistic.roi import BaseROI
from .base import MediaPipeEstimator, get_mediapipe_model
from .pose_args import MediaPipePoseKey, MediaPipePoseArgs
from .mp_constants import PoseLandmark


MEDIA_PIPE_POSE_KEY: MediaPipePoseKey = "mediapipe.pose"


# Deprecated: Use PoseLandmark from constants module instead
# This alias is kept for backward compatibility
MediaPipePoseNames = PoseLandmark
"""Deprecated: Use ``PoseLandmark`` from ``constants`` module instead.

This is an alias to MediaPipe's official ``PoseLandmark`` enum for backward
compatibility. New code should import directly from the constants module::

    from cslrtools2.plugins.mediapipe.lmpipe.constants import PoseLandmark

.. deprecated:: 0.1.0
    Use :class:`~cslrtools2.plugins.mediapipe.lmpipe.constants.PoseLandmark` instead.
"""


class MediaPipePosePartROI(BaseROI):
    """Base class for MediaPipe Pose-based ROI extraction.
    
    Provides common affine transformation matrix operations for extracting
    and normalizing ROI regions from pose landmarks.
    """

    return_none: bool = False
    center_weight: float  # Must be defined in subclass
    size_weight: float    # Must be defined in subclass

    def _safe_init(self, landmarks: list[NDArrayFloat], height: int, width: int) -> bool:
        """Safe initialization with NaN checking and exception handling.
        
        Args:
            landmarks: List of landmark arrays to check for NaN
            height: Frame height in pixels
            width: Frame width in pixels
            
        Returns:
            True if initialization succeeded, False otherwise
        """

        # 1. Check for NaN in landmarks
        for lm in landmarks:
            if np.isnan(lm).any():
                self.return_none = True
                return False

        # 3. Calculate ROI parameters
        try:
            roi_center_px, roi_angle, roi_size_px = self._calculate_roi_parameters(height, width)
        except (ValueError, ZeroDivisionError):
            self.return_none = True
            return False
        
        self.roi_dsize = roi_size_px
        self.inv_dsize = (width, height)

        # 4. Calculate affine matrix
        self.affine_transform = cv2.getRotationMatrix2D(
            center=roi_center_px,
            angle=np.degrees(roi_angle),
            scale=1
        ) + cast(
            NDArrayFloat,
            np.array([
                [0.0, 0.0, roi_size_px[0] / 2 - roi_center_px[0]],
                [0.0, 0.0, roi_size_px[1] / 2 - roi_center_px[1]]
            ], dtype=np.float32)
        )

        self.inv_affine_transform = cv2.invertAffineTransform(
            self.affine_transform
        )

        return True


    @abstractmethod
    def _calculate_roi_parameters(
        self, height: int, width: int
    ) -> tuple[tuple[float, float], float, tuple[int, int]]:
        """Calculate ROI center, size, and angle from landmarks.
        
        Args:
            height: Frame height in pixels
            width: Frame width in pixels
        
        Returns:
            Tuple of (center_px, angle_rad, size_px)
            - center_px: ROI center in pixel coordinates [x, y]
            - angle_rad: ROI rotation angle in radians
            - size_px: ROI size as (width, height) pair in pixels
        """
        ...

    def apply_roi(self, frame_src: MatLike) -> MatLike | None:
        
        if self.return_none:
            return None

        roi_frame = cv2.warpAffine(
            frame_src,
            self.affine_transform,
            dsize=self.roi_dsize,
        )

        return roi_frame

    def apply_world_coords[K: str](
        self, local_coords: Mapping[K, NDArrayFloat] # key: Array(N, >=2)
        ) -> Mapping[K, NDArrayFloat | None]:
        
        if self.return_none:
            return {
                klm: None
                for klm in local_coords.keys()
            }

        result: dict[K, NDArrayFloat] = {}

        norm2pix = np.array(self.roi_dsize)
        pix2norm = 1 / np.array(self.inv_dsize)

        weight = self.inv_affine_transform[:, :2]
        bias = self.inv_affine_transform[:, 2]

        for klm, lm in local_coords.items():

            xy = lm[:, :2]
            ex = lm[:, 2:]

            roi_px = norm2pix * xy
            world_px = roi_px @ weight.T + bias
            world_norm = pix2norm * world_px

            result[klm] = np.concatenate([world_norm, ex], axis=1)

        return result
    

class MediaPipePoseBothHandsROI(MediaPipePosePartROI):
    """Both hands ROI using left and right wrist landmarks from pose estimation.
    
    Generates normalized affine transformation matrix for both hands region extraction.
    ROI vector: right wrist -> left wrist (reversed)
    ROI size: max of left hand and right hand ROI sizes
    ROI center: adjusted to contain both hands within ROI bounds
    """

    center_weight: float = 0.25
    size_weight: float = 2.2
    pad_weight: float = 0.1

    def __init__(
        self,
        left_wrist: NDArrayFloat,
        left_elbow: NDArrayFloat,
        right_wrist: NDArrayFloat,
        right_elbow: NDArrayFloat,
        height: int,
        width: int
        ):
        """Initialize Both Hands ROI with pose landmarks.
        
        Args:
            left_wrist: Left wrist landmark coordinates [x, y, z, c]
            left_elbow: Left elbow landmark coordinates [x, y, z, c]
            right_wrist: Right wrist landmark coordinates [x, y, z, c]
            right_elbow: Right elbow landmark coordinates [x, y, z, c]
            height: Frame height in pixels
            width: Frame width in pixels
        """
        self.left_wrist = left_wrist
        self.left_elbow = left_elbow
        self.right_wrist = right_wrist
        self.right_elbow = right_elbow
        
        # Use common initialization framework
        self._safe_init([left_wrist, left_elbow, right_wrist, right_elbow], height, width)
    
    def _calculate_roi_parameters(
        self, height: int, width: int
    ) -> tuple[tuple[float, float], float, tuple[int, int]]:
        """Calculate ROI center, size, and angle from landmarks.
        
        ROI is oriented from right wrist to left wrist, with the center
        adjusted so that the line segment fits within size/2 from the center.
        Returns a rectangular ROI with width based on wrist distance and height
        based on hand size.
        """

        # Determine long side for padding calculation
        long_side = max(height, width)

        # Convert normalized coordinates to pixel coordinates
        left_wrist_px = np.array([self.left_wrist[0] * width, self.left_wrist[1] * height])
        left_elbow_px = np.array([self.left_elbow[0] * width, self.left_elbow[1] * height])
        right_wrist_px = np.array([self.right_wrist[0] * width, self.right_wrist[1] * height])
        right_elbow_px = np.array([self.right_elbow[0] * width, self.right_elbow[1] * height])
        
        # Calculate hand sizes (for height)
        left_distance = np.linalg.norm(left_wrist_px - left_elbow_px)
        right_distance = np.linalg.norm(right_wrist_px - right_elbow_px)
        max_distance = max(left_distance, right_distance)
        size_px = int(self.size_weight * max_distance + self.pad_weight * long_side)
        
        # Calculate wrist-to-wrist distance (for width)
        wrist_distance = int(np.linalg.norm(left_wrist_px - right_wrist_px))
        size_px_pair = (size_px + wrist_distance, size_px)
        
        # Calculate center (midpoint between left and right wrists)
        center_px = (left_wrist_px + right_wrist_px) / 2
        
        # Calculate rotation angle (right wrist -> left wrist, reversed)
        diff = left_wrist_px - right_wrist_px
        angle_rad = np.arctan2(diff[1], diff[0])

        return ((center_px[0], center_px[1]), angle_rad, size_px_pair)


class MediaPipePoseHandROI(MediaPipePosePartROI):
    """Hand ROI using wrist and elbow landmarks from pose estimation.
    
    Generates normalized affine transformation matrix for hand region extraction.
    """

    center_weight: float = 0.25
    size_weight: float = 2.2
    pad_weight: float = 0.1

    def __init__(
        self,
        wrist: NDArrayFloat,
        elbow: NDArrayFloat,
        height: int,
        width: int
        ):
        """Initialize Hand ROI with pose landmarks.
        
        Args:
            wrist: Wrist landmark coordinates [x, y, z, c]
            elbow: Elbow landmark coordinates [x, y, z, c]
            height: Frame height in pixels
            width: Frame width in pixels
        """
        self.wrist = wrist
        self.elbow = elbow
        
        # Use common initialization framework
        self._safe_init([wrist, elbow], height, width)
    
    def _calculate_roi_parameters(
        self, height: int, width: int
    ) -> tuple[tuple[float, float], float, tuple[int, int]]:
        """Calculate ROI center, size, and angle from landmarks."""

        # Determine long side for square ROI
        long_side = max(height, width)

        # Convert normalized coordinates to pixel coordinates
        wrist_px = np.array([self.wrist[0] * width, self.wrist[1] * height])
        elbow_px = np.array([self.elbow[0] * width, self.elbow[1] * height])
        
        # Calculate center (weighted towards wrist)
        center_px = wrist_px + self.center_weight * (wrist_px - elbow_px)

        # Calculate size based on wrist-elbow distance
        distance = np.linalg.norm(wrist_px - elbow_px)
        size_px = int(self.size_weight * distance + self.pad_weight * long_side)
        
        # Calculate rotation angle
        diff = wrist_px - elbow_px
        angle_rad = np.arctan2(diff[1], diff[0])

        return ((center_px[0], center_px[1]), angle_rad, (size_px, size_px))


class MediaPipePoseFaceROI(MediaPipePosePartROI):
    """Face ROI using ear and nose landmarks from pose estimation.
    
    Generates normalized affine transformation matrix for face region extraction.
    """

    center_weight: float = 0.1
    size_weight: float = 2.0
    pad_weight: float = 0.1

    def __init__(
        self,
        left_ear: NDArrayFloat,
        right_ear: NDArrayFloat,
        nose: NDArrayFloat,
        height: int,
        width: int
    ):
        """Initialize Face ROI with pose landmarks.
        
        Args:
            left_ear: Left ear landmark coordinates [x, y, z, c]
            right_ear: Right ear landmark coordinates [x, y, z, c]
            nose: Nose landmark coordinates [x, y, z, c]
            height: Frame height in pixels
            width: Frame width in pixels
        """
        self.left_ear = left_ear
        self.right_ear = right_ear
        self.nose = nose
        
        # Use common initialization framework
        self._safe_init([left_ear, right_ear, nose], height, width)
    
    def _calculate_roi_parameters(
        self, height: int, width: int
    ) -> tuple[tuple[float, float], float, tuple[int, int]]:
        """Calculate ROI center, size, and angle from landmarks."""

        # Determine long side for square ROI
        long_side = max(height, width)

        # Convert normalized coordinates to pixel coordinates
        left_ear_px = np.array([self.left_ear[0] * width, self.left_ear[1] * height])
        right_ear_px = np.array([self.right_ear[0] * width, self.right_ear[1] * height])
        nose_px = np.array([self.nose[0] * width, self.nose[1] * height])
        
        # Calculate ear center
        ear_center = (left_ear_px + right_ear_px) / 2
        
        # Calculate ROI center (weighted towards nose)
        center_px = ear_center + self.center_weight * (nose_px - ear_center)
        
        # Calculate size based on ear distance
        ear_distance = np.linalg.norm(right_ear_px - left_ear_px)
        size_px = int(self.size_weight * ear_distance + self.pad_weight * long_side)

        # Calculate rotation angle (based on ear alignment)
        diff = right_ear_px - left_ear_px
        angle_rad = np.arctan2(diff[1], diff[0])

        return ((center_px[0], center_px[1]), angle_rad, (size_px, size_px))

class MediaPipePoseEstimator(
    MediaPipeEstimator[MediaPipePoseKey],
    HolisticPoseEstimator[MediaPipePoseKey]
    ):

    _setuped: bool = False

    def __init__(self, pose_args: MediaPipePoseArgs.T = MediaPipePoseArgs.T()):

        super().__init__(base_args=pose_args)
        self.pose_args = pose_args

        self.model_asset_path = get_mediapipe_model("pose", pose_args.pose_model)

        self.landmarker_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=self.delegate
            ),
            num_poses=1,
            running_mode=self.running_mode,
            min_pose_detection_confidence=self.pose_args.min_pose_detection_confidence,
            min_tracking_confidence=self.pose_args.min_pose_tracking_confidence,
            min_pose_presence_confidence=self.pose_args.min_pose_presence_confidence
        )

    def setup(self):

        if self._setuped:
            return
        self._setuped = True

        self._enable_suppress_stderr()
        self.landmarker = PoseLandmarker.create_from_options(
            self.landmarker_options
        )

    def configure_estimator_name(self) -> Literal['mediapipe.pose']:
        return MEDIA_PIPE_POSE_KEY


    @property
    @shape
    @cache
    def shape(self) -> tuple[int, int]:
        return (len(MediaPipePoseNames), self.channels)
    
    @property
    @headers
    @cache
    def headers(self) -> NDArrayStr:
        # item: "{landmark_name}__{axis}"
        return np.array([
            f"{lm_name.name.lower()}__{axis}"
            for lm_name, axis in product(
                MediaPipePoseNames,
                self.axis_names
            )
        ])

    @estimate
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> NDArrayFloat | None:

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_src)
        )

        detection_result = self.landmarker.detect(mp_image) # pyright: ignore[reportUnknownMemberType]
        landmarks: list[list[NormalizedLandmark]] = detection_result.pose_landmarks # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        self._disable_suppress_stderr()

        if not landmarks:
            return None
        
        return np.array([
            self._get_array_from_landmarks(lm)
            for lm in landmarks[0]
        ])

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: Mapping[MediaPipePoseKey, NDArrayFloat]
        ) -> MatLike:

        # TODO: implement annotation drawing
        return frame_src


    def configure_left_hand_roi(
        self,
        landmarks: Mapping[MediaPipePoseKey, NDArrayFloat],
        height: int, width: int
        ) -> MediaPipePoseHandROI:

        left_wrist = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.LEFT_WRIST]
        left_elbow = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.LEFT_ELBOW]

        return MediaPipePoseHandROI(
            wrist=left_wrist,
            elbow=left_elbow,
            height=height,
            width=width
        )

    def configure_right_hand_roi(
        self,
        landmarks: Mapping[MediaPipePoseKey, NDArrayFloat],
        height: int, width: int
        ) -> MediaPipePoseHandROI:

        right_wrist = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.RIGHT_WRIST]
        right_elbow = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.RIGHT_ELBOW]

        return MediaPipePoseHandROI(
            wrist=right_wrist,
            elbow=right_elbow,
            height=height,
            width=width
        )

    def configure_both_hands_roi(
        self,
        landmarks: Mapping[MediaPipePoseKey, NDArrayFloat],
        height: int, width: int
        ) -> MediaPipePoseBothHandsROI:

        left_wrist = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.LEFT_WRIST]
        left_elbow = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.LEFT_ELBOW]
        right_wrist = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.RIGHT_WRIST]
        right_elbow = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.RIGHT_ELBOW]

        return MediaPipePoseBothHandsROI(
            left_wrist=left_wrist,
            left_elbow=left_elbow,
            right_wrist=right_wrist,
            right_elbow=right_elbow,
            height=height,
            width=width
        )

    def configure_face_roi(
        self,
        landmarks: Mapping[MediaPipePoseKey, NDArrayFloat],
        height: int, width: int
        ) -> MediaPipePoseFaceROI:

        left_ear = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.LEFT_EAR]
        right_ear = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.RIGHT_EAR]
        nose = landmarks[MEDIA_PIPE_POSE_KEY][MediaPipePoseNames.NOSE]

        return MediaPipePoseFaceROI(
            left_ear=left_ear,
            right_ear=right_ear,
            nose=nose,
            height=height,
            width=width
        )
