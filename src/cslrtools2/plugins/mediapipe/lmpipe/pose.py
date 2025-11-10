from typing import Literal, Mapping
from abc import abstractmethod
from enum import IntEnum
from functools import cache
from itertools import product

import numpy as np
import cv2

from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat


from ....lmpipe.typings import MatLike, NDArrayFloat, NDArrayStr
from ....lmpipe.estimator import shape, headers, estimate, annotate
from ....lmpipe.app.holistic.estimator import HolisticPoseEstimator
from ....lmpipe.app.holistic.roi import BaseROI
from .base import MediaPipeEstimator, get_mediapipe_model
from .pose_args import MediaPipePoseKey, MediaPipePoseArgs


MEDIA_PIPE_POSE_KEY: MediaPipePoseKey = "mediapipe.pose"


class MediaPipePoseNames(IntEnum):
    """Enum for pose landmark names based on MediaPipe Pose model."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class MediaPipePosePartROI(BaseROI):
    """Base class for MediaPipe Pose-based ROI extraction.
    
    Provides common affine transformation matrix operations for extracting
    and normalizing ROI regions from pose landmarks.
    """

    return_none: bool = False
    center_weight: float  # Must be defined in subclass
    size_weight: float    # Must be defined in subclass
    
    # ROI parameters (set by _calculate_roi_parameters)
    height: int
    width: int
    roi_center_px: NDArrayFloat
    roi_size_px: float
    roi_angle: float
    
    def _safe_init(self, landmarks: list[NDArrayFloat], height: int, width: int) -> bool:
        """Safe initialization with NaN checking and exception handling.
        
        Args:
            landmarks: List of landmark arrays to check for NaN
            height: Frame height in pixels
            width: Frame width in pixels
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        self.height = height
        self.width = width
        
        # Check for NaN values in landmarks
        if any(np.isnan(lm).any() for lm in landmarks):
            self.return_none = True
            return False
        
        # Calculate ROI parameters (implemented by subclass)
        try:
            self._calculate_roi_parameters()
        except (ValueError, ZeroDivisionError):
            self.return_none = True
            return False
        
        # Initialize affine matrices
        self._init_affine_matrices()
        return True
    
    def _init_affine_matrices(self):
        """Initialize and cache affine transformation matrices."""
        # Create normalized affine transformation matrix (3x3)
        self.affine_matrix = self._create_affine_matrix()
        # Inverse matrix for converting ROI coords back to world coords
        self.affine_matrix_inv = np.linalg.inv(self.affine_matrix)
        
        # Cache transformation matrices for apply_roi and apply_world_coords
        self.affine_matrix_2x3 = self.affine_matrix[:2, :]  # For cv2.warpAffine
        self.roi_output_size = max(1, int(self.roi_size_px * 2))  # Output size for ROI
        
        # Extended affine matrix cache for arbitrary dimensions
        # Will be expanded on-demand in apply_world_coords
        self._max_dims = 3  # Start with [x, y, 1] (homogeneous)
        self._extended_affine_inv = self.affine_matrix_inv.copy()  # 3x3 initially
    
    @abstractmethod
    def _calculate_roi_parameters(self):
        """Calculate ROI center, size, and angle from landmarks.
        
        Must set:
            - self.roi_center_px: NDArrayFloat
            - self.roi_size_px: float
            - self.roi_angle: float
        """
        ...
    
    def _create_affine_matrix(self) -> NDArrayFloat:
        """Create 3x3 affine transformation matrix for ROI normalization.
        
        Returns:
            :class:`NDArrayFloat`: 3x3 homogeneous transformation matrix
        """
        # Translation to ROI center
        T = np.array([
            [1, 0, -self.roi_center_px[0]],
            [0, 1, -self.roi_center_px[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation by -roi_angle (to align with horizontal)
        cos_a = np.cos(-self.roi_angle)
        sin_a = np.sin(-self.roi_angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Scale to normalize ROI size
        scale = 1.0 / self.roi_size_px if self.roi_size_px > 0 else 1.0
        S = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined transformation: M = S @ R @ T
        return S @ R @ T
    
    def apply_roi(self, frame_src: MatLike) -> MatLike | None:
        """Apply ROI transformation to extract and normalize region.
        
        Args:
            frame_src (`MatLike`): Input frame image
            
        Returns:
            :code:`MatLike | None`: Normalized ROI image, or None if ROI is invalid
        """
        if self.return_none:
            return None
        
        return cv2.warpAffine(
            frame_src,
            self.affine_matrix_2x3,
            (self.roi_output_size, self.roi_output_size)
        )
    
    def apply_world_coords[K: str](
        self, local_coords: Mapping[K, NDArrayFloat]
        ) -> Mapping[K, NDArrayFloat | None]:
        """Transform ROI coordinates back to world coordinates.
        
        Args:
            local_coords (``Mapping[K, NDArrayFloat]``): Landmark coordinates in ROI space

        Returns:
            :code:`Mapping[K, NDArrayFloat | None]`: Landmark coordinates in world (frame) space
        """
        if self.return_none:
            return {k: None for k in local_coords.keys()}
        
        result: dict[K, NDArrayFloat | None] = {}
        for k, coords in local_coords.items():
            # Determine required dimensions
            num_points = coords.shape[0]
            num_dims = coords.shape[1]
            required_dims = num_dims + 1  # +1 for homogeneous coordinate
            
            # Expand cached affine matrix if necessary
            if required_dims > self._max_dims:
                self._expand_affine_matrix(required_dims)
            
            # Create homogeneous coordinates [x, y, z, ..., 1]
            ones = np.ones((num_points, 1), dtype=coords.dtype)
            homogeneous = np.hstack([coords, ones])
            
            # Apply inverse transformation using sliced extended matrix
            # Use [:num_dims, :required_dims] to get the appropriate submatrix
            transform_matrix = self._extended_affine_inv[:num_dims, :required_dims]
            transformed = homogeneous @ transform_matrix.T
            
            result[k] = transformed
        
        return result
    
    def _expand_affine_matrix(self, new_dims: int):
        """Expand the cached affine matrix to support more dimensions.
        
        Args:
            new_dims (`int`): New total dimensions (including homogeneous coordinate)
        """
        new_size = new_dims
        
        # Create new extended matrix with identity for extra dimensions
        new_matrix = np.eye(new_size, dtype=np.float32)
        
        # Copy existing 2x2 affine transformation (top-left)
        new_matrix[:2, :2] = self.affine_matrix_inv[:2, :2]
        # Copy translation (top-right, excluding homogeneous coord)
        new_matrix[:2, new_size-1] = self.affine_matrix_inv[:2, 2]
        # Keep identity for additional dimensions (z, visibility, etc.)
        # Already set by np.eye
        
        self._extended_affine_inv = new_matrix
        self._max_dims = new_size


class MediaPipePoseHandROI(MediaPipePosePartROI):
    """Hand ROI using wrist and elbow landmarks from pose estimation.
    
    Generates normalized affine transformation matrix for hand region extraction.
    """

    center_weight: float = 0.25
    size_weight: float = 2.2

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
    
    def _calculate_roi_parameters(self):
        """Calculate ROI center, size, and angle from landmarks."""
        # Convert normalized coordinates to pixel coordinates
        wrist_px = np.array([self.wrist[0] * self.width, self.wrist[1] * self.height])
        elbow_px = np.array([self.elbow[0] * self.width, self.elbow[1] * self.height])
        
        # Calculate center (weighted towards wrist)
        self.roi_center_px = wrist_px + self.center_weight * (wrist_px - elbow_px)
        
        # Calculate size based on wrist-elbow distance
        distance = np.linalg.norm(wrist_px - elbow_px)
        self.roi_size_px = float(self.size_weight * distance)
        
        # Calculate rotation angle
        diff = wrist_px - elbow_px
        self.roi_angle = float(np.arctan2(diff[1], diff[0]))


class MediaPipePoseFaceROI(MediaPipePosePartROI):
    """Face ROI using ear and nose landmarks from pose estimation.
    
    Generates normalized affine transformation matrix for face region extraction.
    """

    center_weight: float = 0.1
    size_weight: float = 2.0

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
    
    def _calculate_roi_parameters(self):
        """Calculate ROI center, size, and angle from landmarks."""
        # Convert normalized coordinates to pixel coordinates
        left_ear_px = np.array([self.left_ear[0] * self.width, self.left_ear[1] * self.height])
        right_ear_px = np.array([self.right_ear[0] * self.width, self.right_ear[1] * self.height])
        nose_px = np.array([self.nose[0] * self.width, self.nose[1] * self.height])
        
        # Calculate ear center
        ear_center = (left_ear_px + right_ear_px) / 2
        
        # Calculate ROI center (weighted towards nose)
        self.roi_center_px = ear_center + self.center_weight * (nose_px - ear_center)
        
        # Calculate size based on ear distance
        ear_distance = np.linalg.norm(right_ear_px - left_ear_px)
        self.roi_size_px = float(self.size_weight * ear_distance)
        
        # Calculate rotation angle (based on ear alignment)
        diff = right_ear_px - left_ear_px
        self.roi_angle = float(np.arctan2(diff[1], diff[0]))

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


# abstruct method implementations test
estim = MediaPipePoseEstimator()