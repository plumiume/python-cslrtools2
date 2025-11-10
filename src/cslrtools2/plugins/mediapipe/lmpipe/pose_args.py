from typing import Literal
from clipar import namespace, mixin

from ....lmpipe.app.plugins import Info
from .base_args import MediaPipeBaseArgs

type MediaPipePoseKey = Literal["mediapipe.pose"]

@namespace
class MediaPipePoseArgs(MediaPipeBaseArgs, mixin.ReprMixin):

    pose_model: Literal["lite", "full", "heavy"] = "full"
    "The pose model variant to use. 'lite' for speed, 'heavy' for accuracy."
    # num_poses: int = 1
    min_pose_detection_confidence: float = 0.0
    "The minimum confidence score for the pose detection to be considered successful."
    min_pose_presence_confidence: float = 0.0
    "The minimum confidence score of pose presence score in the pose landmark detection."
    min_pose_tracking_confidence: float = 0.0
    "The minimum confidence score for the pose tracking to be considered successful."

def get_pose_estimator(ns: MediaPipePoseArgs.T):
    from .pose import MediaPipePoseEstimator
    return MediaPipePoseEstimator(ns)

pose_info: Info[MediaPipePoseArgs.T, MediaPipePoseKey] = (
    MediaPipePoseArgs,
    get_pose_estimator
)

