from typing import Literal
from clipar import namespace, mixin

from ....lmpipe.app.plugins import Info
from .base_args import MediaPipeBaseArgs

type MediaPipeHandKey = Literal["mediapipe.left_hand", "mediapipe.right_hand"]
type MediaPipeHandCategory = Literal["left", "right", "both"]


@namespace
class MediaPipeHandArgs(MediaPipeBaseArgs, mixin.ReprMixin):

    hand_model: Literal["full"] = "full"
    "The hand landmark model to use. Currently, only 'full' is supported."
    # num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    "The minimum confidence score for the hand detection to be considered successful."
    min_hand_presence_confidence: float = 0.5
    "The minimum confidence score of hand presence score in the hand landmark detection."
    min_hand_tracking_confidence: float = 0.5
    "The minimum confidence score for the hand tracking to be considered successful."

def get_both_hands_estimator(ns: MediaPipeHandArgs.T):
    from .hand import MediaPipeHandEstimator
    return MediaPipeHandEstimator(ns, category="both")

def get_left_hand_estimator(ns: MediaPipeHandArgs.T):
    from .hand import MediaPipeHandEstimator
    return MediaPipeHandEstimator(ns, category="left")

def get_right_hand_estimator(ns: MediaPipeHandArgs.T):
    from .hand import MediaPipeHandEstimator
    return MediaPipeHandEstimator(ns, category="right")

both_hands_info: Info[MediaPipeHandArgs.T, MediaPipeHandKey] = (
    MediaPipeHandArgs.copy(),
    get_both_hands_estimator
)

left_hand_info: Info[MediaPipeHandArgs.T, MediaPipeHandKey] = (
    MediaPipeHandArgs.copy(),
    get_left_hand_estimator
)

right_hand_info: Info[MediaPipeHandArgs.T, MediaPipeHandKey] = (
    MediaPipeHandArgs.copy(),
    get_right_hand_estimator
)
