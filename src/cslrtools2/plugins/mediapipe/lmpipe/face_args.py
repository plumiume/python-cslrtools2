from typing import Literal
from clipar import namespace, mixin

from ....lmpipe.app.plugins import Info
from .base_args import MediaPipeBaseArgs

type MediaPipeFaceKey = Literal["mediapipe.face"]

@namespace
class MediaPipeFaceArgs(MediaPipeBaseArgs, mixin.ReprMixin):

    face_model: Literal["full"] = "full"
    "The face landmark model to use. Currently, only 'full' is supported."
    # num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    "The minimum confidence score for the face detection to be considered successful."
    min_face_presence_confidence: float = 0.5
    "The minimum confidence score of face presence score in the face landmark detection."
    min_face_tracking_confidence: float = 0.5
    "The minimum confidence score for the face tracking to be considered successful."

def get_face_estimator(ns: MediaPipeFaceArgs.T):
    from .face import MediaPipeFaceEstimator
    return MediaPipeFaceEstimator(ns)

face_info: Info[MediaPipeFaceArgs.T, MediaPipeFaceKey] = (
    MediaPipeFaceArgs,
    get_face_estimator
)
