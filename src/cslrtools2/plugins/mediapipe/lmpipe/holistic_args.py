from clipar import namespace, mixin

from ....lmpipe.app.plugins import Info
from .base_args import MediaPipeBaseArgs
from .pose_args import MediaPipePoseKey
from .hand_args import MediaPipeHandKey
from .face_args import MediaPipeFaceKey

type MediaPipeHolisticKey = (
    MediaPipePoseKey |
    MediaPipeHandKey |
    MediaPipeFaceKey
)

@namespace
class MediaPipeHolisticArgs(MediaPipeBaseArgs, mixin.ReprMixin):

    model_complexity: int = 1
    "The complexity of the model: 0, 1, or 2. Higher complexity means better accuracy but slower speed."
    smooth_landmarks: bool = True
    "Whether to smooth the landmarks."
    enable_segmentation: bool = False
    "Whether to enable segmentation."
    smooth_segmentation: bool = True
    "Whether to smooth the segmentation mask."
    refine_face_landmarks: bool = False
    "Whether to refine the face landmarks around the eyes and lips."

def get_holistic_estimator(ns: MediaPipeHolisticArgs.T):
    from .holistic import MediaPipeHolisticEstimator
    return MediaPipeHolisticEstimator(ns)

holistic_info: Info[MediaPipeHolisticArgs.T, MediaPipeHolisticKey] = (
    MediaPipeHolisticArgs,
    get_holistic_estimator
)
