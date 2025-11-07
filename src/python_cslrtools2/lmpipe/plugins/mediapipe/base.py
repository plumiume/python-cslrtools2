import requests

try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required to use this plugin. "
        "Install it with: pip install cslrtools2[mediapipe]"
    ) from exc

from ...._root import PACKAGE_ROOT
from ...estimator import Estimator
from .base_args import MediaPipeBaseArgs

MODELS: dict[str, dict[str, str]] = {
    "pose": {
        "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    },
    "hand": {
        "full": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    },
    "face": {
        "full": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    }
}

ASSETS_PATH = PACKAGE_ROOT / "assets"
ASSETS_PATH.mkdir(parents=True, exist_ok=True)

def get_mediapipe_model(part: str, size: str) -> str:

    part_map = MODELS.get(part)
    if part_map is None:
        raise ValueError(
            f"Invalid model part: {part}. Available parts: {list(MODELS.keys())}"
        )

    model_url = part_map.get(size)
    if model_url is None:
        raise ValueError(
            f"Invalid model size: {size} for part {part}. Available sizes: {list(part_map.keys())}"
        )

    model_dir = ASSETS_PATH / part
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = (model_dir / size).with_suffix(".task")

    if model_file.exists():
        return str(model_file)

    response = requests.get(model_url)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download model from {model_url}. Status code: {response.status_code}"
        )

    with model_file.open("wb") as f:

        f.write(response.content)

    return str(model_file)

class MediaPipeEstimator[K: str](Estimator[K]):

    def _get_array_from_landmarks(
        self, lm: NormalizedLandmark
        ) -> list[float]:
        return [
            lm.x or self.missing_value,
            lm.y or self.missing_value,
            lm.z or self.missing_value,
            max(0.0, (lm.visibility or 0.0) * (lm.presence or 0.0))
        ]

    def __init__(self, base_args: MediaPipeBaseArgs):

        self.base_args = base_args

        self.delegate = (
            BaseOptions.Delegate.GPU
            if base_args.delegate == "gpu"
            else BaseOptions.Delegate.CPU
        )

        self.running_mode = (
            VisionTaskRunningMode.VIDEO
            if base_args.running_mode == "video"
            else VisionTaskRunningMode.LIVE_STREAM
            if base_args.running_mode == "live_stream"
            else VisionTaskRunningMode.IMAGE
        )