from typing import Literal
from clipar import mixin

class MediaPipeBaseArgs(mixin.ReprMixin):
    """Common configuration arguments for MediaPipe estimators.
    
    This class defines shared settings that apply to all MediaPipe-based
    landmark estimators including device selection, running mode, output
    dimensions, and visualization options.
    """

    delegate: Literal["cpu", "gpu"] = "cpu"
    "Delegate to run MediaPipe models. Default is 'cpu'."
    running_mode: Literal["image", "video", "live_stream"] = "image"
    "Running mode for MediaPipe models. Default is 'image'."
