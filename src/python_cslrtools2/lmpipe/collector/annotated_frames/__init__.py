"""Annotated frames save and show collectors.

This subpackage provides collectors that save or display annotated frames
using various visualization frameworks.
"""

from .base import af_save_aliases, af_show_aliases
from .base import AnnotatedFramesSaveCollector, AnnotatedFramesShowCollector
from .cv2_af import Cv2AnnotatedFramesSaveCollector, Cv2AnnotatedFramesShowCollector
from .matplotlib_af import (
    MatplotlibAnnotatedFramesSaveCollector,
    MatplotlibAnnotatedFramesShowCollector,
)
from .pil_af import PilAnnotatedFramesShowCollector
from .torchvision_af import TorchVisionAnnotatedFramesShowCollector

__all__ = [
    "AnnotatedFramesSaveCollector",
    "AnnotatedFramesShowCollector",
    "Cv2AnnotatedFramesSaveCollector",
    "Cv2AnnotatedFramesShowCollector",
    "MatplotlibAnnotatedFramesSaveCollector",
    "MatplotlibAnnotatedFramesShowCollector",
    "PilAnnotatedFramesShowCollector",
    "TorchVisionAnnotatedFramesShowCollector",
    "af_save_aliases",
    "af_show_aliases",
]
