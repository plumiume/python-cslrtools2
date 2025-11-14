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
