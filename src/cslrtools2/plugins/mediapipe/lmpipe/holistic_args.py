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


from __future__ import annotations

from clipar import namespace, mixin

from ....lmpipe.app.plugins import Info
from .base_args import MediaPipeBaseArgs
from .pose_args import MediaPipePoseKey
from .hand_args import MediaPipeHandKey
from .face_args import MediaPipeFaceKey

type MediaPipeHolisticKey = MediaPipePoseKey | MediaPipeHandKey | MediaPipeFaceKey


@namespace
class MediaPipeHolisticArgs(MediaPipeBaseArgs, mixin.ReprMixin):
    model_complexity: int = 1
    """The complexity of the model: 0, 1, or 2. Higher complexity means better
    accuracy but slower speed."""
    smooth_landmarks: bool = True
    "Whether to smooth the landmarks."
    enable_segmentation: bool = False
    "Whether to enable segmentation."
    smooth_segmentation: bool = True
    "Whether to smooth the segmentation mask."
    refine_face_landmarks: bool = False
    "Whether to refine the face landmarks around the eyes and lips."
    min_holistic_detection_confidence: float = 0.5
    """Minimum confidence value ([0.0, 1.0])
        for the holistic detection to be considered successful."""
    min_holistic_tracking_confidence: float = 0.5
    """Minimum confidence value ([0.0, 1.0])
        for the holistic landmarks to be considered tracked successfully."""


def get_holistic_estimator(ns: MediaPipeHolisticArgs.T):
    from .holistic import MediaPipeHolisticEstimator

    return MediaPipeHolisticEstimator(ns)


holistic_info: Info[MediaPipeHolisticArgs.T, MediaPipeHolisticKey] = (
    MediaPipeHolisticArgs,
    get_holistic_estimator,
)
