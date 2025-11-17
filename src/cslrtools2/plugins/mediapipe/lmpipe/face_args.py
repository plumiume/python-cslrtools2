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
    """The minimum confidence score for the face detection to be considered
    successful."""
    min_face_presence_confidence: float = 0.5
    """The minimum confidence score of face presence score in the face landmark
    detection."""
    min_face_tracking_confidence: float = 0.5
    "The minimum confidence score for the face tracking to be considered successful."


def get_face_estimator(ns: MediaPipeFaceArgs.T):
    from .face import MediaPipeFaceEstimator

    return MediaPipeFaceEstimator(ns)


face_info: Info[MediaPipeFaceArgs.T, MediaPipeFaceKey] = (
    MediaPipeFaceArgs,
    get_face_estimator,
)
