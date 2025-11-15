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
