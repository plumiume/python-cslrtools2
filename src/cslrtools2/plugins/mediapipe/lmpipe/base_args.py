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
