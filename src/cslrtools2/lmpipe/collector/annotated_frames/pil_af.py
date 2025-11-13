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

# pyright: reportUnnecessaryIsInstance=false

import numpy as np

from ...estimator import ProcessResult
from .base import AnnotatedFramesShowCollector, af_show_aliases


class PilAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using PIL/Pillow image viewer.

    Shows frames using PIL's default image viewer.
    """

    def __init__(self) -> None:
        """Initialize the PIL frame viewer."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PilAnnotatedFramesShowCollector requires the 'Pillow' package."
            ) from exc
        self._Image = Image

    def _update(self, result: ProcessResult[K]):
        # Convert numpy array to PIL Image if needed
        if isinstance(result.annotated_frame, np.ndarray):
            img = self._Image.fromarray(result.annotated_frame)
        else:
            img = result.annotated_frame

        img.show(title=f"Frame {result.frame_id}")


def pil_af_show_collector_creator[K: str](
    key_type: type[K]
    ) -> AnnotatedFramesShowCollector[K]:
    """Create a PilAnnotatedFramesShowCollector instance.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`AnnotatedFramesShowCollector[K]`: PIL/Pillow-based annotated frames viewer.
    """
    return PilAnnotatedFramesShowCollector[K]()

af_show_aliases["pil"] = pil_af_show_collector_creator
af_show_aliases["pillow"] = pil_af_show_collector_creator
