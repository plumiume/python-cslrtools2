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

from pathlib import Path
from typing import TYPE_CHECKING

from ...estimator import ProcessResult
from .base import (
    AnnotatedFramesSaveCollector, af_save_aliases,
    AnnotatedFramesShowCollector, af_show_aliases
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
else:
    Axes = None  # pyright: ignore
    Figure = None  # pyright: ignore
    AxesImage = None  # pyright: ignore


class MatplotlibAnnotatedFramesSaveCollector[K: str](AnnotatedFramesSaveCollector[K]):
    """Save annotated frames using Matplotlib backend.

    Saves frames as image files using Matplotlib's image save functionality.
    """

    def __init__(self, *, extension: str = ".png", dpi: int = 100) -> None:
        """Initialize the Matplotlib annotated frames saver.

        Args:
            extension (`str`, optional): File extension for saved images. Defaults to ".png".
            dpi (`int`, optional): Dots per inch for saved images. Defaults to 100.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "MatplotlibAnnotatedFramesSaveCollector requires the 'matplotlib' package."
            ) from exc
        self._plt = plt
        self.extension = extension
        self.dpi = dpi
        self._frames_dir: Path | None = None

    @property
    def is_video(self) -> bool:
        """Not a video collector: saves image sequence."""
        return False

    @property
    def is_image_sequence(self) -> bool:
        """Image sequence mode: saves individual frame images."""
        return True

    @property
    def file_ext(self) -> str:
        """File extension for saved images."""
        return self.extension

    def _open_file(self, path: Path):
        self._frames_dir = path / "annotated_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def _append_result(self, result: ProcessResult[K]):
        if self._frames_dir is None:
            return
        frame_path = self._frames_dir / f"frame_{result.frame_id:06d}{self.extension}"
        self._plt.imsave(str(frame_path), result.annotated_frame, dpi=self.dpi)

    def _close_file(self):
        self._frames_dir = None


class MatplotlibAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using Matplotlib interactive viewer.

    Shows frames in a Matplotlib figure window with interactive updates.
    """

    def __init__(self, *, figsize: tuple[int, int] = (10, 8)) -> None:
        """Initialize the Matplotlib frame viewer.

        Args:
            figsize (`tuple[int, int]`, optional): Figure size in inches. Defaults to (10, 8).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "MatplotlibAnnotatedFramesShowCollector requires the 'matplotlib' package."
            ) from exc
        self._plt = plt
        self.figsize = figsize
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._im: AxesImage | None = None

    def _setup(self):
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=self.figsize)
        self._ax.axis('off')  # pyright: ignore[reportAttributeAccessIssue]

    def _update(self, result: ProcessResult[K]):
        if self._ax is not None:
            if self._im is None:
                self._im = self._ax.imshow(result.annotated_frame)  # pyright: ignore[reportUnknownMemberType]
            else:
                self._im.set_data(result.annotated_frame)

            self._ax.set_title(f"Frame {result.frame_id}")  # pyright: ignore[reportUnknownMemberType]

        if self._fig is not None:
            self._fig.canvas.draw()  # pyright: ignore[reportUnknownMemberType]
            self._fig.canvas.flush_events()  # pyright: ignore[reportUnknownMemberType]
        self._plt.pause(0.001)

    def _cleanup(self):
        self._plt.ioff()
        self._plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._im = None


def matplotlib_af_save_collector_creator[K: str](
    file_ext: str,
    key_type: type[K]
    ) -> AnnotatedFramesSaveCollector[K]:
    """Create a MatplotlibAnnotatedFramesSaveCollector instance.

    Args:
        file_ext (`str`): File extension for output (e.g., '.png', '.jpg').
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`AnnotatedFramesSaveCollector[K]`: Matplotlib-based annotated frames saver.
    """
    return MatplotlibAnnotatedFramesSaveCollector[K](
        extension=file_ext
    )

def matplotlib_af_show_collector_creator[K: str](
    key_type: type[K]
    ) -> AnnotatedFramesShowCollector[K]:
    """Create a MatplotlibAnnotatedFramesShowCollector instance.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`AnnotatedFramesShowCollector[K]`: Matplotlib-based annotated frames viewer.
    """
    return MatplotlibAnnotatedFramesShowCollector[K]()

af_save_aliases["matplotlib"] = matplotlib_af_save_collector_creator
af_show_aliases["matplotlib"] = matplotlib_af_show_collector_creator
