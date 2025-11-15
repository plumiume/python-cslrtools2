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

from pathlib import Path

from ...estimator import ProcessResult
from ...utils import is_video_ext_from_mimetype
from .base import (
    AnnotatedFramesSaveCollector, af_save_aliases,
    AnnotatedFramesShowCollector, af_show_aliases
)


class Cv2AnnotatedFramesSaveCollector[K: str](AnnotatedFramesSaveCollector[K]):
    """Save annotated frames using OpenCV (cv2) backend.

    Supports two modes:
    - Image sequence mode: Saves frames as image files (PNG, JPG, etc.) in ``annotated_frames/`` directory
    - Video mode: Saves frames as a video file (MP4, AVI, etc.) as ``annotated_frames{extension}``
    """

    def __init__(
        self,
        *,
        extension: str = ".png",
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        fourcc: str | None = None,
    ) -> None:
        """Initialize the OpenCV annotated frames saver.

        Args:
            extension (`str`, optional): File extension for output.
                Image formats: .png, .jpg, .bmp, etc.
                Video formats: .mp4, .avi, .mov, .mkv, etc.
                Defaults to ".png".
            height (`int | None`, optional): Video height in pixels.
                Required for video output. Defaults to None.
            width (`int | None`, optional): Video width in pixels.
                Required for video output. Defaults to None.
            fps (`int | None`, optional): Video frames per second.
                Required for video output. Defaults to None.
            fourcc (`str | None`, optional): Video codec FourCC code (e.g., 'mp4v', 'XVID', 'MJPG').
                Only used for video output. Auto-selected based on extension if None. Defaults to None.

        Raises:
            ValueError: If extension is a video format but height, width, or fps is not provided.
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Cv2AnnotatedFramesSaveCollector requires the 'opencv-python' package."
            ) from exc
        self._cv2 = cv2
        self.extension = extension

        # Determine mode based on extension
        self._is_video_mode = is_video_ext_from_mimetype(extension)

        # Validate video parameters
        if self._is_video_mode:
            if height is None or width is None or fps is None:
                raise ValueError(
                    f"Video mode (extension={extension}) requires height, width, and fps to be specified. "
                    f"Got: height={height}, width={width}, fps={fps}"
                )

        self.height = height
        self.width = width
        self.fps = fps
        self.fourcc = fourcc

        # State variables
        self._frames_dir: Path | None = None
        self._video_writer: "cv2.VideoWriter | None" = None
        self._video_path: Path | None = None

    @property
    def is_video(self) -> bool:
        """Whether this collector saves as video (based on extension)."""
        return self._is_video_mode

    @property
    def is_image_sequence(self) -> bool:
        """Whether this collector saves as image sequence (based on extension)."""
        return not self._is_video_mode

    @property
    def file_ext(self) -> str:
        """File extension for output."""
        return self.extension

    def _get_fourcc_code(self) -> int:
        """Get the FourCC code for video encoding.

        Returns:
            :class:`int`: FourCC code as integer.
        """
        if self.fourcc is not None:
            return self._cv2.VideoWriter.fourcc(*self.fourcc)  # type: ignore

        # Auto-select codec based on extension
        ext_lower = self.extension.lower()
        if ext_lower == ".mp4":
            return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        elif ext_lower == ".avi":
            return self._cv2.VideoWriter.fourcc(*"XVID")  # type: ignore
        elif ext_lower == ".mov":
            return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        elif ext_lower == ".mkv":
            return self._cv2.VideoWriter.fourcc(*"X264")  # type: ignore
        else:
            # Default fallback
            return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore

    def _open_file(self, path: Path):
        if self._is_video_mode:
            # Video mode: create video writer statically
            assert self.width is not None and self.height is not None and self.fps is not None

            self._video_path = path / f"annotated_frames{self.extension}"
            self._video_path.parent.mkdir(parents=True, exist_ok=True)

            # Create VideoWriter with specified parameters
            fourcc_code = self._get_fourcc_code()
            self._video_writer = self._cv2.VideoWriter(
                str(self._video_path),
                fourcc_code,
                self.fps,
                (self.width, self.height)
            )
        else:
            # Image sequence mode: prepare directory
            self._frames_dir = path / "annotated_frames"
            self._frames_dir.mkdir(parents=True, exist_ok=True)

    def _append_result(self, result: ProcessResult[K]):
        if self._is_video_mode:
            # Video mode: write to video file
            if self._video_writer is not None:
                self._video_writer.write(result.annotated_frame)
        else:
            # Image sequence mode: save individual frames
            if self._frames_dir is None:
                return
            frame_path = self._frames_dir / f"frame_{result.frame_id:06d}{self.extension}"
            self._cv2.imwrite(str(frame_path), result.annotated_frame)

    def _close_file(self):
        if self._is_video_mode:
            # Release video writer
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self._video_path = None
        else:
            # Clean up image sequence state
            self._frames_dir = None


class Cv2AnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using OpenCV (cv2) highgui.

    Shows frames in an interactive window using cv2.imshow().
    """

    def __init__(self, *, window_name: str = "Annotated Frame", wait_key: int = 1) -> None:
        """Initialize the OpenCV frame viewer.

        Args:
            window_name (`str`, optional): Name of the display window. Defaults to "Annotated Frame".
            wait_key (`int`, optional): Milliseconds to wait between frames. Defaults to 1.
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Cv2AnnotatedFramesShowCollector requires the 'opencv-python' package."
            ) from exc
        self._cv2 = cv2
        self.window_name = window_name
        self.wait_key = wait_key

    def _setup(self):
        self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)

    def _update(self, result: ProcessResult[K]):
        self._cv2.imshow(self.window_name, result.annotated_frame)
        self._cv2.waitKey(self.wait_key)

    def _cleanup(self):
        self._cv2.destroyWindow(self.window_name)


def cv2_af_save_collector_creator[K: str](
    file_ext: str,
    key_type: type[K]
    ) -> AnnotatedFramesSaveCollector[K]:
    """Create a Cv2AnnotatedFramesSaveCollector instance.

    Args:
        file_ext (`str`): File extension for output (e.g., '.png', '.mp4').
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`AnnotatedFramesSaveCollector[K]`: OpenCV-based annotated frames saver.
    """
    return Cv2AnnotatedFramesSaveCollector[K](
        extension=file_ext
    )

def cv2_af_show_collector_creator[K: str](
    key_type: type[K]
    ) -> AnnotatedFramesShowCollector[K]:
    """Create a Cv2AnnotatedFramesShowCollector instance.

    Args:
        key_type (`type[K]`): Type of the key for type checking.

    Returns:
        :class:`AnnotatedFramesShowCollector[K]`: OpenCV-based annotated frames viewer.
    """
    return Cv2AnnotatedFramesShowCollector[K]()

# Register backend name aliases
af_save_aliases["cv2"] = cv2_af_save_collector_creator
af_show_aliases["cv2"] = cv2_af_show_collector_creator
