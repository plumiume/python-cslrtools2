"""Utility helpers for LMPipe image and video handling.

This module provides reusable helpers for the LMPipe pipeline, including
simple path classification functions and frame generators used throughout
``interface.py``.
"""

from pathlib import Path
from typing import Iterable

import cv2

from .typings import MatLike

__all__ = [
    "is_video_file",
    "is_images_dir",
    "is_image_file",
    "capture_to_frames",
    "seq_imgs_to_frames",
    "image_file_to_frame",
]


def is_video_file(path: Path) -> bool:
    """Check if the given path is a video file.
    
    Args:
        path (Path): Path to check.
        
    Returns:
        :code:`bool`: True if the path is a video file, False otherwise.
    """
    # TODO: Improve detection by inspecting MIME types or probing headers for ambiguous cases.
    return path.is_file() and path.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }


def is_images_dir(path: Path) -> bool:
    """Check if the given path is a directory containing image sequences.
    
    Args:
        path (Path): Path to check.
        
    Returns:
        :code:`bool`: True if the path is a directory with image sequences, False otherwise.
    """
    # TODO: Support nested directory traversal and configurable image extension sets.
    if not path.is_dir():
        return False
    return any(is_image_file(child) for child in path.iterdir() if child.is_file())


def is_image_file(path: Path) -> bool:
    """Check if the given path is an image file.
    
    Args:
        path (Path): Path to check.
        
    Returns:
        :code:`bool`: True if the path is an image file, False otherwise.
    """
    # TODO: Detect additional image formats and consider leveraging imghdr or Pillow.
    return path.is_file() and path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".gif",
    }


def capture_to_frames(capture: cv2.VideoCapture) -> Iterable[MatLike]:
    """Convert OpenCV VideoCapture to frame iterator.
    
    Args:
        capture (cv2.VideoCapture): OpenCV VideoCapture object.
        
    Yields:
        :class:`MatLike`: Individual frames as matrices.
    """
    # TODO: Add backpressure control and support for asynchronous frame generation.
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield frame


def seq_imgs_to_frames(src: Path) -> Iterable[MatLike]:
    """Convert image sequence directory to frame iterator.
    
    Args:
        src (Path): Path to directory containing image sequence.
        
    Yields:
        :class:`MatLike`: Individual frames as matrices.
    """
    # TODO: Add natural sorting and handle large datasets via streaming pagination.
    for entry in sorted(src.iterdir()):
        if entry.is_file() and is_image_file(entry):
            yield cv2.imread(str(entry))


def image_file_to_frame(src: Path) -> MatLike:
    """Load single image file as frame matrix.
    
    Args:
        src (Path): Path to image file.
        
    Returns:
        :class:`MatLike`: Image as matrix.
    """
    # TODO: Validate color space requirements and expose configurable read flags.
    frame = cv2.imread(str(src))
    return frame
