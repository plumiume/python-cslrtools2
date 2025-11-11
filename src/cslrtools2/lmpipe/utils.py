"""Utility helpers for LMPipe image and video handling.

This module provides reusable helpers for the LMPipe pipeline, including
simple path classification functions and frame generators used throughout
``interface.py``.
"""

from pathlib import Path
from typing import Iterable

import cv2

from ..typings import MatLike

__all__ = [
    "is_video_file",
    "is_images_dir",
    "is_image_file",
    "capture_to_frames",
    "seq_imgs_to_frames",
    "image_file_to_frame",
    "is_video_ext_from_mimetype",
    "is_image_ext_from_mimetype",
]


def is_video_file(path: Path) -> bool:
    """Check if the given path is a video file.
    
    Uses simple extension-based detection. For MIME type based detection,
    see :func:`is_video_ext_from_mimetype`.
    
    Args:
        path (`Path`): Path to check.
        
    Returns:
        :code:`bool`: :code:`True` if the path is a video file, :code:`False` otherwise.
    """
    return path.is_file() and path.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }


def is_images_dir(path: Path) -> bool:
    """Check if the given path is a directory containing image sequences.
    
    Only checks the immediate children of the directory (non-recursive).
    Hidden files (starting with '.') are ignored.
    
    Args:
        path (`Path`): Path to check.
        
    Returns:
        :code:`bool`: :code:`True` if the path is a directory with image sequences, :code:`False` otherwise.
        
    Note:
        Future versions may support nested directory traversal and configurable extension sets.
    """
    if not path.is_dir():
        return False
    return all(
        is_image_file(child)
        for child in path.iterdir()
        if child.is_file() and not child.name.startswith('.')
    )


def is_image_file(path: Path) -> bool:
    """Check if the given path is an image file.
    
    Uses simple extension-based detection for common image formats.
    For MIME type based detection, see :func:`is_image_ext_from_mimetype`.
    
    Args:
        path (`Path`): Path to check.
        
    Returns:
        :code:`bool`: :code:`True` if the path is an image file, :code:`False` otherwise.
    """
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
    
    Yields frames sequentially until the video stream ends.
    
    Args:
        capture (`cv2.VideoCapture`): OpenCV VideoCapture object.

    Yields:
        :class:`MatLike`: Individual frames as matrices.
        
    Note:
        Future versions may add backpressure control and asynchronous frame generation.
    """
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield frame


def seq_imgs_to_frames(src: Path) -> Iterable[MatLike]:
    """Convert image sequence directory to frame iterator.
    
    Sorts files alphabetically and yields frames from valid image files.
    
    Args:
        src (`Path`): Path to directory containing image sequence.

    Yields:
        :class:`MatLike`: Individual frames as matrices.
        
    Note:
        Future versions may add natural sorting (e.g., img1, img2, img10)
        and streaming pagination for large datasets.
    """
    for entry in sorted(src.iterdir()):
        if entry.is_file() and is_image_file(entry):
            yield cv2.imread(str(entry))


def image_file_to_frame(src: Path) -> MatLike:
    """Load single image file as frame matrix.
    
    Reads the image using OpenCV's default settings (BGR color space).
    
    Args:
        src (`Path`): Path to image file.
        
    Returns:
        :class:`MatLike`: Image as matrix.
        
    Note:
        Future versions may validate color space requirements and expose
        configurable read flags (e.g., grayscale, unchanged).
    """
    frame = cv2.imread(str(src))
    return frame


def is_video_ext_from_mimetype(extension: str) -> bool:
    """Check if the given file extension is a video format using mimetypes.
    
    This function uses Python's mimetypes module to determine if an extension
    corresponds to a video MIME type (e.g., 'video/mp4', 'video/x-msvideo').
    
    Args:
        extension (`str`): File extension including the dot (e.g., '.mp4', '.avi').

    Returns:
        :class:`bool`: :code:`True` if the extension is a video format, :code:`False` otherwise.

    Examples:
        >>> is_video_ext_from_mimetype('.mp4')
        True
        >>> is_video_ext_from_mimetype('.png')
        False
    """
    import mimetypes
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Guess the MIME type from the extension
    mime_type, _ = mimetypes.guess_type(f'dummy{extension}')
    
    # Check if the MIME type starts with 'video/'
    if mime_type is None:
        return False
    
    return mime_type.startswith('video/')


def is_image_ext_from_mimetype(extension: str) -> bool:
    """Check if the given file extension is an image format using mimetypes.
    
    This function uses Python's mimetypes module to determine if an extension
    corresponds to an image MIME type (e.g., 'image/png', 'image/jpeg').
    
    Args:
        extension (`str`): File extension including the dot (e.g., '.png', '.jpg').

    Returns:
        :class:`bool`: :code:`True` if the extension is an image format, :code:`False` otherwise.

    Examples:
        >>> is_image_ext_from_mimetype('.png')
        True
        >>> is_image_ext_from_mimetype('.mp4')
        False
    """
    import mimetypes
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Guess the MIME type from the extension
    mime_type, _ = mimetypes.guess_type(f'dummy{extension}')
    
    # Check if the MIME type starts with 'image/'
    if mime_type is None:
        return False
    
    return mime_type.startswith('image/')
