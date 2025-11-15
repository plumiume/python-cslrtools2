"""Unit tests for lmpipe/utils.py

Tests for path classification and frame generation utilities.
Coverage target: 31% → 100%
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from cslrtools2.lmpipe.utils import (
    is_video_file,
    is_images_dir,
    is_image_file,
    capture_to_frames,
    seq_imgs_to_frames,
    image_file_to_frame,
    is_video_ext_from_mimetype,
    is_image_ext_from_mimetype,
)


@pytest.fixture
def temp_video_file(tmp_path: Path) -> Path:
    """Create a temporary video file."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary image file."""
    image_file = tmp_path / "test_image.png"
    image_file.write_bytes(b"fake image content")
    return image_file


@pytest.fixture
def temp_images_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple image files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img001.jpg").write_bytes(b"image1")
    (images_dir / "img002.png").write_bytes(b"image2")
    (images_dir / "img003.bmp").write_bytes(b"image3")
    return images_dir


class TestIsVideoFile:
    """Test is_video_file() function."""

    def test_mp4_video_file(self, temp_video_file: Path):
        """Test detection of MP4 video file."""
        assert is_video_file(temp_video_file) is True

    def test_various_video_extensions(self, tmp_path: Path):
        """Test detection of various video formats."""
        video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        for ext in video_exts:
            video_file = tmp_path / f"video{ext}"
            video_file.write_bytes(b"content")
            assert is_video_file(video_file) is True, f"Failed for {ext}"

    def test_uppercase_extension(self, tmp_path: Path):
        """Test case-insensitive extension matching."""
        video_file = tmp_path / "VIDEO.MP4"
        video_file.write_bytes(b"content")
        assert is_video_file(video_file) is True

    def test_non_video_file_returns_false(self, temp_image_file: Path):
        """Test that image files return False."""
        assert is_video_file(temp_image_file) is False

    def test_directory_returns_false(self, tmp_path: Path):
        """Test that directories return False."""
        assert is_video_file(tmp_path) is False

    def test_nonexistent_path_returns_false(self, tmp_path: Path):
        """Test that nonexistent paths return False."""
        nonexistent = tmp_path / "nonexistent.mp4"
        assert is_video_file(nonexistent) is False


class TestIsImageFile:
    """Test is_image_file() function."""

    def test_png_image_file(self, temp_image_file: Path):
        """Test detection of PNG image file."""
        assert is_image_file(temp_image_file) is True

    def test_various_image_extensions(self, tmp_path: Path):
        """Test detection of various image formats."""
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"]
        
        for ext in image_exts:
            image_file = tmp_path / f"image{ext}"
            image_file.write_bytes(b"content")
            assert is_image_file(image_file) is True, f"Failed for {ext}"

    def test_uppercase_extension(self, tmp_path: Path):
        """Test case-insensitive extension matching."""
        image_file = tmp_path / "IMAGE.PNG"
        image_file.write_bytes(b"content")
        assert is_image_file(image_file) is True

    def test_non_image_file_returns_false(self, temp_video_file: Path):
        """Test that video files return False."""
        assert is_image_file(temp_video_file) is False

    def test_directory_returns_false(self, tmp_path: Path):
        """Test that directories return False."""
        assert is_image_file(tmp_path) is False


class TestIsImagesDir:
    """Test is_images_dir() function."""

    def test_directory_with_images(self, temp_images_dir: Path):
        """Test detection of directory containing images."""
        assert is_images_dir(temp_images_dir) is True

    def test_empty_directory(self, tmp_path: Path):
        """Test empty directory returns True (vacuously true - all zero elements match)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert is_images_dir(empty_dir) is True

    def test_directory_with_mixed_files(self, tmp_path: Path):
        """Test directory with mixed image and non-image files."""
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "image.png").write_bytes(b"image")
        (mixed_dir / "text.txt").write_bytes(b"text")
        
        # Should return False because not all files are images
        assert is_images_dir(mixed_dir) is False

    def test_directory_with_hidden_files(self, tmp_path: Path):
        """Test that hidden files are ignored."""
        dir_with_hidden = tmp_path / "with_hidden"
        dir_with_hidden.mkdir()
        (dir_with_hidden / "image.png").write_bytes(b"image")
        (dir_with_hidden / ".hidden.txt").write_bytes(b"hidden")
        
        # Should return True because hidden files are ignored
        assert is_images_dir(dir_with_hidden) is True

    def test_file_path_returns_false(self, temp_image_file: Path):
        """Test that file paths return False."""
        assert is_images_dir(temp_image_file) is False


class TestCaptureToFrames:
    """Test capture_to_frames() generator."""

    def test_yields_frames_from_capture(self):
        """Test that frames are yielded from VideoCapture."""
        mock_capture = Mock()
        fake_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Mock read() to return frames then end
        mock_capture.read.side_effect = [
            (True, fake_frame1),
            (True, fake_frame2),
            (False, None),
        ]
        
        frames = list(capture_to_frames(mock_capture))
        
        assert len(frames) == 2
        assert np.array_equal(frames[0], fake_frame1)
        assert np.array_equal(frames[1], fake_frame2)

    def test_empty_capture_yields_nothing(self):
        """Test that empty capture yields no frames."""
        mock_capture = Mock()
        mock_capture.read.return_value = (False, None)
        
        frames = list(capture_to_frames(mock_capture))
        
        assert len(frames) == 0


class TestSeqImgsToFrames:
    """Test seq_imgs_to_frames() generator."""

    @patch('cv2.imread')
    def test_yields_frames_from_image_sequence(self, mock_imread, temp_images_dir: Path):
        """Test that frames are yielded from image directory."""
        # Mock imread to return fake frames
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = fake_frame
        
        frames = list(seq_imgs_to_frames(temp_images_dir))
        
        # Should have 3 frames (3 image files in fixture)
        assert len(frames) == 3
        assert mock_imread.call_count == 3

    @patch('cv2.imread')
    def test_sorted_order(self, mock_imread, tmp_path: Path):
        """Test that images are processed in sorted order."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        
        # Create files in non-alphabetical order
        (images_dir / "img3.png").write_bytes(b"3")
        (images_dir / "img1.png").write_bytes(b"1")
        (images_dir / "img2.png").write_bytes(b"2")
        
        mock_imread.return_value = np.zeros((10, 10, 3))
        
        list(seq_imgs_to_frames(images_dir))
        
        # Check that imread was called in sorted order
        calls = [str(call[0][0]) for call in mock_imread.call_args_list]
        assert calls == [
            str(images_dir / "img1.png"),
            str(images_dir / "img2.png"),
            str(images_dir / "img3.png"),
        ]


class TestImageFileToFrame:
    """Test image_file_to_frame() function."""

    @patch('cv2.imread')
    def test_loads_single_image(self, mock_imread, temp_image_file: Path):
        """Test that single image is loaded correctly."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = fake_frame
        
        result = image_file_to_frame(temp_image_file)
        
        assert np.array_equal(result, fake_frame)
        mock_imread.assert_called_once_with(str(temp_image_file))


class TestIsVideoExtFromMimetype:
    """Test is_video_ext_from_mimetype() function."""

    def test_video_extensions(self):
        """Test common video extensions."""
        video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        for ext in video_exts:
            assert is_video_ext_from_mimetype(ext) is True, f"Failed for {ext}"

    def test_without_leading_dot(self):
        """Test extension without leading dot."""
        assert is_video_ext_from_mimetype("mp4") is True

    def test_image_extension_returns_false(self):
        """Test that image extensions return False."""
        assert is_video_ext_from_mimetype(".png") is False
        assert is_video_ext_from_mimetype(".jpg") is False

    def test_unknown_extension_returns_false(self):
        """Test that unknown extensions return False."""
        assert is_video_ext_from_mimetype(".xyz") is False


class TestIsImageExtFromMimetype:
    """Test is_image_ext_from_mimetype() function."""

    def test_image_extensions(self):
        """Test common image extensions."""
        image_exts = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]
        
        for ext in image_exts:
            assert is_image_ext_from_mimetype(ext) is True, f"Failed for {ext}"

    def test_without_leading_dot(self):
        """Test extension without leading dot."""
        assert is_image_ext_from_mimetype("png") is True

    def test_video_extension_returns_false(self):
        """Test that video extensions return False."""
        assert is_image_ext_from_mimetype(".mp4") is False
        assert is_image_ext_from_mimetype(".avi") is False

    def test_unknown_extension_returns_false(self):
        """Test that unknown extensions return False."""
        assert is_image_ext_from_mimetype(".xyz") is False
