"""Main script to download all test resources."""

from __future__ import annotations

import sys
from pathlib import Path

from .config import DATA_DIR, DATASETS_DIR, LANDMARKS_DIR, VIDEOS_DIR
from .download_github import download_all_github_landmarks
from .download_pexels import download_all_pexels_videos
from .download_zenodo import download_all_zenodo_datasets


def calculate_directory_size(directory: Path) -> float:
    """Calculate total size of directory in MB.

    Args:
        directory: Directory path

    Returns:
        Size in MB
    """
    if not directory.exists():
        return 0.0

    total_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)


def print_summary() -> None:
    """Print summary of downloaded resources."""
    print("\n" + "=" * 60)
    print("RESOURCE DOWNLOAD SUMMARY")
    print("=" * 60)

    directories = [
        ("Videos", VIDEOS_DIR),
        ("Landmarks", LANDMARKS_DIR),
        ("Datasets", DATASETS_DIR),
    ]

    total_size = 0.0
    for name, directory in directories:
        if directory.exists():
            file_count = len(list(directory.rglob("*.*")))
            size_mb = calculate_directory_size(directory)
            total_size += size_mb
            print(f"{name:12} : {file_count:3} files, {size_mb:6.2f} MB")
        else:
            print(f"{name:12} : (not created)")

    print("-" * 60)
    print(f"{'Total':12} : {total_size:6.2f} MB")
    print("=" * 60)


def create_sources_document(
    videos: list[Path],
    landmarks: list[Path],
    datasets: list[Path],
) -> None:
    """Create SOURCES.md documentation file.

    Args:
        videos: List of downloaded video paths
        landmarks: List of downloaded landmark paths
        datasets: List of downloaded dataset paths
    """
    sources_path = DATA_DIR / "SOURCES.md"

    content = """# Test Resource Sources

This file documents the sources of all test resources used in integration tests.

## Videos

### Pexels (Free to Use)

1. **hand_gesture_stop.mp4**
   - Source: https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/
   - Description: A Person Doing a Hand Gesture (19sec, HD 2048x1080 30fps)
   - License: Free to use (Pexels License)
   - Downloaded: Yes

2. **hand_gesture_man.mp4**
   - Source: https://www.pexels.com/video/man-s-hand-gesture-7123940/
   - Description: Man's Hand Gesture (HD 1080x2048 30fps)
   - License: Free to use (Pexels License)
   - Downloaded: Yes

## Landmark Data

### GitHub Repositories

1. **hand-gesture-recognition-mediapipe**
   - Repository: https://github.com/kinivi/hand-gesture-recognition-mediapipe
   - License: Apache-2.0
   - Files:
     - keypoint.csv (hand sign keypoints)
     - keypoint_classifier_label.csv (labels)
     - point_history.csv (finger gesture history)
     - point_history_classifier_label.csv (labels)
   - Description: MediaPipe hand landmark data (21 points x,y,z) + TFLite models

## Datasets

### Zenodo

1. **pointing.csv**
   - Source: https://zenodo.org/records/16420298
   - Description: Pointing Gesture Classification Dataset (13,575 samples)
   - License: CC-BY-4.0
   - Size: 5.1 MB
   - Features: 20 MediaPipe landmark distance features + binary label

## License Compliance

All resources in this directory are used in accordance with their respective licenses:

- **Pexels**: Free to use under Pexels License
- **Apache-2.0**: Open source, allows modification and distribution
- **CC-BY-4.0**: Allows sharing and adaptation with attribution

## Attribution

When using these resources in publications or derived works, please provide
appropriate attribution to the original sources listed above.

## Download

To re-download all resources, run:

```bash
uv run python -m tests.resource.setup_resources
```

## Size Summary

- Videos: ~2-5 MB
- Landmarks: ~1-2 MB
- Datasets: ~5 MB
- **Total: < 20 MB**
"""

    sources_path.write_text(content, encoding="utf-8")
    print(f"\n✓ Created documentation: {sources_path.relative_to(DATA_DIR.parent)}")


def setup_all_resources() -> int:
    """Download all test resources.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("TEST RESOURCE SETUP")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print()

    try:
        # Check if requests is available
        import requests  # pyright: ignore[reportUnusedImport]  # noqa: F401
    except ImportError:
        print("ERROR: requests library not installed")
        print("Install with: uv pip install requests")
        return 1

    # Download resources
    videos = download_all_pexels_videos()
    landmarks = download_all_github_landmarks()
    datasets = download_all_zenodo_datasets()

    # Create documentation
    create_sources_document(videos, landmarks, datasets)

    # Print summary
    print_summary()

    # Check if all downloads succeeded
    total_expected = 2 + 4 + 1  # videos + landmark files + datasets
    total_downloaded = len(videos) + len(landmarks) + len(datasets)

    if total_downloaded == total_expected:
        print("\n✓ All resources downloaded successfully!")
        return 0
    else:
        print(
            f"\n⚠ Warning: Only {total_downloaded}/{total_expected} resources "
            f"downloaded"
        )
        return 1


def main() -> int:
    """Main entry point."""
    return setup_all_resources()


if __name__ == "__main__":
    sys.exit(main())
