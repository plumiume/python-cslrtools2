"""Resource collection configuration."""

from __future__ import annotations

from pathlib import Path

# Base paths
TESTS_DIR = Path(__file__).parent.parent
DATA_DIR = TESTS_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
LANDMARKS_DIR = DATA_DIR / "landmarks"
DATASETS_DIR = DATA_DIR / "datasets"

# Create directories
for directory in [DATA_DIR, VIDEOS_DIR, LANDMARKS_DIR, DATASETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Pexels videos
PEXELS_VIDEOS = [
    {
        "name": "hand_gesture_stop",
        "url": "https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/",
        "video_id": "7123947",
        "description": "A Person Doing a Hand Gesture (19sec, HD 2048x1080 30fps)",
        "resolution": "hd_2048_1080_30fps",
    },
    {
        "name": "hand_gesture_man",
        "url": "https://www.pexels.com/video/man-s-hand-gesture-7123940/",
        "video_id": "7123940",
        "description": "Man's Hand Gesture (HD 1080x2048 30fps)",
        "resolution": "hd_1080_2048_30fps",
    },
]

# GitHub repositories
GITHUB_REPOS = [
    {
        "name": "hand-gesture-recognition-mediapipe",
        "owner": "kinivi",
        "repo": "hand-gesture-recognition-mediapipe",
        "files": [
            "model/keypoint_classifier/keypoint.csv",
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            "model/point_history_classifier/point_history.csv",
            "model/point_history_classifier/point_history_classifier_label.csv",
        ],
        "license": "Apache-2.0",
    },
]

# Zenodo datasets
ZENODO_DATASETS = [
    {
        "name": "pointing_gesture",
        "record_id": "16420298",
        "url": "https://zenodo.org/records/16420298",
        "file": "pointing.csv",
        "size_mb": 5.1,
        "description": "Pointing Gesture Classification Dataset (13,575 samples)",
        "license": "CC-BY-4.0",
    },
]

# Size limits (in MB)
MAX_VIDEO_SIZE_MB = 1.0
MAX_DATASET_SIZE_MB = 2.0
MAX_TOTAL_SIZE_MB = 20.0
