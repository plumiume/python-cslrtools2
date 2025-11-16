# Test Resource Collection

This directory contains scripts to download test resources for integration tests.

## Quick Start

### Step 1: Download non-video resources (automated)

```bash
uv pip install requests
uv run python -m tests.resource.setup_resources
```

This will download:
- ✓ GitHub landmark data (~5 MB)
- ✓ Zenodo datasets (~5 MB)
- ✗ Pexels videos (manual download required)

### Step 2: Download Pexels videos (manual)

Pexels requires manual download due to bot detection:

1. **hand_gesture_stop.mp4**
   - URL: https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/
   - Click "Free Download" → Select "HD 2048x1080 30fps"
   - Save to: `tests/data/videos/hand_gesture_stop.mp4`

2. **hand_gesture_man.mp4**
   - URL: https://www.pexels.com/video/man-s-hand-gesture-7123940/
   - Click "Free Download" → Select "HD 1080x2048 30fps"
   - Save to: `tests/data/videos/hand_gesture_man.mp4`

**Note**: Videos are optional for most tests

## What Gets Downloaded

### 1. Videos (from Pexels) - **MANUAL DOWNLOAD REQUIRED**
- `hand_gesture_stop.mp4` - Hand gesture video (19sec, HD 2048x1080, ~4.8 MB)
- `hand_gesture_man.mp4` - Man's hand gesture (HD 1080x2048, ~3.6 MB)

**License**: Free to use (Pexels License)  
**Size**: ~8.4 MB total

⚠️ **Note**: Pexels blocks automated downloads. You must download these manually:

1. Visit https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/
2. Click "Free Download" → Select "HD 2048x1080 30fps"
3. Save to `tests/data/videos/hand_gesture_stop.mp4`
4. Visit https://www.pexels.com/video/man-s-hand-gesture-7123940/
5. Click "Free Download" → Select "HD 1080x2048 30fps"
6. Save to `tests/data/videos/hand_gesture_man.mp4`

**Alternative**: Skip videos if you only need landmark/dataset tests

### 2. Landmark Data (from GitHub)
From `kinivi/hand-gesture-recognition-mediapipe`:
- `keypoint.csv` - Hand sign keypoints (MediaPipe 21 points)
- `keypoint_classifier_label.csv` - Labels for hand signs
- `point_history.csv` - Finger gesture history
- `point_history_classifier_label.csv` - Labels for gestures

**License**: Apache-2.0  
**Size**: ~1-2 MB total

### 3. Datasets (from Zenodo)
- `pointing.csv` - Pointing gesture classification dataset (13,575 samples)

**License**: CC-BY-4.0  
**Size**: 5.1 MB

## Individual Downloads

You can also download specific resource types:

```bash
# Videos only
uv run python -m tests.resource.download_pexels

# Landmarks only
uv run python -m tests.resource.download_github

# Datasets only
uv run python -m tests.resource.download_zenodo
```

## Directory Structure

After running the download script:

```
tests/
├── resource/           # Download scripts (this directory)
│   ├── __init__.py
│   ├── config.py       # Resource configuration
│   ├── download_pexels.py
│   ├── download_github.py
│   ├── download_zenodo.py
│   └── setup_resources.py
└── data/              # Downloaded resources
    ├── SOURCES.md     # Documentation of all sources
    ├── videos/        # Pexels videos
    ├── landmarks/     # GitHub landmark CSVs
    └── datasets/      # Zenodo datasets
```

## Configuration

Resource URLs and settings are defined in `config.py`. To add new resources:

1. Add entry to `PEXELS_VIDEOS`, `GITHUB_REPOS`, or `ZENODO_DATASETS`
2. Run the download script
3. Update `SOURCES.md` documentation

## Size Limits

- Videos: < 1 MB each
- Datasets: < 2 MB each
- **Total: < 20 MB**

These limits ensure the test resources remain lightweight for CI/CD pipelines.

## Requirements

The download scripts require the `requests` library:

```bash
uv pip install requests
```

## License Compliance

All resources are used in accordance with their respective licenses:

- **Pexels**: Free to use under Pexels License
- **Apache-2.0**: Open source, allows modification and distribution  
- **CC-BY-4.0**: Allows sharing and adaptation with attribution

See `tests/data/SOURCES.md` for full attribution details.

## Troubleshooting

**Error: requests library not installed**
```bash
uv pip install requests
```

**Pexels download fails with 403 Forbidden**

Pexels may block automated downloads. Manual download options:

1. **Manual Download (Recommended)**:
   - Visit: https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/
   - Click "Free Download" button
   - Save as `tests/data/videos/hand_gesture_stop.mp4`
   - Visit: https://www.pexels.com/video/man-s-hand-gesture-7123940/
   - Click "Free Download" button  
   - Save as `tests/data/videos/hand_gesture_man.mp4`

2. **Alternative**: Use any other CC0/Free hand gesture videos from Pexels

3. **Skip videos**: The landmark and dataset resources are sufficient for most integration tests

**Download fails with 404**
- Check if the resource URL is still valid
- Update the URL in `config.py`

**File size exceeds limit**
- Verify the file hasn't been updated to a larger version
- Consider finding an alternative smaller resource
- Update size limits in `config.py` if necessary
