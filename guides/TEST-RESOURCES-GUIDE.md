# Test Resources Search Guide

**Purpose**: Find existing test videos and datasets for integration testing  
**Target**: AI Assistants & Developers

## Required Resources

### 1. Videos (Priority 1) - **Manual Download Required**
- **Hand gestures**: Pexels videos require manual download (bot detection)
- **Automated search blocked**: Use direct URLs provided in resource table below
- **Target**: `tests/data/videos/`
- **Note**: Videos optional for landmark/dataset tests

### 2. Landmark Data
- **Format**: JSON/NPY/NPZ/CSV with x,y,z coordinates
- **Search**: Kaggle "sign language", GitHub "mediapipe landmarks"
- **Target**: `tests/data/landmarks/`

### 3. Small Datasets
- **Size**: <5MB, 2-10 samples
- **Format**: Zarr/HDF5/NPZ
- **Search**: Kaggle "gesture recognition", GitHub landmark repos
- **Target**: `tests/data/datasets/`

## Search Strategy

**Videos**: Pexels.com → "hand gesture" → filter CC0 → download → trim with ffmpeg if needed  
**Datasets**: Kaggle → "Sign Language MNIST" / GitHub → "mediapipe landmarks dataset"  
**Conversion**: CSV→NPZ: `np.savez()`, NPZ→Zarr: `zarr.open_group()`

## Constraints
- Video: <1MB each, <5MB total
- Datasets: <2MB each
- **Total: <20MB**

## Validation Checklist
- [ ] CC0/Public Domain license
- [ ] Size within limits
- [ ] Playable/loadable
- [ ] No PII
- [ ] Document source in `tests/data/SOURCES.md`

## AI Assistant Quick Start
1. **Automated**: Run `uv run python -m tests.resource.setup_resources` (downloads GitHub + Zenodo)
2. **Manual**: Download Pexels videos from URLs in table below (save to `tests/data/videos/`)
3. **Alternative**: Skip videos, use landmark data only for tests
4. **Verify**: Check `tests/data/SOURCES.md` for all downloaded resources

**Note**: Pexels blocks automated downloads - manual download required from web browser

---

## Available Resources (Pre-searched)

| 種類 | 名前・説明 | ライセンス / 備考 | サイズやフォーマット | URL / 出典 |
|------|-----------|------------------|-------------------|-----------|
| **動画（手ジェスチャー）** | A Person Doing a Hand Gesture | Free to use (Pexels) | 19秒, HD 2048x1080 30fps | [Link](https://www.pexels.com/video/a-person-doing-a-hand-gesture-7123947/) |
| 動画 | Man's Hand Gesture | Free to use (Pexels) | HD 1080x2048 30fps | [Link](https://www.pexels.com/video/man-s-hand-gesture-7123940/) |
| 動画 | Close-Up of a Person Making Hand Gestures | Free to use (Pexels) | 要確認 | [Link](https://www.pexels.com/video/close-up-vide-of-a-person-making-hand-gestures-9019581/) |
| 動画 | Hand Gesture Pointing Upward (3Dアニメ) | Free to use (Pexels) | 要確認 | [Link](https://www.pexels.com/video/hand-gesture-pointing-upward-8820122/) |
| 動画 | Hand Gesture "OK" ジェスチャー | Free to use (Pexels) | 要確認 | [Link](https://www.pexels.com/video/hand-gesture-8055717/) |
| **動画（空背景）** | Plain Background Videos | Free (Pexels) | 要確認 | [Link](https://www.pexels.com/search/videos/plain%20background/) |
| **データセット** | American Sign Language (ASL) MNIST | Public (Kaggle) | 28x28 グレースケール画像 | [Link](https://www.kaggle.com/datasets/sm261998/mnist-sign-language) |
| データセット | ASL Dataset（マルチクラス手サイン） | CC0: Public Domain (Kaggle) | 要確認 | [Link](https://www.kaggle.com/datasets/ayuraj/asl-dataset) |
| **ランドマーク** | hand-gesture-recognition-mediapipe | Apache-2.0 (GitHub) | MediaPipe keypoints CSV (21点x,y,z) + TFLite models | [Link](https://github.com/kinivi/hand-gesture-recognition-mediapipe) |
| ランドマーク | Hand Landmark Recognition using MediaPipe | MIT (GitHub) | MediaPipe 21点 (x,y,z) | [Link](https://github.com/prashver/hand-landmark-recognition-using-mediapipe) |
| データセット | Pointing Gesture Classification Dataset | CC-BY-4.0 (Zenodo) | CSV 5.1MB, 13,575 samples, 20 features + label | [Link](https://zenodo.org/records/16420298) |
