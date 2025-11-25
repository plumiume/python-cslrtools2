# MMPose Plugin for cslrtools2

This plugin provides MMPose landmark extraction capabilities for cslrtools2.

## Model Management

### Download Models

Use the `download_models.py` script to download pre-trained MMPose models without requiring `openmim`:

```bash
# List all available models
uv run python src/cslrtools2/plugins/mmpose/lmpipe/download_models.py --list

# Download a specific model
uv run python src/cslrtools2/plugins/mmpose/lmpipe/download_models.py --model rtmpose-m-body8

# Download all models
uv run python src/cslrtools2/plugins/mmpose/lmpipe/download_models.py --all
```

### Available Models

**Body 2D (17 keypoints - COCO format):**
- `rtmpose-t-body8` - RTMPose-tiny (256x192)
- `rtmpose-s-body8` - RTMPose-small (256x192)
- `rtmpose-m-body8` - RTMPose-medium (256x192) ⭐ Recommended
- `rtmpose-l-body8` - RTMPose-large (256x192)
- `rtmpose-m-body8-384` - RTMPose-medium (384x288)
- `rtmpose-l-body8-384` - RTMPose-large (384x288)

**WholeBody 2D (133 keypoints - COCO-WholeBody format):**
- `rtmw-m` - RTMW-medium (256x192)
- `rtmw-l` - RTMW-large (256x192) ⭐ Recommended
- `rtmw-l-384` - RTMW-large (384x288)

**DWPose (Distilled WholeBody, 133 keypoints):**
- `dwpose-t` - DWPose-tiny (256x192)
- `dwpose-s` - DWPose-small (256x192)
- `dwpose-m` - DWPose-medium (256x192)
- `dwpose-l` - DWPose-large (256x192)
- `dwpose-l-384` - DWPose-large (384x288)

**Hand 2D (21 keypoints):**
- `rtmpose-m-hand` - RTMPose-medium (256x256)

**Face 2D (106 keypoints - LaPa format):**
- `rtmpose-m-face` - RTMPose-medium (256x256)

### Model Storage

Models are downloaded to:
```
src/cslrtools2/assets/mmpose/{task}/{name}/{resolution}.pth
```

Example:
```
src/cslrtools2/assets/mmpose/body/rtmpose-m/256x192.pth
src/cslrtools2/assets/mmpose/wholebody/rtmw-l/384x288.pth
```

## Usage

*Implementation guide will be added after estimator implementation*

## References

- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [RTMPose Project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html)
