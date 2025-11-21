# Test Resource Sources

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
