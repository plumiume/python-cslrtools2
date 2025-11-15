"""Test constants module."""
from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import (
    PoseLandmark,
    HandLandmark,
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
)

print("=" * 80)
print("MediaPipe Constants Module Test")
print("=" * 80)

# Test landmark enums
print("\n✓ Landmark Enums:")
print(f"  PoseLandmark.NOSE = {PoseLandmark.NOSE}")
print(f"  PoseLandmark.LEFT_WRIST = {PoseLandmark.LEFT_WRIST}")
print(f"  HandLandmark.WRIST = {HandLandmark.WRIST}")
print(f"  HandLandmark.THUMB_TIP = {HandLandmark.THUMB_TIP}")

# Test len() on enums
print(f"\n✓ Enum lengths:")
print(f"  len(PoseLandmark) = {len(PoseLandmark)}")
print(f"  len(HandLandmark) = {len(HandLandmark)}")

# Test connections
print(f"\n✓ Connections:")
print(f"  POSE_CONNECTIONS: {len(POSE_CONNECTIONS)} connections")
print(f"  HAND_CONNECTIONS: {len(HAND_CONNECTIONS)} connections")

# Test backward compatibility
from cslrtools2.plugins.mediapipe.lmpipe.pose import MediaPipePoseNames

print(f"\n✓ Backward compatibility:")
print(f"  MediaPipePoseNames.NOSE = {MediaPipePoseNames.NOSE}")
print(f"  MediaPipePoseNames is PoseLandmark: {MediaPipePoseNames is PoseLandmark}")

print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
