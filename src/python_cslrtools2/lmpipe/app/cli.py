from typing import Any
from clipar import NotSelected, NotSelectedType
from ..options import lm_pipe_options_group_to_dict, LMPipeOptions
from ..estimator import Estimator
from ..collector import Collector
from .args import (
    GlobalArgs, plugins,
    PoseArgs, HandArgs, LeftHandArgs, RightHandArgs, FaceArgs, HolisticArgs,
)


def _get_estimator_from_args(
    args: GlobalArgs.T,
    options: LMPipeOptions
    ) -> Estimator[Any] | None:

    holistic_args: HolisticArgs.T | NotSelectedType = args.holistic
    pose_args: PoseArgs.T | NotSelectedType = args.pose
    hand_args: HandArgs.T | NotSelectedType = args.hand
    left_hand_args: LeftHandArgs.T | NotSelectedType = args.left_hand
    right_hand_args: RightHandArgs.T | NotSelectedType = NotSelected
    face_args: FaceArgs.T | NotSelectedType = args.face

    if holistic_args:
        assert holistic_args.command
        holistic_plugins = plugins["holistic"] # KeyError?
        holistic_info = holistic_plugins[holistic_args.command] # KeyError?
        holistic_creator = holistic_info["creator"]
        holistic_group = getattr(holistic_args, holistic_info["name"])
        assert holistic_group
        return holistic_creator(holistic_group)

    if pose_args:
        assert pose_args.command
        pose_plugins = plugins["pose"] # KeyError?
        pose_info = pose_plugins[pose_args.command] # KeyError?
        pose_creator = pose_info["creator"]
        pose_group = getattr(pose_args, pose_info["name"])
        assert pose_group
        pose_init_pair = (pose_creator, pose_group)
        hand_args = getattr(pose_args, "hand")
        left_hand_args = getattr(pose_args, "left_hand")
        face_args = getattr(pose_args, "face")
    else:
        pose_init_pair = None

    if hand_args:
        assert hand_args.command
        hand_plugins = plugins["hand"] # KeyError?
        hand_info = hand_plugins[hand_args.command] # KeyError?
        hand_creator = hand_info["creator"]
        hand_group = getattr(hand_args, hand_info["name"])
        assert hand_group
        hand_init_pair = (hand_creator, hand_group)
        left_hand_init_pair = (hand_creator, hand_group)
        right_hand_init_pair = (hand_creator, hand_group)
        face_args = getattr(hand_args, "face")
    else:
        hand_init_pair = None
        left_hand_init_pair = None
        right_hand_init_pair = None

    if left_hand_args:
        assert left_hand_args.command
        left_hand_plugins = plugins["lefthand"] # KeyError?
        left_hand_info = left_hand_plugins[left_hand_args.command] # KeyError?
        left_hand_creator = left_hand_info["creator"]
        left_hand_group = getattr(left_hand_args, left_hand_info["name"])
        assert left_hand_group
        left_hand_init_pair = (left_hand_creator, left_hand_group)
        right_hand_args = getattr(left_hand_args, "right_hand")
    else:
        left_hand_init_pair = None

    if right_hand_args:
        assert right_hand_args.command
        right_hand_plugins = plugins["righthand"] # KeyError?
        right_hand_info = right_hand_plugins[right_hand_args.command] # KeyError?
        right_hand_creator = right_hand_info["creator"]
        right_hand_group = getattr(right_hand_args, right_hand_info["name"])
        assert right_hand_group
        right_hand_init_pair = (right_hand_creator, right_hand_group)
        face_args = getattr(right_hand_args, "face")
    else:
        right_hand_init_pair = None

    if face_args:
        assert face_args.command
        face_plugins = plugins["face"] # KeyError?
        face_info = face_plugins[face_args.command] # KeyError?
        face_creator = face_info["creator"]
        face_group = getattr(face_args, face_info["name"])
        assert face_group
        face_init_pair = (face_creator, face_group)
    else:
        face_init_pair = None

    if (
        pose_init_pair
        and (
            face_init_pair
            or left_hand_init_pair
            or right_hand_init_pair
        )
        ):
        from .holistic.estimator import HolisticPartEstimator, HolisticPoseEstimator, HolisticEstimator
        pose_estimator = pose_init_pair[0](pose_init_pair[1])
        if not isinstance(pose_estimator, HolisticPoseEstimator):
            raise TypeError(
                "Pose estimator for HolisticEstimator must be a HolisticPoseEstimator"
            )
        left_hand_estimator = left_hand_init_pair and left_hand_init_pair[0](left_hand_init_pair[1])
        if not isinstance(left_hand_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Left hand estimator for HolisticEstimator must be a HolisticPartEstimator or None"
            )
        right_hand_estimator = right_hand_init_pair and right_hand_init_pair[0](right_hand_init_pair[1])
        if not isinstance(right_hand_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Right hand estimator for HolisticEstimator must be a HolisticPartEstimator or None"
            )
        face_estimator = face_init_pair and face_init_pair[0](face_init_pair[1])
        if not isinstance(face_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Face estimator for HolisticEstimator must be a HolisticPartEstimator or None"
            )
        return HolisticEstimator(
            pose_estimator=pose_estimator,
            left_hand_estimator=left_hand_estimator,
            right_hand_estimator=right_hand_estimator,
            face_estimator=face_estimator,
        )

    if pose_init_pair:
        return pose_init_pair[0](pose_init_pair[1])
    if hand_init_pair:
        return hand_init_pair[0](hand_init_pair[1])
    if left_hand_init_pair:
        return left_hand_init_pair[0](left_hand_init_pair[1])
    if right_hand_init_pair:
        return right_hand_init_pair[0](right_hand_init_pair[1])
    if face_init_pair:
        return face_init_pair[0](face_init_pair[1])
    return None

def _get_collectors_from_args(options: LMPipeOptions) -> list[Collector[Any]]:

    from ..collector.landmark_matrix import lmsc_aliases
    from ..collector.annotated_frames import af_save_aliases, af_show_aliases

    collectors: list[Collector[Any]] = []

    lmsc_format = options["landmark_matrix_save_file_format"]
    if lmsc_format:
        lmsc = lmsc_aliases[lmsc_format] # KeyError?
        collectors.append(lmsc(str))

    afsc_save_framework = options["annotated_frames_save_framework"]
    afsc_save_format = options["annotated_frames_save_file_format"]
    if afsc_save_framework:
        afsc = af_save_aliases[afsc_save_framework] # KeyError?
        collectors.append(afsc(afsc_save_format, str))

    afsc_show_framework = options["annotated_frames_show_framework"]
    if afsc_show_framework:
        afsc = af_show_aliases[afsc_show_framework] # KeyError?
        collectors.append(afsc(str))

    return collectors


def main():

    args = GlobalArgs.parse_args()

    lmpipe_options_group = args.lmpipe_options
    assert lmpipe_options_group

    lmpipe_options = lm_pipe_options_group_to_dict(
        lmpipe_options_group
    )

    estimator = _get_estimator_from_args(args, lmpipe_options)
    if estimator is None:
        print("No estimator specified. Exiting.")
        return 1
    collectors = _get_collectors_from_args(lmpipe_options)

    from ..interface import LMPipeInterface
    from .runner import CliAppRunner

    try:
        interface = LMPipeInterface(
            estimator=estimator,
            collectors=collectors,
            options=lmpipe_options,
            runner_type=CliAppRunner,
        )
    except KeyboardInterrupt:
        print("Initialization interrupted by user.")
        return 0
    except Exception as e:
        print(f"Error initializing LMPipeInterface: {e}")
        return 1

    try:
        interface.run(args.src, args.dst)
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0