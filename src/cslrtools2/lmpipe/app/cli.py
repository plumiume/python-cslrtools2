# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

"""Command-line interface for LMPipe landmark extraction.

This module provides the main CLI entry point for running landmark
extraction pipelines from the command line. It handles argument parsing,
estimator selection, and pipeline execution.

The CLI supports multiple estimator types (holistic, pose, hands, face)
and various output formats through the collector system.

Example:
    Run pose estimation on a video::

        $ lmpipe pose mediapipe --model-size lite input.mp4 output/

    Run holistic estimation with multiple collectors::

        $ lmpipe holistic mediapipe --csv --show input.mp4 output/
"""

from typing import Any
import logging
from multiprocessing import freeze_support
from clipar import NotSelected, NotSelectedType

import cv2

from ..options import lm_pipe_options_group_to_dict, LMPipeOptions
from ..estimator import Estimator
from ..collector import Collector
from .args import (
    GlobalArgs,
    plugins,
    PoseArgs,
    BothHandsArgs,
    LeftHandArgs,
    RightHandArgs,
    FaceArgs,
    HolisticArgs,
)


freeze_support()


def _get_estimator_from_args(
    args: GlobalArgs.T, options: LMPipeOptions
) -> Estimator[Any] | None:
    holistic_args: HolisticArgs.T | NotSelectedType = args.holistic
    pose_args: PoseArgs.T | NotSelectedType = args.pose
    both_hands_args: BothHandsArgs.T | NotSelectedType = args.both_hands
    left_hand_args: LeftHandArgs.T | NotSelectedType = args.left_hand
    right_hand_args: RightHandArgs.T | NotSelectedType = NotSelected
    face_args: FaceArgs.T | NotSelectedType = args.face

    if holistic_args:
        assert holistic_args.command
        holistic_plugins = plugins["holistic"]  # KeyError?
        holistic_info = holistic_plugins[holistic_args.command]  # KeyError?
        holistic_creator = holistic_info["creator"]
        holistic_group = getattr(holistic_args, holistic_info["name"])
        assert holistic_group
        return holistic_creator(holistic_group)

    if pose_args:
        assert pose_args.command
        pose_plugins = plugins["pose"]  # KeyError?
        pose_info = pose_plugins[pose_args.command]  # KeyError?
        pose_creator = pose_info["creator"]
        pose_group = getattr(pose_args, pose_info["name"])
        assert pose_group
        pose_init_pair = (pose_creator, pose_group)
        both_hands_args = getattr(pose_group, "both_hands", NotSelected)
        left_hand_args = getattr(pose_group, "left_hand", NotSelected)
        face_args = getattr(pose_group, "face", NotSelected)
    else:
        pose_init_pair = None

    if both_hands_args:
        assert both_hands_args.command
        both_hands_plugins = plugins["both_hands"]  # KeyError?
        both_hands_info = both_hands_plugins[both_hands_args.command]  # KeyError?
        both_hands_creator = both_hands_info["creator"]
        both_hands_group = getattr(both_hands_args, both_hands_info["name"])
        assert both_hands_group
        both_hands_init_pair = (both_hands_creator, both_hands_group)
        face_args = getattr(both_hands_group, "face", NotSelected)
    else:
        both_hands_init_pair = None

    if left_hand_args:
        assert left_hand_args.command
        left_hand_plugins = plugins["left_hand"]  # KeyError?
        left_hand_info = left_hand_plugins[left_hand_args.command]  # KeyError?
        left_hand_creator = left_hand_info["creator"]
        left_hand_group = getattr(left_hand_args, left_hand_info["name"])
        assert left_hand_group
        left_hand_init_pair = (left_hand_creator, left_hand_group)
        right_hand_args = getattr(left_hand_group, "right_hand", NotSelected)
    else:
        left_hand_init_pair = None

    if right_hand_args:
        assert right_hand_args.command
        right_hand_plugins = plugins["right_hand"]  # KeyError?
        right_hand_info = right_hand_plugins[right_hand_args.command]  # KeyError?
        right_hand_creator = right_hand_info["creator"]
        right_hand_group = getattr(right_hand_args, right_hand_info["name"])
        assert right_hand_group
        right_hand_init_pair = (right_hand_creator, right_hand_group)
        face_args = getattr(right_hand_group, "face", NotSelected)
    else:
        right_hand_init_pair = None

    if face_args:
        assert face_args.command
        face_plugins = plugins["face"]  # KeyError?
        face_info = face_plugins[face_args.command]  # KeyError?
        face_creator = face_info["creator"]
        face_group = getattr(face_args, face_info["name"])
        assert face_group
        face_init_pair = (face_creator, face_group)
    else:
        face_init_pair = None

    if pose_init_pair and (
        face_init_pair
        or both_hands_init_pair
        or left_hand_init_pair
        or right_hand_init_pair
    ):
        from .holistic.estimator import (
            HolisticPartEstimator,
            HolisticPoseEstimator,
            HolisticEstimator,
        )

        pose_estimator = pose_init_pair[0](pose_init_pair[1])
        if not isinstance(pose_estimator, HolisticPoseEstimator):
            raise TypeError(
                "Pose estimator for HolisticEstimator must be a HolisticPoseEstimator"
            )
        both_hands_estimator = both_hands_init_pair and both_hands_init_pair[0](
            both_hands_init_pair[1]
        )
        if not isinstance(both_hands_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Both hands estimator for HolisticEstimator must be a "
                "HolisticPartEstimator or None"
            )
        left_hand_estimator = left_hand_init_pair and left_hand_init_pair[0](
            left_hand_init_pair[1]
        )
        if not isinstance(left_hand_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Left hand estimator for HolisticEstimator must be a "
                "HolisticPartEstimator or None"
            )
        right_hand_estimator = right_hand_init_pair and right_hand_init_pair[0](
            right_hand_init_pair[1]
        )
        if not isinstance(right_hand_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Right hand estimator for HolisticEstimator must be a "
                "HolisticPartEstimator or None"
            )
        face_estimator = face_init_pair and face_init_pair[0](face_init_pair[1])
        if not isinstance(face_estimator, HolisticPartEstimator | None):
            raise TypeError(
                "Face estimator for HolisticEstimator must be a "
                "HolisticPartEstimator or None"
            )
        return HolisticEstimator(
            pose_estimator=pose_estimator,
            both_hands_estimator=both_hands_estimator,
            left_hand_estimator=left_hand_estimator,
            right_hand_estimator=right_hand_estimator,
            face_estimator=face_estimator,
        )

    if pose_init_pair:
        return pose_init_pair[0](pose_init_pair[1])
    if both_hands_init_pair:
        return both_hands_init_pair[0](both_hands_init_pair[1])
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
        lmsc = lmsc_aliases[lmsc_format]  # KeyError?
        cllctr = lmsc(str)
        cllctr.exist_rule = options["landmark_matrix_save_exist_rule"]
        collectors.append(cllctr)

    afsc_save_framework = options["annotated_frames_save_framework"]
    afsc_save_format = options["annotated_frames_save_file_format"]
    if afsc_save_framework:
        afsc = af_save_aliases[afsc_save_framework]  # KeyError?
        cllctr = afsc(afsc_save_format, str)
        cllctr.exist_rule = options["annotated_frames_save_exist_rule"]
        collectors.append(cllctr)

    afsc_show_framework = options["annotated_frames_show_framework"]
    if afsc_show_framework:
        afsc = af_show_aliases[afsc_show_framework]  # KeyError?
        collectors.append(afsc(str))

    return collectors


def main():
    args = GlobalArgs.parse_args()

    lmpipe_options_group = args.lmpipe_options
    assert lmpipe_options_group

    lmpipe_options = lm_pipe_options_group_to_dict(lmpipe_options_group)

    estimator = _get_estimator_from_args(args, lmpipe_options)
    if estimator is None:
        print("No estimator specified. Exiting.")
        return 1
    collectors = _get_collectors_from_args(lmpipe_options)

    from ..logger import lmpipe_logger, lmpipe_formatter
    from ..interface import LMPipeInterface
    from .runner import CliAppRunner

    lmpipe_logger.setLevel(lmpipe_options["log_level"].upper())
    if lmpipe_options["log_file"]:
        handler = logging.FileHandler(lmpipe_options["log_file"], encoding="utf-8")
        handler.setFormatter(lmpipe_formatter)
        lmpipe_logger.addHandler(handler)

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


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        print(f"Fatal error: {e}")
        code = 1
    finally:
        cv2.destroyAllWindows()
    exit(code)
