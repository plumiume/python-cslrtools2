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

from typing import Any
from clipar import namespace, mixin
from clipar.entities import NamespaceWrapper
from ..options import LMPipeOptionsGroup
from .plugins import loader, PluginInfo

def _set_ns_chain(
    plugins: dict[str, PluginInfo[Any, Any]],
    parent: NamespaceWrapper[Any],
    *children: tuple[str, NamespaceWrapper[Any]],
    ):
    for pl_info in plugins.values():
        for child_name, child in children:
            pl_info["nswrapper"].add_wrapper(child_name, child)
        parent.add_wrapper(pl_info["name"], pl_info["nswrapper"])

plugins = loader()

@namespace
class FaceArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(plugins["face"], FaceArgs)

@namespace
class BothHandsArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(
    plugins["both_hands"], BothHandsArgs,
    ("face", FaceArgs)
)

@namespace
class RightHandArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(
    plugins["right_hand"], RightHandArgs,
    ("face", FaceArgs)
)

@namespace
class LeftHandArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(
    plugins["left_hand"], LeftHandArgs,
    ("right_hand", RightHandArgs),
)

@namespace
class PoseArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(
    plugins["pose"], PoseArgs,
    ("face", FaceArgs),
    ("left_hand", LeftHandArgs),
    ("both_hands", BothHandsArgs),
)

@namespace
class HolisticArgs(mixin.ReprMixin, mixin.CommandMixin): pass
_set_ns_chain(plugins["holistic"], HolisticArgs)

@namespace
class GlobalArgs(mixin.ReprMixin, mixin.CommandMixin):

    src: str
    "source path"
    dst: str
    "destination directory path"

    lmpipe_options = LMPipeOptionsGroup
    "lmpipe options group"

    holistic = HolisticArgs
    "holistic estimator arguments"
    pose = PoseArgs
    "pose estimator arguments"
    both_hands = BothHandsArgs
    "both hands estimator arguments"
    left_hand = LeftHandArgs # requires right_hand
    "left hand estimator arguments"
    face = FaceArgs
    "face estimator arguments"
