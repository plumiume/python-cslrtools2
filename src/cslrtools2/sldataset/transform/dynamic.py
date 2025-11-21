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

"""Data augmentation and transformation utilities for sign language datasets.

This module provides dynamic (runtime) transformations for continuous sign language
recognition (CSLR) datasets. All transforms operate on TensorSLDatasetItem with
the following tensor shapes:
    - videos: [N, T, C, H, W] (batch, time, channel, height, width)
    - landmarks: [N, T, V, C] (batch, time, vertices, coordinates)
    - targets: [N, S] (batch, sequence_length)

Implementation philosophy:
    - Lightweight transforms only (heavy preprocessing in frozen.py)
    - Missing values handled separately (see transform/frozen.py)
    - Error handling via NameError/KeyError for missing keys
    - Shape validation delegated to item constructor

For transform selection rationale and CSLR-specific considerations, see:
    guides/DATA_AUGUMENTATION.md
"""

# INFO: this module is under construction

from __future__ import annotations

from typing import (
    Sequence,
    Callable
)

import torch
import torchvision.transforms.v2 as transforms_v2

from ..dataset.core import SLDatasetItem, TensorSLDatasetItem
from .core import Transform


def return_true() -> bool:
    return True


if return_true():
    raise NotImplementedError("This module is under construction.")


# ==============================================================================
# Random Value Factories
# ==============================================================================


def uniform_rand_factory(gen: torch.Generator) -> torch.Tensor:
    """Default random value factory using uniform distribution.

    Factory function for transforms that require random values in ``[0, 1]``.

    Args:
        gen (`torch.Generator`): Random number generator for reproducibility.

    Returns:
        :class:`torch.Tensor`: Scalar tensor with value sampled from U(0, 1).

    Example::

        >>> gen = torch.Generator()
        >>> gen.manual_seed(42)
        >>> value = uniform_rand_factory(gen)
        >>> value.shape
        torch.Size([])
    """
    return torch.rand([], generator=gen)


class UniformSpeedChange[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
    """Apply uniform temporal scaling to simulate speed variations.

    Simulates speed variations in sign language videos by temporal scaling.

    This transform uniformly stretches or compresses the temporal dimension of
    videos and landmarks using interpolation, changing the number of frames while
    preserving spatial structure. Useful for augmenting sign language datasets
    without breaking temporal grammar.

    Args:
        video_keys (`Sequence[Kvid]`): Keys of videos to transform.
        landmark_keys (`Sequence[Klm]`): Keys of landmarks to transform.
        min_scale (`float`): Minimum scale factor (default: 0.5).
            Scale < 1.0 slows down motion (increases frame count).
        max_scale (`float`): Maximum scale factor (default: 2.0).
            Scale > 1.0 speeds up motion (decreases frame count).
        rand_factory (`Callable`): Function to generate random values in [0, 1].
            Returns :class:`torch.Tensor` with scalar value sampled from U(0, 1).
        mode (`str`): Interpolation mode for
            :func:`torch.nn.functional.interpolate` (default: "nearest").
        gen (`torch.Generator` | :obj:`None`): Random number generator for
            reproducibility. If :obj:`None`, uses default RNG.

    Raises:
        :exc:`ValueError`: If scale values are non-positive or
            ``min_scale > max_scale``.

    Note:
        - Input shapes: videos ``[N, T, C, H, W]``, landmarks ``[N, T, V, C]``
        - Output frame count T' is determined by: ``T' = int(T * scale)``
        - Extreme scale values may disrupt linguistic structure; use with caution
        - Uses :func:`torch.nn.functional.interpolate` for temporal resizing

    Example::

        >>> transform = UniformSpeedChange(
        ...     video_keys=["rgb"],
        ...     landmark_keys=["pose", "hand"],
        ...     min_scale=0.8,
        ...     max_scale=1.2
        ... )
        >>> transformed_item = transform(item)
    """
    def __init__(
        self,
        video_keys: Sequence[Kvid],
        landmark_keys: Sequence[Klm],
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        rand_factory: Callable[[torch.Generator], torch.Tensor] = uniform_rand_factory,
        mode: str = "nearest",
        gen: torch.Generator | None = None,
    ):

        # Validate scale parameters
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError(
                f"Scale values must be positive, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )
        if min_scale > max_scale:
            raise ValueError(
                f"min_scale must be <= max_scale, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )

        self.video_keys = video_keys
        self.landmark_keys = landmark_keys

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rand_factory = rand_factory
        self.gen = self._ensure_generator(gen)
        self.mode = mode

    def __call__(
        self,
        item: TensorSLDatasetItem[Kvid, Klm, Ktgt],
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:
        """Apply uniform speed change to the input item.

        Args:
            item (`TensorSLDatasetItem`): Input dataset item with videos and
                landmarks.

        Returns:
            :class:`TensorSLDatasetItem`: Transformed item with scaled temporal
                dimension. Frame count changes according to sampled scale.

        Raises:
            :exc:`KeyError`: If specified :attr:`video_keys` or
                :attr:`landmark_keys` not found in item.
        """
        rand_val = self.rand_factory(self.gen).clamp(0, 1).item()
        scale = self.min_scale + (self.max_scale - self.min_scale) * rand_val

        videos = {**item.videos}

        for kvid in self.video_keys:

            vvid = item.videos[kvid]  # NameError?
            "with shape [N, T, C, H, W]"

            time_indices = torch.arange(0, vvid.shape[1] * scale) / scale

            vvid_ncthw = vvid.permute(0, 2, 1, 3, 4)  # [N, C, T, H, W]
            videos[kvid] = torch.nn.functional.interpolate(
                vvid_ncthw,
                size=(
                    time_indices.shape[0],  # T
                    vvid_ncthw.shape[3],    # H
                    vvid_ncthw.shape[4],    # W
                ),
                mode=self.mode,
            ).permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:

            vlm = item.landmarks[klm]  # NameError?
            "with shape [N, T, V, C]"

            time_indices = torch.arange(0, vlm.shape[1] * scale) / scale

            vlm_ncth = vlm.permute(0, 3, 1, 2)  # [N, C, T, V]
            landmarks[klm] = torch.nn.functional.interpolate(
                vlm_ncth,
                size=(
                    time_indices.shape[0],  # T
                    vlm_ncth.shape[3],      # V
                ),
                mode=self.mode,
            ).permute(0, 2, 3, 1)  # [N, T, V, C]

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)


class RandomResizePaddingCrop[
    Kvid: str, Klm: str, Ktgt: str
](Transform[Kvid, Klm, Ktgt]):
    """Apply random spatial scaling with automatic padding or cropping.

    This transform applies center-based affine scaling to videos and
    corresponding coordinate transformations to landmarks. The tensor shape
    remains unchanged:

    - ``scale > 1.0``: zoom in (crop effect - outer regions cut off)
    - ``scale < 1.0``: zoom out (padding effect - filled with zeros)

    The class name reflects both behaviors: ResizePaddingCrop handles both
    cases depending on the randomly sampled scale value.

    Args:
        video_keys (`Sequence[Kvid]`): Keys of videos to transform.
        landmark_keys (`Sequence[Klm]`): Keys of landmarks to transform.
        min_scale (`float`): Minimum scale factor (default: 0.8).
        max_scale (`float`): Maximum scale factor (default: 1.2).
        gen (`torch.Generator | None`): Random number generator for
            reproducibility. If :obj:`None`, uses default RNG.

    Raises:
        :exc:`ValueError`: If scale values are non-positive or
            ``min_scale > max_scale``.

    Note:
        - Input/output shapes: videos ``[N, T, C, H, W]``,
          landmarks ``[N, T, V, C]``
        - Tensor dimensions do NOT change (only content is transformed)
        - Uses :func:`torchvision.transforms.v2.functional.affine` for videos
        - Landmark coordinates ``(x, y)`` are transformed:
          ``(xy - 0.5) * scale + 0.5``
        - Other landmark dimensions (z, visibility, etc.) remain unchanged
        - Transform is center-invariant (pivot point at image center)
        - See :mod:`torchvision.transforms.v2.functional` for affine details

    Example::

        >>> transform = RandomResizePaddingCrop(
        ...     video_keys=["rgb"],
        ...     landmark_keys=["pose"],
        ...     min_scale=0.9,
        ...     max_scale=1.1
        ... )
        >>> transformed_item = transform(item)
    """
    def __init__(
        self,
        video_keys: Sequence[Kvid],
        landmark_keys: Sequence[Klm],
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        gen: torch.Generator | None = None,
    ):

        # Validate scale parameters
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError(
                f"Scale values must be positive, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )
        if min_scale > max_scale:
            raise ValueError(
                f"min_scale must be <= max_scale, "
                f"got min_scale={min_scale}, max_scale={max_scale}"
            )

        self.video_keys = video_keys
        self.landmark_keys = landmark_keys

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.gen = self._ensure_generator(gen)

    def __call__(
        self,
        item: TensorSLDatasetItem[Kvid, Klm, Ktgt],
    ) -> TensorSLDatasetItem[Kvid, Klm, Ktgt]:
        """Apply random spatial scaling to the input item.

        Args:
            item (`TensorSLDatasetItem`): Input dataset item with videos and
                landmarks.

        Returns:
            :class:`TensorSLDatasetItem`: Transformed item with scaled spatial
                content. Tensor shapes remain unchanged (only content is
                transformed via :func:`torchvision.transforms.v2.functional.affine`).

        Raises:
            :exc:`KeyError`: If specified :attr:`video_keys` or
                :attr:`landmark_keys` not found in item.
        """
        scale = (
            torch.empty(1)
            .uniform_(self.min_scale, self.max_scale, generator=self.gen)
            .item()
        )

        videos = {**item.videos}

        for kvid in self.video_keys:

            vvid = item.videos[kvid]  # NameError?
            "with shape [N, T, C, H, W]"

            # transforms_v2.functional.affine behavior:
            # - Input: [..., C, H, W] with arbitrary leading batch dimensions
            # - Output: Same shape as input (tensor size unchanged)
            # - scale > 1.0: Image zooms in → crop effect (outer regions cut off)
            # - scale < 1.0: Image zooms out → padding effect (filled with fill value)
            # - Transform is center-invariant by default
            # See: torchvision/transforms/v2/functional/_geometry.py::affine()
            vvid = transforms_v2.functional.affine(
                inpt=vvid,
                angle=0.0,
                translate=[0.0, 0.0],
                scale=scale,
                shear=[0.0, 0.0],
            )

            videos[kvid] = vvid

        landmarks = {**item.landmarks}

        for klm in self.landmark_keys:

            vlm = item.landmarks[klm]  # NameError?
            "with shape [N, T, V, C]"
            vlm_xy = vlm[..., :2]
            vlm_other = vlm[..., 2:]

            # simple impletementation
            vlm_xy = (vlm_xy - 0.5) * scale + 0.5
            landmarks[klm] = torch.cat([vlm_xy, vlm_other], dim=-1)

        return SLDatasetItem(videos=videos, landmarks=landmarks, targets=item.targets)


# ==============================================================================
# Planned Implementations (see guides/DATA_AUGUMENTATION.md)
# ==============================================================================

# class RandomTemporalCrop[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
#     """Random temporal cropping with linguistic structure preservation.
#
#     Priority: Mid
#     Implementation: Extract random time window from [N, T, ...] tensors
#     Constraint: Window size must not break linguistic structure (sentence boundaries)
#     Reference: DATA_AUGUMENTATION.md - "Random Temporal Crop"
#     """


# class ColorJitter[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
#     """Random color jittering for RGB videos (landmark-independent).
#
#     Priority: Mid
#     Implementation: Wrapper for torchvision.transforms.v2.ColorJitter
#     Applies to: video_keys only (landmarks unaffected)
#     Parameters: brightness, contrast, saturation, hue
#     Reference: DATA_AUGUMENTATION.md - "Color Jitter"
#     """


# class RandomGrayscale[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
#     """Randomly convert RGB videos to grayscale.
#
#     Priority: Low
#     Implementation: Wrapper for torchvision.transforms.v2.RandomGrayscale
#     Applies to: video_keys only (landmarks unaffected)
#     Reference: DATA_AUGUMENTATION.md - "Random Grayscale"
#     """


# class JointCoordinateNoise[
#     Kvid: str, Klm: str, Ktgt: str
# ](Transform[Kvid, Klm, Ktgt]):
#     """Add Gaussian noise to landmark coordinates for robustness.
#
#     Priority: Mid
#     Implementation: Sample noise ~ N(0, sigma^2), add to landmark coordinates
#     Applies to: landmark_keys only (videos unaffected)
#     Parameters: sigma (noise standard deviation)
#     Reference: DATA_AUGUMENTATION.md - "Joint Coordinate Noise"
#     """


# class DropJoint[Kvid: str, Klm: str, Ktgt: str](Transform[Kvid, Klm, Ktgt]):
#     """Randomly drop (mask) specific joints to simulate occlusion.
#
#     Priority: Low
#     Implementation: Set landmark coordinates to missing_value or NaN
#     Applies to: landmark_keys only
#     Use case: Evaluate model robustness to missing joints
#     Reference: DATA_AUGUMENTATION.md - "Drop Joint / Mask Joint"
#     """


# class SymmetricAdjacencyNormalization[
#     Kvid: str, Klm: str, Ktgt: str
# ](Transform[Kvid, Klm, Ktgt]):
#     """Normalize adjacency matrix for GCN (D^{-1/2} A D^{-1/2}).
#
#     Priority: Mid
#     Implementation: Compute symmetric normalization of adjacency matrix
#     Applies to: graph_keys (if using dense GCN, not PyG sparse)
#     Note: Only useful for naive GCN implementations (PyG handles this internally)
#     Reference: DATA_AUGUMENTATION.md - "Symmetric Adjacency Normalization"
#     """


# ==============================================================================
# Future Implementations (High Complexity)
# ==============================================================================

# class BoneLengthConstrainedPerturbation[
#     Kvid: str, Klm: str, Ktgt: str
# ](Transform[Kvid, Klm, Ktgt]):
#     """Perturb bone lengths while respecting anatomical constraints.
#
#     Priority: Low (Future Goal)
#     Complexity: Very High
#     Requirements:
#       - Physical constraints: joint angles, bone length ratios
#       - Optimization solver to satisfy constraints
#       - Significant computational cost
#     Expected benefit: Enhanced landmark robustness
#     Reference: DATA_AUGUMENTATION.md - "Bone-Length Constrained Perturbation"
#     """
