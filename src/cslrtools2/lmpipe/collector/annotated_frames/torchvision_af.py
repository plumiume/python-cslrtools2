# pyright: reportUnnecessaryIsInstance=false

import numpy as np

from ...interface import ProcessResult
from .base import AnnotatedFramesShowCollector, af_show_aliases


class TorchVisionAnnotatedFramesShowCollector[K: str](AnnotatedFramesShowCollector[K]):
    """Display annotated frames using TorchVision utilities.

    Shows frames using TorchVision's visualization tools.
    """

    def __init__(self, *, figsize: tuple[int, int] = (10, 8)) -> None:
        """Initialize the TorchVision frame viewer.

        Args:
            figsize (`tuple[int, int]`, optional): Figure size in inches. Defaults to (10, 8).
        """
        try:
            import torch
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "TorchVisionAnnotatedFramesShowCollector requires 'torch' and 'matplotlib' packages."
            ) from exc
        self._torch = torch
        self._plt = plt
        self.figsize = figsize

    def _update(self, result: ProcessResult[K]):
        # Convert to tensor if needed
        if isinstance(result.annotated_frame, np.ndarray):
            frame_tensor = self._torch.tensor(result.annotated_frame)
        else:
            frame_tensor = result.annotated_frame

        self._plt.figure(figsize=self.figsize)
        self._plt.imshow(frame_tensor.permute(1, 2, 0) if frame_tensor.ndim == 3 and frame_tensor.shape[0] in [1, 3] else frame_tensor)
        self._plt.title(f"Frame {result.frame_id}")
        self._plt.axis('off')
        self._plt.show()


def torchvision_af_show_collector_creator[K: str](
    key_type: type[K]
    ) -> AnnotatedFramesShowCollector[K]:
    """Create a TorchVisionAnnotatedFramesShowCollector instance.
    
    Args:
        key_type (`type[K]`): Type of the key for type checking.
    
    Returns:
        :class:`AnnotatedFramesShowCollector[K]`: TorchVision-based annotated frames viewer.
    """
    return TorchVisionAnnotatedFramesShowCollector[K]()

af_show_aliases["torchvision"] = torchvision_af_show_collector_creator
