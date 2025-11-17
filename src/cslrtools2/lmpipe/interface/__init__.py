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

"""LMPipe interface module for computer vision processing pipeline.

This module provides the main interface and runner classes for the LMPipe system,
which enables batch processing of images, videos, and streams using machine learning
estimators with configurable execution modes and collection strategies.

The module contains:
    - LMPipeInterface: Main user interface for running CV processing tasks
    - LMPipeRunner: Internal runner class for handling execution and coordination
    - Utility functions for file type detection and frame processing

Example:
    Basic usage with an estimator:

    >>> from python_cslrtools2.lmpipe import LMPipeInterface
    >>> interface = LMPipeInterface(my_estimator)
    >>> interface.run('input.mp4', 'output/')
"""
from __future__ import annotations

from typing import (
    Callable, Unpack
)

from ...typings import PathLike
from ..logger import lmpipe_logger
from ..options import LMPipeOptions, LMPipeOptionsPartial, DEFAULT_LMPIPE_OPTIONS
from ..estimator import Estimator
from ..collector import Collector
from ..runspec import RunSpec

from .runner import LMPipeRunner


def _update_lmpipe_options(
    base: LMPipeOptions,
    *updates: LMPipeOptions | LMPipeOptionsPartial
) -> LMPipeOptions:
    """Update LMPipe options by merging partial options into base options.

    Args:
        base (`LMPipeOptions`): Base options to start with.
        *updates (`LMPipeOptions | LMPipeOptionsPartial`):
            Variable number of partial options to merge.

    Returns:
        :class:`LMPipeOptions`: Updated options with all partial options
            merged.
    """
    ret = base.copy()
    for opt in updates:
        ret.update(opt)
    return ret


class LMPipeInterface[K: str]:
    """Main interface for LMPipe computer vision processing pipeline.

    This class provides the primary interface for users to execute various types
    of computer vision processing tasks including single images, image sequences,
    videos, and live streams. It handles automatic detection of input types and
    delegates to appropriate processing methods.

    The interface supports multiple execution modes and can be configured with
    various options for parallel processing, output collection, and error handling.

    Type Parameters:
        K: String type for landmark keys identifying different body parts.

    Example:
        Basic usage with video processing::

            >>> interface = LMPipeInterface(my_estimator)
            >>> interface.run('input.mp4', 'output_dir/')

        Custom options::

            >>> interface = LMPipeInterface(
            ...     my_estimator,
            ...     max_cpus=4,
            ...     executor_type='thread'
            ... )
            >>> interface.run_batch('images/', 'results/')
    """

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    def __init__(
        self,
        estimator: Estimator[K],
        collectors: list[Collector[K]] | Callable[[], list[Collector[K]]] = [],
        options: LMPipeOptions | LMPipeOptionsPartial = {},
        runner_type: type[LMPipeRunner[K]] | None = None,
        **kwargs: Unpack[LMPipeOptionsPartial]
    ):
        """Initialize LMPipeInterface with estimator and options.

        Creates a new pipeline interface configured with the specified estimator,
        collectors, and execution options. The interface serves as the main entry
        point for running various types of computer vision processing tasks.

        Args:
            estimator (:class:`~cslrtools2.lmpipe.estimator.Estimator`\\[:obj:`K`\\]):
                The machine learning estimator to use for processing.
            collectors (
                :obj:`list`\\[:class:`~cslrtools2.lmpipe.collector.Collector`\\[
                    :obj:`K`\\]\\
                ] |
                :class:`~typing.Callable`\\[
                    \\[\\],
                    :obj:`list`\\[
                        :class:`~cslrtools2.lmpipe.collector.Collector`\\[:obj:`K`\\]
                    \\]
                \\]):
                Result collectors or a factory function to create them.
                Defaults to empty list.
            options (
                :class:`~cslrtools2.lmpipe.options.LMPipeOptions` |
                :class:`~cslrtools2.lmpipe.options.LMPipeOptionsPartial`
                ):
                Configuration options dictionary. Defaults to empty dict.
            runner_type (
                :obj:`type`\\[:class:`LMPipeRunner`\\[:obj:`K`\\]\\] | :obj:`None`
                ):
                Custom runner class for advanced use cases. Defaults to :obj:`None`.
            **kwargs (:class:`~typing.Unpack`\\[
                    :class:`~cslrtools2.lmpipe.options.LMPipeOptionsPartial`
                \\]):
                Additional configuration parameters to override defaults.
                Common options include
                    ``max_cpus``, ``executor_type``, ``executor_mode``.

        Example:
            Basic initialization with an estimator::

                from cslrtools2.lmpipe import LMPipeInterface
                from my_module import MyEstimator

                interface = LMPipeInterface(MyEstimator())

            With custom options::

                interface = LMPipeInterface(
                    MyEstimator(),
                    max_cpus=4,
                    executor_type='process'
                )

            With collectors::

                from python_cslrtools2.lmpipe.collector import (
                    CsvLandmarkMatrixSaveCollector
                )

                interface = LMPipeInterface(
                    MyEstimator(),
                    collectors=[CsvLandmarkMatrixSaveCollector()]
                )
        """

        self.estimator = estimator
        "Core estimator implementation provided by the caller."
        self.lmpipe_options = _update_lmpipe_options(
            self.lmpipe_options, options, kwargs
        )
        "Effective interface configuration after overrides."
        self.collectors_or_factory = collectors
        "Result collectors or a factory function to create them."
        self.runner_type: type[LMPipeRunner[K]] = runner_type or LMPipeRunner
        "Runner class used to execute processing tasks."

        lmpipe_logger.debug(
            f"LMPipeInterface initialized: estimator={type(estimator).__name__}, "
            f"options={self.lmpipe_options}"
        )

    def run(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing pipeline with automatic input type detection.

        This is the primary entry point for processing. It automatically detects
        whether the source is a directory (batch processing) or a file (single file
        processing) and delegates to the appropriate specialized method.

        Args:
            src (`PathLike`): Source path (file or directory) to process.
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Returns:
            Processing results from the appropriate runner method.

        Raises:
            FileNotFoundError: If source path does not exist.
            ValueError: If source path type is not supported.

        Example:
            Process a video file::

                interface = LMPipeInterface(my_estimator)
                interface.run('input_video.mp4', 'output/')

            Process a directory of files::

                interface.run('input_dir/', 'output_dir/')

            With custom options::

                interface.run(
                    'input.mp4',
                    'output/',
                    max_cpus=8,
                    executor_mode='parallel'
                )
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        lmpipe_logger.info(f"Starting run: src={src}, dst={dst}, mode=auto-detect")
        lmpipe_logger.debug(f"Run options: {updated_options}")
        return self.runner_type(self, updated_options).run(runspec)

    def run_batch(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run batch processing on multiple files in a directory.

        Processes all supported files in the source directory using parallel
        execution based on the configured executor settings. Each file is processed
        independently, making this ideal for large-scale batch processing tasks.

        Args:
            src (`PathLike`): Source directory containing files to process.
            dst (`PathLike`): Destination directory for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Example:
            Process all files in a directory::

                interface = LMPipeInterface(my_estimator)
                interface.run_batch('input_videos/', 'output_results/')

            With parallel processing::

                interface.run_batch(
                    'input_videos/',
                    'output_results/',
                    max_cpus=8,
                    executor_type='process'
                )
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        lmpipe_logger.info(f"Starting batch processing: src={src}, dst={dst}")
        lmpipe_logger.debug(f"Batch options: {updated_options}")
        return self.runner_type(self, updated_options).run_batch(runspec)

    def run_single(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing on a single file with automatic type detection.

        Automatically detects if the file is a video, image directory, or single image
        and processes accordingly. This method serves as a smart dispatcher that
        determines the appropriate processing strategy based on the input file type.

        Args:
            src (`PathLike`): Source file path to process.
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Raises:
            ValueError: If file type is not supported.

        Example:
            Process with automatic detection::

                interface = LMPipeInterface(my_estimator)

                # Automatically detects and processes video
                interface.run_single('video.mp4', 'output/')

                # Automatically detects and processes single image
                interface.run_single('image.jpg', 'output/')

                # Automatically detects and processes image sequence directory
                interface.run_single('frames_dir/', 'output/')
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        return self.runner_type(self, updated_options).run_single(runspec)

    def run_video(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing on a video file.

        Processes each frame of the video file sequentially or in parallel
        depending on configuration. The estimator is applied to each frame,
        and results are collected according to the configured collectors.

        Args:
            src (`PathLike`): Source video file path.
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Example:
            Process a video file::

                interface = LMPipeInterface(my_estimator)
                interface.run_video('input.mp4', 'output/')

            With frame-level parallelization::

                interface.run_video(
                    'input.mp4',
                    'output/',
                    executor_mode='parallel',
                    max_cpus=4
                )
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        return self.runner_type(self, updated_options).run_video(runspec)

    def run_sequence_images(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing on a directory of image sequences.

        Processes image files in the directory as a sequence, maintaining
        temporal order where relevant. Images are typically processed in
        alphabetical/numerical order based on filenames.

        Args:
            src (`PathLike`): Source directory containing image sequence.
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Example:
            Process an image sequence::

                interface = LMPipeInterface(my_estimator)
                interface.run_sequence_images('frames/', 'output/')

            Image sequence from numbered frames::

                # frames/ contains: frame_0001.png, frame_0002.png, ...
                interface.run_sequence_images('frames/', 'results/')
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        return self.runner_type(self, updated_options).run_sequence_images(runspec)

    def run_single_image(
        self,
        src: PathLike,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing on a single image file.

        Processes a single image file using the configured estimator. This is
        the most straightforward processing mode, suitable for testing or
        processing individual images.

        Args:
            src (`PathLike`): Source image file path.
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Example:
            Process a single image::

                interface = LMPipeInterface(my_estimator)
                interface.run_single_image('input.jpg', 'output/')

            With specific collectors::

                from python_cslrtools2.lmpipe.collector import (
                    NpyLandmarkMatrixSaveCollector
                )

                interface = LMPipeInterface(
                    my_estimator,
                    collectors=[NpyLandmarkMatrixSaveCollector()]
                )
                interface.run_single_image('test.png', 'results/')
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_pathlikes(src, dst)
        return self.runner_type(self, updated_options).run_single_image(runspec)

    def run_stream(
        self,
        src: int,
        dst: PathLike,
        **options: Unpack[LMPipeOptionsPartial]
    ):
        """Run processing on a live video stream.

        Processes frames from a live video stream (e.g., webcam) in real-time.
        The stream index typically corresponds to the device index, where 0 is
        the default camera, 1 is the second camera, and so on.

        Args:
            src (`int`): Video stream index (e.g., 0 for default camera).
            dst (`PathLike`): Destination path for output results.
            **options (`Unpack[LMPipeOptionsPartial]`):
                Optional configuration parameters to override instance defaults.

        Raises:
            ValueError: If video stream cannot be opened.

        Example:
            Process from default webcam::

                interface = LMPipeInterface(my_estimator)
                interface.run_stream(0, 'output/')

            Process from second camera::

                interface.run_stream(1, 'camera2_output/')

            With real-time display::

                from python_cslrtools2.lmpipe.collector import (
                    Cv2AnnotatedFramesShowCollector
                )

                interface = LMPipeInterface(
                    my_estimator,
                    collectors=[Cv2AnnotatedFramesShowCollector()]
                )
                interface.run_stream(0, 'stream_output/')
        """
        updated_options = _update_lmpipe_options(self.lmpipe_options, options)
        runspec = RunSpec.from_index(src, dst)
        return self.runner_type(self, updated_options).run_stream(runspec)


__all__ = ["LMPipeInterface", "LMPipeRunner"]
