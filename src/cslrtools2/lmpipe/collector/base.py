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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from ..estimator import ProcessResult
from ..runspec import RunSpec

if TYPE_CHECKING:
    from ..options import LMPipeOptions


class Collector[K: str](ABC):
    """Abstract base class for result collectors.
    
    Collectors are responsible for saving or displaying the results from
    landmark detection pipelines. Implementations handle different output
    formats (CSV, JSON, video annotations, etc.).
    
    Type Parameters:
        K: String type for landmark keys identifying different body parts.
    
    Note:
        Subclasses must implement both :meth:`configure_from_options` and
        :meth:`collect_results` methods. The :meth:`log_start` and :meth:`log_end`
        methods are optional hooks for logging.
    """

    @abstractmethod
    def configure_from_options(self, options: "LMPipeOptions") -> None:
        """Configure collector from LMPipe options.
        
        This method is called after collector instantiation to set up
        configuration based on the provided options.
        
        Args:
            options (:class:`~cslrtools2.lmpipe.options.LMPipeOptions`):
                The pipeline options containing configuration.
        """
        ...

    @abstractmethod
    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect and process the results from the estimator.
        
        Args:
            runspec (:class:`~cslrtools2.lmpipe.runspec.RunSpec`\\[:obj:`~typing.Any`\\]):
                The run specification for the current task.
            results (:class:`~typing.Iterable`\\[:class:`~cslrtools2.lmpipe.estimator.ProcessResult`\\[:obj:`K`\\]\\]):
                An iterable of :class:`~cslrtools2.lmpipe.estimator.ProcessResult`
                objects to be collected.
        """
        ...

    # overridable method
    def apply_exist_rule(self, runspec: RunSpec[Any]) -> bool:
        """Decide whether processing should proceed for the given run specification.

        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.

        Returns:
            :class:`bool`: :code:`True` to continue processing, :code:`False` to skip the task.
        """
        return True
