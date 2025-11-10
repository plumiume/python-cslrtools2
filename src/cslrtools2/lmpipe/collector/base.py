from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from ..estimator import ProcessResult
from ..runspec import RunSpec

if TYPE_CHECKING:
    from ..options import LMPipeOptions


class Collector[K: str](ABC):
    """Abstract base class for result collectors."""

    @abstractmethod
    def configure_from_options(self, options: "LMPipeOptions") -> None:
        """Configure collector from LMPipe options.
        
        This method is called after collector instantiation to set up
        configuration based on the provided options.
        
        Args:
            options (`LMPipeOptions`): The pipeline options containing configuration.
        """
        ...

    @abstractmethod
    def collect_results(self, runspec: RunSpec[Any], results: Iterable[ProcessResult[K]]):
        """Collect and process the results from the estimator.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.
            results (`Iterable[ProcessResult[K]]`): An iterable of :class:`ProcessResult` objects to be collected.
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
