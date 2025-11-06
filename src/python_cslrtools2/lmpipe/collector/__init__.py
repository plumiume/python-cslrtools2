from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..estimator import ProcessResult
from ..runspec import RunSpec

class Collector[K: str](ABC):

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
        """Determine whether to skip processing based on existing results.
        
        Args:
            runspec (`RunSpec[Any]`): The run specification for the current task.

        Raises:
            ValueError: If you want to stop processing due to existing results and the rule is set to 'error'.
        
        Returns:
            :class:`bool`: :code:`True` if processing should be skipped, :code:`False` otherwise.

        
        """
        return True