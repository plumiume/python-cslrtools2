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

from typing import Callable
from concurrent.futures import Executor, Future
from loky import ProcessPoolExecutor as LokyExecutor # pyright: ignore[reportMissingTypeStubs]

class DummyExecutor(Executor):
    """A dummy executor that executes tasks sequentially in the current thread.
    
    This executor mimics the interface of ProcessPoolExecutor and ThreadPoolExecutor
    but executes all tasks immediately in the calling thread. Useful for debugging
    or when parallel processing is not desired.
    """

    # like 2nd overload of ThreadPoolExecutor.__init__
    # and 2nd overload of ProcessPoolExecutor.__init__

    def __init__(
        self,
        max_workers: int | None = None,
        *,
        initializer: Callable[[], object]
        ):
        """Initialize the dummy executor.
        
        Args:
            max_workers (`int | None`, optional): Ignored, kept for compatibility. Defaults to None.
            initializer (`(() -> object)`): Function to call for initialization.
        """

        initializer()

    def submit[**P, T](
        self,
        fn: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs
        ) -> Future[T]:
        """Submit a callable to be executed immediately.
        
        Args:
            fn (`((...) -> T)`): The callable to execute.
            *args: Positional arguments to pass to the callable.
            **kwargs: Keyword arguments to pass to the callable.
            
        Returns:
            :class:`Future[T]`: A Future object representing the execution result.
        """

        ftr = Future[T]()

        try:
            ret = fn(*args, **kwargs)
            ftr.set_result(ret)
        except Exception as e:
            ftr.set_exception(e)

        return ftr

class ProcessPoolExecutor(LokyExecutor):
    """A ProcessPoolExecutor that supports cancelling futures on shutdown.
    
    This subclass of Loky's ProcessPoolExecutor adds the ability to cancel
    pending futures when shutting down the executor, similar to the behavior
    of ThreadPoolExecutor.
    """

    def shutdown( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False
        ) -> None:
        """Shut down the executor, optionally cancelling pending futures.
        
        Args:
            wait (`bool`, optional):
                If :code:`True`, wait for all running tasks to complete.
                Defaults to :code:`True`.
            cancel_futures (`bool`, optional):
                If :code:`True`, cancel all pending futures.
                Defaults to :code:`False`.
        """
        super().shutdown(wait=wait, kill_workers=cancel_futures)
