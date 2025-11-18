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

"""Unit tests for concurrent execution in LMPipe.

This module tests multiprocessing, thread safety, and executor fallback behavior.
Supplements existing executor tests with concurrency-focused scenarios.

The key finding is that cslrtools2 uses :mod:`loky.ProcessPoolExecutor`,
which supports nested and local functions unlike the standard library's
:class:`concurrent.futures.ProcessPoolExecutor`.

Test Coverage:
    - ProcessPoolExecutor with nested functions (loky compatibility)
    - ThreadPoolExecutor thread safety
    - DummyExecutor sequential execution
    - Executor interface consistency

Example:
    Run concurrent execution tests::

        >>> pytest tests/unit/lmpipe/test_concurrent.py -v
"""

from __future__ import annotations

from typing import cast
from concurrent.futures import ThreadPoolExecutor

import pytest  # pyright: ignore[reportUnusedImport]

from cslrtools2.lmpipe.interface.executor import (
    DummyExecutor,
    ProcessPoolExecutor,
)


class TestProcessPoolExecutorCompatibility:
    """Test ProcessPoolExecutor compatibility and behavior."""

    def test_process_pool_executor_creation(self):
        """Test that ProcessPoolExecutor can be created."""
        with ProcessPoolExecutor(max_workers=2) as executor:
            assert executor is not None
            assert hasattr(executor, "submit")
            assert hasattr(executor, "map")
            assert hasattr(executor, "shutdown")

    def test_process_pool_task_submission(self):
        """Test task submission to ProcessPoolExecutor.

        Note: This works because cslrtools2 uses loky.ProcessPoolExecutor,
        which supports nested/local functions.
        """

        def simple_task(x: int) -> int:
            return x * 2

        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit single task
            future = executor.submit(simple_task, 5)  # pyright: ignore[reportUnknownMemberType] # noqa: E501
            result = cast(int, future.result())

            assert result == 10

    def test_process_pool_map(self):
        """Test map operation with ProcessPoolExecutor.

        Note: This works because cslrtools2 uses loky.ProcessPoolExecutor,
        which supports nested/local functions.
        """

        def square(x: int) -> int:
            return x**2

        inputs = [1, 2, 3, 4, 5]

        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(square, inputs))  # type: ignore[reportUnknownMemberType] # noqa: E501

        assert results == [1, 4, 9, 16, 25]

    def test_process_pool_multiple_tasks(self):
        """Test submitting multiple tasks to ProcessPoolExecutor.

        Note: This works because cslrtools2 uses loky.ProcessPoolExecutor,
        which supports nested/local functions.
        """

        def task(x: int) -> int:
            return x * 2

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(task, i) for i in range(5)]  # pyright: ignore[reportUnknownMemberType] # noqa: E501
            results = [cast(int, f.result()) for f in futures]

        assert results == [0, 2, 4, 6, 8]


class TestThreadPoolExecutorCompatibility:
    """Test ThreadPoolExecutor compatibility and behavior."""

    def test_thread_pool_executor_creation(self):
        """Test that ThreadPoolExecutor can be created."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert executor is not None
            assert hasattr(executor, "submit")
            assert hasattr(executor, "map")
            assert hasattr(executor, "shutdown")

    def test_thread_pool_task_submission(self):
        """Test task submission to ThreadPoolExecutor."""

        def simple_task(x: int) -> int:
            return x * 2

        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(simple_task, 5)
            result = future.result()

            assert result == 10

    def test_thread_pool_shared_state(self):
        """Test that ThreadPoolExecutor shares state correctly."""
        shared_list: list[int] = []

        def append_task(value: int) -> None:
            shared_list.append(value)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(append_task, i) for i in range(5)]

            # Wait for all tasks to complete
            for future in futures:
                future.result()

        # All values should be appended (order may vary)
        assert len(shared_list) == 5
        assert set(shared_list) == {0, 1, 2, 3, 4}

    def test_thread_pool_exception_handling(self):
        """Test exception handling in ThreadPoolExecutor."""

        def failing_task() -> None:
            raise ValueError("Intentional error")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(failing_task)

            with pytest.raises(ValueError, match="Intentional error"):
                future.result()


class TestDummyExecutorConcurrency:
    """Test DummyExecutor concurrency behavior (sequential execution)."""

    def test_dummy_executor_sequential_execution(self):
        """Test that DummyExecutor executes tasks sequentially."""
        execution_order: list[int] = []

        def task(value: int) -> int:
            execution_order.append(value)
            return value * 2

        def noop_init() -> None:
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        # Submit multiple tasks
        futures = [executor.submit(task, i) for i in range(5)]

        # Get results
        results = [f.result() for f in futures]

        # Verify sequential execution
        assert execution_order == [0, 1, 2, 3, 4]
        assert results == [0, 2, 4, 6, 8]

    def test_dummy_executor_exception_handling(self):
        """Test exception handling in DummyExecutor."""

        def failing_task() -> None:
            raise ValueError("Intentional error")

        def noop_init() -> None:
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        future = executor.submit(failing_task)

        with pytest.raises(ValueError, match="Intentional error"):
            future.result()

    def test_dummy_executor_map_sequential(self):
        """Test that DummyExecutor map executes sequentially."""
        execution_order: list[int] = []

        def task(value: int) -> int:
            execution_order.append(value)
            return value**2

        def noop_init() -> None:
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        inputs = [1, 2, 3, 4, 5]
        results = list(executor.map(task, inputs))

        # Verify sequential execution
        assert execution_order == [1, 2, 3, 4, 5]
        assert results == [1, 4, 9, 16, 25]


class TestExecutorInterfaceConsistency:
    """Test that all executor types provide consistent interface."""

    def test_executor_submit_interface(self):
        """Test that all executors support submit method.

        Note: This works because cslrtools2 uses loky.ProcessPoolExecutor,
        which supports nested/local functions.
        """

        def simple_task(x: int) -> int:
            return x * 2

        def noop_init() -> None:
            pass

        # Test ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            future = executor.submit(simple_task, 5)  # pyright: ignore[reportUnknownMemberType] # noqa: E501
            assert future.result() == 10

        # Test ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(simple_task, 5)
            assert future.result() == 10

        # Test DummyExecutor
        executor = DummyExecutor(max_workers=1, initializer=noop_init)
        future = executor.submit(simple_task, 5)
        assert future.result() == 10

    def test_executor_map_interface(self):
        """Test that all executors support map method.

        Note: This works because cslrtools2 uses loky.ProcessPoolExecutor,
        which supports nested/local functions.
        """

        def simple_task(x: int) -> int:
            return x * 2

        def noop_init() -> None:
            pass

        inputs = [1, 2, 3]
        expected = [2, 4, 6]

        # Test ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(simple_task, inputs))  # type: ignore[reportUnknownMemberType] # noqa: E501
            assert results == expected

        # Test ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(simple_task, inputs))
            assert results == expected

        # Test DummyExecutor
        executor = DummyExecutor(max_workers=1, initializer=noop_init)
        results = list(executor.map(simple_task, inputs))
        assert results == expected

    def test_executor_context_manager(self):
        """Test that all executors work as context managers."""

        def noop_init() -> None:
            pass

        # Test ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            assert executor is not None

        # Test ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert executor is not None

        # Test DummyExecutor
        with DummyExecutor(max_workers=1, initializer=noop_init) as executor:
            assert executor is not None
