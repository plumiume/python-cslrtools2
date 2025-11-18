"""Tests for LMPipe executor utilities."""

from __future__ import annotations

from typing import cast
import time
from concurrent.futures import Future

import pytest

from cslrtools2.lmpipe.interface.executor import (
    DummyExecutor,
    ProcessPoolExecutor,
)


class TestDummyExecutor:
    """Test DummyExecutor for sequential task execution."""

    def test_initialization(self):
        """Test DummyExecutor initialization with initializer."""
        init_called: list[bool] = []

        def initializer():
            init_called.append(True)

        DummyExecutor(max_workers=4, initializer=initializer)

        assert len(init_called) == 1

    def test_submit_successful_task(self):
        """Test submitting a task that succeeds."""

        def noop_init():
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        def add(a: int, b: int) -> int:
            return a + b

        future = executor.submit(add, 2, 3)

        assert isinstance(future, Future)
        assert future.done()
        assert future.result() == 5

    def test_submit_task_with_exception(self):
        """Test submitting a task that raises an exception."""

        def noop_init():
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        def failing_task():
            raise ValueError("Test error")

        future = executor.submit(failing_task)

        assert future.done()
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_submit_with_kwargs(self):
        """Test submitting task with keyword arguments."""

        def noop_init():
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        future = executor.submit(greet, "World", greeting="Hi")

        assert future.result() == "Hi, World!"

    def test_sequential_execution(self):
        """Test that tasks are executed sequentially (immediately)."""

        def noop_init():
            pass

        executor = DummyExecutor(max_workers=1, initializer=noop_init)

        execution_order: list[int] = []

        def task(task_id: int):
            execution_order.append(task_id)
            return task_id

        # Submit multiple tasks
        futures = [executor.submit(task, i) for i in range(5)]

        # All should be done immediately
        assert all(f.done() for f in futures)
        # Execution order should match submission order
        assert execution_order == [0, 1, 2, 3, 4]


class TestProcessPoolExecutor:
    """Test ProcessPoolExecutor with cancel_futures support."""

    def test_initialization(self):
        """Test ProcessPoolExecutor can be created."""
        executor = ProcessPoolExecutor(max_workers=2)
        executor.shutdown(wait=True)

    def test_submit_task(self):
        """Test submitting and executing a task."""
        executor = ProcessPoolExecutor(max_workers=2)

        def square(x: int) -> int:
            return x * x

        future: Future[int] = executor.submit(square, 5)  # pyright: ignore[reportUnknownMemberType] # noqa: E501
        result: int = cast(int, future.result(timeout=5.0))

        assert result == 25
        executor.shutdown(wait=True)

    def test_shutdown_with_wait(self):
        """Test shutdown waits for running tasks."""
        executor = ProcessPoolExecutor(max_workers=2)

        def slow_task():
            time.sleep(0.1)
            return "done"

        future: Future[str] = executor.submit(slow_task)  # pyright: ignore[reportUnknownMemberType] # noqa: E501
        executor.shutdown(wait=True)

        # Task should have completed
        assert future.done()
        assert future.result() == "done"

    def test_shutdown_with_cancel_futures(self):
        """Test shutdown with cancel_futures=True."""
        executor = ProcessPoolExecutor(max_workers=1)

        def quick_task(x: int) -> int:
            return x * 2

        # Submit a task
        executor.submit(quick_task, 10)  # pyright: ignore[reportUnknownMemberType]

        # Shutdown with cancel_futures
        executor.shutdown(wait=False, cancel_futures=True)

        # The specific behavior depends on timing, but shutdown should not raise
        # This tests that the parameter is accepted
