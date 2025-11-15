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

import pytest  # pyright: ignore[reportUnusedImport]
import logging
# from unittest.mock import Mock

from cslrtools2.logger import (
    root_logger,
    standard_formatter,
    detailed_formatter,
    configure_logger,
)


class TestRootLogger:
    """Tests for root_logger."""

    def test_root_logger_name(self):
        """Test root_logger has correct name."""
        assert root_logger.name == "cslrtools2"

    def test_root_logger_is_logger_instance(self):
        """Test root_logger is a Logger instance."""
        assert isinstance(root_logger, logging.Logger)


class TestFormatters:
    """Tests for logging formatters."""

    def test_standard_formatter_exists(self):
        """Test standard_formatter is a Formatter instance."""
        assert isinstance(standard_formatter, logging.Formatter)

    def test_standard_formatter_format(self):
        """Test standard_formatter produces expected format."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None
        )
        formatted = standard_formatter.format(record)
        assert "[INFO    ]" in formatted
        assert "test" in formatted
        assert "test message" in formatted

    def test_detailed_formatter_exists(self):
        """Test detailed_formatter is a Formatter instance."""
        assert isinstance(detailed_formatter, logging.Formatter)

    def test_detailed_formatter_includes_pathname(self):
        """Test detailed_formatter includes pathname and lineno."""
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=42,
            msg="debug message",
            args=(),
            exc_info=None
        )
        formatted = detailed_formatter.format(record)
        assert "test.py" in formatted
        assert ":42" in formatted


class TestConfigureLogger:
    """Tests for configure_logger function."""

    def test_configure_logger_sets_level(self):
        """Test configure_logger sets logger level."""
        logger = logging.getLogger("test_logger_level")
        configure_logger(logger, level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_configure_logger_adds_handler(self):
        """Test configure_logger adds a handler."""
        logger = logging.getLogger("test_logger_handler")
        # Clear any existing handlers
        logger.handlers.clear()

        configure_logger(logger)

        assert len(logger.handlers) >= 1

    def test_configure_logger_uses_custom_handler(self):
        """Test configure_logger uses provided custom handler."""
        logger = logging.getLogger("test_logger_custom")
        logger.handlers.clear()

        custom_handler = logging.NullHandler()
        configure_logger(logger, handler=custom_handler)

        assert custom_handler in logger.handlers

    def test_configure_logger_sets_formatter_on_handler(self):
        """Test configure_logger sets formatter on handler."""
        logger = logging.getLogger("test_logger_formatter")
        logger.handlers.clear()

        configure_logger(logger, formatter=detailed_formatter)

        # Check that at least one handler has the detailed formatter
        handler = logger.handlers[0]
        assert handler.formatter is detailed_formatter

    def test_configure_logger_default_level_is_info(self):
        """Test configure_logger uses INFO level by default."""
        logger = logging.getLogger("test_logger_default")
        configure_logger(logger)

        assert logger.level == logging.INFO

    def test_configure_logger_with_all_parameters(self):
        """Test configure_logger with all parameters specified."""
        logger = logging.getLogger("test_logger_all_params")
        logger.handlers.clear()

        custom_handler = logging.StreamHandler()
        configure_logger(
            logger,
            level=logging.WARNING,
            formatter=detailed_formatter,
            handler=custom_handler
        )

        assert logger.level == logging.WARNING
        assert custom_handler in logger.handlers
        assert custom_handler.formatter is detailed_formatter
