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

"""Logging utilities for cslrtools2.

This module provides unified logging configuration for all cslrtools2
subpackages.

Logger Hierarchy::

    cslrtools2 (root)
    ├── cslrtools2.lmpipe
    └── cslrtools2.sldataset

Example:
    Get a logger::

        >>> import logging
        >>> logger = logging.getLogger("cslrtools2.mymodule")
        >>> logger.info("Starting operation")

    Configure logging level::

        >>> import logging
        >>> logging.getLogger("cslrtools2").setLevel(logging.DEBUG)
"""

import logging

__all__ = [
    "root_logger",
    "standard_formatter",
    "detailed_formatter",
    "configure_logger",
]

# Root logger for cslrtools2
root_logger = logging.getLogger("cslrtools2")

# Standard formatter for production use
standard_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Detailed formatter for debugging
detailed_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s (%(pathname)s:%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def configure_logger(
    logger: logging.Logger,
    level: int = logging.INFO,
    formatter: logging.Formatter = standard_formatter,
    handler: logging.Handler | None = None,
) -> None:
    """Configure a logger with standard settings.

    Args:
        logger: Logger instance to configure.
        level: Logging level (default: INFO).
        formatter: Formatter to use (default: standard_formatter).
        handler: Handler to use (default: StreamHandler to stdout).

    Example:
        >>> import logging
        >>> from cslrtools2.logger import configure_logger, detailed_formatter
        >>> logger = logging.getLogger("cslrtools2.mymodule")
        >>> configure_logger(logger, level=logging.DEBUG, formatter=detailed_formatter)
    """
    logger.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
