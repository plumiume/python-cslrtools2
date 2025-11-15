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

"""Command-line interface for sign language dataset tools.

This module provides the main CLI entry point for processing and
managing sign language datasets. It supports various dataset formats
and preprocessing operations through a plugin system.

Example:
    Process FluentSigners50 dataset::

        $ sldataset fluentsigners50 --input-dir data/ --output-dir processed/
"""

from ...exceptions import ConfigurationError
from ..logger import sldataset_logger
from .args import CliArgs, plugins

def main():
    """Main entry point for the sldataset CLI.

    Parses command-line arguments and dispatches to the appropriate
    dataset processor based on the selected command.

    Raises:
        :exc:`ConfigurationError`: If an unknown command is specified.
    """

    args = CliArgs.parse_args()

    sldataset_logger.info(f"Starting sldataset CLI with command: {args.command}")

    assert args.command

    pl_info = plugins.get(args.command)

    if pl_info is None:
        available_commands = ", ".join(plugins.keys())
        sldataset_logger.error(f"Unknown command '{args.command}'. Available commands: {available_commands}")
        raise ConfigurationError(
            f"Unknown command: {args.command}. "
            f"Available commands: {available_commands}"
        )

    sldataset_logger.debug(f"Executing plugin processor for command: {args.command}")
    pl_info["processor"](args)

if __name__ == "__main__":
    main()
