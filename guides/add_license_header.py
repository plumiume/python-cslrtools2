#!/usr/bin/env python3
"""Add Apache 2.0 license header to Python files missing it.

This script scans Python files in the specified directory and adds
the Apache 2.0 license header to files that don't have it.

Usage:
    python add_license_header.py [directory] [--dry-run]

Arguments:
    directory: Directory to scan (default: tests/)
    --dry-run: Show what would be changed without modifying files
"""

import sys
from pathlib import Path
from typing import List

LICENSE_HEADER = """# Copyright 2025 cslrtools2 contributors
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

"""


def has_license_header(file_path: Path) -> bool:
    """Check if file already has a license header."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Read first 30 lines to check for license header
            lines = [f.readline() for _ in range(30)]

        # Look for license indicators
        content = "".join(lines)
        return "Licensed under the Apache License" in content or \
               "Copyright 2025 cslrtools2 contributors" in content
    except Exception:
        return False


def add_license_header(file_path: Path, dry_run: bool = False) -> bool:
    """Add license header to a file.

    Returns:
        True if header was added, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file starts with shebang
        if content.startswith("#!"):
            lines = content.split("\n", 1)
            new_content = lines[0] + "\n" + LICENSE_HEADER
            if len(lines) > 1:
                new_content += lines[1]
        else:
            new_content = LICENSE_HEADER + content

        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively."""
    return list(directory.rglob("*.py"))


def main():
    """Main entry point."""
    # Parse arguments
    dry_run = "--dry-run" in sys.argv

    # Get directory
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        directory = Path(sys.argv[1])
    else:
        # Default to tests/ directory relative to script location
        script_dir = Path(__file__).parent
        directory = script_dir.parent / "tests"

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {directory}")
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()

    # Find Python files
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")

    # Process files
    missing_header: list[Path] = []
    for file_path in python_files:
        if not has_license_header(file_path):
            missing_header.append(file_path)

    print(f"Files without license header: {len(missing_header)}")
    print()

    if not missing_header:
        print("All files have license headers!")
        return

    # Show files that will be modified
    print("Files to be modified:")
    for file_path in missing_header:
        rel_path = file_path.relative_to(directory.parent)
        print(f"  - {rel_path}")
    print()

    if dry_run:
        print("DRY RUN - No files were modified")
        return

    # Add headers
    print("Adding license headers...")
    success_count = 0
    for file_path in missing_header:
        if add_license_header(file_path, dry_run=False):
            success_count += 1
            rel_path = file_path.relative_to(directory.parent)
            print(f"  ✓ {rel_path}")

    print()
    total = len(missing_header)
    print(f"Successfully added license headers to {success_count}/{total} files")


if __name__ == "__main__":
    main()
