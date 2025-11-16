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

from pathlib import Path

from cslrtools2._root import PACKAGE_ROOT


class TestPackageRoot:
    """Tests for PACKAGE_ROOT constant."""

    def test_package_root_returns_path(self):
        """Test PACKAGE_ROOT is a Path object."""
        assert isinstance(PACKAGE_ROOT, Path)

    def test_package_root_is_absolute(self):
        """Test PACKAGE_ROOT is an absolute path."""
        assert PACKAGE_ROOT.is_absolute()

    def test_package_root_exists(self):
        """Test PACKAGE_ROOT is an existing directory."""
        assert PACKAGE_ROOT.exists()
        assert PACKAGE_ROOT.is_dir()

    def test_package_root_points_to_cslrtools2(self):
        """Test PACKAGE_ROOT points to the cslrtools2 package root."""
        # Should contain __init__.py
        init_file = PACKAGE_ROOT / "__init__.py"
        assert init_file.exists()
        # Name should be cslrtools2
        assert PACKAGE_ROOT.name == "cslrtools2"
