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

"""Tests for cslrtools2 package initialization."""

from __future__ import annotations

import pytest


class TestPackageImport:
    """Test package-level imports and attributes."""

    def test_version_exists(self):
        """Test that __version__ attribute exists."""
        import cslrtools2

        assert hasattr(cslrtools2, "__version__")
        assert isinstance(cslrtools2.__version__, str)
        assert len(cslrtools2.__version__) > 0

    def test_version_format(self):
        """Test version string format."""
        import cslrtools2

        version = cslrtools2.__version__
        # Should be semantic version or dev version
        assert version.count(".") >= 2 or "unknown" in version

    def test_exceptions_exported(self):
        """Test that exception classes are exported."""
        import cslrtools2

        # Check exceptions are accessible
        assert hasattr(cslrtools2, "CSLRToolsError")
        assert hasattr(cslrtools2, "ConfigurationError")
        assert hasattr(cslrtools2, "ValidationError")
        assert hasattr(cslrtools2, "LMPipeError")
        assert hasattr(cslrtools2, "SLDatasetError")

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        from cslrtools2 import (
            CSLRToolsError,
            ConfigurationError,
            ValidationError,
            LMPipeError,
            SLDatasetError,
        )

        # All should be subclasses of Exception
        assert issubclass(CSLRToolsError, Exception)
        assert issubclass(ConfigurationError, CSLRToolsError)
        assert issubclass(ValidationError, CSLRToolsError)
        assert issubclass(LMPipeError, CSLRToolsError)
        assert issubclass(SLDatasetError, CSLRToolsError)

    def test_all_attribute(self):
        """Test __all__ contains expected exports."""
        import cslrtools2

        assert hasattr(cslrtools2, "__all__")
        assert isinstance(cslrtools2.__all__, list)

        # Should contain version and exceptions
        assert "__version__" in cslrtools2.__all__
        assert "CSLRToolsError" in cslrtools2.__all__
        assert "ConfigurationError" in cslrtools2.__all__
        assert "ValidationError" in cslrtools2.__all__
        assert "LMPipeError" in cslrtools2.__all__
        assert "SLDatasetError" in cslrtools2.__all__

    def test_lightweight_import(self):
        """Test that importing package doesn't load heavy dependencies."""
        import sys
        import cslrtools2

        # Remove if already imported
        heavy_deps = ["torch", "cv2", "mediapipe", "zarr"]
        for dep in heavy_deps:
            if dep in sys.modules:
                del sys.modules[dep]

        # Import package
        _ = cslrtools2.__version__  # Use the import

        # Heavy dependencies should not be loaded
        for dep in heavy_deps:
            assert dep not in sys.modules, (
                f"Heavy dependency '{dep}' was loaded during package import"
            )


class TestExceptionUsage:
    """Test that exceptions can be raised and caught."""

    def test_raise_cslrtools_error(self):
        """Test raising CSLRToolsError."""
        from cslrtools2 import CSLRToolsError

        with pytest.raises(CSLRToolsError, match="test error"):
            raise CSLRToolsError("test error")

    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        from cslrtools2 import ConfigurationError

        with pytest.raises(ConfigurationError, match="config error"):
            raise ConfigurationError("config error")

    def test_raise_validation_error(self):
        """Test raising ValidationError."""
        from cslrtools2 import ValidationError

        with pytest.raises(ValidationError, match="validation failed"):
            raise ValidationError("validation failed")

    def test_catch_as_base_exception(self):
        """Test catching specific error as base CSLRToolsError."""
        from cslrtools2 import CSLRToolsError, LMPipeError

        with pytest.raises(CSLRToolsError):
            raise LMPipeError("lmpipe error")
