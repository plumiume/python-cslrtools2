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

"""Basic import tests for cslrtools2 package.

This module verifies that all major modules can be imported without errors.
"""

from __future__ import annotations

import pytest


class TestBasicImports:
    """Test basic module imports."""

    def test_import_cslrtools2(self) -> None:
        """Test that the main package can be imported."""
        import cslrtools2

        assert hasattr(cslrtools2, "__version__")

    def test_import_convsize(self) -> None:
        """Test that the convsize module can be imported."""
        from cslrtools2 import convsize

        assert hasattr(convsize, "conv_size")

    def test_import_lmpipe(self) -> None:
        """Test that the lmpipe module can be imported."""
        from cslrtools2 import lmpipe

        # lmpipe module exists (lightweight design, no re-exports)
        assert lmpipe is not None

    def test_import_lmpipe_estimator(self) -> None:
        """Test that the lmpipe.estimator module can be imported."""
        from cslrtools2.lmpipe import estimator

        assert hasattr(estimator, "Estimator")
        assert hasattr(estimator, "ProcessResult")

    def test_import_lmpipe_options(self) -> None:
        """Test that the lmpipe.options module can be imported."""
        from cslrtools2.lmpipe import options

        assert hasattr(options, "LMPipeOptions")

    def test_import_lmpipe_runspec(self) -> None:
        """Test that the lmpipe.runspec module can be imported."""
        from cslrtools2.lmpipe import runspec

        assert hasattr(runspec, "RunSpec")

    def test_import_sldataset(self) -> None:
        """Test that the sldataset module can be imported."""
        from cslrtools2 import sldataset

        # sldataset module exists (lightweight design, no re-exports)
        assert sldataset is not None

    def test_import_sldataset_dataset(self) -> None:
        """Test that the sldataset.dataset module can be imported."""
        from cslrtools2.sldataset import dataset

        assert hasattr(dataset, "SLDataset")
        assert hasattr(dataset, "SLDatasetItem")

    def test_import_lmpipe_collector(self) -> None:
        """Test that the lmpipe.collector module can be imported."""
        from cslrtools2.lmpipe import collector

        assert hasattr(collector, "Collector")


class TestPluginImports:
    """Test plugin module imports (MediaPipe may not be installed)."""

    def test_import_plugins(self) -> None:
        """Test that the plugins module can be imported."""
        from cslrtools2 import plugins

        # plugins module exists but may be empty if optional deps not installed
        assert plugins is not None

    @pytest.mark.skipif(
        condition=True,
        reason="MediaPipe is optional dependency - skip if not installed",
    )
    def test_import_mediapipe_plugin(self) -> None:
        """Test that the MediaPipe plugin can be imported (optional)."""
        try:
            from cslrtools2.plugins.mediapipe import lmpipe

            assert lmpipe is not None
        except ImportError:
            pytest.skip("MediaPipe plugin not installed")
