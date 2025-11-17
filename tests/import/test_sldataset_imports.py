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

"""Basic tests for the sldataset module.

Note: Full dataset tests require creating Zarr files, which are deferred
to integration tests. These tests verify import and basic structure.
"""

from __future__ import annotations


class TestSLDatasetImports:
    """Test that sldataset classes can be imported."""

    def test_import_sldataset(self) -> None:
        """Test that SLDataset can be imported from dataset module."""
        from cslrtools2.sldataset.dataset import SLDataset

        assert SLDataset is not None

    def test_import_sldatasetitem(self) -> None:
        """Test that SLDatasetItem can be imported from dataset module."""
        from cslrtools2.sldataset.dataset import SLDatasetItem

        assert SLDatasetItem is not None

    def test_import_dataset_module(self) -> None:
        """Test that dataset module can be imported."""
        from cslrtools2.sldataset import dataset

        assert hasattr(dataset, "SLDataset")
        assert hasattr(dataset, "SLDatasetItem")

    def test_import_array_loader(self) -> None:
        """Test that array_loader module can be imported."""
        from cslrtools2.sldataset import array_loader

        assert array_loader is not None


class TestSLDatasetStructure:
    """Test basic SLDataset class structure."""

    def test_sldataset_is_class(self) -> None:
        """Test that SLDataset is a class."""
        from cslrtools2.sldataset.dataset import SLDataset

        assert isinstance(SLDataset, type)

    def test_sldatasetitem_is_class(self) -> None:
        """Test that SLDatasetItem is a class."""
        from cslrtools2.sldataset.dataset import SLDatasetItem

        assert isinstance(SLDatasetItem, type)


# NOTE: Full dataset functionality tests (creating/loading Zarr datasets)
# require test data and are deferred to integration tests in tests/integration/
# or similar location.
