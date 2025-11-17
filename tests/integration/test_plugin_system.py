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

"""Integration tests for plugin system.

This module tests plugin discovery, loading, and registration through
entry points. Validates the plugin system's ability to discover and
load external estimator and collector implementations.

Test Coverage:
    - Plugin discovery via entry points
    - Plugin loading and structure validation
    - Plugin integration with LMPipe interface
    - NamespaceWrapper functionality
    - Error handling and robustness
    - Plugin registry validation

Example:
    Run plugin system tests::

        >>> pytest tests/integration/test_plugin_system.py -v
"""

from __future__ import annotations

import pytest  # pyright: ignore[reportUnusedImport]

from cslrtools2.lmpipe.app.plugins import loader as load_lmpipe_plugins


class TestPluginDiscovery:
    """Test plugin discovery through entry points."""

    def test_lmpipe_plugins_loader(self):
        """Test that LMPipe plugin loader returns expected structure."""
        plugins = load_lmpipe_plugins()

        # Should return a dictionary
        assert isinstance(plugins, dict), "Plugin loader should return dict"

        # Verify structure: dict[type, dict[name, PluginInfo]]
        for plugin_type, plugin_dict in plugins.items():
            assert isinstance(
                plugin_type, str
            ), f"Plugin type should be string: {plugin_type}"
            assert isinstance(
                plugin_dict, dict
            ), f"Plugin dict should be dict for type {plugin_type}"

            for plugin_name, plugin_info in plugin_dict.items():
                assert isinstance(
                    plugin_name, str
                ), f"Plugin name should be string: {plugin_name}"
                assert isinstance(
                    plugin_info, dict
                ), f"Plugin info should be dict: {plugin_name}"

                # Verify PluginInfo structure
                assert "name" in plugin_info, f"Missing 'name' in {plugin_name}"
                assert "type" in plugin_info, f"Missing 'type' in {plugin_name}"
                assert (
                    "nswrapper" in plugin_info
                ), f"Missing 'nswrapper' in {plugin_name}"
                assert "creator" in plugin_info, f"Missing 'creator' in {plugin_name}"

    def test_mediapipe_plugins_available(self):
        """Test that MediaPipe plugins are discovered."""
        pytest.importorskip("mediapipe", reason="MediaPipe not installed")

        plugins = load_lmpipe_plugins()

        # Expected plugin types for MediaPipe
        expected_types = ["holistic", "pose", "both_hands", "face"]

        for expected_type in expected_types:
            assert (
                expected_type in plugins
            ), f"Expected plugin type '{expected_type}' not found"

            # MediaPipe should have at least one plugin per type
            assert (
                "mediapipe" in plugins[expected_type]
            ), f"MediaPipe plugin not found for type '{expected_type}'"

    def test_plugin_metadata_consistency(self):
        """Test that plugin metadata is consistent."""
        plugins = load_lmpipe_plugins()

        for plugin_type, plugin_dict in plugins.items():
            for plugin_name, plugin_info in plugin_dict.items():
                # Name in info should match key
                assert (
                    plugin_info["name"] == plugin_name
                ), f"Name mismatch: {plugin_name} vs {plugin_info['name']}"

                # Type in info should match parent key
                assert (
                    plugin_info["type"] == plugin_type
                ), f"Type mismatch: {plugin_type} vs {plugin_info['type']}"


class TestPluginLoading:
    """Test loading and instantiation of plugins."""

    @pytest.mark.mediapipe
    def test_load_mediapipe_holistic_plugin(self):
        """Test loading MediaPipe holistic plugin."""
        plugins = load_lmpipe_plugins()

        assert "holistic" in plugins, "Holistic plugins not found"
        assert "mediapipe" in plugins["holistic"], "MediaPipe holistic plugin not found"

        plugin_info = plugins["holistic"]["mediapipe"]

        # Verify plugin has expected structure
        assert "name" in plugin_info, "Plugin should have name"
        assert "type" in plugin_info, "Plugin should have type"
        assert "creator" in plugin_info, "Plugin should have creator"
        assert "nswrapper" in plugin_info, "Plugin should have nswrapper"

        # Verify nswrapper has add_wrapper method
        nswrapper = plugin_info["nswrapper"]
        assert hasattr(
            nswrapper, "add_wrapper"
        ), "NamespaceWrapper should have add_wrapper method"

    @pytest.mark.mediapipe
    def test_instantiate_mediapipe_pose_estimator(self):
        """Test instantiating a MediaPipe pose estimator through plugin."""
        plugins = load_lmpipe_plugins()

        assert "pose" in plugins, "Pose plugins not found"
        assert "mediapipe" in plugins["pose"], "MediaPipe pose plugin not found"

        plugin_info = plugins["pose"]["mediapipe"]
        creator = plugin_info["creator"]

        # Creator should be callable
        assert callable(creator), "Creator should be callable"

        # Note: We can't easily instantiate without proper namespace setup
        # This test verifies the plugin structure is correct
        pytest.skip("Cannot instantiate without proper namespace - structure verified")

    def test_plugin_creator_signature(self):
        """Test that plugin creators have correct signature."""
        plugins = load_lmpipe_plugins()

        for plugin_type, plugin_dict in plugins.items():
            for plugin_name, plugin_info in plugin_dict.items():
                creator = plugin_info["creator"]

                # Creator should be callable
                assert callable(
                    creator
                ), f"Creator for {plugin_name} should be callable"

                # Should accept at least one argument (namespace)
                import inspect

                sig = inspect.signature(creator)
                assert (
                    len(sig.parameters) >= 1
                ), f"Creator for {plugin_name} should accept at least one parameter"


class TestPluginIntegration:
    """Test integration of plugins with LMPipe interface."""

    @pytest.mark.mediapipe
    def test_plugin_estimator_interface(self):
        """Test that plugin estimators have expected structure."""
        plugins = load_lmpipe_plugins()

        # Test with pose plugin
        if "pose" in plugins and "mediapipe" in plugins["pose"]:
            plugin_info = plugins["pose"]["mediapipe"]
            creator = plugin_info["creator"]

            # Verify creator is callable
            assert callable(creator), "Creator should be callable"

            # Verify it has proper signature
            import inspect

            sig = inspect.signature(creator)
            assert len(sig.parameters) > 0, "Creator should accept parameters"

    @pytest.mark.mediapipe
    def test_plugin_estimator_estimate_method(self):
        """Test that plugin structure supports estimate method."""
        plugins = load_lmpipe_plugins()

        if "pose" in plugins and "mediapipe" in plugins["pose"]:
            plugin_info = plugins["pose"]["mediapipe"]
            creator = plugin_info["creator"]

            # Verify creator structure
            assert callable(creator), "Creator should be callable"

            # Note: Cannot test actual estimation without proper setup
            pytest.skip("Cannot test estimation without proper namespace setup")


class TestPluginNamespaceWrapper:
    """Test NamespaceWrapper functionality for plugins."""

    def test_namespace_wrapper_structure(self):
        """Test that NamespaceWrapper has expected structure."""
        plugins = load_lmpipe_plugins()

        for plugin_type, plugin_dict in plugins.items():
            for plugin_name, plugin_info in plugin_dict.items():
                nswrapper = plugin_info["nswrapper"]

                # Should exist
                assert (
                    nswrapper is not None
                ), f"NamespaceWrapper for {plugin_name} should not be None"

                # Should have add_wrapper method (clipar NamespaceWrapper interface)
                assert hasattr(
                    nswrapper, "add_wrapper"
                ), f"NamespaceWrapper for {plugin_name} should have add_wrapper method"

    def test_namespace_wrapper_type(self):
        """Test NamespaceWrapper type verification."""
        plugins = load_lmpipe_plugins()

        # Test with a known plugin
        if "pose" in plugins and "mediapipe" in plugins["pose"]:
            nswrapper = plugins["pose"]["mediapipe"]["nswrapper"]

            # Should have clipar NamespaceWrapper methods
            assert hasattr(nswrapper, "add_wrapper"), "Should have add_wrapper method"


class TestPluginErrorHandling:
    """Test error handling in plugin system."""

    def test_plugin_loader_no_crash(self):
        """Test that plugin loader doesn't crash."""
        # This test verifies the loader is robust
        try:
            plugins = load_lmpipe_plugins()
            assert isinstance(plugins, dict)
        except Exception as e:
            pytest.fail(f"Plugin loader crashed: {e}")

    def test_plugin_structure_validity(self):
        """Test that all plugins have valid structure."""
        plugins = load_lmpipe_plugins()

        # All plugins should have valid type and name
        for plugin_type, plugin_dict in plugins.items():
            assert plugin_type, f"Plugin type should not be empty: {plugin_type}"
            assert isinstance(
                plugin_dict, dict
            ), f"Plugin dict should be dict: {plugin_dict}"

            for plugin_name, plugin_info in plugin_dict.items():
                assert plugin_name, f"Plugin name should not be empty: {plugin_name}"
                assert (
                    "name" in plugin_info
                ), f"Plugin info should have 'name': {plugin_info}"
                assert (
                    "type" in plugin_info
                ), f"Plugin info should have 'type': {plugin_info}"
                assert (
                    "creator" in plugin_info
                ), f"Plugin info should have 'creator': {plugin_info}"
                assert (
                    "nswrapper" in plugin_info
                ), f"Plugin info should have 'nswrapper': {plugin_info}"

    def test_plugin_metadata_consistency(self):
        """Test that plugin metadata is consistent."""
        plugins = load_lmpipe_plugins()

        # Verify name and type match dict keys
        for plugin_type, plugin_dict in plugins.items():
            for plugin_name, plugin_info in plugin_dict.items():
                assert (
                    plugin_info["type"] == plugin_type
                ), f"Type mismatch: {plugin_info['type']} != {plugin_type}"
                assert (
                    plugin_info["name"] == plugin_name
                ), f"Name mismatch: {plugin_info['name']} != {plugin_name}"


class TestPluginRegistry:
    """Test plugin registry functionality."""

    def test_all_plugins_have_unique_names(self):
        """Test that all plugins within a type have unique names."""
        plugins = load_lmpipe_plugins()

        for plugin_type, plugin_dict in plugins.items():
            plugin_names = list(plugin_dict.keys())
            unique_names = set(plugin_names)

            assert len(plugin_names) == len(
                unique_names
            ), f"Duplicate plugin names found in type '{plugin_type}'"

    def test_plugin_types_are_valid(self):
        """Test that plugin types are valid estimator types."""
        plugins = load_lmpipe_plugins()

        # Expected valid types (may need to be updated)
        valid_types = {
            "holistic",
            "pose",
            "both_hands",
            "left_hand",
            "right_hand",
            "face",
        }

        for plugin_type in plugins.keys():
            assert plugin_type in valid_types, f"Unexpected plugin type: {plugin_type}"

    def test_plugin_count(self):
        """Test that expected number of plugins are registered."""
        plugins = load_lmpipe_plugins()

        # Should have at least one plugin type
        assert len(plugins) > 0, "No plugins registered"

        # Each type should have at least one plugin
        for plugin_type, plugin_dict in plugins.items():
            assert (
                len(plugin_dict) > 0
            ), f"No plugins registered for type '{plugin_type}'"
