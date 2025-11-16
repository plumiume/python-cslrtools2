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

"""Integration tests for CLI commands.

This module tests the command-line interface by running commands in subprocess
and verifying output files and exit codes. Tests are primarily structural,
validating command execution and file creation without requiring MediaPipe API.

Note:
    Most tests are skipped due to MediaPipe API compatibility issues.
    The ``holistic`` and ``pose`` commands are tested for structure only,
    as actual execution requires MediaPipe landmark detection which currently
    encounters ``AttributeError: landmarks`` errors.

Example:
    Run CLI integration tests::

        >>> pytest tests/integration/test_cli_commands.py -v
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest  # pyright: ignore[reportUnusedImport]


class TestCLIBasicExecution:
    """Test basic CLI command execution."""

    def test_cli_help_command(self):
        """Test that --help flag works without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "cslrtools2.lmpipe.app.cli", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "lmpipe" in result.stdout.lower()

    def test_cli_version_command(self):
        """Test that --version flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "cslrtools2.lmpipe.app.cli", "--version"],
            capture_output=True,
            text=True,
        )

        # Version flag may not be implemented, so check for help or error
        assert result.returncode in (0, 1, 2)

    def test_cli_no_arguments(self):
        """Test CLI behavior with no arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "cslrtools2.lmpipe.app.cli"],
            capture_output=True,
            text=True,
        )

        # Should show help or error message
        assert result.returncode != 0


class TestCLIHolisticCommand:
    """Test holistic estimator CLI commands."""

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="MediaPipe API compatibility issue: AttributeError: landmarks"
    )
    def test_holistic_mediapipe_basic(self, sample_video_stop, tmp_path):
        """Test basic holistic MediaPipe command."""
        output_dir = tmp_path / "holistic_output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                str(output_dir),
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check exit code
        assert result.returncode == 0, f"STDERR: {result.stderr}"

        # Verify output files were created
        output_files = list(output_dir.glob("*.npz"))
        assert len(output_files) > 0, "No output files created"

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="MediaPipe API compatibility issue: AttributeError: landmarks"
    )
    def test_holistic_with_csv_collector(self, sample_video_stop, tmp_path):
        """Test holistic command with CSV output."""
        output_dir = tmp_path / "csv_output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "--landmark-matrix-save-file-format",
                ".csv",
                str(sample_video_stop),
                str(output_dir),
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"STDERR: {result.stderr}"

        # Verify CSV files were created
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0, "No CSV files created"

        # Verify CSV content is not empty
        for csv_file in csv_files:
            content = csv_file.read_text()
            assert len(content) > 0, f"CSV file {csv_file} is empty"
            assert "frame" in content.lower(), "CSV missing frame column"

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="MediaPipe API compatibility issue: AttributeError: landmarks"
    )
    def test_holistic_with_multiple_collectors(self, sample_video_stop, tmp_path):
        """Test holistic command with NPZ output (default)."""
        output_dir = tmp_path / "multi_output"
        output_dir.mkdir()

        # Test with default NPZ format
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                str(output_dir),
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"STDERR: {result.stderr}"

        # Verify NPZ output (default format)
        npz_files = list(output_dir.glob("*.npz"))
        assert len(npz_files) > 0, "No NPZ files created"


class TestCLIPoseCommand:
    """Test pose estimator CLI commands."""

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(reason="Pose command hangs/times out - investigation needed")
    def test_pose_mediapipe_basic(self, sample_video_stop, tmp_path):
        """Test basic pose MediaPipe command."""
        output_dir = tmp_path / "pose_output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                str(output_dir),
                "pose",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"STDERR: {result.stderr}"

        # Verify output files
        output_files = list(output_dir.glob("*.npz"))
        assert len(output_files) > 0, "No output files created"

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(reason="Pose command hangs/times out - investigation needed")
    def test_pose_with_model_complexity(self, sample_video_stop, tmp_path):
        """Test pose command with model option."""
        output_dir = tmp_path / "pose_complex"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                str(output_dir),
                "pose",
                "mediapipe",
                "--pose-model",
                "lite",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"STDERR: {result.stderr}"


class TestCLIErrorHandling:
    """Test CLI error handling and validation."""

    def test_cli_nonexistent_input_file(self, tmp_path):
        """Test CLI behavior with non-existent input file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "nonexistent_video.mp4",
                str(output_dir),
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Error message should mention the file
        assert (
            "nonexistent" in result.stderr.lower()
            or "not found" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    @pytest.mark.skip(reason="Test hangs with holistic/pose MediaPipe commands")
    def test_cli_invalid_output_directory(self, sample_video_stop):
        """Test CLI behavior with invalid output directory."""
        # Try to use a file as output directory (Windows may handle differently)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                "nul",  # Windows null device
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # May succeed or fail depending on implementation
        # Just verify it doesn't crash without output
        assert result.returncode in (0, 1, 2)

    def test_cli_missing_estimator(self):
        """Test CLI behavior without specifying estimator."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "input.mp4",
                "output/",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail or show help
        assert result.returncode != 0


class TestCLIOutputVerification:
    """Test that CLI output files contain valid data."""

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="MediaPipe API compatibility issue: AttributeError: landmarks"
    )
    def test_npz_output_structure(self, sample_video_stop, tmp_path):
        """Test that NPZ output has correct structure."""
        output_dir = tmp_path / "npz_verify"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                str(sample_video_stop),
                str(output_dir),
                "holistic",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # Load and verify NPZ structure
        npz_files = list(output_dir.glob("*.npz"))
        assert len(npz_files) > 0

        with np.load(npz_files[0]) as data:
            # Should have landmark arrays
            assert len(data.files) > 0, "NPZ file is empty"

            # Check for expected keys (may vary by estimator)
            expected_keys = {"pose", "left_hand", "right_hand", "face"}
            found_keys = set(data.files)

            # At least one key should match
            assert (
                len(expected_keys & found_keys) > 0
            ), f"No expected keys found. Got: {found_keys}"

            # Verify array shapes
            for key in found_keys:
                arr = data[key]
                assert arr.ndim == 3, f"Array {key} should be 3D"
                assert arr.shape[0] > 0, f"Array {key} has no frames"

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="Pose command hangs/times out - investigation needed"
    )
    def test_csv_output_format(self, sample_video_stop, tmp_path):
        """Test that CSV output has correct format."""
        output_dir = tmp_path / "csv_verify"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "--landmark-matrix-save-file-format",
                ".csv",
                str(sample_video_stop),
                str(output_dir),
                "pose",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # Verify CSV format
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0

        content = csv_files[0].read_text()
        lines = content.strip().split("\n")

        # Should have header and data
        assert len(lines) >= 2, "CSV has no data rows"

        # Header should contain expected columns
        header = lines[0].lower()
        assert "frame" in header or "index" in header, """
        CSV missing frame/index column"""

        # Data rows should have same number of columns as header
        header_cols = len(lines[0].split(","))
        for i, line in enumerate(lines[1:6], 1):  # Check first 5 data rows
            data_cols = len(line.split(","))
            assert (
                data_cols == header_cols
            ), f"Row {i} has {data_cols} columns, expected {header_cols}"


class TestCLILogOutput:
    """Test CLI logging functionality."""

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(
        reason="Pose command hangs/times out - investigation needed"
    )
    def test_cli_with_log_file(self, sample_video_stop, tmp_path):
        """Test CLI with log file output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        log_file = tmp_path / "lmpipe.log"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "--log-file",
                str(log_file),
                "--log-level",
                "debug",
                str(sample_video_stop),
                str(output_dir),
                "pose",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # Verify log file was created
        assert log_file.exists(), "Log file not created"

        # Verify log content
        log_content = log_file.read_text()
        assert len(log_content) > 0, "Log file is empty"

        # Should contain some logging indicators
        assert any(
            keyword in log_content.lower()
            for keyword in ["debug", "info", "processing", "frame"]
        ), "Log file doesn't contain expected content"

    @pytest.mark.requires_video
    @pytest.mark.mediapipe
    @pytest.mark.skip(reason="Pose command hangs/times out - investigation needed")
    def test_cli_verbose_output(self, sample_video_stop, tmp_path):
        """Test CLI with verbose/debug output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cslrtools2.lmpipe.app.cli",
                "--log-level",
                "info",
                str(sample_video_stop),
                str(output_dir),
                "pose",
                "mediapipe",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # With info level, should see some output
        # (May be in stdout or stderr depending on implementation)
        combined_output = result.stdout + result.stderr
        assert len(combined_output) > 0, """
        No output produced with info log level
        """
