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

"""Custom exceptions for cslrtools2.

This module defines the exception hierarchy for cslrtools2, providing
better error classification and handling capabilities.

Exception Hierarchy::

    CSLRToolsError (base)
    ├── ConfigurationError
    ├── ValidationError
    ├── LMPipeError
    │   ├── EstimatorError
    │   ├── CollectorError
    │   ├── VideoProcessingError
    │   └── ModelDownloadError
    └── SLDatasetError
        ├── DataLoadError
        ├── DataFormatError
        └── PluginError

Example:
    Catch all cslrtools2 errors::

        >>> from cslrtools2.exceptions import CSLRToolsError
        >>> try:
        ...     # Some operation
        ...     pass
        ... except CSLRToolsError as e:
        ...     print(f"Error: {e}")

    Use specific exceptions::

        >>> from cslrtools2.exceptions import ValidationError
        >>> if value < 0:
        ...     raise ValidationError(f"Expected positive value, got {value}")
"""

__all__ = [
    "CSLRToolsError",
    "ConfigurationError",
    "ValidationError",
    "LMPipeError",
    "EstimatorError",
    "CollectorError",
    "VideoProcessingError",
    "ModelDownloadError",
    "SLDatasetError",
    "DataLoadError",
    "DataFormatError",
    "PluginError",
]


class CSLRToolsError(Exception):
    """Base exception for all cslrtools2 errors.
    
    All custom exceptions in cslrtools2 inherit from this class.
    This allows users to catch all cslrtools2-specific errors with
    a single except clause.
    
    Example:
        >>> try:
        ...     # Some cslrtools2 operation
        ...     pass
        ... except CSLRToolsError as e:
        ...     print(f"cslrtools2 error: {e}")
    """
    pass


# ============================================================================
# Common Exceptions
# ============================================================================

class ConfigurationError(CSLRToolsError):
    """Raised when configuration is invalid or inconsistent.
    
    This includes:
    - Invalid option combinations
    - Missing required configuration
    - Malformed configuration files
    
    Example:
        >>> raise ConfigurationError(
        ...     "Invalid estimator configuration: missing model_path"
        ... )
    """
    pass


class ValidationError(CSLRToolsError):
    """Raised when input validation fails.
    
    This includes:
    - Invalid argument values
    - Type mismatches
    - Out-of-range values
    
    Example:
        >>> raise ValidationError(
        ...     f"Expected positive integer, got {value}"
        ... )
    """
    pass


# ============================================================================
# LMPipe Exceptions
# ============================================================================

class LMPipeError(CSLRToolsError):
    """Base exception for landmark pipeline errors.
    
    All lmpipe-specific exceptions inherit from this class.
    """
    pass


class EstimatorError(LMPipeError):
    """Raised when landmark estimation fails.
    
    This includes:
    - Model initialization failures
    - Estimation computation errors
    - Invalid estimator state
    
    Example:
        >>> raise EstimatorError(
        ...     "MediaPipe model initialization failed: invalid model file"
        ... )
    """
    pass


class CollectorError(LMPipeError):
    """Raised when result collection fails.
    
    This includes:
    - Output file write errors
    - Format conversion failures
    - Collector initialization errors
    
    Example:
        >>> raise CollectorError(
        ...     f"Failed to write output to {path}: {reason}"
        ... )
    """
    pass


class VideoProcessingError(LMPipeError):
    """Raised when video processing fails.
    
    This includes:
    - Cannot open video file
    - Video decode errors
    - Frame extraction failures
    
    Example:
        >>> raise VideoProcessingError(
        ...     f"Cannot open video file: {path}. "
        ...     f"Ensure the file exists and is a valid video format."
        ... )
    """
    pass


class ModelDownloadError(LMPipeError):
    """Raised when model download fails.
    
    This includes:
    - Network errors
    - HTTP errors
    - File write errors
    
    Example:
        >>> raise ModelDownloadError(
        ...     f"Failed to download model from {url}. "
        ...     f"Status code: {status_code}. "
        ...     f"Ensure you have internet connectivity."
        ... )
    """
    pass


# ============================================================================
# SLDataset Exceptions
# ============================================================================

class SLDatasetError(CSLRToolsError):
    """Base exception for dataset errors.
    
    All sldataset-specific exceptions inherit from this class.
    """
    pass


class DataLoadError(SLDatasetError):
    """Raised when data loading fails.
    
    This includes:
    - File not found
    - Data key not found
    - File format errors
    
    Example:
        >>> raise DataLoadError(
        ...     f"Failed to load array from {path}: {reason}"
        ... )
    """
    pass


class DataFormatError(SLDatasetError):
    """Raised when data format is unexpected.
    
    This includes:
    - Unexpected data types
    - Invalid data shapes
    - Missing required fields
    
    Example:
        >>> raise DataFormatError(
        ...     f"Expected Tensor in file {path}, got {type(data).__name__}. "
        ...     f"Ensure the file was saved with torch.save(tensor, path)."
        ... )
    """
    pass


class PluginError(SLDatasetError):
    """Raised when plugin loading or execution fails.
    
    This includes:
    - Invalid plugin structure
    - Plugin load failures
    - Plugin execution errors
    
    Example:
        >>> raise PluginError(
        ...     f"Plugin entry point {name} does not return a valid processor. "
        ...     f"Expected a callable, got {type(processor).__name__}."
        ... )
    """
    pass
