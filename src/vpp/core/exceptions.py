"""Exception hierarchy for the VPP system.

A single, shallow hierarchy rooted at VPPError so callers can catch broadly
(except VPPError) or narrowly (except NoSurfaceFoundError).
"""

from __future__ import annotations


class VPPError(Exception):
    """Base class for all errors raised deliberately by this application."""


class ConfigError(VPPError):
    """Configuration is missing, malformed, or internally inconsistent."""


class VideoIOError(VPPError):
    """A video file could not be read, decoded, written, or encoded."""


class ModelError(VPPError):
    """Base class for failures originating in a model-backed service."""


class ModelLoadError(ModelError):
    """A model's weights or runtime could not be loaded (e.g. onto the GPU)."""


class InferenceError(ModelError):
    """A model ran but failed to produce a usable result."""


class PipelineError(VPPError):
    """A pipeline stage failed or received input it cannot process."""


class NoSurfaceFoundError(PipelineError):
    """No usable placement surface was found in a shot."""


class PlacementError(PipelineError):
    """A placement could not be computed, tracked, or rendered."""
