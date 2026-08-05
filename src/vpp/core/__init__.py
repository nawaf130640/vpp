"""VPP core domain: data types, model contracts, config, and errors."""

from __future__ import annotations

from vpp.core.contracts import (
    DepthEstimator,
    Detector,
    Renderer,
    Segmenter,
    Service,
    Tracker,
)
from vpp.core.exceptions import (
    ConfigError,
    InferenceError,
    ModelError,
    ModelLoadError,
    NoSurfaceFoundError,
    PipelineError,
    PlacementError,
    VideoIOError,
    VPPError,
)
from vpp.core.types import (
    BoundingBox,
    Detection,
    DepthMap,
    Frame,
    Mask,
    PlacementPlan,
    PlacementSpec,
    PlacementTrack,
    Point2D,
    ProductAsset,
    Quad,
    Shot,
    Track,
)

__all__ = [
    "Service", "Detector", "Segmenter", "DepthEstimator", "Tracker", "Renderer",
    "VPPError", "ConfigError", "VideoIOError", "ModelError", "ModelLoadError",
    "InferenceError", "PipelineError", "NoSurfaceFoundError", "PlacementError",
    "Frame", "Point2D", "BoundingBox", "Detection", "Mask", "DepthMap", "Quad",
    "Shot", "ProductAsset", "PlacementSpec", "PlacementPlan", "Track", "PlacementTrack",
]
