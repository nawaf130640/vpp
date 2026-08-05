"""Model contracts — the interfaces every model must satisfy.

The pipeline depends only on these ABCs, never on a concrete model. A real
model and a fake stub both implement the same contract, so swapping is a config
change, not a code change. Every contract extends Service (load/unload).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import ClassVar

import numpy as np

from vpp.core.types import (
    BoundingBox,
    Detection,
    DepthMap,
    Frame,
    Mask,
    PlacementTrack,
    ProductAsset,
    Track,
)


class Service(ABC):
    """Base for every model-backed service, with a load/unload lifecycle."""

    name: ClassVar[str] = "service"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def __enter__(self) -> "Service":
        if not self._loaded:
            self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.unload()


class Detector(Service, ABC):
    """Finds candidate regions in a frame, prompted by text labels."""

    name: ClassVar[str] = "detector"

    @abstractmethod
    def detect(self, frame: Frame, prompts: Sequence[str]) -> list[Detection]:
        ...


class Segmenter(Service, ABC):
    """Produces a precise mask for a region of a frame."""

    name: ClassVar[str] = "segmenter"

    @abstractmethod
    def segment(self, frame: Frame, box: BoundingBox) -> Mask:
        ...


class DepthEstimator(Service, ABC):
    """Estimates per-pixel relative depth for a frame."""

    name: ClassVar[str] = "depth"

    @abstractmethod
    def estimate(self, frame: Frame) -> DepthMap:
        ...


class Tracker(Service, ABC):
    """Follows a set of query points across a sequence of frames."""

    name: ClassVar[str] = "tracker"

    @abstractmethod
    def track(self, frames: Sequence[Frame], queries: np.ndarray) -> Track:
        ...


class Renderer(Service, ABC):
    """Composites the product into each frame according to a placement track."""

    name: ClassVar[str] = "renderer"

    @abstractmethod
    def render(
        self,
        frames: Sequence[Frame],
        track: PlacementTrack,
        product: ProductAsset,
    ) -> list[Frame]:
        ...
