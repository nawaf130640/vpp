"""Core data types — the vocabulary of the pipeline.

Framework-light domain objects: NumPy only, never torch/OpenCV. Array
conventions: colour images (H,W,3) uint8 BGR; alpha images (H,W,4) uint8 RGBA;
masks (H,W) bool; depth (H,W) float32; pixel coords x=column, y=row, origin
top-left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

ImageBGR = np.ndarray
ImageRGBA = np.ndarray
BoolMask = np.ndarray
DepthArray = np.ndarray


@dataclass(frozen=True, slots=True)
class Point2D:
    """A pixel coordinate. x is the column, y is the row."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates, corners inclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid box: {self.as_xyxy()}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """A single decoded video frame plus its position in time."""

    index: int
    timestamp: float
    image: ImageBGR

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Frame.image must be (H, W, 3); got {self.image.shape}")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected region: a box, a text label, and a confidence in [0, 1]."""

    box: BoundingBox
    label: str
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Detection.score must be in [0, 1]; got {self.score}")


@dataclass(frozen=True, slots=True, eq=False)
class Mask:
    """A binary segmentation mask over a frame."""

    data: BoolMask
    score: float | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"Mask.data must be 2-D; got {self.data.shape}")
        if self.data.dtype != np.bool_:
            raise ValueError(f"Mask.data must be bool; got {self.data.dtype}")

    @property
    def area(self) -> int:
        return int(self.data.sum())

    @property
    def bounding_box(self) -> BoundingBox | None:
        ys, xs = np.where(self.data)
        if ys.size == 0:
            return None
        return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


@dataclass(frozen=True, slots=True, eq=False)
class DepthMap:
    """A per-pixel relative depth map for one frame."""

    data: DepthArray

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"DepthMap.data must be 2-D; got {self.data.shape}")

    def normalized(self) -> np.ndarray:
        d = self.data.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d)
        return (d - lo) / (hi - lo)


@dataclass(frozen=True, slots=True, eq=False)
class Quad:
    """A quadrilateral placement region: four corners in TL, TR, BR, BL order."""

    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.top_left.as_tuple(),
                self.top_right.as_tuple(),
                self.bottom_right.as_tuple(),
                self.bottom_left.as_tuple(),
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, corners: np.ndarray) -> "Quad":
        if corners.shape != (4, 2):
            raise ValueError(f"Quad expects (4, 2); got {corners.shape}")
        tl, tr, br, bl = (Point2D(float(x), float(y)) for x, y in corners)
        return cls(tl, tr, br, bl)

    @classmethod
    def from_box(cls, box: BoundingBox) -> "Quad":
        return cls(
            top_left=Point2D(box.x1, box.y1),
            top_right=Point2D(box.x2, box.y1),
            bottom_right=Point2D(box.x2, box.y2),
            bottom_left=Point2D(box.x1, box.y2),
        )


@dataclass(frozen=True, slots=True)
class Shot:
    """A continuous camera take. end_frame is inclusive."""

    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(f"Shot {self.index}: end < start")

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True, slots=True, eq=False)
class ProductAsset:
    """The logo or product PNG to insert, carrying its alpha channel."""

    image: ImageRGBA
    name: str = "product"

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 4:
            raise ValueError(f"ProductAsset.image must be (H, W, 4); got {self.image.shape}")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    """How the caller wants a placement chosen. MVP supports only 'auto'."""

    mode: Literal["auto"] = "auto"
    target_prompts: tuple[str, ...] = (
        "wall",
        "table",
        "floor",
        "billboard",
        "tv screen",
        "poster",
    )


@dataclass(frozen=True, slots=True, eq=False)
class PlacementPlan:
    """The chosen placement on a shot's reference frame."""

    shot_index: int
    reference_frame: int
    quad: Quad
    label: str
    score: float
    mask: Mask | None = None
    normal: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Track:
    """Raw point trajectories: points (T,N,2) float32, visibility (T,N) bool."""

    start_frame: int
    points: np.ndarray
    visibility: np.ndarray

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[2] != 2:
            raise ValueError(f"Track.points must be (T, N, 2); got {self.points.shape}")
        if self.visibility.shape != self.points.shape[:2]:
            raise ValueError("Track.visibility must match points[:2]")

    @property
    def num_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[1])


@dataclass(frozen=True, slots=True, eq=False)
class PlacementTrack:
    """The placement quad resolved for every frame of a shot."""

    start_frame: int
    quads: tuple[Quad, ...]
    visibility: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.quads) != len(self.visibility):
            raise ValueError("quads and visibility must be the same length")

    @property
    def num_frames(self) -> int:
        return len(self.quads)

    def quad_at(self, frame_index: int) -> Quad | None:
        offset = frame_index - self.start_frame
        if offset < 0 or offset >= len(self.quads):
            return None
        return self.quads[offset] if self.visibility[offset] else None


ServiceKey = str
ClassVarStr = ClassVar[str]
