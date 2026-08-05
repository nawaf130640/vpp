"""Unit tests for the core data types."""

from __future__ import annotations

import numpy as np
import pytest

from vpp.core.types import (
    BoundingBox, Detection, DepthMap, Frame, Mask, PlacementSpec,
    PlacementTrack, Point2D, ProductAsset, Quad, Shot, Track,
)


def _blank_bgr(h=4, w=6):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_bounding_box_geometry():
    box = BoundingBox(10, 20, 40, 60)
    assert box.width == 30 and box.height == 40 and box.area == 1200
    assert box.center == Point2D(25, 40)
    assert box.to_int() == (10, 20, 40, 60)


def test_bounding_box_rejects_inverted():
    with pytest.raises(ValueError):
        BoundingBox(40, 0, 10, 10)


def test_detection_score_bounds():
    Detection(BoundingBox(0, 0, 1, 1), "wall", 0.9)
    with pytest.raises(ValueError):
        Detection(BoundingBox(0, 0, 1, 1), "wall", 1.5)


def test_frame_shape_and_props():
    frame = Frame(index=3, timestamp=0.1, image=_blank_bgr(4, 6))
    assert frame.height == 4 and frame.width == 6 and frame.size == (6, 4)
    with pytest.raises(ValueError):
        Frame(0, 0.0, np.zeros((4, 6), dtype=np.uint8))


def test_mask_area_and_bbox():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:8] = True
    mask = Mask(data)
    assert mask.area == 15
    assert mask.bounding_box.to_int() == (3, 2, 7, 4)


def test_empty_mask_has_no_bbox():
    assert Mask(np.zeros((5, 5), dtype=bool)).bounding_box is None


def test_mask_rejects_non_bool():
    with pytest.raises(ValueError):
        Mask(np.zeros((5, 5), dtype=np.uint8))


def test_depth_normalized_range():
    depth = DepthMap(np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32))
    norm = depth.normalized()
    assert float(norm.min()) == 0.0 and float(norm.max()) == 1.0


def test_quad_roundtrip_and_from_box():
    box = BoundingBox(1, 2, 5, 8)
    quad = Quad.from_box(box)
    arr = quad.as_array()
    assert arr.shape == (4, 2)
    assert tuple(arr[0]) == (1, 2) and tuple(arr[2]) == (5, 8)
    assert Quad.from_array(arr).as_array().tolist() == arr.tolist()


def test_shot_num_frames_and_validation():
    shot = Shot(0, 10, 19, 0.4, 0.8)
    assert shot.num_frames == 10
    with pytest.raises(ValueError):
        Shot(1, 20, 10, 0.0, 0.0)


def test_product_asset_requires_alpha():
    ProductAsset(np.zeros((3, 3, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        ProductAsset(np.zeros((3, 3, 3), dtype=np.uint8))


def test_placement_spec_default_is_auto():
    spec = PlacementSpec()
    assert spec.mode == "auto" and "wall" in spec.target_prompts


def test_track_shape_validation():
    pts = np.zeros((5, 4, 2), dtype=np.float32)
    vis = np.ones((5, 4), dtype=bool)
    track = Track(0, pts, vis)
    assert track.num_frames == 5 and track.num_points == 4
    with pytest.raises(ValueError):
        Track(0, pts, np.ones((5, 3), dtype=bool))


def test_placement_track_quad_lookup():
    q = Quad.from_box(BoundingBox(0, 0, 2, 2))
    track = PlacementTrack(10, (q, q, q), (True, False, True))
    assert track.num_frames == 3
    assert track.quad_at(10) is q
    assert track.quad_at(11) is None
    assert track.quad_at(99) is None
