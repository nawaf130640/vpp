"""Unit tests for the model contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vpp.core.contracts import Detector, Service
from vpp.core.types import BoundingBox, Detection, Frame


class _StubDetector(Detector):
    name = "stub"

    def detect(self, frame: Frame, prompts: Sequence[str]) -> list[Detection]:
        box = BoundingBox(0, 0, float(frame.width), float(frame.height))
        return [Detection(box, label=p, score=1.0) for p in prompts]


def _frame():
    return Frame(0, 0.0, np.zeros((8, 8, 3), dtype=np.uint8))


def test_contract_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Detector()


def test_stub_detector_implements_contract():
    det = _StubDetector()
    results = det.detect(_frame(), ["wall", "table"])
    assert [d.label for d in results] == ["wall", "table"]
    assert results[0].box.to_int() == (0, 0, 8, 8)


def test_service_lifecycle_flag():
    det = _StubDetector()
    assert det.is_loaded is False
    det.load()
    assert det.is_loaded is True
    det.unload()
    assert det.is_loaded is False


def test_service_context_manager_loads_and_unloads():
    det = _StubDetector()
    with det as active:
        assert isinstance(active, Service)
        assert det.is_loaded is True
    assert det.is_loaded is False
