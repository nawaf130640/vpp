"""Phase 0 smoke test: the app boots and /health responds."""

from fastapi.testclient import TestClient

from vpp.main import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}
    assert set(body["services"]) == {
        "detector", "segmenter", "depth", "tracker", "renderer"
    }
