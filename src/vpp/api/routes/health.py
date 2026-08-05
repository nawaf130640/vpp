"""Health and readiness endpoint.

A trivial route that confirms the app booted and surfaces the active device
and service selection. Useful as the very first end-to-end check and later as
a liveness probe.
"""

from __future__ import annotations

from fastapi import APIRouter

from vpp import __version__
from vpp.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    """Return basic service status and the active configuration summary."""
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "device": settings.device,
        "services": settings.services.model_dump(),
    }
