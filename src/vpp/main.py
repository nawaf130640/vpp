"""Application entrypoint.

Exposes ``create_app()`` (an app factory, so tests can build isolated
instances) and a module-level ``app`` for ``uvicorn vpp.main:app``. At this
phase the app only mounts the health route; pipeline and job routes arrive in
Phase 3.
"""

from __future__ import annotations

from fastapi import FastAPI

from vpp import __version__
from vpp.api.routes import health
from vpp.core.config import get_settings
from vpp.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    """Build and configure a FastAPI application instance."""
    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger(__name__)

    app = FastAPI(title=settings.app.name, version=__version__)
    app.include_router(health.router)

    logger.info(
        "VPP app initialized (device=%s, services=%s)",
        settings.device,
        settings.services.model_dump(),
    )
    return app


app = create_app()
