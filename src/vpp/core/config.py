"""Typed application configuration.

The YAML file in ``config/`` is the single source of truth for anything that
varies between machines or experiments. This module loads it into validated
Pydantic models so the rest of the codebase works with typed objects and fails
loudly at startup if something is missing or malformed — never with a silent
``KeyError`` deep inside a pipeline stage.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Resolve the project root from this file's location: src/vpp/core/config.py -> root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

Device = Literal["cpu", "cuda"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class AppConfig(BaseModel):
    """General application settings."""

    name: str = "vpp"
    log_level: LogLevel = "INFO"


class PathsConfig(BaseModel):
    """Filesystem locations, stored relative to the project root."""

    uploads: Path = Path("data/uploads")
    work: Path = Path("data/work")
    outputs: Path = Path("data/outputs")

    def resolved(self, root: Path) -> "PathsConfig":
        """Return a copy with every path made absolute against ``root``."""
        return PathsConfig(
            uploads=root / self.uploads,
            work=root / self.work,
            outputs=root / self.outputs,
        )

    def ensure_exist(self) -> None:
        """Create the directories if they are missing (idempotent)."""
        for path in (self.uploads, self.work, self.outputs):
            path.mkdir(parents=True, exist_ok=True)


class ServicesConfig(BaseModel):
    """Selects which implementation backs each model contract.

    Each value is a registry key (e.g. ``"fake"`` or ``"grounding_dino"``).
    The service registry, added in a later phase, maps these to classes.
    """

    detector: str = "fake"
    segmenter: str = "fake"
    depth: str = "fake"
    tracker: str = "fake"
    renderer: str = "fake"


class Settings(BaseModel):
    """Root configuration object for the whole application."""

    app: AppConfig = Field(default_factory=AppConfig)
    device: Device = "cpu"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    # Absolute project root, injected at load time (not read from YAML).
    project_root: Path = _PROJECT_ROOT


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from a YAML file.

    Args:
        config_path: Path to the YAML config. Defaults to ``config/config.yaml``
            at the project root.

    Returns:
        A validated :class:`Settings` instance with absolute paths.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the file contents fail validation.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)

    # Make paths absolute and ensure the working directories exist.
    settings = settings.model_copy(
        update={"paths": settings.paths.resolved(settings.project_root)}
    )
    settings.paths.ensure_exist()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance.

    FastAPI dependencies and services call this so configuration is loaded
    and validated exactly once per process.
    """
    return load_settings()
