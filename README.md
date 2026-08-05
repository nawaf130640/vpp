# VPP — Virtual Product Placement (research prototype)

Backend-only prototype that inserts a logo or product PNG into a video and
keeps it locked to a surface as the camera moves. No frontend; driven by REST
endpoints and scripts.

The architecture splits cleanly: this code runs on a **laptop or a Codespace**
for development (CPU, stubbed AI), and heavy model inference runs on a rented
**cloud GPU** later. The contracts don't change between them — only the
`device` and `services` entries in `config/config.yaml`.

## Quick start — GitHub Codespaces (recommended)

No local setup. On the repo page: **Code → Codespaces → Create codespace**.
It builds for a minute or two and installs everything automatically. Then, in
the Codespace terminal:

```bash
uv run pytest -q                         # smoke test — expect 1 passed
uv run uvicorn vpp.main:app --reload     # boots the API
```

When the API starts, Codespaces pops a notification to open port 8000. Open it
and add `/health` to the URL. You should see `{"status": "ok", ... }`.

## Alternative — local laptop (WSL2 on Windows)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv, then reopen terminal
uv sync --extra dev                                # Python 3.12 + deps (CPU torch)
uv run pytest -q
uv run uvicorn vpp.main:app --reload               # http://127.0.0.1:8000/health
```

## Layout

```
config/         YAML configuration (single source of truth)
src/vpp/
  core/         contracts, data types, config, logging  (the spine)
  gpu/          CUDA device + VRAM lifecycle             (Phase 4)
  services/     one adapter per model, behind a contract (Phase 3 fakes -> Phase 5 real)
  video/        decode / encode / shot detection         (Phase 2)
  pipelines/    typed stages + orchestrator              (Phase 3)
  jobs/         async job queue + runner                 (Phase 3)
  storage/      artifact persistence behind an interface (Phase 3)
  api/          thin FastAPI layer
  main.py       app factory + entrypoint
scripts/        headless CLI runs (same pipeline, no API)
tests/          unit + integration
```

## Cloud GPU (later)

On the rented box: clone this repo, uncomment the `[tool.uv.sources]` CUDA
block in `pyproject.toml` (set `cuXXX` to match `nvidia-smi`), set
`device: "cuda"` in the config, then `uv sync`.
