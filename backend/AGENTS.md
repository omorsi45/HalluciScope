# Backend instructions

Applies to `backend/` and descendants. Follow the repository-level `AGENTS.md` as well.

## Boundaries
- `config.py` defines configuration through Pydantic Settings and `HALLUCISCOPE_` environment variables. Add settings there rather than scattering environment lookups.
- `api/app.py` builds the testable FastAPI app; `api/startup.py` wires real models, HTTP clients, database connections, and shutdown via lifespan. Keep heavyweight initialization out of import time and out of unit-test app construction.
- API routes under `api/routes/` are mounted at `/api`. Request/response contracts live in `api/schemas.py`. Keep `frontend/src/api/client.ts` aligned when these change.
- `cli/main.py` exposes the `halluciscope check` and `halluciscope evaluate` commands. Keep business logic in shared modules instead of duplicating it in routes or CLI handlers.
- `db/repository.py` owns async SQLite access and stores analysis and feedback. Preserve parameterized queries and connection lifecycle; use temporary databases in tests.

## Changes and validation
- Preserve async behavior where I/O is involved. Reuse application-scoped clients and dependencies rather than adding per-request model loads.
- Avoid leaking document text, answers, or user questions in diagnostics.
- Run affected tests, commonly `python -m pytest tests/test_api.py tests/test_cli.py tests/test_db.py tests/test_config.py -q`, from the repository root.
