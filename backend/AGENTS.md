# Backend instructions

Applies to `backend/` and descendants. Follow the repository-level `AGENTS.md` as well.

## Boundaries
- `config.py` defines configuration through Pydantic Settings and `HALLUCISCOPE_` environment variables. Add settings there rather than scattering environment lookups.
- `api/app.py` builds the testable FastAPI app; `api/startup.py` wires real models, HTTP clients, the database pool, and shutdown via lifespan. Keep heavyweight initialization out of import time and out of unit-test app construction.
- Routes under `api/routes/` are mounted at `/api/v1`, except `routes/legacy.py`, the deprecated `/api` adapters over the same engine. Contracts live in `api/schemas_v1.py` (and `api/schemas.py` for the deprecated routes). Keep `frontend/src/api/client.ts` aligned when these change.
- `services.py` holds the process-wide dependencies (database, models, LLM, calibrator). The worker and the `check` command build them with real models; tests build them from the fakes in `tests/fakes.py`.
- `cli/main.py` exposes `check` (a quick analysis run in-process), `migrate`, and `worker`. Keep business logic in shared modules instead of duplicating it in routes or CLI handlers. Benchmarks are not CLI commands: they run as `python -m evaluation.run_*`.
- `db/migrations/` holds the numbered SQL migrations and `db/repos/` the repositories. Use parameterized SQL only, and open a tenant transaction (`Database.tenant_tx`) for every tenant table so row level security applies.
- `jobs/` is the Postgres queue. Job payloads carry ids only; handlers read the data they need inside a tenant transaction.
- `verify/` extracts claims, gathers evidence, scores it with NLI, and decides verdicts with the calibrator loaded from `calibration/`. `verify/consistency.py` is a diagnostic only: it runs for ask analyses that request it, is stored in `claims.diagnostics`, and never reaches a verdict.
- `core/document.py` is the only module left of the earlier pipeline: it reads a file to text through the ingest parsers.

## Changes and validation
- Preserve async behavior where I/O is involved. Reuse application-scoped clients and dependencies rather than adding per-request model loads. Model libraries are imported inside the functions that use them, so importing `models/loader.py` stays cheap.
- Verdicts come only from the NLI features and the calibrator, offsets stay exact, and no document text, question, answer, or exception message reaches a log line or a job payload. Analyses fail with a stable error code, never with a message.
- Run affected tests, for example `python -m pytest tests/test_run_analysis.py tests/test_cli.py tests/test_config.py -q`, from the repository root. Most of them need Postgres.
