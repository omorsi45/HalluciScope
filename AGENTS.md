# EvidenceGraph agent instructions

## Purpose and scope
EvidenceGraph is a multi-tenant document intelligence system. Teams upload documents into collections; every version is kept as exactly addressable passages. The system answers questions from them (ask) and checks an answer, given or generated, claim by claim against them (verify). Each claim gets a calibrated verdict (supported, contradicted, or insufficient) with the evidence passages and exact spans it rests on.

This file applies across the repository. Also follow any `AGENTS.md` in the directory being edited or its ancestors. More local rules supplement these shared rules. Keep new instructions close to the code they govern; do not duplicate this file in nested directories.

## Repository map
- `backend/api/`: FastAPI app, routes (`/api/v1` plus the deprecated `/api` adapters), schemas, middleware, and startup.
- `backend/auth/`, `backend/authz/`: sessions, API keys, passwords, and the collection access policy.
- `backend/db/`: Postgres pool, SQL migrations in `migrations/`, and repositories in `repos/`.
- `backend/jobs/`: the Postgres job queue and the worker.
- `backend/ingest/`: parsers, normalization, sentence segmentation, passages, versions, diffs, and upload intake.
- `backend/retrieval/`: BM25, dense and full-text search, fusion, reranking, and visibility filters.
- `backend/llm/`, `backend/generate/`: the Ollama client; answer generation and `[P#]` citations.
- `backend/verify/`: claim extraction, evidence gathering, NLI scoring, features, the verification engine, and the self-consistency diagnostic.
- `backend/calibration/`: calibrator fitting and the artifact the engine loads.
- `backend/provenance/`: passage resolution, citations, the evidence graph, and re-verification.
- `backend/analysis/`: ask, verify, and quick analyses (request, run, handlers).
- `backend/models/`: embedding, reranking, and NLI adapters. `backend/core/document.py` reads a file to text through the ingest parsers.
- `backend/obs/`: logging and metrics.
- `backend/cli/`: Typer commands `check`, `migrate`, and `worker`.
- `frontend/`: React, TypeScript, Vite, and Tailwind app.
- `evaluation/`: benchmark runners, metrics, and manifests.
- `docs/results/`: result reports written by the evaluation scripts.
- `tests/`: pytest suite.

## Setup and checks
Run Python commands from the repository root (Python 3.11+). The app and the test suite need Postgres 17 with pgvector. Start it with `scripts/dev-postgres.sh start` (macOS, Homebrew), or with Docker:
```bash
docker run -d --name eg-pg -p 54329:5432 -e POSTGRES_HOST_AUTH_METHOD=trust pgvector/pgvector:pg17
```
The script also creates the `eg_owner` and `eg_app` roles and the `evidencegraph` database that the app commands use; with Docker, run the same statements (see the script). The test suite creates its own throwaway databases and roles on the server at `HALLUCISCOPE_TEST_ADMIN_URL` (default `postgresql://postgres@localhost:54329/postgres`).

```bash
python -m pip install -e ".[dev,eval]"
python -m backend.cli.main migrate
python -m backend.cli.main worker
python -m pytest -q
ruff check .
```
`worker` runs until SIGTERM or SIGINT, so use its own terminal. `pytest -q` needs Postgres and skips the tests marked `models`; run those with `-m models` when real model downloads are acceptable.

Run frontend commands from `frontend/`:
```bash
npm ci
npm run lint
npm run build
```

For an actual end-to-end run, start Ollama with the `llama3.2:3b` model available, start the worker, then start the API from the repository root:
```bash
uvicorn backend.api.startup:create_configured_app --factory --reload
```
The Vite dev server runs with `npm run dev` in `frontend/`. Unit tests should not require Ollama or live Hugging Face downloads.

## Invariants
- Offsets are exact. Passage, span, claim, and citation offsets index the stored text of one document version or the stored answer, and slicing that text with them returns the quoted text. Never trim, normalize, or re-encode text after offsets were computed from it.
- Verdicts come only from NLI-derived features and the calibrator. The LLM writes answers and splits them into claims; it never decides a verdict, and a diagnostic such as self-consistency is never a feature.
- No text in logs or job payloads. Never write document text, questions, answers, passwords, tokens, or exception messages to a log line or a job payload; payloads carry ids, and log lines carry constant messages with whitelisted fields.
- Tenant data is isolated by row level security, and every foreign key between tenant tables includes `tenant_id`. Do not read or write tenant tables outside a tenant transaction.
- The only committed benchmark artifacts are `docs/results/*.md`, `evaluation/manifests/*.json`, and `backend/calibration/artifacts/default.json`. Benchmark text, caches, and raw results stay in the gitignored `evaluation/data/` and `evaluation/results/`.

## Working rules
- Make the smallest change that answers the task. Read the affected implementation and nearby tests before editing; do not refactor unrelated code.
- Keep Python APIs, FastAPI schemas, CLI arguments, and frontend response types in sync when changing a public contract.
- Preserve claim order and verdict semantics across extraction, verification, API, CLI, and UI.
- Add or update focused tests when behavior changes. Run the relevant checks and report what was run or could not be run.
- Treat source documents and user questions as untrusted, potentially sensitive input. Do not log, commit, or expose their contents unnecessarily. Never commit credentials, local databases, model weights, or generated evaluation outputs.
- Do not invent benchmark numbers or silently change labeled evaluation examples to make results look better. No number in a document may come from anywhere but a committed script's output.
- Address the cause of failures. Do not bypass tests, lint checks, or error handling to make a task appear complete.
