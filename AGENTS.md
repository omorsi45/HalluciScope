# HalluciScope agent instructions

## Purpose and scope
HalluciScope checks hallucinations in single-document RAG answers. It retrieves source chunks, generates an answer with Ollama, decomposes the answer into claims, evaluates claims with NLI, self-consistency, and semantic similarity, then reports per-claim hallucination scores.

This file applies across the repository. Also follow any `AGENTS.md` in the directory being edited or its ancestors. More local rules supplement these shared rules. Keep new instructions close to the code they govern; do not duplicate this file in nested directories.

## Repository map
- `backend/api/`: FastAPI routes, request/response schemas, and application startup.
- `backend/cli/`: Typer commands.
- `backend/core/`: document parsing, chunk retrieval, generation, claim decomposition, verifiers, and ensemble scoring.
- `backend/db/`: SQLite schema and async persistence.
- `backend/models/`: local ML model loading.
- `frontend/`: React, TypeScript, Vite, and Tailwind app.
- `evaluation/`: labeled JSONL datasets, benchmarks, ablations, and calibration.
- `tests/`: pytest suite.

## Setup and checks
Run Python commands from the repository root (Python 3.11+):
```bash
python -m pip install -e ".[dev,eval]"
python -m pytest -q
ruff check .
```

Run frontend commands from `frontend/`:
```bash
npm ci
npm run lint
npm run build
```

For an actual end-to-end run, start Ollama with the `llama3.1:8b` model available, then start the API from the repository root:
```bash
uvicorn backend.api.startup:create_configured_app --factory --reload
```
The Vite dev server runs with `npm run dev` in `frontend/`. Unit tests should not require Ollama or live Hugging Face downloads.

## Working rules
- Make the smallest change that answers the task. Read the affected implementation and nearby tests before editing; do not refactor unrelated code.
- Keep Python APIs, FastAPI schemas, CLI arguments, and frontend response types in sync when changing a public contract.
- Preserve claim order and score semantics across retrieval, verifiers, ensemble, API, CLI, and UI.
- Add or update focused tests when behavior changes. Run the relevant checks and report what was run or could not be run.
- Treat source documents and user questions as untrusted, potentially sensitive input. Do not log, commit, or expose their contents unnecessarily. Never commit credentials, local databases, model weights, or generated evaluation outputs.
- Do not invent benchmark numbers or silently change labeled evaluation examples to make results look better.
- Address the cause of failures. Do not bypass tests, lint checks, or error handling to make a task appear complete.
