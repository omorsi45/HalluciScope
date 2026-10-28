# Test instructions

Applies to `tests/`, in addition to the root instructions.

## Testing conventions
- Use pytest and pytest-asyncio; `pyproject.toml` sets `asyncio_mode = "auto"`. Tests marked `pg` need Postgres; `python -m pytest -q` needs it running (see the root instructions). Tests marked `models` load real models and are skipped by default (`-m models` runs them).
- Use the fixtures in `conftest.py` and `pg_fixtures.py` (`db`, `services`, `client`, `seed_docs`, `drain`) and add focused fixtures for new cases. Keep tests deterministic and independent of external network services.
- Use the fakes in `fakes.py` instead of real models: `HashEmbedder`, `KeywordReranker`, `RuleNLIScorer`, `ScriptedLLM` (replies and failures in call order), and `make_test_calibrator`. Do not download model weights or reach an Ollama server in routine tests.
- Run jobs through the real worker with `drain(db, services)` rather than calling handlers directly when the job flow matters.
- Test FastAPI routes through `httpx.AsyncClient` with `ASGITransport` and the lightweight `backend.api.app.create_app()` where possible, so startup does not load production models.
- Use `tmp_path` for file tests. The `db` fixture empties every table after each test; never point a test at a developer's database or at labeled evaluation data.
- Check the invariants where they apply: exact offsets, verdicts decided only by the features and the calibrator (a passage that tells the model what to answer must not change a verdict), and no text in logs or job payloads.
- Cover normal results, empty inputs, error handling, and the claim-order contract when relevant. Do not weaken assertions or skip tests to hide failures.

## Commands
From the repository root, with Postgres running:
```bash
python -m pytest tests/test_run_analysis.py -q
python -m pytest -q
```
Run the affected test module first, then the full suite.
