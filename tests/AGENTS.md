# Test instructions

Applies to `tests/`, in addition to the root instructions.

## Testing conventions
- Use pytest and pytest-asyncio; `pyproject.toml` sets `asyncio_mode = "auto"`.
- Use the fixtures in `conftest.py` and add focused fixtures for new cases. Keep tests deterministic and independent of external network services.
- Mock Ollama calls and ML model interfaces rather than downloading model weights or loading real models in routine tests. Existing tests use `unittest.mock`, `AsyncMock`, and `MagicMock`.
- Test FastAPI routes through `httpx.AsyncClient` with `ASGITransport` and the lightweight `backend.api.app.create_app()` where possible, so startup does not load production models.
- Use `tmp_path` or the `tmp_db_path` fixture for file and SQLite tests. Never mutate actual user databases or labeled evaluation data.
- Cover normal results, empty inputs, error handling, and the claim-order/score contract when relevant. Do not weaken assertions or skip tests to hide failures.

## Commands
From the repository root:
```bash
python -m pytest tests/test_pipeline.py -q
python -m pytest -q
```
Run the affected test module first, then the full suite if dependencies are available.
