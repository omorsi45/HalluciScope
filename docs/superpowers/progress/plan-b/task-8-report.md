# Plan B Task 8 report: model protocols, adapters, Services, fakes

Status: DONE_WITH_CONCERNS (one scope decision on legacy imports, see Decisions)
Commit: `6e115b9 feat: add embedder and reranker adapters with test fakes` on `feature/evidencegraph` (not pushed)

## What was built

- `backend/models/protocols.py`: `Embedder` and `Reranker` Protocols (attributes and method signatures exactly as in the brief).
- `backend/models/loader.py` additions: `BGE_QUERY_PREFIX`, `resolve_device`, `SentenceTransformerEmbedder`, `CrossEncoderReranker`, `load_embedder`, `load_reranker`. Existing `get_settings`, `get_embedding_model`, `get_nli_model` kept with the same behavior.
- `backend/config.py`: `embedder_model`, `embedding_dim`, `reranker_model`, `model_device`, `embed_batch_size`, `max_upload_bytes`, `parse_timeout_s` with the brief's defaults. Legacy `embedding_model_name` untouched.
- `backend/services.py`: `@dataclass Services(settings, db, embedder, reranker=None)`.
- `tests/fakes.py`: `HashEmbedder` (`model_id = "fake-hash-384"`, blake2b digest_size=8 per lowercase `\w+` token, signed counts, L2-normalized, fixed unit vector for empty text) and `KeywordReranker` (`model_id = "fake-keyword"`, distinct lowercase query tokens present in the passage).
- `tests/test_model_adapters.py`: 13 default tests plus 2 `@pytest.mark.models` tests.

Files: modified `backend/config.py`, `backend/models/loader.py`; created `backend/models/protocols.py`, `backend/services.py`, `tests/fakes.py`, `tests/test_model_adapters.py`. Modified files stay CRLF in the working copy (`i/lf w/crlf`), new files are LF. No em dashes (checked by byte scan).

## Tests written (all before implementation)

Default run (13): `test_hash_embedder_normalized_and_deterministic`, `test_hash_embedder_same_vectors_across_processes` (two subprocesses with `PYTHONHASHSEED=1` and `2` must print identical vectors), `test_hash_embedder_similarity_tracks_overlap`, `test_hash_embedder_queries_and_passages_are_embedded_identically`, `test_hash_embedder_ignores_case_and_punctuation`, `test_hash_embedder_empty_text_and_empty_input` (empty text is a unit vector, `[]` gives `(0, 384)`), `test_hash_embedder_dim_is_configurable`, `test_keyword_reranker_counts`, `test_resolve_device_explicit` (parametrized cpu, cuda, mps), `test_importing_loader_does_not_import_ml_libraries` (subprocess import, asserts torch, sentence_transformers, transformers not in `sys.modules`), `test_settings_model_defaults`.

Models run (2): `test_real_embedder_dim_and_prefix` (dim 384, float32, unit norm, query vector differs from passage vector, and `embed_queries(x)` equals `embed_passages([BGE_QUERY_PREFIX + x])` within 1e-4, which proves the exact prefix; empty input shapes), `test_real_reranker_orders` (refund passage outscores the unrelated one, reversed input gives reversed scores, `[]` gives `[]`, floats returned).

## RED evidence

Nothing was implemented when these ran, so no model could be downloaded.

1. All tests written, `tests/fakes.py` and loader additions absent:

```
$ ... wsl.sh timeout 300 python -m pytest tests/test_model_adapters.py -q
E   ModuleNotFoundError: No module named 'tests.fakes'
ERROR tests/test_model_adapters.py
1 error in 0.84s            (exit 2)

$ ... wsl.sh timeout 300 python -m pytest tests/test_model_adapters.py -q -m models
E   ModuleNotFoundError: No module named 'tests.fakes'
1 error in 0.80s            (exit 2)
```

2. After adding only `protocols.py` and `tests/fakes.py` (to get per-test failures for the rest):

```
default:
FAILED test_resolve_device_explicit[cpu] - ImportError
FAILED test_resolve_device_explicit[cuda] - ImportError
FAILED test_resolve_device_explicit[mps] - ImportError
FAILED test_settings_model_defaults - AttributeError: 'Settings' object has no attribute 'embedder_model'
4 failed, 8 passed, 2 deselected in 15.34s

models:
FAILED test_real_embedder_dim_and_prefix - ImportError: cannot import name 'BGE_QUERY_PREFIX' from 'backend.models.loader'
FAILED test_real_reranker_orders - ImportError: cannot import name 'load_reranker' from 'backend.models.loader'
2 failed, 12 deselected in 13.59s
```

3. After adding the loader/config/services code, the import-cheapness test (added at this point) failed for the right reason, because the legacy loader imported `sentence_transformers` and `transformers` at module top:

```
E   AssertionError: assert ['torch', 'se...transformers'] == []
FAILED tests/test_model_adapters.py::test_importing_loader_does_not_import_ml_libraries
1 failed, 12 passed, 2 deselected in 28.13s
```

## GREEN evidence

```
$ ... wsl.sh timeout 300 python -m pytest tests/test_model_adapters.py -q
13 passed, 2 deselected in 3.13s

$ ... wsl.sh timeout 900 python -m pytest tests/test_model_adapters.py -q -m models
2 passed, 13 deselected in 22.92s       (exit 0, models already in the HF cache from the first run)
```

The first models run downloaded the weights (30.25 s total) and showed a `FutureWarning` for `get_sentence_embedding_dimension`; I switched to `get_embedding_dimension()` and re-ran clean.

Full suite: `... wsl.sh timeout 900 python -m pytest -q` gives `728 passed, 2 deselected, 2 warnings in 183.32s` (exit 0). The 2 warnings are pre-existing `RuntimeWarning: coroutine ... never awaited` in `tests/test_generator.py` (untouched). The 2 deselected are this task's `models` tests, run separately above.

Lint: `... wsl.sh timeout 120 ruff check .` gives `All checks passed!` (exit 0).

Extra check (one-off, not committed): real reranker raw logits on the refund query were `[8.79, -11.11]` (outside [0, 1], so no sigmoid is applied). `Services(settings=Settings(), db=None, embedder=HashEmbedder())` defaults `reranker` to `None`.

Commit hooks ran (gitleaks: no leaks found). Staged by name only: the 6 files above.

## Decisions

1. Legacy loader imports made lazy. The task said importing `backend.models.loader` must stay cheap, but the existing loader imported `sentence_transformers` and `transformers` at module top, so it was not cheap (test 3 above proves it). To make the requirement true I moved those two imports into `get_embedding_model` / `get_nli_model` (the former annotated via `TYPE_CHECKING` and a quoted return type). Behavior of both functions is unchanged. `grep` showed no test or code patches `backend.models.loader.SentenceTransformer`; `backend/api/startup.py` and `backend/cli/main.py` only import the two getter functions. If this is considered out of scope it is a 10-line revert plus dropping `test_importing_loader_does_not_import_ml_libraries`.
2. Reranker returns raw logits by passing `activation_fn=torch.nn.Identity()` to `CrossEncoder.predict`, so the result does not depend on a default activation in the checkpoint config.
3. `HashEmbedder` index comes from the first 4 digest bytes mod `dim`, sign from the low bit of byte 5. A text whose tokens all cancel to a zero vector also gets the fixed unit vector (same as empty text), so the "rows are unit length" contract always holds.
4. `SentenceTransformerEmbedder` handles empty input itself (returns `(0, dim)`) because `encode([])` does not return a 2-D array. `CrossEncoderReranker.score` returns `[]` for no passages for the same reason.
5. `dim` comes from `get_embedding_dimension()`. `load_embedder` does not compare it to `settings.embedding_dim` (not in the brief); the real-model test asserts they match for the defaults.

## Concerns

- `settings.embedding_dim` is not enforced against the loaded model. A mismatched `HALLUCISCOPE_EMBEDDER_MODEL` would only show up when inserting into the `vector(384)` column. Later tasks may want a startup check.
- `resolve_device('auto')` is untested for `mps`/`cuda` selection, as instructed (this WSL torch is CPU-only).
- The real-model tests need the Hugging Face cache or network; they are excluded by default (`addopts = "-m 'not models'"`).
