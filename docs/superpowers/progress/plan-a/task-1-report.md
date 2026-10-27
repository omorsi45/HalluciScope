# Task 1 report: remove FAISS and pin dependencies

Status: DONE

## What was built
- `backend/core/chunker.py`: removed `faiss` import and the `faiss_index` field. `DocumentIndex(chunks, embeddings)`. `retrieve` now computes `scores = doc_index.embeddings @ query_vec[0]` and takes `np.argsort(-scores, kind="stable")[:k]` (descending dot product, ties by lower chunk index).
- `pyproject.toml`: all dependencies pinned with `==` to the versions given in the dispatch. `langchain-text-splitters==1.1.3` and `aiosqlite==0.22.1` come from `pip show`. Added `numpy`, `psycopg[binary]`, `psycopg-pool`, `pgvector`. Removed `faiss-cpu`, `datasets`, `scikit-learn`. Added `markers` and `addopts = "-m 'not models'"` under `[tool.pytest.ini_options]`.
- `evaluation/benchmarks.py`: removed sklearn import; local `compute_metrics` (precision/recall/F1 with zero_division 0) and `_auc_roc` (rank-sum formula with average ranks for ties, `None` when one class).
- `tests/test_chunker.py`: replaced the `faiss_index is not None` assertion with `"faiss_index" not in DocumentIndex.__dataclass_fields__`; added `test_retrieve_orders_by_cosine` (DocumentIndex built directly with fixed vectors, no `hash()` dependence).

## Files changed
pyproject.toml, backend/core/chunker.py, evaluation/benchmarks.py, tests/test_chunker.py

## RED
`.venv/bin/python -m pytest tests/test_chunker.py -q` before implementation:
```
FAILED tests/test_chunker.py::test_build_index_returns_document_index - AssertionError: assert 'faiss_index' not in {...}
FAILED tests/test_chunker.py::test_retrieve_orders_by_cosine - TypeError: DocumentIndex.__init__() missing 1 required positional argument: 'faiss_index'
2 failed, 4 passed
```

## GREEN
`tests/test_chunker.py`: 6 passed. After reinstall (`pip uninstall -y faiss-cpu scikit-learn datasets`, `pip install -e ".[dev,eval]"`), full suite in one process: `48 passed, 2 warnings in 7.04s`. No OMP abort.

Extra check (before uninstalling sklearn): ran 200 random cases comparing the new `compute_metrics` (including `auc_roc` with tied scores and single-class inputs) against sklearn; all matched. Throwaway script, not committed.

## Lint
`ruff check` on the three changed Python files: before 6 findings, after 5. Nothing new. Remaining are pre-existing (I001 in chunker.py and test_chunker.py, F401 `pytest` unused in test_chunker.py, 2x ASYNC230 in benchmarks.py), left for Task 4. benchmarks.py I001 disappeared as a side effect of removing the sklearn import.

## Decisions / concerns
- `scikit-learn==1.9.1` is still installed in the venv because `sentence-transformers` requires it (`pip show`: Required-by sentence-transformers). It is no longer a declared dependency of this project and is not imported by project code. FAISS and `datasets` are gone, which is what fixes the double libomp load; suite confirms it.
- `grep faiss` finds only the new test assertion.
- Pre-existing RuntimeWarning (un-awaited AsyncMock in tests/test_generator.py) appears in the full run; not touched.
- Git commit used `PATH=/usr/bin:$PATH`.

## Commit
See final reply.
