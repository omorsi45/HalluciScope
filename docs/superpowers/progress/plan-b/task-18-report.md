# Plan B Task 18: Retrieval metrics

Status: DONE
Commit: `2d41a6f feat: add retrieval metrics` (branch `feature/evidencegraph`, not pushed)

## What was built

New package `evaluation/metrics/` with original implementations (no metric libraries):

- `ndcg_at_k(ranked, qrels, k=10)`: linear gain (gain = grade), discount `log2(rank + 1)`, ideal DCG from the `k` best positive grades sorted descending. Grades `<= 0` are not relevant. No relevant document raises `ValueError`.
- `recall_at_k(ranked, qrels, k)`: relevant retrieved in the top k over all relevant. Empty ranking with relevant documents gives `0.0`.
- `mrr_at_k(ranked, qrels, k)`: reciprocal rank of the first relevant document in the top k, else `0.0`.
- `maxp(passage_ranking, passage_to_doc)`: documents ordered by first passage occurrence.
- `evaluate_run(run, qrels)`: keys `ndcg@10`, `recall@10`, `recall@100` (k=100), `mrr@10`, `n_queries`. Averages over queries with at least one positive grade. A judged query missing from `run` counts as an empty ranking.

Legacy modules (`benchmarks.py`, `ablation.py`, `calibration.py`) and `evaluation/AGENTS.md` untouched.

## Files changed (3, all new, LF line endings, no em dashes verified by byte scan)

- `evaluation/metrics/__init__.py`
- `evaluation/metrics/retrieval.py`
- `tests/test_retrieval_metrics.py` (31 test cases, 24 test functions)

## RED evidence

Tests written first. Command:

```
wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 300 python -m pytest tests/test_retrieval_metrics.py -q
```

Output (exit 2, fast, no expensive behavior involved):

```
ERROR collecting tests/test_retrieval_metrics.py
tests/test_retrieval_metrics.py:5: in <module>
    from evaluation.metrics.retrieval import (
E   ModuleNotFoundError: No module named 'evaluation.metrics'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.74s
```

Failure cause is the missing feature (module does not exist), not a typo.

## GREEN evidence

Same command after implementing:

```
...............................                                          [100%]
31 passed in 0.31s
```

The nDCG literal `ndcg_at_k(["d1","d2","d3"], {"d1": 1, "d3": 2}, 3) == 0.7601875334318685` passes within 1e-12.

## Full suite and lint (run once, before commit)

- `timeout 900 python -m pytest -q`: `972 passed, 2 deselected, 2 warnings in 146.54s`, exit 0. The two warnings are pre-existing `RuntimeWarning: coroutine ... never awaited` in `tests/test_generator.py` (backend/core/generator.py:50), unrelated to this task.
- `timeout 120 ruff check .`: `All checks passed!`, exit 0.
- `ruff format --check` on the 3 new files: `3 files already formatted`.
- Commit hooks ran (gitleaks: no leaks found), commit succeeded without `--no-verify`.

## Decisions (brief did not specify these; all conservative)

1. `recall_at_k` with no relevant document raises `ValueError` (division by zero otherwise). `mrr_at_k` returns `0.0` in that case, per the brief's "else 0".
2. `k < 1` raises `ValueError` in all three metrics (nDCG and recall would otherwise divide by zero).
3. The three single-query metrics also reject a ranking with a repeated document (`ValueError`). Without that, `recall_at_k` could exceed 1.0 on a run that skipped MaxP collapse.
4. `evaluate_run` validates every ranking in `run`, including queries that are not judged, so a missing MaxP collapse cannot slip through on an unjudged query. Run queries absent from qrels are otherwise ignored.
5. `evaluate_run` raises `ValueError` when no query has a positive grade (mean over zero queries is undefined).
6. `maxp` raises `KeyError` for a passage id missing from `passage_to_doc` (no silent drop).
7. `n_queries` is returned as a float (`dict[str, float]` per the interface).
8. `evaluation/metrics/__init__.py` re-exports the five public functions.

## Mutation check (reasoned, per writing-good-tests)

- Recall@100 key using k=10: caught by `test_evaluate_run_recall_100_uses_depth_100`.
- Ideal DCG over all grades instead of the k best: caught by `test_ndcg_ideal_uses_only_k_best_grades`.
- Counting grades `<= 0` as relevant: caught by `test_ndcg_ignores_nonpositive_grades`, `test_recall_counts_only_positive_grades_and_the_top_k`, `test_queries_without_relevant_excluded`.
- Missing query not counted as zero: caught by `test_evaluate_run_missing_query_counts_zero`.
- Averaging over unjudged queries: caught by `test_queries_without_relevant_excluded`.
- maxp keeping last instead of first occurrence: caught by `test_maxp_first_occurrence`.
- Dropping the duplicate check: caught by `test_duplicate_docs_*`.

I did not execute these mutations; they are reasoned from the assertions.

## Concerns

None blocking. Decisions 3 to 5 add small guard behavior beyond the brief; the controller can drop them if a stricter reading of "exactly the brief" is wanted.
