# Plan B Task 16 report: PostgresRetriever

Status: DONE

Commit: `d55301a feat: add hybrid Postgres retriever with access-controlled scope` (branch `feature/evidencegraph`, not pushed). Staged by name: `backend/retrieval/service.py`, `tests/test_retriever.py`. The gitleaks pre-commit hook ran and passed.

## What was built

`backend/retrieval/service.py`: `PostgresRetriever(db, embedder, reranker=None, *, k1=0.9, b=0.4, rrf_k=60)` with `async search(query, scope, *, k=10, mode="hybrid", rerank=True, candidates=50, exact=False)`, the exact signature from the brief and the `Retriever` protocol.

Flow of `search`:
1. Validate `1 <= k <= 300`, `1 <= candidates <= 1000`, `mode in get_args(SearchMode)` (`ValueError` otherwise).
2. `rerank=True` with no reranker raises `RerankUnavailable`, before any work.
3. Empty `scope.collection_ids` returns `[]` (no embedding, no transaction).
4. Embed the query with `asyncio.to_thread(embedder.embed_queries, [query])`, only for `hybrid` and `dense`, before the transaction opens.
5. `n = max(k, candidates)` if mode is `hybrid` or `rerank` is set, else `k`.
6. One `db.tenant_tx(scope.tenant_id)`: `visible_version_ids`, then `bm25_search` / `fts_search` first and `dense_search` last (binding note 1), `rrf_fuse` for hybrid, then one hydration query (passages join document_versions join documents) that preserves the fused order.
7. After the transaction closes: `rerank_hits(reranker, query, hits, k)` when reranking (so no pooled connection is held during CPU-bound scoring); otherwise the hydrated hits are the result.
8. `STAGE_SECONDS{stage="retrieve"}` is observed in a `finally` for the whole call after validation (embedding, transaction, rerank).

Scores dict: every component present (`bm25`, `dense`, `fts`) as floats, plus `rrf` for hybrid; `rerank` is added by `rerank_hits`. `PassageHit.text` is the stored passage text (the migration trigger guarantees it equals `version_text[start_char:end_char]`).

## Files

- Created `backend/retrieval/service.py` (183 lines, LF)
- Created `tests/test_retriever.py` (14 tests, all `pg`, LF)

## RED evidence

Command: `wsl ... wsl.sh timeout 300 python -m pytest tests/test_retriever.py -q`

```
ERROR collecting tests/test_retriever.py
tests/test_retriever.py:11: in <module>
    from backend.retrieval import service as service_module
E   ImportError: cannot import name 'service' from 'backend.retrieval' (.../backend/retrieval/__init__.py). Did you mean: 'services'?
1 error in 0.71s
```

The feature (module) did not exist, so the whole file failed at collection. No expensive behavior is reachable in this RED.

Because RED was a collection error, each test's ability to fail was proven afterwards by mutating the finished implementation and restoring it (restore verified byte for byte). Each realistic mutation failed the named test(s):

| Mutation | Failing test(s) |
|---|---|
| `n = k` always (candidates ignored) | test_rerank_reorders, test_component_limits_exact_flag_and_lexical_before_dense |
| rerank pool hydrated as `k` instead of `n` | test_rerank_reorders |
| query embedded on the event loop | test_query_embedded_off_loop_before_transaction_and_only_for_dense_modes |
| `exact` not passed to dense | test_component_limits_exact_flag_and_lexical_before_dense |
| `STAGE_SECONDS` not observed | test_retrieve_stage_is_timed |
| `rrf_k` ignored (60 hard-coded) | test_hybrid_combines_lexical_and_dense |
| `k1`/`b` not passed to BM25 | test_bm25_parameters_reach_the_scorer |
| hydration loses fused order | test_hybrid_combines_lexical_and_dense, test_scores_carry_each_component, test_rerank_reorders |
| empty-scope short circuit removed | test_empty_scope_returns_empty |
| rerank check moved after empty-scope check | test_rerank_without_reranker_raises |
| `rrf` score dropped | test_hybrid_combines_lexical_and_dense, test_scores_carry_each_component, test_rerank_reorders |

## GREEN evidence

Focused: `14 passed in 1.83s` (after `ruff format` of the test file).

Tests (all `pg`, fakes `HashEmbedder` / `KeywordReranker` / local spies):
- The nine from the brief: `test_hybrid_combines_lexical_and_dense`, `test_rerank_reorders`, `test_rerank_without_reranker_raises`, `test_scope_enforced_in_every_mode`, `test_as_of_in_every_mode`, `test_hit_fields_resolve_exactly`, `test_cross_tenant_identical_text_isolated`, `test_validation`, `test_empty_scope_returns_empty`.
- Five more for behavior the brief and binding notes require: `test_scores_carry_each_component` (score keys per mode, rrf equals the sum of `1/(60+rank)` over standalone rankings, ranks 1..n), `test_bm25_parameters_reach_the_scorer` (k1 and b change the score and match `bm25_term_score`), `test_query_embedded_off_loop_before_transaction_and_only_for_dense_modes` (event order `embed` then one `tx`; no embed for bm25/fts; embed runs off the loop thread), `test_component_limits_exact_flag_and_lexical_before_dense` (call-through spies on the real search functions: limits 7/3/7, `exact` passthrough, bm25 before dense), `test_retrieve_stage_is_timed`.

Design of the key tests:
- Hybrid: with query `what is the harbor`, BM25 alone returns only the lexical passage, dense alone returns only the two stopword passages, and hybrid with `k=2, candidates=2` returns the lexical passage and the best dense passage, each with `rrf = 1/61` (and `1/11` for `rrf_k=10`).
- Scope (Review Focus 3): collection Y holds the query terms in bulk and one X document is soft-deleted; for bm25, dense, fts, hybrid, with and without rerank, no hit leaves the readable set; dense returns exactly the readable set. BM25 scores of the readable passages are identical (approx abs 1e-12) before and after Y and the deleted file exist. Controls prove the checks can see a leak: scope (X, Y) returns Y passages, and a `known_at` before the delete returns the deleted file, in every mode.
- As-of (Review Focus 4): every mode and rerank value returns only v1 text between versions, nothing before the first version, a July document is absent in March, and a v3 uploaded later with `effective_from` backdated to July does not change results for the earlier `known_at`.
- Hit fields: Unicode text (combining accents, CJK, emoji) in three-passage versions; every hit satisfies `text == version_text[start:end]`; version_number, title, and effective_from match literals; non-zero offsets and non-ASCII text are asserted to occur.
- Cross-tenant: identical text in tenants A and B; hits stay within the tenant's own passage ids for live and future as-of scopes; naming the other tenant's collection returns `[]` in every mode.

Full suite: `python -m pytest -q` (timeout 900): `901 passed, 2 deselected, 2 warnings in 125.63s`. The 2 deselected are the `models` marker tests excluded by `addopts`. The 2 warnings are the existing `RuntimeWarning` in `tests/test_generator.py` (`coroutine 'AsyncMockMixin._execute_mock_call' was never awaited`), unrelated to this task.

Lint: `ruff check .` -> `All checks passed!`. `ruff format --check` on both new files: already formatted.

## Decisions

1. Hydration size: the brief says to hydrate the top `n` ids. I hydrate `n` only when reranking and `k` otherwise. The returned hits are identical (no rerank means the result is the first `k` of the fused order), but the text of up to `n - k` passages is not fetched for nothing. Reranking still scores the full hydrated top `n`, and the test asserts the reranker receives `max(k, candidates)` passages.
2. Check order: validation, then the rerank-availability check, then the empty-scope short circuit. A missing reranker is a configuration error and is raised even for an empty scope; invalid arguments are raised even for an empty scope.
3. `STAGE_SECONDS{stage="retrieve"}` covers the whole call after validation (embedding, database work, rerank), observed in `finally` so failed searches are timed too. Validation errors and the empty-scope return are not observed.
4. Reranking runs after the transaction closes so the pooled connection is not held while the CPU-bound model scores on a worker thread.
5. `float(score)` on each component score so the scores dict only holds plain floats.
6. No signature change for the later `hybrid_rerank` benchmark: it can call `search(..., rerank=False)` per mode as the binding note says.

## Concerns

- None blocking. Behavior notes for callers: hybrid mode does not include `fts` (per the brief); `exact` only affects dense search (bm25 and fts ignore it); `candidates` is only used in hybrid mode or with `rerank=True`.
- Each statement runs at the default READ COMMITTED level inside the transaction, so a concurrent soft delete between the visibility query and hydration cannot remove a hydrated row (passages and versions are immutable), but a document deleted mid-search may still appear in that one result. This is inherited from the earlier tasks' design and not changed here.
