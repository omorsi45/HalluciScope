# Plan B Task 14 report: filtered dense search and full-text baseline

Status: DONE_WITH_CONCERNS (concerns are observations about HNSW behavior, no open defects)

Commit: `74380f7 feat: add filtered pgvector search and full-text baseline` on `feature/evidencegraph` (not pushed).

## What was built

- `backend/retrieval/dense.py`: `dense_search(conn, query_vec, scope, *, limit, version_ids, exact=False)`.
  - Non-exact: `set local hnsw.ef_search = <int>` where the integer is `min(1000, max(100, int(limit)))`, then `set local hnsw.iterative_scan = relaxed_order`.
  - Exact: `set local enable_indexscan = off` and `set local enable_bitmapscan = off` instead.
  - One statement: `p` joined to `d` on `d.id = p.document_id`, `passage_filter` in WHERE, `order by p.embedding <=> %s limit %s` (bare distance operator so the planner can use the HNSW index). The query vector is passed as a numpy float32 array, once for the select list and once for the ORDER BY.
  - Returns `(passage_id, 1 - distance)` re-sorted by similarity desc, then passage id.
  - A temporal scope with no visible versions returns `[]` without running SQL (same guard as `bm25_search`).
- `backend/retrieval/fts.py`: `fts_search(conn, query, scope, *, limit, version_ids)`.
  - The OR tsquery is built in SQL: `array_to_string(array(select quote_literal(lexeme) from unnest(to_tsvector('english', %s))), ' | ')::tsquery`, wrapped in `nullif(..., '')` so a stopword-only query yields a NULL tsquery (matches nothing, no server NOTICE) instead of an empty tsquery.
  - Filter `p.tsv @@ q`, score `ts_rank_cd(p.tsv, q)::float8`, `order by score desc, p.id`.
- `tests/test_dense_fts.py`: 10 tests, all `pg`, `HashEmbedder` for the query vector.

Files: `backend/retrieval/dense.py`, `backend/retrieval/fts.py`, `tests/test_dense_fts.py` (all LF, no em dashes).

## Tests

The six from the brief: `test_dense_nearest_first`, `test_exact_and_hnsw_agree`, `test_dense_filter_does_not_starve`, `test_dense_respects_as_of`, `test_fts_or_semantics`, `test_fts_stopwords_empty`.

Four extra tests (they cover behavior the brief states but does not list):
- `test_dense_session_settings`: the SET LOCAL values after a call (ef_search 100 / 250 / clamped 1000, iterative_scan, exact turning index and bitmap scans off).
- `test_dense_empty_temporal_scope_returns_nothing`: no visible versions and empty collection list return `[]`.
- `test_fts_operators_in_query_are_plain_text`: `& ! | ( ) :* <->` and a stray quote in the query neither raise nor act as operators.
- `test_fts_respects_scope_and_as_of`: an unreadable collection never appears; as-of picks v1, live picks v2.

## RED

Command: `wsl.sh timeout 300 python -m pytest tests/test_dense_fts.py -q`, before the modules existed:

```
tests/test_dense_fts.py:9: in <module>
    from backend.retrieval.dense import dense_search
E   ModuleNotFoundError: No module named 'backend.retrieval.dense'
ERROR tests/test_dense_fts.py
1 error in 0.75s            (exit 2)
```

Feature missing, as intended.

## GREEN

After implementing: `9 passed` on the first six-plus-three version, then three test-design corrections (see Decisions 1 to 3), final `10 passed in 3.22s`.

Mutation checks (proof the tests can fail for the right reason), done with the final tests:
- Removing the `hnsw.iterative_scan = relaxed_order` statement makes `test_dense_filter_does_not_starve` fail with `assert set() == {three X passage ids}` (the filtered scan starves: 0 rows) and `test_dense_session_settings` fail with `{'iterative_scan': 'off'} != {'iterative_scan': 'relaxed_order'}`. Restored afterwards (`git status` clean apart from the three new files, `MUTATION` marker count 0).
- Stability: `test_dense_filter_does_not_starve` run 15 times in separate processes: 15 passed, 0 failed (HNSW level assignment is random, so the graph differs per run).

## Full suite and lint

- `wsl.sh timeout 900 python -m pytest -q`: `878 passed, 2 deselected, 2 warnings in 142.13s`, exit 0. The 2 warnings are the existing `tests/test_generator.py` `AsyncMockMixin` RuntimeWarnings; the 2 deselected are the `models` marker. No failures.
- `wsl.sh timeout 120 ruff check .`: `All checks passed!`. `ruff format --check` on the three files: already formatted.
- Commit hooks ran (secret scan: no leaks found).

## Decisions

1. `test_exact_and_hnsw_agree`: the brief's setup with `HashEmbedder` (bag of words, +-1 components) produces many exactly tied similarities at rank 5 (for example dozens of passages at 0.218218), so two correct searches return different id sets. I confirmed this by printing both similarity lists (identical values, different ids). The corpus now uses repeated words plus a per-passage amount of filler, and the test asserts a gap above 1e-6 between rank 5 and rank 6 of the exact result before comparing. It compares the ordered top 5 (stricter than the id sets the brief names) and the similarities.
2. Forcing the HNSW path in tests: on these small tables the planner joins `documents` first and sorts, which never touches the index, so `test_dense_filter_does_not_starve` passed even with iterative scans removed. EXPLAIN of the real statement showed that `enable_seqscan = off` alone is not enough; `enable_seqscan = off` plus `enable_sort = off` yields `Nested Loop -> Index Scan using passages_embedding_hnsw` with the documents filter as a join filter. The helper `run_dense(..., force_index=True)` sets both in the test transaction. This is test-only; production code does not change planner settings except the two exact-mode ones the brief names.
3. `test_dense_filter_does_not_starve` data: the brief's picture (3 passages in X, 300 closer ones in Y) needs the 3 X passages to be reachable in the HNSW graph. With X as an unrelated outlier cluster the unfiltered index scan returned only 300 of 303 rows in 3 of 4 trials, and the filtered query returned 0 even with iterative scan (the X nodes have no incoming edges). With X passages sharing two query words with Y (Refund window plus filler) reachability was 303/303 in all trials and at least 298 Y passages are still closer than the best X passage (more than ef_search = 100). The test asserts both preconditions (the 100 exact nearest are all unreadable; an unfiltered forced-index scan reaches all 303) so a graph artifact fails loudly on a precondition and not as a false starvation result.
4. `ef_search` is `min(1000, max(100, int(limit)))`: pgvector rejects `hnsw.ef_search` above 1000. For `limit > 1000` the iterative scan supplies the remaining rows.
5. Empty or stopword-only FTS query: handled with `nullif(..., '')` so the tsquery is NULL and the statement returns no rows without the server notice an empty tsquery raises. Probed a few hostile queries directly in Postgres (backslashes, quotes, `& | ! :* <->`, URLs): the english parser never emits lexemes that break `quote_literal` plus `::tsquery`, and the extra test covers operator characters.
6. Tests import the helpers `make_scope`, `add_version`, `version_of`, `VOCAB`, `D_JAN`, `D_JUN` from `tests/test_bm25.py` (existing precedent: `test_diff.py` imports from other test modules).

## Concerns

- HNSW can leave outlier vectors unreachable: nodes with no incoming edge are never returned, with or without iterative scan (observed above). Exact mode is the correct fallback when recall must be guaranteed; the recall calibration planned for retrieval should account for it. Not a defect in this task's code.
- `SET LOCAL` settings outlive the call: they apply to the rest of the caller's transaction. An `exact=True` call leaves `enable_indexscan` and `enable_bitmapscan` off for later statements in that transaction, and an approximate call leaves `ef_search` and `iterative_scan` set. Callers that mix modes in one transaction (the Task 15/16 retriever) should use one transaction per mode or reset the settings. I did not add resets because the brief does not ask for them.
- On small tables the planner does not use the HNSW index for the production statement (it joins `documents` first and sorts, which is exact). The index matters at scale; the HNSW path is only exercised in tests through the forced-planner helper.
