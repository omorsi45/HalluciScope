# Plan B Task 13 report: retrieval types, visible passage set, BM25 in SQL

Status: DONE
Commit: `d0f506c feat: add bm25 scoring in sql over the visible passage set` (branch `feature/evidencegraph`, not pushed)

## What was built

- `backend/retrieval/__init__.py`: package docstring.
- `backend/retrieval/types.py`: `SearchMode`, `RerankUnavailable(RuntimeError)`, frozen `SearchScope` (with `is_live`), frozen `PassageHit` (with `effective_from`), `Retriever` protocol (with `exact`).
- `backend/retrieval/visible.py`:
  - `visible_version_ids(conn, scope)`: `None` for live scopes, else the R9 `distinct on (v.document_id)` query with `known = coalesce(known_at, now())`, `valid = coalesce(as_of, known)`. `as_of`, `known_at` and the collection list are query parameters.
  - `passage_filter(scope, version_ids)`: live `p.is_current and d.deleted_at is null and d.collection_id = any(%s)`; temporal `p.version_id = any(%s)`. A temporal scope with `version_ids=None` raises `ValueError`.
- `backend/retrieval/bm25.py`:
  - `bm25_term_score` and `bm25_reference` (Python reference; N and avgdl over every passage in `doc_lens`; repeated query terms count once; returns only passages with at least one query term, like the SQL).
  - `query_terms`: sorted distinct lexemes of `to_tsvector('english', query)`.
  - `bm25_search`: one SQL statement with CTEs `vis` (visible ids and `token_count`), `stats` (`count(*)::float8`, `avg(token_count)::float8` over `vis`), `df` (per term, over `passage_terms` joined to `vis`), then the scored sum joined to `vis`, `df`, `stats`. Ordered by score desc, passage id; `limit`. k1 and b are bound parameters. Empty terms, or a temporal scope with an empty version list, return `[]` without querying.
- `tests/pg_fixtures.py`: `seed_docs(db, services, tenant_id, collection_id, texts, *, effective_from=None) -> list[UUID]` (one document per text via `parse_text` and `create_version`; default `effective_from` is now). Added to `__all__`. File stays CRLF in the working copy (checked: 0 LF-only lines), diff is +49 lines only.
- `tests/test_bm25.py`: 10 tests (below).

## Files changed (committed)

```
backend/retrieval/__init__.py  |   1 +
backend/retrieval/bm25.py      | 134 ++++
backend/retrieval/types.py     |  63 ++
backend/retrieval/visible.py   |  54 ++
tests/pg_fixtures.py           |  49 ++
tests/test_bm25.py             | 344 ++++
```

## Tests

| Test | What it proves |
|---|---|
| `test_term_score_literals` | both literals from the brief within 1e-12 |
| `test_reference_counts_every_passage_and_distinct_terms` (extra, unit) | N and avgdl come from all of `doc_lens`; repeated term counts once; only matching passages returned |
| `test_sql_matches_reference` (pg) | 12 passages over 3 documents (asserted), 5 queries (multi-term, stopwords, repeated words, 7 terms); `passage_terms` and `token_count` read back, `bm25_reference` computed, every SQL score within 1e-9 (abs) and identical order; corpus has no score ties (asserted); `limit=3` is the top 3 |
| `test_query_terms_are_distinct_sorted_lexemes` (extra, pg) | `"Velvet velvet EMBER the granite"` -> `["ember", "granit", "velvet"]`; stopword-only -> `[]` |
| `test_stats_only_over_visible_set` (pg) | scores for collection X identical (1e-12) before and after adding 50 matching documents to Y outside the scope; control: with Y in scope X's scores change by more than 1e-3, so the test can detect a leak |
| `test_deleted_documents_excluded` (pg) | deleted document's 4 passages absent; live scores equal a control collection holding only the live texts (so statistics exclude the deleted document); a `known_at` after the deletion also excludes it |
| `test_stopword_only_query_empty` (pg) | `"the and of"` -> `[]` |
| `test_empty_scope_returns_nothing` (extra, pg) | empty `collection_ids` (live and as-of) -> `[]` |
| `test_as_of_versions` (pg) | v1 2026-01-01, v2 2026-06-01: as_of 2025-12-01 -> no versions and no hits; 2026-03-01 -> v1 (not current); live -> `None` and v2 passages. After capturing `known`, v3 effective 2026-07-01: as_of 2026-08-01 with `known_at=known` -> v2, with `known_at=None` -> v3 |
| `test_deleted_after_known_at_still_visible` (pg) | deleted after `known`: `known_at=known` still sees it, live does not, `known_at` after the deletion does not |

Time handling in the as-of and deletion tests: no Python clock. The helper `known_after` reads `greatest(max(document_versions.created_at), max(documents.created_at), max(documents.deleted_at))` from Postgres, then polls `clock_timestamp() > known` (50 ms steps, 20 s cap) so anything created afterwards has a strictly larger `created_at` even if the clock stepped back. `effective_from` values are the explicit dates from the brief.

## RED evidence

Run 1 (modules missing):

```
> wsl.sh timeout 300 python -m pytest tests/test_bm25.py -q
ERROR collecting tests/test_bm25.py
tests/test_bm25.py:15: in <module>
    from backend.retrieval.bm25 import bm25_reference, bm25_search, bm25_term_score, query_terms
E   ModuleNotFoundError: No module named 'backend.retrieval'
1 error in 0.65s
```

Run 2 (types complete, `visible.py` and `bm25.py` as signature stubs raising `NotImplementedError`, so every test fails inside the code under test and `seed_docs` is proven to work):

```
FAILED tests/test_bm25.py::test_term_score_literals - NotImplementedError
FAILED tests/test_bm25.py::test_reference_counts_every_passage_and_distinct_terms
FAILED tests/test_bm25.py::test_sql_matches_reference - NotImplementedError
FAILED tests/test_bm25.py::test_query_terms_are_distinct_sorted_lexemes - Not...
FAILED tests/test_bm25.py::test_stats_only_over_visible_set - NotImplementedE...
FAILED tests/test_bm25.py::test_deleted_documents_excluded - NotImplementedError
FAILED tests/test_bm25.py::test_stopword_only_query_empty - NotImplementedError
FAILED tests/test_bm25.py::test_empty_scope_returns_nothing - NotImplementedE...
FAILED tests/test_bm25.py::test_as_of_versions - NotImplementedError
FAILED tests/test_bm25.py::test_deleted_after_known_at_still_visible - NotImp...
10 failed in 1.66s
```

## GREEN evidence

```
> wsl.sh timeout 300 python -m pytest tests/test_bm25.py -q
..........   10 passed in 1.47s
```

Re-run 3 more times after the lint edits: 10 passed each time (1.4 to 1.9 s).

Mutation checks (temporary edits to production code, reverted, then re-run green):

- `stats` CTE reading from `passages` instead of `vis` -> `test_stats_only_over_visible_set` fails (the parity test does not, as expected: everything is visible there).
- Removing `v.created_at <= t.known` from `visible_version_ids` -> `test_as_of_versions` fails (`assert visible(scope(as_of=august, known_at=known)) == [v2]`).

## Full suite and lint

```
> wsl.sh timeout 900 python -m pytest -q
868 passed, 2 deselected, 2 warnings in 128.71s (0:02:08)      exit 0
```

The 2 warnings are the existing `tests/test_generator.py` "coroutine ... was never awaited" ones; the 2 deselected are the `models` marker tests (default `-m 'not models'`).

```
> wsl.sh timeout 120 ruff check .
All checks passed!
> ruff format --check tests/test_bm25.py backend/retrieval tests/pg_fixtures.py
6 files already formatted
```

`ruff format --check .` over the whole repo reports 30 files that would be reformatted. None are mine (all pre-existing); I did not touch them.

## Decisions

- Placeholders carry explicit casts (`%s::uuid[]`, `%s::text[]`, `%s::float8`, `%s::timestamptz`) so empty lists and Python floats have a fixed type; the fragments are otherwise the ones in the brief. `{visible}` in the BM25 SQL is `str.format` of a code-owned fragment from `passage_filter`; every value is still a bound parameter.
- `passage_filter` for a live scope ignores `version_ids`; for a temporal scope it requires them (`ValueError` otherwise) instead of silently returning everything.
- The temporal fragment is exactly `p.version_id = any(%s)` as in the brief; collection and deletion rules for temporal scopes live in `visible_version_ids`.
- `bm25_reference` returns only passages with at least one query term. The SQL does the same, so key sets match.
- Three tests beyond the brief's list (reference unit test, `query_terms`, empty scope) cover brief requirements stated in the interface text (N and avgdl over all passages given, distinct terms, empty `vis` returns `[]`). A fourth extra (backdated version) was dropped as redundant with `test_as_of_versions`.
- Corpus design: words of 6+ letters, 60 to 100 words per paragraph, so each paragraph is exactly one passage (the 800-char target cannot hold two) while token counts and term frequencies vary.

## Concerns

- None blocking. One environmental note: the as-of tests depend on the database clock being monotonic between statements; `known_after` waits out a backwards step but cannot help if the clock steps back between the wait and the next insert (milliseconds window).
- `seed_docs` defaults `effective_from` to Python's `datetime.now(UTC)`; no test in this task relies on that default (as-of tests pass explicit dates).
- The retrieval-mode wrappers (dense, fts, fusion, `PassageHit` construction) are later tasks; `types.py` has no dedicated test because it is declarative and exercised through `SearchScope` in every test.
