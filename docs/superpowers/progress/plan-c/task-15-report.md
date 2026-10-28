# Plan C Task 15 report: passage resolution and evidence validation

Status: DONE (one commit, not pushed)
Commit: `4ec6772 feat: add passage provenance resolution and evidence validation` (branch `feature/evidencegraph`)

## What was built

- `backend/provenance/resolve.py`: frozen `ResolvedPassage` (the spec section 15 fields plus `collection_id`, `ordinal`, `section`) and `async resolve_passage(conn, passage_id)`. One SQL statement joins the passage, its version, its document, and the document's current version.
  - `context_before`: up to 300 characters of the version text ending at `start_char`, clipped at the text start: `substr(v.text, start_char - least(start_char, 300) + 1, least(start_char, 300))`.
  - `context_after`: `substr(v.text, end_char + 1, 300)`.
  - `is_current_version = (v.id = d.current_version_id) is true`.
  - `superseded_by_version`: current version number when the passage's version is not the current one, else null.
  - `current_equivalent_passage_id`: null when the document is deleted; the passage itself when its version is current; otherwise the lowest-ordinal passage of the current version with the same `text_sha256`, or null.
- `backend/provenance/citations.py`: frozen `LinkProblem(kind, link_id)`, frozen `CitationReport`, `async validate_analysis(conn, analysis_id)`. Two read-only queries (evidence links, citations); counts and problems are assembled in Python.
- `backend/api/routes/passages.py`: `GET /api/v1/passages/{passage_id}` returning `PassageOut`. Resolves, then requires `collection_id in readable_subset(principal, [collection_id])`. A missing id, another tenant's id, and an unreadable collection all give 404 `passage_not_found`.
- `backend/api/schemas_v1.py`: `PassageOut` (all `ResolvedPassage` fields). `backend/api/app.py`: router included in v1.
- `tests/test_provenance.py`: 15 tests (all `pg`).

Files changed (7): `backend/api/app.py` (+2), `backend/api/schemas_v1.py` (+28), new `backend/api/routes/passages.py`, `backend/provenance/__init__.py`, `backend/provenance/resolve.py`, `backend/provenance/citations.py`, `tests/test_provenance.py`. Existing CRLF working copies of `app.py` and `schemas_v1.py` kept their line endings (`git ls-files --eol`: `i/lf w/crlf`); new files are LF. No em dashes.

## Tests (15)

Brief list: `test_resolve_current_passage`, `test_resolve_superseded_with_equivalent`, `test_resolve_changed_passage_has_no_equivalent`, `test_passages_route_requires_read`, `test_validate_clean_analysis`, `test_validate_detects_outside_span` (parametrized: `past_the_end`, `before_the_start`, so 2 cases).

Added for behaviors the brief states or that the implementation defines:
- `test_resolve_context_is_clipped_at_text_edges` (first passage: empty `context_before`; last passage: empty `context_after`)
- `test_resolve_equivalent_prefers_lowest_ordinal` (duplicate text in the current version; a current passage is its own equivalent)
- `test_resolve_deleted_document_has_no_equivalent`
- `test_resolve_unknown_or_foreign_passage_is_none` (missing id and another tenant's connection)
- `test_validate_detects_span_text_mismatch`, `test_validate_detects_missing_passage`, `test_validate_detects_link_naming_the_wrong_version`, `test_validate_detects_citation_version_mismatch`

Fixtures: documents are built from ~500 character sentences containing an emoji and an accented letter, so each sentence is its own passage (verified by a precondition assert on the stored passage texts) and code point offsets differ from byte offsets. The analysis comes from the real run_analysis flow over the HTTP routes with the services fakes (`ANSWER = "Refunds take 30 days [P1, P9]."`: one claim, 2 evidence links, citations P1 valid and P9 unknown). The out-of-span link is inserted through `tenant_tx` with valid FK values. The three corruption tests (mismatch, missing passage, version mismatch) cannot be produced under the schema's triggers and FKs, so they use a throwaway superuser connection to the test database with `session_replication_role = replica`, the same `postgres` access the fixtures already use.

## RED evidence

Command: `... wsl.sh timeout 280 python -m pytest tests/test_provenance.py -q` (log `.local/t15-red.log`)

```
ERROR collecting tests/test_provenance.py
tests/test_provenance.py:15: in <module>
    from backend.provenance.citations import CitationReport, LinkProblem, validate_analysis
E   ModuleNotFoundError: No module named 'backend.provenance'
1 error in 0.96s
exit=2
```

The 12 tests written first failed for the right reason (feature missing). The 3 later additions (`prefers_lowest_ordinal`, `link_naming_the_wrong_version`, `before_the_start` case) were written after the first GREEN; they are proven by the mutation run below (each is the only or one of the killers of its mutation).

## GREEN evidence

First implementation, 12 tests: `12 passed in 5.04s`. Final code, 15 tests (`.local/t15-green3.log`): `15 passed in 5.94s`.

## Mutation check (tests can fail)

`.local/t15-mutate.py` applies 22 single-line mutations to the implementation, runs `tests/test_provenance.py`, and restores each file byte for byte (final run `.local/t15-mutation2.log`): `survivors: []`. Mutations killed: context_after and context_before off by one, context_before not clipped, context window 299, equivalent ignores hash, equivalent highest ordinal, current is not its own equivalent, deleted keeps equivalent, never superseded, always current, end span unchecked, start span unchecked, slice compare inverted, slice offset, link pair ignores version, citation version unchecked, unknown marker flagged, unknown markers uncounted, citation problems dropped, route skips read check, route wrong error code. An earlier run found one survivor (`greatest(...)` clip): `substr` with a negative start already clips, so that mutant was behavior-equivalent; I replaced the clip with the explicit `least(start_char, 300)` form (no reliance on negative-start semantics, no lateral join) and the equivalent mutant became killable.

## Full suite and lint

- Full suite (`timeout 900 python -m pytest -q`, `.local/t15-full.log`): `1380 passed, 3 deselected, 2 warnings in 231.23s`. The 3 deselected are the `models` marker (default `addopts`). The 2 warnings are pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in `tests/test_generator.py` (legacy code, not touched).
- `ruff check .`: `All checks passed!`
- `ruff format --check` on the 7 touched files: `7 files already formatted`.
- Commit hook: gitleaks, no leaks found.

## Decisions

1. Current-version passage: `current_equivalent_passage_id` is the passage itself, not "lowest ordinal with the same hash". The brief says "a passage of the current version with the same text_sha256 (lowest ordinal)"; for unique text both give the same result. With duplicate text in the current version the literal reading would point a current passage at a different passage; pinned by `test_resolve_equivalent_prefers_lowest_ordinal`. Easy to change (remove one `when` line) if the literal reading is wanted.
2. Evidence link joined to its passage on `(passage_id, version_id)`: a pair no passage matches is `missing_passage` (the brief's kinds have no link-level version mismatch). A citation joins on `passage_id` alone, so a passage of another version is `citation_version_mismatch` and a gone passage is `missing_passage`.
3. For a citation problem, `LinkProblem.link_id` holds the citation id. `n_valid_links = n_links - link problems`; `n_valid_citations` counts `valid` citations without a problem; unknown markers are counted in `n_unknown_markers` only and never produce a problem.
4. A deleted document's passages still resolve through the route when the collection is readable (past analyses cite them); only the equivalent is nulled, as the brief says. Superseded number for a deleted document still shows the current version number if different.
5. Authorization order in the route: resolve first, then `readable_subset`, inside one `tenant_tx`; nothing is returned or logged unless readable. No logging added (no document text in logs).
6. The brief's "Review Focus" numbering is from an earlier draft; followed this task's own test list as instructed.
7. Tests import helpers (`collection_with_docs`, `post_ask`, `wait_done`, `background_worker`, ...) from `tests/test_analysis_routes.py`, following the existing precedent of `test_analysis_routes` importing from `test_collections`. Not importing its autouse `_shared_services` fixture: the passages route does not use `app.state.services`.

## Concerns

- Corruption tests need superuser access to the test Postgres (already required by `pg_fixtures`: `ADMIN_URL`); they would not work against a managed database without superuser.
- WSL returned a transient `Wsl/Service/0x8007274c` connection error once (during the first ruff call, right after the mutation run); a probe and retry succeeded, no code impact.
- `validate_analysis` takes a connection and an id and does no authorization (internal API for the temporal benchmark and reverify work, as in the brief). A caller exposing it over HTTP must check analysis visibility first (`require_analysis`).
