# Plan C Task 14 report: ask, verify, and analysis read routes with redaction

Status: DONE_WITH_CONCERNS (concerns are small judgment calls, listed at the end).

Commit: `4e7f692 feat: add ask, verify, and analysis routes with evidence redaction`
(on top of 05433e5; 5 files, 1247 insertions, 2 deletions; not pushed). Pre-commit gitleaks hook ran: no leaks.

## What was built

- `backend/authz/policy.py`
  - `readable_subset(conn, principal, collection_ids) -> set[UUID]`: any collection kind; owners and admins read every collection of the tenant, others need a grant; unknown or foreign ids are simply absent.
  - `require_analysis(conn, principal, analysis_id) -> dict`: visible to the creator or a tenant owner/admin (R8); missing, someone else's, and cross-tenant (RLS) all give `404 analysis_not_found`.
- `backend/api/schemas_v1.py`: `AskQuestion` (same rules as `SearchQuery`), `VerifyQuestion`, `Answer` string types; `AnalysisOptionsIn`, `AskIn`, `VerifyIn` (field validator: `consistency_samples` must be 0 for verify, else 422), `AnalysisAcceptedOut`, `AnalysisSummaryOut`, `NLIOut`, `EvidenceOut`, `ClaimCheckOut`, `ClaimOut`, `CitationOut`, `AnalysisOut`, with the field names from the brief.
- `backend/api/routes/analyses.py` (new), mounted in `backend/api/app.py` under `/api/v1`:
  - `POST /ask`, `POST /verify`: 202 `AnalysisAcceptedOut` through `create_analysis_request` (so 429 `too_many_analyses` and 404 `collection_not_found` come from the service). Options are stored as the validated, fully expanded options object.
  - `GET /analyses?mine=&limit=&cursor=`: `Page[AnalysisSummaryOut]`; `mine=false` is 403 `forbidden` for non-owner/admin; cursor key validated with `decode_uuid_cursor` first (bad key is 400 `bad_cursor`, not a 500 from the SQL cast); `question_preview` is the first 200 characters; `live = as_of is null`; `changed_since` = any `claim_checks.changed` for the analysis (one query per page).
  - `GET /analyses/{id}`: `AnalysisOut`. Claims, evidence (`get_evidence`), latest checks, and citations (`get_citations` plus a join of passages/versions/documents for collection, title, version number) are read in one tenant transaction; `readable_subset` runs once over every collection id seen.
  - Redaction (R8): an evidence item or citation whose passage collection is not readable by the viewer comes back `redacted: true` with `text`, `span_text`, `document_title`, `document_id`, `version_number`, `effective_from`, `page_start`, `page_end` (and `passage_is_current`, see decisions) null; ids, relation, rank, span offsets, and NLI numbers stay. `span_text` is `substr(document_versions.text, span_start + 1, span_end - span_start)`, fetched for readable items only in one batched `unnest` query, so a withheld span never leaves the database. `conflict` is true when a claim has at least one `supports` and one `contradicts` item, redacted or not.
  - Nothing is logged by the routes (no question or answer text anywhere).

Files: created `backend/api/routes/analyses.py`, `tests/test_analysis_routes.py`; modified `backend/authz/policy.py`, `backend/api/schemas_v1.py`, `backend/api/app.py`. Line endings: existing files stay CRLF in the working copy (only my lines added); new files are LF (`git ls-files --eol` shows `i/lf w/lf`).

## Tests (`tests/test_analysis_routes.py`, 16 tests, all `pg`)

The brief's 10: `test_ask_then_get_full_analysis`, `test_verify_route`, `test_redaction_after_revoked_grant`, `test_other_member_cannot_see`, `test_cross_tenant_404`, `test_mine_false_requires_admin`, `test_validation_limits`, `test_unreadable_collection_404`, `test_backpressure_route_429`, `test_conflict_flag`.
Added because the brief's behaviors need them to be pinned: `test_admin_without_grant_reads_everything` (admin role without a grant reads unredacted and lists the tenant), `test_list_pages_newest_first` (limit, cursor, bad cursor 400), `test_list_item_fields` (summary keys, 200-char preview, `live=false` with `as_of`), `test_changed_since_and_latest_check` (an unchanged re-check keeps `changed_since` false, a changed one flips it; `latest_check` is the newest row), `test_options_are_stored`, `test_blank_verify_question_is_not_stored`.

Harness: a background drain loop (50 ms) is started per test through an async context manager that cancels and awaits the task in `finally`; `GET` polls are bounded (30 s deadline, then the last body is returned so the assertion shows the real status and error code). `api_app.state.services` is set to the `services` fixture by an autouse fixture (routes themselves do not read it). A local `client4` fixture provides Dana. The conflict test uses documents `Refunds take 30 days.` and `Refunds take 45 days.` and gets one `supports` and one `contradicts` item (R-C2). The redaction test uses two collections (one revoked, one kept) and checks both directions: nothing redacted before the revoke, only the revoked collection's items redacted after, owner unaffected, and ids/relation/rank/NLI kept.

## RED evidence

```
.venv\Scripts\python.exe .local\run.py 340 .local\t14-red.log -- wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 280 python -m pytest tests/test_analysis_routes.py -q --no-header -p no:cacheprovider --tb=line
```
Result: `14 failed in 4.83s`, exit=1 (the 14 tests that existed then; 2 more were added afterwards, see below). Every failure is the missing route: `assert 404 == 202` on `POST /api/v1/ask` / `/verify` (`{"code":"not_found"}`), `'not_found' == 'collection_not_found'`, `(404, 'not_found') != (422, 'validation_error')`, `(404, 404) == (202, 202)`. Nothing hung and nothing touched a model.

## GREEN evidence

- New file, first implementation: `14 passed in 5.13s`.
- After adding `test_admin_without_grant_reads_everything` and `test_blank_verify_question_is_not_stored` and tightening the changed-since test: `16 passed in 5.89s`.
- Mutation check (`.local/t14-mutate.py`, gitignored, files restored byte for byte after each run): 17 mutations, 17 killed, 0 survivors. Killed by the expected tests: redaction off, citation never redacted, evidence text always shown (`test_redaction_after_revoked_grant`); conflict any-relation (`test_ask_then_get_full_analysis`); analysis visible to all (`test_other_member_cannot_see`); admin not allowed (3 tests); `mine` ignored / `mine=false` open (`test_mine_false_requires_admin`, admin test); span offset off by one (`test_ask_then_get_full_analysis`); `readable_subset` ignoring admin (`test_admin_without_grant_reads_everything`); `changed` filter dropped (`test_changed_since_and_latest_check`); page trim and cursor check removed (`test_list_pages_newest_first`); samples upper bound, verify samples allowed (`test_validation_limits`); blank verify question (`test_blank_verify_question_is_not_stored`); preview length (`test_list_item_fields`). An early draft of the changed-since test survived the `changed` mutation (it never inserted an unchanged check); that is why it now inserts one first.

## Full suite and lint

- `timeout 900 python -m pytest -q`: `1365 passed, 3 deselected, 2 warnings in 195.53s`, exit=0. The 3 deselected are the `models` marker (excluded by `addopts`). The 2 warnings are the pre-existing `coroutine ... was never awaited` in `tests/test_generator.py`, unrelated.
- `ruff check .`: `All checks passed!`
- `ruff format --check` on the 5 files I touched: `5 files already formatted`. Repository-wide `ruff format --check .` reports 31 other files that would be reformatted (for example `tests/test_schema_global.py`, `tests/test_similarity.py`, `tests/test_worker.py`); none is mine and `git status` shows I changed none of them, so this is pre-existing and I left it alone.

## Decisions (where the brief left room or conflicted)

1. `as_of` is `AwareDatetime | None` (as `SearchIn.as_of`), not bare `datetime`: a naive timestamp would be silently read in the database session time zone. Strictly narrower than the brief.
2. `passage_is_current` is also nulled when redacted. The brief types it `bool | None` (the only way to get null is redaction) but omits it from the null list; leaving it would tell a viewer who lost access whether the passage's document was updated since. One-line change if the controller prefers it kept.
3. `NLIOut` fields and `EvidenceOut.rerank_score` are `float | None` because the database columns are nullable (the writer always sets the NLI numbers), so a null can never turn into a 500.
4. `options` is stored as `body.options.model_dump()` (both keys, defaults filled), which is what `run_analysis` reads (`options.get("extraction", "llm_atomic")`).
5. The brief's file list excludes `backend/db/repos/analyses.py`, so the three small read queries it needs beyond the repo (citation passage details, `changed_since`, batched `span_text`) are private helpers in the routes module, all parameterized.
6. Verify `question` is stripped and a blank one is stored as null; verify `answer` is stored byte for byte (no strip) because claim offsets point into it; both reject NUL and lone surrogates like the other request strings.
7. Review Focus numbering in the brief is from an earlier draft; I followed this task's own test list as instructed.

## Concerns

- `AnalysisOut.collection_ids` is returned as stored, including ids of collections the viewer can no longer read. R8 only redacts evidence, citations, and passages, so I left it; these are ids from the viewer's own request, but say so if you want them filtered too.
- Redaction is computed at read time against current grants only; it does not look at collection deletion or document soft deletion (a soft-deleted document's evidence still shows for readable collections, as `get_evidence` already joins without a `deleted_at` filter).
- Importing `backend.analysis.service` into the API app is cheap (no torch), confirmed by grep of module imports; the full suite and the app factory tests pass.
