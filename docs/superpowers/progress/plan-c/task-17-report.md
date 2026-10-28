# Plan C Task 17 report: re-verification when sources change, and manual re-verification

Status: DONE_WITH_CONCERNS
Commit: `587cf57 feat: re-verify live claims when cited sources change` (branch `feature/evidencegraph`, not pushed)
Tests: 1434 passed, 3 deselected (`-m 'not models'`), 0 failed; `ruff check .` clean. 40 new tests in `tests/test_reverify.py`.

## What I built

- `backend/auth/service.py`: `Principal.auth` gains `"system"`; `async principal_for_user(db, tenant_id, user_id) -> Principal | None`
  (system tx, user must exist and not be disabled, role from the current membership, `auth="system"`).
  Nothing in HTTP authentication produces `"system"` (`resolve_session`, `resolve_api_key`, `login`, `switch_tenant` are untouched).
- `backend/provenance/reverify.py` (new):
  - `RecheckCandidate`, `claims_to_recheck` (R3): candidate analyses are `succeeded`, `as_of is null`, and contain the document's
    collection. Precedence per claim: (a) `evidence_changed`, else only when changed passages exist (b) `thin_evidence`
    (links < `evidence_k`), else (c) `new_passage` (best rerank logit against the changed texts >= the claim's lowest non-null
    stored rerank score; a claim whose links all have null scores qualifies without a model call). Ordered by analysis
    `created_at, id`, then claim ordinal. Reranker scoring runs in `asyncio.to_thread`, one call per claim that reaches (c).
  - `recheck_claims`: loads the analysis (returns 0 when it is gone, not `succeeded`, or has `as_of`), `principal_for_user` for the
    creator (None -> 0, nothing written), scope = stored `collection_ids` intersected with `readable_subset(creator)` (R8),
    rebuilds `Claim` objects (no pins), `VerificationEngine.verify` with no transaction open, then one transaction appends a
    `claim_checks` row per claim: `reason`, verdict, probabilities, `evidence` JSON
    (`passage_id, version_id, relation, rank, span_start, span_end, nli{entail, contra, neutral}`, ids only), distinct
    `checked_version_ids`, `changed = verdict != (latest check verdict, else the claim's original verdict)`.
  - `handle_reverify_document` (payload `document_id, old_version_id, new_version_id`), `handle_reverify_analysis`
    (payload `analysis_id`, reason `manual`). Both raise `PermanentJobError("calibrator_missing")` or
    `PermanentJobError("calibrator_mismatch")` before any work.
  - `log_reverify_failure`: the failure hook for both kinds. Logs one constant message, `kind` and `error_code`
    (whitelisted extras) plus `tenant_id` and `job_id` context vars (str). Never touches the database, never raises.
- `backend/jobs/registry.py`: `reverify_document` and `reverify_analysis` registered in `build_handlers` and `build_failure_handlers`.
- `backend/api/routes/analyses.py` + `backend/api/schemas_v1.py`: `POST /api/v1/analyses/{id}/reverify`.
  `require_analysis` (creator or owner/admin, everyone else 404 `analysis_not_found`), then 409 `not_live` when `as_of` is set,
  409 `not_finished` when the status is not `succeeded`, else enqueue `reverify_analysis` (priority 5, payload `{analysis_id}`)
  and answer 202 `{"status": "queued"}` (`ReverifyAcceptedOut`, new small response model).
- Tests: `tests/test_reverify.py` (new), `tests/test_worker.py` (registry test now lists the five kinds),
  `tests/test_ingest_upload.py` (one line, see Decisions).

## Files changed

```
 backend/api/routes/analyses.py   |  41 +++
 backend/api/schemas_v1.py        |   4 +
 backend/auth/service.py          |  27 +-
 backend/jobs/registry.py         |  15 +-
 backend/provenance/reverify.py   | 312 (new)
 tests/test_ingest_upload.py      |   3 +-
 tests/test_reverify.py           | 748 (new, 40 tests)
 tests/test_worker.py             |   4 +-
 8 files changed, 1148 insertions(+), 6 deletions(-)
```

## RED evidence

Commands run from the worktree root through the Windows killer, e.g.
`.venv\Scripts\python.exe .local\run.py 340 .local\t17-red.log -- wsl -d Ubuntu -- bash <wt>/.local/wsl.sh timeout 280 python -m pytest ...`

1. All tests written first, nothing implemented. `pytest tests/test_reverify.py tests/test_worker.py -k "reverify or registries or principal"`:
   ```
   ERROR collecting tests/test_reverify.py
   E   ImportError: cannot import name 'principal_for_user' from 'backend.auth.service'
   Interrupted: 1 error during collection   (exit=2)
   ```
2. Registry test alone (`pytest tests/test_worker.py -k registries`):
   ```
   AssertionError: assert {'ingest_uplo...is', 'search'} == {'ingest_uplo...is', 'search'}
   Extra items in the right set: 'reverify_analysis', 'reverify_document'      (1 failed)
   ```
3. After implementing `principal_for_user` and `reverify.py` but before registering the jobs and adding the route, so the
   behavior tests fail for their own reasons (`pytest tests/test_reverify.py`): 23 failed, 15 passed.
   - The 15 passes are `principal_for_user` (5) and `claims_to_recheck` (10), whose code existed by then.
   - End-to-end failures: `ValueError: not enough values to unpack (expected 1, got 0)` on `(check,) = await checks_of(...)`
     (the `reverify_document` job was never claimed, so no check row), and `assert 'queued' == 'succeeded'` for the job row.
   - Route tests: `assert 404 == 409`, `assert (404, {...}) == (202, {'status': 'queued'})`.
   - Hook tests: `KeyError: 'reverify_analysis'` from `build_failure_handlers(None)[kind]`.
   - Job failure code tests: no `calibrator_*` job status because the kinds were not registered.

## GREEN evidence

- After registering the jobs and adding the route: `pytest tests/test_reverify.py tests/test_worker.py -k "reverify or registries or principal or trigger or import_paths or ..."`
  -> `46 passed` (exit 0). Slowest test 2.2 s.
- `test_worker_import_paths_do_not_load_ml_libraries` still passes: `backend.provenance.reverify` imports no torch.
- Full suite, first run: `1 failed, 1433 passed`. The failure was `tests/test_ingest_upload.py::test_new_version_enqueues_reverify`
  (`assert 2 == 1` on `drain(...)`): it pinned the state where `reverify_document` was enqueued but had no handler. Fixed (Decisions).
- Full suite, final (`pytest -q`, inner timeout 900): `1434 passed, 3 deselected, 2 warnings in 261.04s`, exit 0.
  The 2 warnings are pre-existing (`tests/test_generator.py`, `backend/core/generator.py:50`, an un-awaited AsyncMock).
- `ruff check .` -> `All checks passed!` (exit 0).
- `ruff format --check` on every file I touched is clean except `tests/test_worker.py`, which has a pre-existing formatting
  difference at line 1120 (two blank lines before a decorator) that I left alone. I ran `ruff format` on my two files whose
  format check failed (`analyses.py`, `test_reverify.py`); the `analyses.py` diff is still only my additions (41 insertions, 0 deletions).

## Mutation check (production code mutated one change at a time, then restored; none left behind)

| Mutation in `reverify.py` | Test that failed |
|---|---|
| `>= lowest` -> `> lowest` | `test_related_change_in_another_document_rechecks_claim`, `test_trigger[...Orders take 7 days...new_passage]` |
| scope without `readable_subset` intersection | `test_creator_lost_access_scope_shrinks` |
| `changed` compared with the original verdict instead of the latest check | `test_changed_compares_with_the_latest_check` |
| reranker called on the event loop (no `to_thread`) | `test_reranking_runs_off_the_event_loop_once_per_claim` |
| `elif changed_texts:` -> `else:` (thin/new without changed passages) | `test_a_version_with_nothing_new_flags_no_thin_evidence` |
| drop the `as_of` guard in `recheck_claims` | `test_as_of_analysis_never_rechecked` |
| ignore a missing principal | `test_creator_removed_no_check[removed]` |
| `evidence_changed` ignoring the new version's hashes | first run: survived. Added `test_an_evidence_passage_kept_in_the_new_version_is_not_evidence_changed`, which then failed under the mutation |
| drop the analysis/collection filter in the SQL | `test_only_analyses_over_the_documents_collection_are_considered` |

## Test list (tests/test_reverify.py)

Brief's seven: `test_changed_evidence_flips_verdict`, `test_unrelated_change_skips_claim`, `test_new_passage_triggers_recheck`,
`test_as_of_analysis_never_rechecked`, `test_creator_lost_access_scope_shrinks`, `test_creator_removed_no_check` (removed and disabled),
`test_manual_reverify_route`, `test_manual_reverify_as_of_409`.
Added: positive control for the skip test (`test_related_change_in_another_document_rechecks_claim`), `changed` vs the latest check,
409 `not_finished` (queued, running, failed), visibility (404 for other member/other tenant/unknown, 202 for creator and owner),
job failure codes (document/analysis x missing/mismatch), 7 `test_trigger` cases for the precedence and boundaries,
null rerank scores, collection filter, kept evidence passage, nothing-new version, rerank thread/once-per-claim,
`principal_for_user` (member, removed, disabled, unknown user, other tenant), failure hooks (logs only, never raises).
The first test also captures logs and asserts no claim or passage text appears and that the capture works (`claims rechecked`).

## Decisions

1. `test_unrelated_change_skips_claim` uses two documents as the brief says, but with a calibrator whose `evidence_k=1`
   (`make_test_calibrator(evidence_k=1)`). With the default `evidence_k=4` and two documents every claim has fewer than
   4 evidence links, so R3(b) (`thin_evidence`) correctly flags it and the brief's "no check row" would be wrong. With depth 1 the
   claim has exactly one link (score 4), the other document is not evidence, and the changed passage scores 2. I followed R3 as
   written and chose the setup that makes the brief's expectation true. The direct trigger tests use a five-document corpus
   (claim evidence = four documents, the fifth unlinked) with `evidence_k` passed straight to `claims_to_recheck`.
   `test_new_passage_triggers_recheck` also uses depth 1, so it exercises (c) rather than (a) or (b).
2. `tests/test_ingest_upload.py::test_new_version_enqueues_reverify` now drains only `ingest_upload` (`kinds=("ingest_upload",)`).
   It asserts the job Plan B enqueues (payload, priority 5, idempotency key, status `queued`); once the handler exists, an
   unfiltered `drain` also runs it. Intent unchanged, one line.
3. `ReverifyAcceptedOut` added to `schemas_v1.py` (the brief lists only the route file) so the OpenAPI schema is typed like
   the sibling routes.
4. The route checks `as_of` before status, so an as-of analysis that is not finished says `not_live`.
5. `recheck_claims` returns 0 for an analysis that is missing, not `succeeded`, or has `as_of`, so a directly enqueued job cannot
   violate R3/R9 either.
6. Error mapping mirrors `analysis/service.py` only where it applies: `calibrator_missing` and `calibrator_mismatch` are
   permanent. Anything else propagates and the worker records `internal_error` (log has only exception type and frames).
7. No idempotency key on `reverify_analysis`: each manual request is a deliberate new check.
8. Failure hooks: one shared function for both kinds. The brief's "ids as whitelisted fields" is satisfied by the `tenant_id` and
   `job_id` context vars (the hook binds them as `str`, because it also runs from the stale-lease loop where `job_id` is not bound);
   document and analysis ids are not in `ALLOWED_EXTRA` and I did not widen the whitelist.

## Concerns

1. Job timeout: the worker gives every kind except `run_analysis` the 300 s default (`cli/main.py` sets `timeouts_s` for `run_analysis`
   only). `reverify_document` rechecks every touched claim of every touched analysis with the real NLI model, so a popular document
   could exceed 300 s and be retried. Not changed (outside the brief); worth a dedicated timeout in Task 18 or later.
2. Retry duplicates: each analysis commits its checks in its own transaction. If `reverify_document` fails after some analyses
   (timeout, internal error) the retry rechecks them again and appends a second row (`changed=false`, so `changed_since` stays right).
   Rows are append-only, so this is noise, not corruption.
3. A creator who lost access to every stored collection gets a scope with no collections: the check is written with verdict
   `insufficient` and no evidence (and `changed` true). This is R8 as written, but the user may read it as "the source went away".
   If that is unwanted, `recheck_claims` should return 0 for an empty scope.
4. `claims_to_recheck` takes a connection, so `handle_reverify_document` holds one tenant transaction open while the reranker
   scores (one call per claim that reaches R3(c)). Fine for the sizes in the plan; the interface is the brief's.
5. No rate limit on `POST /analyses/{id}/reverify`: a user can queue several jobs for one analysis (each does real model work).
6. `test_worker.py` has a pre-existing `ruff format` difference (line 1120); not mine, not touched.
