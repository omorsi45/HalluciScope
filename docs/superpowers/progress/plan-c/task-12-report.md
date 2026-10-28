# Plan C Task 12 report: analyses schema and repository

Status: DONE_WITH_CONCERNS (concerns are design notes for Task 13, none blocking)

Commit: `c4c02e8 feat: add analyses and evidence graph schema` on `feature/evidencegraph` (not pushed).

## What was built

- `backend/db/migrations/0004_analyses.sql`: `analyses`, `claims`, `evidence_links`, `citations`, `claim_checks`, `feedback`, each with the forced-RLS `tenant_isolation` policy copied from 0002/0003.
  - Columns, checks, and the three `analyses` indexes follow the brief verbatim.
  - R-C1 applied: `unique (tenant_id, id)` on `analyses` and `claims`; `claims` and `citations` reference `analyses` through `(tenant_id, analysis_id)`; `evidence_links`, `claim_checks`, `feedback` reference `claims` through `(tenant_id, claim_id)`, all `on delete cascade`; `evidence_links` and `citations` reference `passages` through `(tenant_id, passage_id, version_id)` (uses the unique already declared in 0003). `users` and `tenants` stay single-column references.
  - `citations` with a null passage or version: the composite FK is not enforced (MATCH SIMPLE), and the check `(status = 'valid') = (passage_id is not null and version_id is not null)` carries that case.
  - `claim_checks`: `before update` trigger `claim_checks_append_only` raises `claim_checks are append-only`. No delete trigger, because deleting an analysis cascades to claims and then to their checks.
  - `feedback`: `unique (claim_id, user_id)`, note check `length(note) <= 1000`, `updated_at`.
- `backend/db/repos/analyses.py`: all functions of the brief's interface (`create_analysis`, `count_active`, `get_analysis`, `list_analyses`, `set_status`, `set_progress`, `save_results`, `get_claims`, `get_evidence`, `get_citations`, `insert_claim_check`, `latest_checks`, `upsert_feedback`). `CitationRecord`, `PageParams`, `ClaimVerdict` are imported under `TYPE_CHECKING`.
- `backend/analysis/__init__.py` and `backend/analysis/service.py` (R-C3): only the frozen dataclass `CitationRecord(CitationRef)` adding required `passage_id: UUID | None` and `version_id: UUID | None`. It subclasses `CitationRef`, so the fields cannot drift.
- `tests/test_schema_analyses.py`: 32 tests.

## Files changed

```
backend/analysis/__init__.py            |   1 +
backend/analysis/service.py             |  17 +
backend/db/migrations/0004_analyses.sql | 187 ++++++
backend/db/repos/analyses.py            | 346 ++++++++++++
tests/test_schema_analyses.py           | 969 ++++++++++++++++++++++++++++++++
```

Nothing else was touched. `docs/superpowers/`, `.superpowers/`, `.local/` untouched.

## TDD evidence

I split the test file into two RED/GREEN cycles so that the SQL tests failed on their own merits rather than on an import error.

### Cycle 1: schema tests (SQL only), RED

Command: `wsl.sh timeout 300 python -m pytest tests/test_schema_analyses.py -q`

```
FAILED tests/test_schema_analyses.py::test_new_tables_forced_rls - AssertionError   (assert tables.keys() >= NEW_TABLES)
FAILED ...::test_same_tenant_rows_accepted
FAILED ...::test_evidence_link_version_must_match_passage
FAILED ...::test_evidence_span_must_be_non_empty
FAILED ...::test_citation_valid_with_passage_and_version
FAILED ...::test_citation_unknown_marker_without_passage
FAILED ...::test_citation_valid_without_passage_rejected
FAILED ...::test_citation_valid_without_version_rejected
FAILED ...::test_citation_unknown_marker_with_passage_rejected
FAILED ...::test_claim_checks_append_only
FAILED ...::test_deleting_analysis_removes_its_rows
FAILED ...::test_row_pointing_at_other_tenants_parent_rejected[...] x7
18 failed in 1.74s
psycopg.errors.UndefinedTable: relation "analyses" does not exist
```

### Cycle 1: GREEN (after writing 0004_analyses.sql)

```
18 passed in 2.00s
```

### Cycle 2: repository tests, RED

Appended the repository tests (they import `backend.analysis.service` and `backend.db.repos.analyses`).

```
E   ModuleNotFoundError: No module named 'backend.analysis'
1 error in 0.90s
```

After creating only `backend/analysis/` (the R-C3 dataclass), still RED for the missing repo:

```
E   ImportError: cannot import name 'analyses' from 'backend.db.repos' ... Did you mean: 'analysis'?
1 error in 0.90s
```

### Cycle 2: GREEN (after writing the repository)

```
32 passed in 2.98s
```

### Mutation checks (tests catch real regressions)

Temporarily (reverted before the full run and the commit):

- `claims` FK changed to single-column `foreign key (analysis_id) references analyses (id)`: `test_row_pointing_at_other_tenants_parent_rejected[claim_to_analysis]` FAILED.
- Removed the `sorted(...)` by ordinal in `save_results`: `test_save_results_returns_claim_ids_in_ordinal_order` FAILED.
- Result of that mutated run: `2 failed, 30 passed`. After revert: 32 passed again (confirmed by the full-suite run below, which ran on the committed content).

## Tests (32)

Brief tests: `test_new_tables_forced_rls`, `test_evidence_link_version_must_match_passage`, citation status checks (split into four tests: valid with passage ok, unknown_marker without passage ok, valid without passage or without version `CheckViolation`, unknown_marker with passage `CheckViolation`), `test_claim_checks_append_only`, `test_save_results_roundtrip`, `test_count_active`, `test_feedback_upsert` (plus `test_feedback_note_limited_to_1000_characters`: 1000 ok, 1001 `CheckViolation`).

R-C1: `test_row_pointing_at_other_tenants_parent_rejected` parametrized over seven cases, each raising `ForeignKeyViolation`: claim to other tenant's analysis, citation to other tenant's analysis, evidence link to other tenant's claim, evidence link to other tenant's passage (and its version), citation to other tenant's passage, claim check to other tenant's claim, feedback to other tenant's claim. `test_same_tenant_rows_accepted` is the positive control showing the same helper inserts succeed inside one tenant.

Extra tests for repository behavior I added: `test_evidence_span_must_be_non_empty`, `test_deleting_analysis_removes_its_rows` (cascade through all five child tables), `test_create_and_get_analysis`, `test_get_analysis_missing_or_other_tenant_is_none`, `test_set_status_sets_completed_at_only_when_terminal`, `test_set_progress_appends_finished_stages`, `test_list_analyses_scopes_and_pages`, `test_save_results_returns_claim_ids_in_ordinal_order` (also covers a claim with no evidence stored as `insufficient`), `test_save_results_keeps_unknown_marker_citation_without_passage` (review focus 1), `test_save_results_unknown_analysis_raises`, `test_latest_checks_returns_newest_per_claim`, `test_insert_claim_check_rejects_unknown_field`.

## Full suite

```
wsl.sh timeout 900 python -m pytest -q
1312 passed, 3 deselected, 2 warnings in 234.24s (0:03:54)   (exit 0)
```

The two warnings are the existing `tests/test_generator.py` "coroutine ... was never awaited" ones, unrelated to this task. The 3 deselected are the existing `models`-marker tests from the project configuration.

## Lint

```
wsl.sh timeout 120 ruff check .      -> All checks passed!
```

`ruff format --check` on my four Python files: clean (I ran `ruff format` on the new test file once). `ruff format --check .` over the whole repo reports 31 other files that were already unformatted before this task (not in my diff; I left them alone).

## Migrate on the dev database

Before: `schema_migrations` had versions 1, 2, 3.

```
wsl.sh timeout 120 python -m backend.cli.main migrate
applied: [4]            (exit 0)
wsl.sh timeout 120 python -m backend.cli.main migrate
up to date              (exit 0)
```

After: versions 1..4 (`4 | analyses`); `analyses`, `claims`, `evidence_links`, `citations`, `claim_checks`, `feedback` all have `relrowsecurity = t` and `relforcerowsecurity = t` on `evidencegraph`.

## Decisions

1. `set_progress(stage)` semantics: `stage` becomes the current stage and the previous current stage is appended to `stages_done` (not appended twice if the same stage is set again). First call gives `{"stage": "retrieve", "stages_done": []}`, then `verify` gives `{"stage": "verify", "stages_done": ["retrieve"]}`. The brief only said "stages_done appended"; this is the reading where `stages_done` holds finished stages.
2. `claim_checks.created_at` defaults to `clock_timestamp()` instead of `now()`. `now()` is the transaction start, so two checks written in one transaction would tie and `latest_checks` would pick by random uuid. Everything else uses `now()`.
3. Three supporting indexes beyond the brief's list, for the repository's own access paths and the cascade deletes: `evidence_links (claim_id, rank)`, `citations (analysis_id)`, `claim_checks (claim_id, created_at desc, id desc)`. Without them `get_evidence`, `get_citations`, and `latest_checks` scan the whole table (Plan A's final review flagged missing FK indexes the same way). The brief's `evidence_links (tenant_id, version_id)` and the three `analyses` indexes are present as specified. Cheap to drop if the controller prefers the exact brief list.
4. `save_results` sorts verdicts by `claim.ordinal` before inserting, so the returned ids are in ordinal order whatever the input order. Probabilities are read from `ClaimVerdict.probs["supported" | "contradicted" | "insufficient"]`. Evidence `rank` is the 1-based position in `verdict.evidence`. Evidence links and citations are bulk inserted with `executemany`. It raises `LookupError` if the analysis is not visible (unknown id or another tenant).
5. `insert_claim_check(conn, **fields)` whitelists field names (the column names are interpolated into the SQL, the values are bound parameters) and raises `TypeError` on an unknown name. `evidence` is wrapped in `Jsonb`.
6. `list_analyses` and `get_analysis` return every `analyses` column except `tenant_id`, and `list_analyses` returns up to `limit + 1` rows with the same cursor contract as `list_documents`.
7. `evidence_links.nli_*` and `dense_cos` are nullable, exactly as the brief writes them (only `rerank_score` is spelled `null`, but none of the others say `not null`). `save_results` always supplies them.
8. `get_evidence` reports `passage_is_current` as `passages.is_current` (the retrieval flag maintained by ingest), and joins by the stored `(passage_id, version_id)` pair, so it still resolves for soft-deleted documents.

## Concerns / notes for later tasks

- `get_citations` returns only `citations` columns, as the brief lists. Review focus 2 (redaction of citations from a revoked collection) and "citations resolved with titles" in Task 13 need the passage's `collection_id` and the document title; Task 13 will have to join `passages`/`documents` (or extend this function). I did not add it because the brief did not ask.
- The spec says evidence span offsets "must lie inside the passage (check in code and test)". The database only checks `span_start < span_end`, and `save_results` does not validate the inside-the-passage condition (the producer is the verify engine from Task 10). No test here asserts it.
- `set_status` does not refuse to move a terminal analysis; the Task 13 failure hook ("unless already terminal") must read the status first.
- `analyses.collection_ids uuid[]` cannot carry a foreign key, so it is not tenant-checked by the database (callers authorize collections before creating the analysis).
- The migration is written with LF endings (new file, per the rules); the checksum is over file bytes, so a checkout that converts it to CRLF would change the checksum, same as for the earlier migrations.
