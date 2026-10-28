# Task 12 fix report: default effective_from must not precede the current version

Status: DONE_WITH_CONCERNS (one minor, pre-existing; see Concerns)
Commit: a57dbb2 `fix: default effective_from to no earlier than the current version`
Files changed (only the two the brief names): `backend/ingest/intake.py`, `tests/test_ingest_upload.py`
Not pushed. Report not committed.

## What changed

`intake_upload` (backend/ingest/intake.py):
- The default is no longer set before the transaction. Only the future check on an explicit value stays up front, so explicit `effective_from` values behave exactly as before (future check at intake; out-of-order still fails in the worker).
- When the upload targets an existing document that has a current version, the current version row is now read once inside the same `tenant_tx` that loads the document, selecting `blob_sha256, effective_from`.
- If `effective_from` is None: `now` for a new document or a document with no current version, otherwise `max(now, current["effective_from"])`. Equal is allowed, matching `create_version`'s `effective_from < previous` check.
- The duplicate-bytes check now reuses that same row instead of issuing a second `select blob_sha256 ...` query (same semantics, one query instead of two).

Test (tests/test_ingest_upload.py): `test_default_effective_from_not_before_current_version`. v1 goes through `intake_upload` with `effective_from = now + 30 s`, drain, then v2 with changed text and no `effective_from`, drain. Asserts v2 upload `status == created_version`, `error_code is None`, two versions stored, and v2's stored `effective_from` equals v1's. No clock stepping, no mocks.

The existing `test_out_of_order_effective_from_fails_the_upload_and_adds_nothing` (explicit backdated v2 fails with `effective_from_out_of_order`) is untouched and still passes.

## RED (before the fix)

```
wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py -q -k default_effective_from
E       AssertionError: assert ('failed', 'e...out_of_order') == ('created_version', None)
E         At index 0 diff: 'failed' != 'created_version'
tests/test_ingest_upload.py:416: AssertionError
FAILED tests/test_ingest_upload.py::test_default_effective_from_not_before_current_version
1 failed, 49 deselected in 2.29s
```

Fails for the expected reason: v2 ends `failed` with `effective_from_out_of_order`. It ran in 2.3 s, so nothing expensive or dangerous is triggered.

## GREEN (after the fix)

```
wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py -q -k default_effective_from
1 passed, 49 deselected in 1.92s

wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py -q
50 passed in 18.79s
```

## Full suite

```
wsl.sh timeout 900 python -m pytest -q
858 passed, 2 deselected, 2 warnings in 136.05s (0:02:16)
```

- The 2 deselected are the `models` marker tests excluded by `addopts = "-m 'not models'"` in pyproject.toml (unchanged config).
- The 2 warnings are `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in `tests/test_generator.py` (backend/core/generator.py:50). Not related to this change.
- Exit code 124 did not occur (136 s of the 900 s limit).
- This was one run. The original failure was intermittent (wall-clock steps under load), so a single green run is consistent with the fix but does not by itself prove it; the deterministic regression test above is the proof.

## Lint

```
wsl.sh timeout 120 ruff check .
All checks passed!
```

`ruff format --check` on the two changed files: `backend/ingest/intake.py` is clean. `tests/test_ingest_upload.py` would be reformatted, but only at line 548 (`test_collection_kinds_are_passed_to_the_policy`, `created = await upload(...)` wrapped over three lines). That is pre-existing code from commit 197f11f, not in my diff, so I left it alone. My added test passes the formatter.

## Commit checks

- Staged the two files by name; no `-A`, no `--no-verify`, no amend, no push.
- Pre-commit hook ran gitleaks: no leaks found.
- Working tree clean after the commit (`git status --short` empty).
- Line endings: both files `i/lf w/lf` before and after (working copy is LF in this worktree; I preserved it). The "LF will be replaced by CRLF" warning is the expected one.
- No U+2014 em dashes in either changed file (counted 0).
- Commit message is exactly the one in the brief; no mention of tools or assistants.

## Decisions

- Merged the "current version" read into one query (`blob_sha256, effective_from`) rather than adding a second query next to the existing duplicate check. Same transaction, same row, one round trip fewer. Behavior of the duplicate path is unchanged and covered by `test_same_bytes_duplicate_without_job` and `test_resaved_same_text_duplicate`, which pass.
- Added only the one test the brief names; the explicit-backdated case is already covered by the existing test, so I did not duplicate it.
- `current: dict | None` is annotated loosely, matching how `documents.get_document` rows are used elsewhere in this module (rows are dicts from the dict row factory).

## Concerns

- Pre-existing `ruff format` drift at tests/test_ingest_upload.py:548 (not mine, not fixed; out of scope for a two-file fix commit). `ruff check` is clean. Worth a separate formatting pass if the project enforces `ruff format --check` in CI.
- The default for an upload targeting an existing document is read when the upload is recorded, not when the worker runs. If a different upload to the same document lands a later-dated version between intake and the worker run, this upload can still fail with `effective_from_out_of_order`. That is the intended order check, and it is outside the brief; noting it only so it is not mistaken for a regression of this fix.
