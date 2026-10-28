# Plan B Task 6 report: PDF parser, type dispatch, killable parse subprocess

Status: DONE_WITH_CONCERNS (concerns are design notes below, nothing blocking).
Commit: `2bd9f09 feat: add pdf parser and isolated parse subprocess` on `feature/evidencegraph` (not pushed).
Raw command output is saved next to this file in `task-6-evidence/` (not committed).

## What was built

- `backend/ingest/parsers/pdf.py`: `parse_pdf(data)`. `pypdf.PdfReader(io.BytesIO(data))`, parser `pdf` / `pdf/1`.
  Encrypted files get `decrypt("")`; `PasswordType.NOT_DECRYPTED` raises `encrypted_pdf`. Each page's
  `extract_text()` (None becomes `""`) goes into `TextAssembler.add(..., page=<1-based>)`. Metadata: `title`,
  `author` (non-empty after `normalize`), `created` (ISO 8601). No text raises `no_text`. `PyPdfError`,
  `ValueError`, `KeyError`, `TypeError` raise `bad_pdf` (`from None`, so no content in tracebacks).
- `backend/ingest/parsers/__init__.py`: `EXTENSIONS` (verbatim from the brief), `mime_for`, `parse`,
  `parse_with_timeout`. `parse` dispatches on the lowercased extension, requires `%PDF-` magic exactly when the
  extension is `.pdf` (either mismatch is `unsupported_type`), then raises `too_long` above 2,000,000 chars and
  `no_text` on empty text.
- `backend/ingest/parse_child.py`: `python -m backend.ingest.parse_child <filename>`; bytes on stdin, one JSON
  object on stdout (`{"ok": true, "doc": {...}}` or `{"ok": false, "code": ...}`), exit 0 either way. Imports
  only `backend.ingest`, stdlib (and pypdf through the parsers).
- `parse_with_timeout(data, filename, timeout_s=60.0)`: `asyncio.create_subprocess_exec(sys.executable, "-m",
  "backend.ingest.parse_child", filename, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)`, `wait_for(proc.communicate(data))`.
  Timeout raises `parse_timeout`. A `finally` block kills and `await proc.wait()`s any child still running, so a
  timeout and a cancelled caller both leave no zombie and no open pipe. Non-zero exit or unparsable / malformed
  output raises `bad_pdf` for `.pdf` and `unsupported_type` otherwise; `{"ok": false}` raises `IngestError(code)`.
- `backend/core/document.py`: `parse_document` signature unchanged. `.pdf` goes through `parse_pdf`, anything else
  through `parse_text`. A `no_text` PDF still returns `""` (the old behavior, and `tests/test_document.py` blank PDF
  case needs it). File stays CRLF in the working copy (`i/lf w/crlf`).
- `tests/test_parsers_pdf.py` (75 tests incl. parametrized): hand-written `make_pdf(pages)` computes every xref
  offset as it writes objects; `test_make_pdf_xref_offsets_point_at_objects` checks each xref entry lands on
  `<n> 0 obj`. No binary fixtures.

## Files changed (5, all in the commit)

`backend/core/document.py` (M), `backend/ingest/parsers/__init__.py` (M, was empty), `backend/ingest/parse_child.py`,
`backend/ingest/parsers/pdf.py`, `tests/test_parsers_pdf.py` (new). `tests/test_document.py` untouched.

## RED evidence

Command: `wsl.sh timeout 300 python -m pytest tests/test_parsers_pdf.py -q` (saved: `task-6-evidence/red1.txt`)

```
tests/test_parsers_pdf.py:13: in <module>
    from backend.ingest.parsers import mime_for, parse, parse_with_timeout
E   ImportError: cannot import name 'mime_for' from 'backend.ingest.parsers' (.../backend/ingest/parsers/__init__.py)
ERROR tests/test_parsers_pdf.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.13s
```

Fails fast for the right reason (feature missing). Nothing dangerous or slow ran in RED.

Because every test passed on the first GREEN run, I proved the key tests can fail with two throwaway mutations
(each reverted, verified by grep and a clean diff):

1. Disabled the kill/reap `finally` block in `parse_with_timeout`
   (`task-6-evidence/mutation-no-kill.txt`): `3 failed, 68 deselected in 1.61s` (`test_parse_with_timeout_kills`,
   `..._kills_a_hung_child_that_never_reads_stdin`, `..._kills_the_child_when_cancelled`), each
   `assert None == -<Signals.SIGKILL: 9>`. Leftover `sleep(600)` children were killed afterwards; `ps` shows none.
2. Replaced `normalize(value)` with `value.strip()` and the `except ValueError` around `creation_date` with a
   different exception (`task-6-evidence/mutation-pdf-metadata.txt`): `test_pdf_metadata_is_normalized_like_body_text`
   and `test_pdf_unparsable_creation_date_is_ignored` failed, `2 failed, 2 passed`.

## GREEN evidence

- Focused: `pytest tests/test_parsers_pdf.py tests/test_document.py -q -W error::ResourceWarning`:
  `75 passed in 7.59s` (the subprocess tests take about 0.6 s each; the kill tests are well under 1 s, no
  ResourceWarning for unclosed pipes or transports).
- Full suite: `wsl.sh timeout 900 python -m pytest -q`: `702 passed, 2 warnings in 65.48s`. The 2 warnings are the
  existing `tests/test_generator.py` `AsyncMockMixin` "never awaited" RuntimeWarnings, unrelated to this task.
  Baseline before Task 6 was not re-measured; the suite grew only by this task's 75 tests plus earlier tasks.
- Lint: `wsl.sh timeout 120 ruff check .`: `All checks passed!` (the first run flagged ASYNC110 on a sleep-poll loop
  in my test; replaced by an `asyncio.Event` and re-ran everything above after that fix).
- Commit hook scan: `no leaks found`.

## Decisions

1. `created` is read in its own `try`: an unparsable `/CreationDate` (pypdf raises `ValueError`) means "metadata
   absent", not `bad_pdf`. The brief's blanket `ValueError` rule would reject a readable PDF over a junk date.
   Test: `test_pdf_unparsable_creation_date_is_ignored`.
2. `title` and `author` go through `normalize` like the HTML and Markdown parsers' metadata, so NUL and other
   control characters (rejected by Postgres) never reach the version row. Test:
   `test_pdf_metadata_is_normalized_like_body_text`.
3. Legacy `parse_document`: `.pdf` uses `parse_pdf`, every other suffix uses `parse_text` (not the full `parse`),
   so the old "default: read as plain text" behavior for any suffix is kept. `no_text` on a PDF maps back to `""`.
   Intentional behavior changes: text is now normalized (BOM dropped, whitespace collapsed), multi-page PDFs join
   with a blank line instead of one newline, and failures are `IngestError` instead of pypdf / `UnicodeDecodeError`.
   Only `backend/cli/main.py check --doc` calls it.
4. Kill/reap lives in a `finally` (not only the `except TimeoutError`), so cancellation of the caller (worker
   shutdown) also kills and reaps the child.
5. Child JSON uses the default `ensure_ascii=True`, so stdout encoding can never fail, including on lone surrogates.

## Concerns (none blocking)

- The child is launched without `cwd=`, exactly as the brief specifies, so `python -m backend.ingest.parse_child`
  only resolves when the worker's cwd contains `backend/` (true for the tests and a repo-root Docker WORKDIR). If a
  deployment starts the worker elsewhere, every parse would fail as `bad_pdf` / `unsupported_type`. Passing
  `cwd=<dir containing backend/>` would remove the dependency; not added because it is outside the brief.
- AES-encrypted PDFs (even with an empty user password) need `cryptography`, which is not installed or pinned.
  `pypdf.errors.DependencyError` is not a `PyPdfError`, so `parse_pdf` lets it propagate; in the child that is a
  non-zero exit, which the parent reports as `bad_pdf`. RC4 files work (tested). Adding the dependency is a
  decision for the plan owner.
- pypdf `extract_text` can in principle return lone surrogates for malformed fonts. They survive `normalize` and
  the JSON hop, and Postgres would reject them at insert time (Task 9 territory). Not reproduced here.
- A filename containing a NUL byte makes `create_subprocess_exec` raise `ValueError` in the parent. Filename
  sanitizing belongs to the upload layer (Tasks 9/10); worth keeping in mind there.
- `too_large` (20 MiB) is not enforced in `parse`; the brief assigns it to intake, so it was not added.
