# Plan B Task 12 report: sentence-level version diff and route

Status: DONE (one minor concern below)
Commit: `e7077ab feat: add sentence-level version diff` on `feature/evidencegraph` (not pushed)

## What was built

- `backend/ingest/diff.py`: `DiffHunk` (frozen dataclass), `diff_texts(old, new)`, and private helpers `_range`, `_hunk`, `_merge_adjacent`.
  - Sentences from `segment_sentences`; `difflib.SequenceMatcher(None, old_sentences, new_sentences, autojunk=False)` over the sentence strings; each opcode becomes one hunk; adjacent same-op hunks merge.
  - Offsets are Python `str` indices. Non-empty ranges run from the first sentence start to the last sentence end in each text.
  - `equal`: `from_text == to_text == ""`, offsets still span the equal sentences in each text.
  - `insert`: `from_start == from_end`, at the end of the preceding old sentence; at the start of the first old sentence when nothing precedes; `len(old)` when appended past the last sentence. `delete` is symmetric in the new text.
  - `old == new` returns exactly one `equal` hunk `(0, len(old), 0, len(new))`, also for a one-sentence text.
- `backend/api/schemas_v1.py`: `DiffHunkOut`, `DiffOut(from_version, to_version, hunks, stats: dict[str, int])`.
- `backend/api/routes/documents.py`: `GET /api/v1/documents/{document_id}/diff?from=<n>&to=<n>`.
  - `from` and `to` are required int query params (aliases onto `from_version`/`to_version`, since `from` is a Python keyword); missing or non-integer values are 422 `validation_error`.
  - Read permission through the existing `_load_document(..., "read")`, so cross-tenant, no-grant, unknown and soft-deleted documents all give the same 404 `document_not_found`.
  - Versions loaded with `documents.get_version` (includes text). Unknown number on either side is 404 `version_not_found`. When `from == to` the version is fetched once and `diff_texts` returns the single equal hunk.
  - `stats` always carries all four keys (`equal`, `insert`, `delete`, `replace`) as hunk counts.
  - The diff runs in `asyncio.to_thread` after the tenant transaction closes (CPU bound, quadratic worst case; same pattern as `auth/service.py` and `ingest/service.py`).
- `tests/test_diff.py`: 39 tests (see below).

## Files changed (committed, staged by name)

- create `backend/ingest/diff.py`
- create `tests/test_diff.py`
- modify `backend/api/routes/documents.py` (imports, `DIFF_OPS`, one route)
- modify `backend/api/schemas_v1.py` (two models; file stays CRLF in the working copy, `i/lf w/crlf` unchanged)

## RED evidence

1. Before any production code, `python -m pytest tests/test_diff.py -q -x`:
   `ModuleNotFoundError: No module named 'backend.ingest.diff'`, `1 error in 0.75s` (collection error, feature missing).
2. With an interface-only skeleton (`DiffHunk` plus `diff_texts`/`_merge_adjacent` raising `NotImplementedError`, no route), `python -m pytest tests/test_diff.py -q`:
   `39 failed in 12.37s`. Unit tests failed with `NotImplementedError`; the route tests failed on behavior (for example `At index 1 diff: 'not_found' != 'document_not_found'`, 404 on the missing route instead of 200, `requires_both_numbers` statuses not 422). No expensive or dangerous behavior is involved, so the RED run was fast.

## GREEN evidence

- First GREEN run: `38 passed, 1 failed`. The failure was a bug in my test data, not in the code: `test_non_ascii_offsets_are_code_points` used `"... opens. <CJK> closes."`, and the segmenter (correctly, by its own rule) does not split before a CJK letter because it is neither uppercase nor a digit, so the text was one sentence and there was no `hunks[1]`. Fixed the test data to `"... opens. Tokyo <CJK> closes."`, which keeps the non-ASCII coverage and now has two sentences.
- `python -m pytest tests/test_diff.py -q`: `39 passed in 10.45s`.
- After the final lint tweak (below): `python -m pytest tests/test_diff.py tests/test_document_routes.py -q`: `61 passed in 35.61s`.

## Full suite and lint

- Full suite, `timeout 900 python -m pytest -q`: `857 passed, 2 deselected, 2 warnings in 124.00s (0:02:04)`, exit 0. The 2 deselected are the existing `models`-marker deselection (not touched by this task); the 2 warnings are the pre-existing `AsyncMockMixin` RuntimeWarnings in `tests/test_generator.py`.
- `ruff check .`: first run flagged one thing in my test (`B905`-adjacent `RUF007`: `zip(hunks, hunks[1:], strict=False)` should be `itertools.pairwise`). Replaced with `pairwise(hunks)`. Re-run: `All checks passed!`. `ruff format --check` on the 4 touched files: `4 files already formatted`.
- Order note: the full suite ran before that one test-only `pairwise` edit. No production code changed after it; I re-ran `tests/test_diff.py` plus `tests/test_document_routes.py` (61 passed) after the edit instead of repeating the whole suite.
- Commit hook: gitleaks ran at commit, `no leaks found`.
- No em dash characters in any touched file; new files are LF.

## Tests in `tests/test_diff.py`

Unit (no DB): `test_identical_texts_single_equal`, `test_identical_one_sentence_text_single_equal`, `test_replace_one_sentence` (replace then equal, slices exact), `test_insert_appended_sentence_sits_at_end_of_old` (`from_start == from_end == len(old)`, old ends with a trailing space so `len(old)` differs from the last sentence end), `test_delete_trailing_sentence_sits_at_end_of_new`, `test_insert_at_beginning_...`, `test_delete_at_beginning_...`, `test_insert_in_middle_...`, `test_delete_in_middle_...` (boundary rules, each with whitespace so the boundary choice is observable), `test_equal_hunk_spans_the_equal_sentences_in_each_text`, `test_non_ascii_offsets_are_code_points` (combining accent, emoji, CJK), `test_adjacent_hunks_with_the_same_op_merge`, and `test_offsets_slice_exactly[0..19]` (20 seeded random edit sets over 40 sentences: every non-equal hunk's slices equal `from_text`/`to_text`; equal hunks have empty texts and equal slices; insert/delete ranges are empty; offsets in bounds and non-decreasing; neighbors never share an op).

`pg` marked: `test_diff_route` (two uploaded versions, 200, stats `{"equal": 1, "insert": 0, "delete": 0, "replace": 1}`, hunk slices exact against the texts served by `/versions/{n}/text`, unknown `to` and unknown `from` give 404 `version_not_found`), `test_diff_route_same_version_is_one_equal_hunk` (exact JSON body; unknown same-number 404), `test_diff_route_requires_both_numbers` (422 x4), `test_diff_route_unreadable_document_is_404[carol|bob]` (cross-tenant and no-grant, body identical to an unknown id, id not echoed), `test_diff_route_read_permission_is_enough` (read-only grant gets 200), `test_diff_route_soft_deleted_document_is_404`.

## Decisions (ambiguities resolved as instructed)

- Empty-range boundary rule order in `_range`: past the last sentence first (`len(text)`), else after a preceding sentence (its end), else the first sentence start. This yields `len(old)` for an append even when the old text has trailing whitespace, as the dispatch required.
- `old == new` shortcut covers "the whole text" literally as `0..len(text)` on both sides. For non-identical texts whose sentences all match (only whitespace between them differs), the single equal hunk spans first sentence start to last sentence end per side.
- Merge step: implemented as specified, but `SequenceMatcher.get_opcodes()` already coalesces matching blocks, so two neighbors with the same op never come out of it. The merge is therefore unreachable from `diff_texts` today; it is covered by a direct test of `_merge_adjacent` with hand built hunks. It is 8 lines; say so if you would rather drop it.
- `stats` is `dict[str, int]` with all four keys present, matching the brief's `{"equal": int, ...}`.
- Moved the CPU-bound diff to `asyncio.to_thread` and out of the DB transaction (not in the brief; follows existing repo pattern and keeps a 2M-character diff off the event loop).

## Concerns

- Worst-case cost: `SequenceMatcher` with `autojunk=False` is quadratic in the number of distinct sentences. A 2,000,000 character version could take a long time. It is off the event loop and the transaction, but there is no timeout or size cap on the diff (none was specified). Worth a limit if Plan C exposes this to untrusted large uploads.
- Response size: a replace hunk returns both full texts, so one hunk over a huge changed region can be large. Not capped (not in the brief).
- `/diff` loads the full version rows via `get_version` (includes `page_map` and `sections`) although only `text` is used. Left as is to use the specified repository function.
