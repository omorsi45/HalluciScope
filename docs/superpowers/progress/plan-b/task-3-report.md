# Plan B Task 3 report: sentence-aligned passage builder

Status: DONE_WITH_CONCERNS (concerns are minor, see the end)

Commit: `b75e2e0 feat: add sentence-aligned passage builder` on `feature/evidencegraph` (not pushed). Working tree clean after the commit. The pre-commit gitleaks hook ran and found no leaks.

## What I built

`build_passages(text, pages=(), sections=(), *, target_chars=800, max_chars=1200, overlap_sentences=1) -> list[PassageDraft]` in `backend/ingest/passages.py`.

1. Regions: sections sorted by start and clipped to the text and to the previous region, plus the gaps before, between and after them as unsectioned regions (`section=None`). No sections gives one region covering the whole text. Empty text gives no regions.
2. Each region is segmented with `segment_sentences(text, region_start, region_end)`, so sentence offsets index the full text.
3. A sentence longer than `max_chars` is split into pieces: cut at the last whitespace at or before `start + max_chars`, hard cut at `max_chars` when the window has no whitespace, each piece trimmed (whitespace after a cut is skipped before the next piece starts). The pieces are then treated as ordinary sentences for accumulation and overlap (ambiguity ruling from the dispatch).
4. Greedy accumulation per region: a passage extends while `next.end - first.start <= target_chars`; it always holds at least one sentence. The next passage starts at the last `overlap_sentences` sentences of the previous one only if the previous passage held more than `overlap_sentences` sentences and `overlap + next sentence` spans at most `max_chars`. The next sentence always joins the new passage (that is what guarantees progress when overlap plus next is over `target_chars` but within `max_chars`). No trailing passage is emitted that would hold only overlap sentences.
5. Each draft: `text = text[start:end]`, `text_sha256 = sha256(text.encode("utf-8")).digest()`, `page_start`/`page_end` = first and last page span intersecting `[start, end)` (bisect over pages sorted by start; `None` when none), `section` = region path, ordinals `0..n-1` in document order.

## Files changed

- Created `backend/ingest/passages.py` (132 lines)
- Created `tests/test_passages.py` (about 190 lines)

Nothing else touched. Both files are LF in the index and working copy (`git ls-files --eol`: `i/lf w/lf`). No em dashes.

## Tests (14 in `tests/test_passages.py`)

Brief tests:
- `test_invariants_on_generated_text` (seeds 1, 2, 3): 30 paragraphs of 1 to 12 sentences of 5 to 40 words, with a mixed-script vocabulary (accents, a combining accent, emoji, CJK) plus one 3,000-character unpunctuated sentence. Asserts `text == text[start:end]`, `0 < len <= 1200`, sha256, consecutive ordinals, non-decreasing starts, `page_start/page_end/section` all `None` without pages or sections, more than 30 passages, and that an over-long sentence really occurs. Coverage per the dispatch ruling: a sentence of at most 1200 characters must lie inside one passage; an over-long sentence must have every non-whitespace character covered by the union of passages.
- `test_overlap_one_sentence`: ten 100-character sentences, `target_chars=350`; hand-derived offsets `(0,302) (202,504) (404,706) (606,908) (808,1009)`.
- `test_no_overlap_when_single_sentence_passages`: five 700-character sentences give five passages, `(0,700) (701,1401) (1402,2102) (2103,2803) (2804,3504)`.
- `test_sections_never_crossed`: unsectioned intro plus sections A and B; section list `[None, A, A, A, B, B, B]`, every passage inside one region, overlap restarts inside a section, first B passage starts at the B start.
- `test_pages_mapping`: pages `(1,0,201)`, `(2,203,404)`; passages `(0,201,1,1) (101,303,1,2) (203,404,2,2)`.
- `test_empty_text`.

Extra tests for algorithm rules the brief states but its six tests do not pin:
- `test_overlap_skipped_when_it_would_exceed_max_chars` (sentences 300, 600, 700, `target_chars=1000`): no overlap because 1301 > 1200.
- `test_overlap_may_exceed_target_chars_within_max_chars` (sentences 300, 300, 600, `target_chars=700`): overlap kept at 901 characters.
- `test_overlong_sentence_is_split` (4 cases): `last_whitespace`, `whitespace_at_limit` (whitespace exactly at index `start + max_chars`), `hard_cut`, `trimmed_pieces` (multiple spaces at the cut).

All expected values are hand-derived literals, not computed by the code under test.

## RED evidence

Run 1, module absent:

```
$ wsl ... wsl.sh timeout 300 python -m pytest tests/test_passages.py -q -p no:cacheprovider
E   ModuleNotFoundError: No module named 'backend.ingest.passages'
ERROR tests/test_passages.py
1 error in 0.54s          (exit 2)
```

Run 2, throwaway stub `build_passages` that returns `[]` (so the tests fail on behavior, not on import; the stub was overwritten by the real implementation):

```
FAILED test_invariants_on_generated_text[1] / [2] / [3]
FAILED test_overlap_one_sentence            - assert [] == [(0, ...
FAILED test_no_overlap_when_single_sentence_passages
FAILED test_overlap_skipped_when_it_would_exceed_max_chars
FAILED test_overlong_sentence_is_split[last_whitespace] / [whitespace_at_limit] / [hard_cut] / [trimmed_pieces]
FAILED test_sections_never_crossed
FAILED test_pages_mapping                   - assert [] == [(0, 201, 1, 1), ...
12 failed, 1 passed in 0.68s      (the one pass is test_empty_text, which a stub satisfies)
```

`test_overlap_may_exceed_target_chars_within_max_chars` was added after the implementation (it pins a rule I only noticed while implementing). I proved it can fail with a temporary mutation: changing the overlap check from `<= max_chars` to `<= target_chars` gave `1 failed, 13 passed`, failing exactly that test. The mutation was reverted before the commit.

## GREEN evidence

```
$ ... pytest tests/test_passages.py -q -p no:cacheprovider
14 passed in 0.32s

$ ... timeout 900 python -m pytest -q -p no:cacheprovider      (full suite)
503 passed, 2 warnings in 57.04s
```

The two warnings are pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in `tests/test_generator.py` (`backend/core/generator.py:50`); not from this task.

```
$ ... timeout 120 ruff check .
All checks passed!
$ ... ruff format --check backend/ingest/passages.py tests/test_passages.py
2 files already formatted
```

## Decisions

- Over-long sentence pieces count as separate sentences for accumulation and overlap, and coverage is by union of passages, as ruled in the dispatch.
- Forced inclusion of the next sentence after an overlap (see step 4). The brief's wording only bounds overlap plus next by `max_chars`; without forcing, a passage made only of overlap sentences could repeat forever when overlap plus next exceeds `target_chars`. A passage can therefore exceed `target_chars` (never `max_chars`) when it starts with overlap.
- `pages` are sorted by start and assumed non-overlapping (the `TextAssembler` guarantees this); sections are sorted and clipped, so overlapping sections cannot produce overlapping regions.
- Tests run with `-p no:cacheprovider` to avoid writing `.pytest_cache` into the worktree from WSL.

## Concerns (minor, not fixed because out of the brief)

- No parameter validation. `max_chars < 1` would loop forever in the piece splitter, and `target_chars > max_chars` can produce passages longer than `max_chars`. The defaults (800, 1200) and the config constants are safe; callers must keep `1 <= target_chars <= max_chars`. A guard (`ValueError`) is a one-line follow-up if wanted.
- The 5,000 passages per version limit (`too_many_passages`) is not enforced here; the brief assigns that to the caller.
