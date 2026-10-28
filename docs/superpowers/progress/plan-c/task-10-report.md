# Plan C Task 10 report: evidence gathering and span localization

Status: DONE_WITH_CONCERNS (concerns are design notes, nothing blocking)

Commit: `b4a4b58 feat: add claim evidence gathering and span windows` on `feature/evidencegraph` (not pushed).

## What was built

`backend/verify/evidence.py` (new, 104 lines):

- `gather_evidence(retriever, reranker, claims, scope, *, pinned={}, depth=8, candidates=20, concurrency=4) -> list[list[PassageHit]]`
  - One task per claim, all gathered with `asyncio.gather`, so the result list is in claim input order. Each task runs inside one `asyncio.Semaphore(concurrency)`.
  - Per claim: `retriever.search(claim.text, scope, k=candidates, mode="hybrid", rerank=False, candidates=candidates)`.
  - Union: pinned hits for `claim.ordinal` first, then the retrieved hits, deduplicated by `passage_id` (first occurrence wins, so the pinned copy beats the retrieved copy).
  - The union is renumbered `rank=1..n` in union order (pinned first) with `dataclasses.replace` before reranking, so `rerank_hits` ties (which break on `hit.rank`) keep pinned-first order instead of depending on ranks from unrelated earlier lists.
  - `rerank_hits(reranker, claim.text, ordered, k=len(ordered))`.
  - Selection: every pinned hit is kept (even when there are more pinned hits than `depth`), then the highest reranked others fill the remaining `max(depth - n_pinned, 0)` slots. Final order is rerank score descending (rerank order), ranks renumbered `1..n`.
  - No shared mutable state: hits are frozen dataclasses and every change goes through `replace` (a new object); the only dict mutated is the per-claim `union` local. The `pinned` default is a read-only `{}` that is never written (ruff B006 accepts it because of the `Mapping` annotation).
  - Empty scope or nothing found and nothing pinned gives `[]` for that claim (the reranker is never called for empty input).
- `window_spans(hit, widths=(1, 2)) -> list[tuple[int, int, str]]`
  - Sentences from `segment_sentences(hit.text)`; for each width in `widths` (in the order given), every run of that many consecutive sentences, left to right. A width larger than the sentence count produces nothing.
  - Offsets are `hit.start_char + local offset`; text is `hit.text[local_start:local_end]`.
  - No sentences at all (whitespace-only passage): one window for the whole passage. A one-sentence passage without terminator gives the sentence span, which is the whole passage.
- `relation_for(r) -> "supports" | "contradicts" | "related"`
  - `supports`: `entail >= neutral and entail > contra`.
  - `contradicts`: `contra >= entail and contra >= neutral`.
  - otherwise `related`.

`tests/test_features_evidence.py` (modified, 313 lines added, 1 import line replaced by the larger import block): 12 new test functions (19 test cases after parametrization); the file went from 9 to 28 tests.

## Files changed

- `backend/verify/evidence.py` (new)
- `tests/test_features_evidence.py` (appended tests, imports extended)

## RED evidence

Tests written first, run before `evidence.py` existed:

```
wsl ... wsl.sh timeout 300 python -m pytest tests/test_features_evidence.py -q
tests/test_features_evidence.py:11: in <module>
    from backend.verify.evidence import gather_evidence, relation_for, window_spans
E   ModuleNotFoundError: No module named 'backend.verify.evidence'
ERROR tests/test_features_evidence.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.86s
```

The RED run touches no database or model, so it fails in under a second.

## GREEN evidence

```
python -m pytest tests/test_features_evidence.py -q
27 passed in 0.97s        (first implementation)
28 passed in 1.01s        (after adding the pinned-first / dedupe test)
```

## Test sensitivity (mutation checks)

After GREEN, each mutation of `evidence.py` was applied temporarily, the file run, then restored byte for byte (verified):

| Mutation | Result |
|---|---|
| `Semaphore(concurrency)` -> `Semaphore(1000)` | `test_gather_bounds_concurrency` FAILED |
| window offsets made relative (no `hit.start_char`) | `test_window_spans_absolute` FAILED |
| keep only `ranked[:depth]` (pinned not guaranteed) | `test_gather_keeps_pinned_and_depth` FAILED |
| union order `(*found, *pinned)` | first run: all 27 passed (gap); added `test_gather_puts_pinned_first_and_dedupes_by_passage`; rerun: that test FAILED |

## Tests added

Pure (no Postgres):

- `test_window_spans_absolute`: `start_char = 100`, three sentences gives 5 windows, exact absolute spans `(100,115) (116,134) (135,151) (100,134) (116,151)`, and `hit.text[start - 100 : end - 100] == window text` for each.
- `test_window_spans_width_larger_than_sentence_count_is_skipped`
- `test_window_spans_without_sentence_boundary_is_whole_passage` (2 cases)
- `test_relation_for_cases` (7 cases, including the three tie cases)
- `test_gather_searches_hybrid_without_rerank`: asserts the exact search kwargs `k=candidates, mode="hybrid", rerank=False, candidates=candidates`.
- `test_gather_bounds_concurrency`: 6 claims, `concurrency=2`, max in flight exactly 2.
- `test_gather_puts_pinned_first_and_dedupes_by_passage`

`@pg` (Docker Postgres, seeded documents, `KeywordReranker`, `HashEmbedder`):

- `test_gather_keeps_pinned_and_depth`: `depth=3`; unpinned gives the three best (rerank 6, 4, 1); with a pinned weather passage (rerank 0) the result is `[6.0, 4.0, 0.0]`, the rerank-1 passage is dropped, ranks `[1, 2, 3]`, ids unique.
- `test_gather_keeps_every_pinned_hit_beyond_depth`: two pinned hits with `depth=1` are both kept and the strong unpinned passage is dropped.
- `test_gather_results_follow_claim_order_with_per_claim_pins`: two claims at `concurrency=2`, pin only on ordinal 1; results in claim order and the pin does not leak into claim 0.
- `test_gather_respects_scope`: out-of-scope collection holds stronger matches; no hit comes from it and both in-scope documents are returned.
- `test_gather_empty_scope`: collection with no documents gives `[[]]`.

## Full suite and lint

```
python -m pytest -q   (timeout 900)
1266 passed, 3 deselected, 2 warnings in 159.47s (0:02:39)
```

The 2 warnings are the existing `tests/test_generator.py` `RuntimeWarning: coroutine ... was never awaited`, unrelated to this task. No failures.

```
ruff check .                                   -> All checks passed!
ruff format --check evidence.py test file      -> 2 files already formatted
```

The commit hook (gitleaks) ran clean: "no leaks found". No em dash (U+2014) in either file.

## Decisions

1. Window order: width-major (all width 1 left to right, then width 2). The brief says "every run of w consecutive sentences for each w in widths" and does not fix an order; the test pins this order so a later consumer can rely on it.
2. `relation_for` ties: the spec says "entail is the max of the three and above contradiction" (the "and above" clause is only meaningful if ties for the max count as the max) and "contradiction is the max". I followed that literally, so ties count as the max. Consequences: `entail == contra > neutral` gives `contradicts`; `entail == neutral > contra` gives `supports`; an exact three-way tie gives `contradicts`. Real softmax output never ties exactly, but if Task 11 prefers ties to be `related`, change the two `>=` comparisons.
3. Pinned hits are kept even when they number more than `depth` ("keep every pinned hit plus the highest-reranked others until depth hits in total"); the list is then longer than `depth` and no unpinned hit is added.
4. Rank renumbering on the union and on the result (see above) is my addition for deterministic ties and for the `PassageHit.rank` contract ("1-based in the returned list"). It does not change any score.
5. LF line endings: the Windows editor tool rewrote `tests/test_features_evidence.py` as CRLF once; I converted it back to LF so the working copy matches the original `w/lf` state (`git ls-files --eol` shows `i/lf w/lf`). The diff against HEAD contains only my additions.

## Concerns (non-blocking)

- `asyncio.gather` without cancellation: if one claim's retrieval raises, the exception propagates to the caller at once but sibling claim tasks keep running until they finish (they hold at most `concurrency` pooled connections for one search each). Task 12 should map the exception to `retrieval_failed`. Switching to a `TaskGroup` would cancel siblings but wraps errors in an `ExceptionGroup`; I kept the plain exception on purpose and did not add speculative handling.
- Pinned hits keep whatever `scores` they came with, plus a new `rerank` key from this claim's rerank. A pinned hit from an earlier ask retrieval can therefore carry a stale `bm25`/`dense`/`rrf` entry from the ask query next to the claim-level `rerank`. Only `rerank` is compared by consumers here, so this is harmless today.
- Pinned hits are not re-checked against `scope` (the brief says they are in scope by construction).
