# NLI length-sorted batching: implementer report

Status: DONE
Commit: c383a19 `perf: batch NLI pairs by length to cut padding` on `feature/evidencegraph` (not pushed)
Files changed (only these two): `backend/verify/nli.py`, `tests/test_nli_scorer.py`
Both stay `i/lf w/lf`. No em dashes in either file. Gitleaks pre-commit hook: no leaks.

## What changed

`DebertaNLIScorer.score` now orders pair indices by `len(premise) + len(hypothesis)` in
characters, longest first (`sorted(..., reverse=True)`, which keeps the original order on ties),
scores `batch_size` pairs at a time in that order, stores each result in a dict keyed by the
original index, and returns `[scored[i] for i in range(len(pairs))]` (input order).
Truncation, max_length, batch size, label mapping, fp32 and inference_mode are untouched.

Sort key: characters, not token count. Reason: a token-length key needs a second tokenization
pass over every pair (the tokenizer is called per batch with padding), which would eat part of
the measured +38%. The brief allows either and asks to state it.

Tests (tests/test_nli_scorer.py), using the existing fake tokenizer and model:
- `FakeModel` takes an optional `probabilities` table (default is the old 5-row `PROBABILITIES`,
  so existing tests are unchanged); `_scorer` passes it through. New constants
  `PROBABILITIES_6` (old 5 rows plus one distinct row) and `WORD_COUNTS = [1, 50, 2, 49, 3, 48]`;
  helper `_pairs_with_word_counts` builds premises of N words ending in `premise-<i>`.
- `test_batches_group_similar_lengths`: batch_size 3, asserts two forward passes of 3 and that
  the tokenizer saw premise word counts `[[50, 49, 48], [3, 2, 1]]`.
- `test_sorted_batching_keeps_input_order`: same 6 pairs; each result must equal row i of
  `PROBABILITIES_6`, so a result left in batch order is visible.

## RED (before any production change)

Command (focused, outer 340 / inner 280):
`.venv\Scripts\python.exe .local\run.py 340 .local\nli-red.log -- wsl -d Ubuntu -- bash <wt>/.local/wsl.sh timeout 280 python -m pytest tests/test_nli_scorer.py -m "not models" -q -p no:cacheprovider`

```
FAILED tests/test_nli_scorer.py::test_batches_group_similar_lengths - assert ...
E       assert [[1, 50, 2], [49, 3, 48]] == [[50, 49, 48], [3, 2, 1]]
1 failed, 27 passed, 1 deselected in 6.31s
exit=1
```

Fails for the right reason: input-order batches mix long and short premises.

`test_sorted_batching_keeps_input_order` PASSED in RED. That is expected: the old code already
returns input order, so this test is a regression guard for the new sort, not a RED driver. To
prove it has teeth I ran a mutation: sort and batch in sorted order but do NOT restore input
order. Result:

```
.........F..................
FAILED tests/test_nli_scorer.py::test_sorted_batching_keeps_input_order
(grouping test passed, order test failed on the zip/approx assertion)
```

The mutation was then replaced by the real implementation (restore via dict).

## GREEN

`... pytest tests/test_nli_scorer.py -m "not models" -v`
```
collected 29 items / 1 deselected / 28 selected
test_batches_group_similar_lengths PASSED
test_sorted_batching_keeps_input_order PASSED
test_scorer_maps_labels_and_batches[layout0] PASSED   (5 pairs, batch 2, [2, 2, 1])
test_scorer_maps_labels_and_batches[layout1] PASSED
======================= 28 passed, 1 deselected in 7.35s =======================
exit=0
```

## Full suite and lint (run once, after GREEN, before commit)

Full suite (outer 960 / inner 900):
`... wsl.sh timeout 900 python -m pytest -q -p no:cacheprovider`
```
1656 passed, 3 deselected in 266.48s (0:04:26)
exit=0
```
The 3 deselected are the `models`-marked tests (default config deselects them; I did not run
them and did not pass any option to include them). No failures, nothing red to disclose.

Lint: `... wsl.sh timeout 120 ruff check .`
```
All checks passed!
exit=0
```

## Decisions

- Character-length key (see above). Stable on ties via Python's stable `sorted` with
  `reverse=True`; ties keep input order, which keeps `test_scorer_maps_labels_and_batches`
  (all premises same length) batching as `[2, 2, 1]` over pairs 0..4.
- Dict keyed by original index for un-permuting, rather than a `list[NLIResult | None]`, so the
  return type stays `list[NLIResult]` without Optional handling.
- Extended the test fake with an optional `probabilities` argument instead of growing
  `PROBABILITIES` to 6 rows, because `test_scorer_maps_labels_and_batches` zips 5 results
  against `PROBABILITIES` with `strict=True`.

## Concerns

- Character length is a proxy: it ranks non-English or digit-heavy text slightly differently
  from token count. Batching quality, not correctness, is the only thing affected. If Task 27
  wants the exact token-length key, it needs a pre-tokenization pass.
- The +38% throughput figure is the controller's spike measurement; I did not re-measure it
  (no real model run, models-marked tests not executed per instructions).
- Padding changes can alter logits at float-rounding level between runs with different batch
  composition; the real-model test uses `abs=1e-3` tolerances, which absorbs that. Not run here.
