# Plan C Task 5: Calibration metrics (report)

Status: DONE
Branch: `feature/evidencegraph`
Commit: `8beabf4 feat: add calibration metrics` (not pushed)

## What was built

New package `backend/calibration/` with numpy-only metrics (no sklearn). Classes are always
(supported, contradicted, insufficient); `y` holds class indices.

- `softmax(z)`: row-wise (last axis), row max subtracted.
- `ece_top_label(probs, y, n_bins=15)`: confidence is the row max, correct is argmax equals `y`, bin index `min(int(conf * n_bins), n_bins - 1)`, result `sum_b (n_b / n) * |mean_conf_b - acc_b|`.
- `classwise_ece(probs, y, n_bins=15)`: mean over `probs.shape[1]` (3) classes of the same binned gap between `probs[:, c]` and `y == c`.
- `brier_multiclass(probs, y)`: mean over rows of the summed squared error against one-hot.
- `nll(probs, y, eps=1e-12)`: mean of `-log(max(p_true, eps))`.
- `reliability_bins(conf, correct, n_bins=15)`: list of `n_bins` dicts with `lo`, `hi`, `count`, `mean_conf`, `accuracy`; empty bins kept with `None` values.
- `risk_coverage(score, gold_ok)`: stable descending order by score, coverage `j / n`, risk is the fraction of accepted with `gold_ok == False`. `aurc(risks)` is the mean of the risk array.

Internals: `_bin_stats` (np.bincount of counts, confidence sums, correct sums) feeds both `_binned_gap` (ECE) and `reliability_bins`, so ECE and the reliability diagram use one binning rule. `_binned_gap` computes `|conf_sum_b - correct_sum_b| / n`, which is algebraically identical to `(n_b / n) * |mean_conf_b - acc_b|` and leaves empty bins at exactly 0.

## Files

Created (all LF, no em dashes, verified by byte scan):
- `backend/calibration/__init__.py`
- `backend/calibration/metrics.py`
- `tests/test_calibration_metrics.py`

Nothing else touched. No dependency added (numpy 2.5.4 was already pinned).

## RED evidence

Tests written first; no production file existed.

```
wsl ... timeout 300 python -m pytest tests/test_calibration_metrics.py -q
ERROR collecting tests/test_calibration_metrics.py
tests/test_calibration_metrics.py:10: in <module>
    from backend.calibration.metrics import (
E   ModuleNotFoundError: No module named 'backend.calibration'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.89s        (pytest exit 2)
```

The failure is the expected one (feature missing). No stub was created to turn the collection error into per-test failures, because a stub would itself be production code written before a failing test.

## GREEN evidence

First GREEN attempt: 13 passed, 1 failed. The failure was `test_brier_and_nll_average_over_rows`, in a test I added beyond the brief:

```
assert 0.8000000000000002 == 0.5 ± 1.0e-12
```

Root cause (systematic-debugging, not a patch over the symptom): my hand derivation of the second row was wrong. For `probs = [0.1, 0.1, 0.8]`, `y = 1`, one-hot `[0, 1, 0]`, the squared errors are `0.1^2 + 0.9^2 + 0.8^2 = 0.01 + 0.81 + 0.64 = 1.46` (I had used 0.2 for the third term, which is not a probability in that row). Mean over rows is `(0.14 + 1.46) / 2 = 0.8`. The implementation was right; I corrected the test literal and its comment from 0.5 to 0.8. No brief literal and no other assertion was changed.

Final focused run:

```
wsl ... timeout 300 python -m pytest tests/test_calibration_metrics.py -v
14 passed in 0.31s   (pytest exit 0)
```

Tests (6 required by the brief, 8 added for behaviour the brief specifies but its list does not exercise):

| Test | Source | Catches |
|---|---|---|
| test_ece_literal | brief | ECE 0.4 with `[0.9, 0.9, 0.6, 0.6]`, correct `[1, 0, 1, 1]` |
| test_perfect_calibration_zero | brief | conf 1.0 must be clamped into the last bin (else index 15 out of range) |
| test_brier_and_nll_literals | brief | Brier 0.14, NLL 0.35667494393873245 |
| test_risk_coverage_literal | brief | risks `[0, 0.5, 1/3, 0.25]`, AURC 0.2708333333333333, plus coverage values |
| test_reliability_bins_counts | brief | 15 bins, exact per-bin counts, sum equals n, 1.0 lands in bin 14 |
| test_softmax_stable | brief | logits `[1000, 0, -1000]` finite, sums to 1 |
| test_top_label_ece_counts_wrong_argmax_as_incorrect | added | wrong argmax counts as incorrect (ECE 0.8) |
| test_classwise_ece_literal | added | classwise ECE had no test in the brief; hand-derived 0.125 with `n_bins=2` (top-label ECE on the same data would be 0.0625, so the two are not interchangeable) |
| test_brier_and_nll_average_over_rows | added | mean over rows, not sum (the brief literal has n = 1) |
| test_nll_floors_a_zero_probability_at_eps | added | `eps` floor: `-ln(1e-12) = 27.631021115928547`, and a custom `eps=1e-6` |
| test_risk_coverage_orders_by_descending_score | added | the brief literal is already sorted, so a missing sort would pass it; this input is unsorted |
| test_risk_coverage_breaks_ties_by_input_order | added | stable tie order (a reversed tie order gives risks `[1, 0.5, 1/3]`) |
| test_reliability_bins_fields_and_empty_bins | added | `lo`/`hi`, per-bin `mean_conf`/`accuracy`, empty bin `None` values |
| test_softmax_is_row_wise | added | softmax acts per row (axis), with hand literals 1/3 and `[0.5, 0.25, 0.25]` |

Mutation check (reasoned, not executed): removing the `n_bins - 1` clamp breaks the perfect-calibration and reliability tests; dropping the sort breaks the unsorted risk-coverage test; an unstable or reversed tie order breaks the tie test; summing instead of averaging breaks the two-row Brier/NLL test; removing the `eps` floor yields `inf` and breaks the eps test; averaging over only some classes breaks the classwise literal; subtracting no row max overflows in the stable softmax test.

## Full suite

```
wsl ... timeout 900 python -m pytest -q
1137 passed, 3 deselected, 2 warnings in 147.68s (0:02:27)   (exit 0)
```

3 deselected are the `models` marker (excluded by the repo's `addopts = "-m 'not models'"`). The 2 warnings are pre-existing and unrelated: `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` from `tests/test_generator.py` (`backend/core/generator.py:50`). No failures.

## Lint

```
wsl ... timeout 120 ruff check .
All checks passed!   (exit 0)
```

(Before the final edit one comment line in `metrics.py` exceeded the 100-column limit; I wrapped it before the lint run, so the reported result is for the committed files.)

## Commit

Staged by name only: `backend/calibration/__init__.py`, `backend/calibration/metrics.py`, `tests/test_calibration_metrics.py`. Hooks ran (gitleaks: no leaks found). Working tree clean afterwards. Not pushed. The "LF will be replaced by CRLF" warnings are the expected autocrlf notice for new LF files.

## Decisions

- The brief's literals were used verbatim (0.4, 0.14, 0.35667494393873245, `[0, 0.5, 1/3, 0.25]`, 0.2708333333333333). I re-derived each by hand before writing the tests and they agree with the formulas.
- `risk_coverage` returns `(coverage, risks)` in that order, matching "tuple[np.ndarray, np.ndarray]" and the order the brief introduces them (coverage `j / n` first, then risk).
- `gold_ok` is cast with `astype(bool)`, so int `[1, 0, 1, 1]` and bool arrays both work; likewise `correct` in `reliability_bins` is cast to float.
- `ece_top_label` breaks argmax ties toward the lower class index (numpy default), i.e. toward `supported` first.
- Module constant `N_BINS = 15` is the default for all three binned functions, so the default cannot drift between ECE and the reliability diagram.

## Concerns

None blocking. Two things downstream callers should know, both outside the brief so not changed:
- No input validation: shape mismatches between `probs` and `y`, or an empty input, are not rejected (an empty input makes `_binned_gap` divide by zero and `aurc` return NaN with a numpy warning). The calibration pipeline in later tasks should not pass empty splits.
- Confidences are assumed to be in [0, 1]. A value above 1 is clamped into the last bin; a value below `-1/n_bins` makes `np.bincount` raise ValueError, and one between `-1/n_bins` and 0 silently lands in bin 0 (the cast truncates toward zero).
