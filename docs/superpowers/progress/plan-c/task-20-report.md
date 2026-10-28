# Plan C Task 20 report: classification metrics and the cluster bootstrap

Status: DONE

Commit: `95951b3 feat: add claim classification metrics and cluster bootstrap` (branch `feature/evidencegraph`, not pushed). Staged by name: the three files below. Pre-commit gitleaks ran clean.

## What I built

- `evaluation/metrics/classification.py`: `CLASSES`, `detection_prf`, `macro_f1`, `false_supported_rate`, `accepted_risk`, `supported_coverage`, `confusion`, `response_level`. Pure Python, no metric libraries.
- `evaluation/metrics/bootstrap.py`: `cluster_bootstrap` (numpy only for the rng, concatenation and `np.percentile`).
- `tests/test_classification_metrics.py`: 21 tests, every expected value a hand-derived literal.

`evaluation/metrics/__init__.py` is untouched (the brief lists only the three files; tests import from the submodules).

## RED

`wsl.sh timeout 280 python -m pytest tests/test_classification_metrics.py -q` before any implementation, under the Windows-level killer (`.local/run.py 340`). Log `.local/t20-red.log`:

```
tests/test_classification_metrics.py:11: in <module>
    from evaluation.metrics.bootstrap import cluster_bootstrap
E   ModuleNotFoundError: No module named 'evaluation.metrics.bootstrap'
1 error in 1.16s      exit=2
```

Fails because the feature is missing. The modules do nothing expensive, so RED returned immediately.

## GREEN

Focused run after implementing: `21 passed in 0.42s` (log `.local/t20-green.log`).

Mutation check (`.local/t20-mutate.py`, untracked, restores each file in a `finally`): 16 mutants. First pass left 3 survivors, all genuine test gaps, fixed by strengthening tests (not by touching the code):

- `false_supported_rate` with `p != supported`: survived because the brief's literal gives 0.5 either way. Added asymmetric data (`[C,I,I,S]` vs `[S,S,I,I]` -> 2/3) and the same for `accepted_risk` (`[C,I,S,S]` vs `[S,S,I,S]` -> `(2/3, 2, 3)`).
- `confusion` indexed `[pred][gold]`: survived because the literal confusion matrix is symmetric. Added an asymmetric case.
- Removed explicit length check: survived because `zip(strict=True)` covers most paths, but `false_supported_rate`/`accepted_risk` return early on a zero denominator before zipping. Added tests for those paths.

Second pass: `survivors: 0` (16/16 KILLED, log `.local/t20-mut2.log`). Mutants covered: fp/fn swap, F1 denominator, absent class included in macro mean, wrong risk numerator, risk-undefined return, first-unit-only response aggregation, confusion transposed, label and length validation removed, empty coverage guard, bootstrap draw count, unseeded rng, None not skipped, wrong percentile tails, row-level instead of group-level resampling, non-linear percentile method.

Literals verified (gold `[S,C,I,S]`, pred `[S,S,I,C]`): tp=fp=fn=tn=1, P=R=F1=0.5, false-supported 0.5, accepted risk `(0.5, 1, 2)`, macro-F1 0.5, confusion `{S:{S:1,C:1,I:0}, C:{S:1,..}, I:{I:1,..}}`, coverage 0.5.

## Full suite and lint

- `wsl.sh timeout 900 python -m pytest -q`: `1462 passed, 3 deselected in 377.06s (0:06:17)`, exit 0 (log `.local/t20-full.log`). The 3 deselected are the `models`-marked tests that `addopts = "-m 'not models'"` in `pyproject.toml` excludes by default, not anything I skipped.
- `ruff check .`: `All checks passed!`
- `ruff format --check evaluation/metrics tests/test_classification_metrics.py`: `5 files already formatted` (I ran `ruff format` on `classification.py` once, two `sum(...)` calls collapsed to one line).
- No em dashes, LF endings confirmed (`git ls-files --eol`: `i/lf w/lf` for all three).

## Decisions (brief silent or ambiguous)

1. Unique groups are taken in sorted order (`np.unique`), so the interval does not depend on row order. Each resample draws `n_unique_groups` groups with `rng.integers(0, n_groups, size=n_groups)`. First-appearance order would give different numbers for the same seed, so if Task 21 or the published numbers assume that order, say so. Rows per group are found with one stable argsort (O(N log N)) instead of a per-group scan.
2. `lo`/`hi` are returned as plain `float`; `point` is whatever the statistic returned. When every resample is None the result is `(point, None, None)` (point kept if defined), and `(None, None, None)` when the point is also None.
3. Input validation (not in the brief): equal lengths and labels in `CLASSES` raise `ValueError`, `response_ids` must match in length. Silent truncation or a mistyped label counted as "not supported" would corrupt the published numbers. Covered by `test_inputs_are_validated`.
4. Empty input: `macro_f1([], [])` and `supported_coverage([])` return 0.0 (the brief's zero-denominator convention); `false_supported_rate` and `accepted_risk` return their documented None values.
5. `detection_prf` counts a unit as a true positive when both gold and pred are not supported, whatever the unsupported class (gold contradicted, pred insufficient is a tp), matching the brief's positive = not supported. F1 is computed as `2tp / (2tp + fp + fn)`, which equals the harmonic mean of precision and recall.

## Concerns

None blocking. Only the group order choice in decision 1 deserves a look from whoever consumes `cluster_bootstrap` in Task 21.
