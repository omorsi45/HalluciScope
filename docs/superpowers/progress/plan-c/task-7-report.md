# Plan C Task 7 report: risk control (Learn-then-Test, Hoeffding-Bentkus)

Status: DONE

Commit: `6eb93a3` feat: add Learn-then-Test threshold selection with Hoeffding-Bentkus p-values
(branch `feature/evidencegraph`, not pushed; pre-commit gitleaks ran clean)

## What was built

- `backend/calibration/risk.py` (new, 174 lines): `GRID`, `binom_cdf`, `clopper_pearson_upper`, `hb_pvalue`,
  `per_source_risk`, `ltt_fixed_sequence`, `ThresholdResult` (frozen dataclass), `select_thresholds`.
  Interfaces exactly as in the brief. Pure Python `math` plus numpy, no scipy.
  - `binom_cdf`: `math.lgamma` log pmf terms combined with log-sum-exp; `k < 0 -> 0.0`, `k >= n -> 1.0`;
    `p <= 0 -> 1.0`, `p >= 1 -> 0.0` (for `0 <= k < n`); result clamped to 1.0.
  - `clopper_pearson_upper`: bisection on `binom_cdf(k, n, p) >= delta` until the bracket is below 1e-12,
    returns the high end (errs conservative); `n == 0` or `k >= n -> 1.0`.
  - `hb_pvalue`: `min(1, exp(-n h(min(r_hat, alpha), alpha)), e * binom_cdf(ceil(n r_hat - 1e-9), n, alpha))`,
    `0 ln 0 = 0`, `n == 0 -> 1.0`.
  - `per_source_risk`: vectorized (`np.unique(..., return_inverse=True)` + `np.bincount`), one pass per call.
  - `ltt_fixed_sequence`: trace row per evaluated threshold including the failing one; stops at first `p > delta`.
  - `select_thresholds`: SUPPORTED on `probs[:, 0]`, `y != 0`; CONTRADICTED on `probs[:, 1]`, `y != 1`,
    `accept_mask = probs[:, 0] < tau_s` (all true when `tau_s is None`).
- `tests/test_risk.py` (new, 21 tests with parametrization, 285 lines).

## RED evidence

Command (focused, before `risk.py` existed):
`wsl -d Ubuntu -- bash .../.local/wsl.sh timeout 300 python -m pytest tests/test_risk.py -q`

```
ERROR collecting tests/test_risk.py
tests/test_risk.py:10: in <module>
    from backend.calibration.risk import (
E   ModuleNotFoundError: No module named 'backend.calibration.risk'
Interrupted: 1 error during collection
1 error in 0.83s
```

Fails fast on the missing module, nothing expensive triggered.

## GREEN evidence

`... timeout 300 python -m pytest tests/test_risk.py -q --durations=3` ->
`21 passed in 0.88s` (slowest: `test_ltt_guarantee_simulation` 0.54 s).

All literals from the brief pass with the brief's tolerances (the `binom_cdf(500, 1000, 0.5)` large-n case
also passes at 1e-12). Expected values were cross-checked with exact rational arithmetic in a scratch script:
`binom_cdf(2,10,.1) = 0.9298091736`, `hb(0,100,.1) = 0.9^100`, `hb(.05,100,.1)` Bentkus = 0.15651020427695314
(Hoeffding 0.18812...), `P(Bin(1000,.5) <= 500) = 0.5126125090891804`.

## Simulation (`test_ltt_guarantee_simulation`)

- Seed `numpy.random.default_rng(20261026)`; 200 repetitions of 300 sources (1 to 3 bad rows ~ Beta(2,5),
  1 to 3 good rows ~ Beta(5,2)); true risk curve over `GRID` estimated once from 200,000 fresh Beta(2,5)
  scores (`true_risk[0.50] = 0.1085 > 0.10`, so the scenario is non-vacuous; the test asserts this).
- Observed violation fraction (true risk at selected `tau_s` above `alpha_s = 0.10`): **0 / 200 = 0.000**
  (limit 0.08). `tau_s` was never None; selected `tau_s` ranged 0.52 to 0.61, mean 0.563, mean true risk
  at the selected threshold 0.0613.
- Runtime: **0.52 s** for the loop (0.54 s test call), far under the 60 s target. Vectorization is the
  per-source bincount in `per_source_risk`; the true risk is computed once, not per repetition.
- Power checks (scratch script outside the repo, same generator, `hb_pvalue` monkeypatched):
  - mutant "p always 0" (never stops): 200 / 200 violations (1.000), caught.
  - mutant "plug-in test, p = 0 iff r_hat <= alpha": 48 / 200 violations (0.240), caught.
  So the 0.08 bound can fail for both a procedure with no stopping rule and one with no concentration bound.
  The real procedure is conservative here (0 violations), as expected for a finite-sample bound at n = 300.

## Mutation checks (scratch, source mutated in memory, not committed)

All 13 mutants were killed by `tests/test_risk.py`: no ceil guard; mask `<=`; mask inverted; None treated as
"nothing accepted"; stop at `p >= delta`; return failing `t`; masked rows dropped from the denominator;
`>=` -> `>` in acceptance; no Hoeffding; no Bentkus; SUPPORTED bad = `y == 1`; CONTRADICTED bad = `y == 0`;
Clopper-Pearson returns the low end. Three mutants initially survived (SUPPORTED bad = `y == 1`,
CONTRADICTED bad = `y == 0`, Clopper-Pearson low end). They led to extra tests before the commit: the no-`tau_s`
leak case is parametrized over gold-contradicted and gold-insufficient, a gold-insufficient row accepted as
contradicted must block `tau_c`, and the Clopper-Pearson test also asserts the bound errs high.

## Full suite and lint

- `timeout 900 python -m pytest -q` -> `1166 passed, 3 deselected, 2 warnings in 147.43s` (exit 0).
  The 3 deselected tests come from the repo's default pytest config, not from this task. The 2 warnings are
  pre-existing (`tests/test_generator.py`, un-awaited `AsyncMock` coroutine in `backend/core/generator.py`),
  unrelated to this change.
- `timeout 120 ruff check .` -> `All checks passed!` (an initial B905 `zip` finding in my test was fixed
  with `itertools.pairwise`).
- `ruff format --check backend/calibration/risk.py tests/test_risk.py` -> 2 files already formatted.
- New files are LF in the index and working copy (`git ls-files --eol`: `i/lf w/lf`), no em dashes.

## Decisions

1. `hb_pvalue` uses the brief's `ceil(n * r_hat - 1e-9)`. The spec R1 text writes `ceil(n * r_hat)`; the brief's
   guard is the same formula protected from float noise (`0.07 * 100 = 7.000000000000001`), test `test_ceil_guard`
   pins it (unguarded k = 8 would give 0.5754, guarded k = 7 gives 0.5601).
2. `accept_mask` ruling implemented as stated: masked-out bad rows stay in the denominator and never count as
   accepted; a source with bad rows all masked out still counts as a source with `L_i = 0`. This matches R1
   (`L_i` denominator is all gold-not-contradicted units).
3. `per_source_risk` with no bad rows returns `(0.0, 0)`; `hb_pvalue(_, 0, _) = 1.0` then makes every threshold
   fail, so `select_thresholds` returns `tau = None` and `n_sources = 0`. Brief was silent; tested.
4. `n_sources_s` / `n_sources_c` are computed directly as the number of distinct groups with at least one bad
   row (not read from the last trace row), so they are correct when the grid is empty or the first test fails.
5. `clopper_pearson_upper` returns the high end of the final bisection bracket (conservative by under 1e-12).
6. `binom_cdf` guards `p <= 0` and `p >= 1` explicitly (log of 0 otherwise); the brief only listed the `k` edges.

## Concerns

- None blocking. Not wired into anything yet (Task 7 is a library module); callers must pass `groups` as one id
  per row (any sortable dtype) and `probs` columns in the fixed (supported, contradicted, insufficient) order.
- `hb_pvalue` assumes `0 < alpha < 1` (no input validation, per the brief); `alpha` outside that range raises
  a math domain or zero-division error rather than returning a value.
