# Plan C Task 6 report: multinomial logistic regression and temperature scaling

Status: DONE
Commit: `de77137` feat: add Newton-fit multinomial logistic regression and temperature scaling
Branch: `feature/evidencegraph` (not pushed)

## What was built

- `backend/calibration/logreg.py`
  - `LogRegModel` (frozen dataclass: `mean`, `std`, `W`, `b`, `n_iter`, `converged`).
  - `objective(theta, Xs, Y, l2)`: stable log-softmax NLL plus `0.5 * l2 * ||W||^2` (bias column unpenalized), flat gradient, and the full `(K(d+1), K(d+1))` Hessian. The Hessian is vectorized: per-row `diag(p) - p p^T` of shape `(n, K, K)` and per-row `x x^T` of shape `(n, d+1, d+1)` are contracted over `n` with one `np.einsum("nab,nkl->akbl", ..., optimize=True)`, then `l2` is added to the weight diagonal entries. No Python loop over rows.
  - `fit_logreg`: standardizes (std 0 replaced by 1), appends ones, zero start, Newton with `(H + 1e-6 I) step = grad`, backtracking from t = 1 with Armijo `c = 1e-4`, at most 30 halvings, stop when `max |grad| < tol * n` or after `max_iter` steps. The accepted candidate's value, gradient, and Hessian are reused as the next iteration's state, so each accepted step costs one extra objective evaluation. After the fit the mean bias across classes is subtracted. If no step size passes Armijo, the loop stops and `converged` stays False.
  - `predict_logits`: standardize with the stored `mean`/`std`, then `Xs @ W.T + b`.
- `backend/calibration/temperature.py`: `fit_temperature` runs golden-section search on `log T` in `[lo, hi]` to width `tol` using `nll(softmax(logits / T), y)` from `metrics.py`, and returns `exp` of the interval midpoint.
- `tests/test_logreg_temperature.py`: 8 tests.

Interface reading: `Xs` passed to `objective` already carries the trailing ones column (shape `(n, d + 1)`), `Y` is one-hot `(n, K)`, and `theta` is `(K, d + 1)` row-major (index `c * (d + 1) + k`). The brief says "standardized `Xs` with an appended column of ones", and this is how I read it. Task 7 onward only needs `fit_logreg`, `predict_logits`, and `fit_temperature`; `objective` is exposed only because the brief lists it.

## Files changed

- `backend/calibration/logreg.py` (new, LF)
- `backend/calibration/temperature.py` (new, LF)
- `tests/test_logreg_temperature.py` (new, LF)

Verified before commit: no CR bytes in the three files, no U+2014 character, staged by name, hooks ran (gitleaks: no leaks).

## RED evidence

Tests written first, before either module existed.

```
wsl.sh timeout 300 python -m pytest tests/test_logreg_temperature.py -q
ERROR collecting tests/test_logreg_temperature.py
tests/test_logreg_temperature.py:12: in <module>
    from backend.calibration.logreg import fit_logreg, objective, predict_logits
E   ModuleNotFoundError: No module named 'backend.calibration.logreg'
1 error in 0.82s   (exit 2)
```

It failed on the missing module, in under a second. No fit ran.

## GREEN evidence

First GREEN run after implementing: 8 passed, 1 failed (`test_recovers_known_logit_differences`, mean abs error 0.228 against the 0.1 bound).

Root cause (systematic debugging, not a code bug): my first `W_true`/`b_true` made class 1 about 85% of the labels (counts `[1906, 17042, 1052]`), so the logit differences were poorly determined. I checked the fit itself with a throwaway script (deleted afterwards):

- `n_iter = 8`, `converged = True`, `max |grad| = 8.7e-9` (threshold `tol * n = 2e-4`).
- Un-standardized fitted parameters were close to the truth (slope differences 1.455 / -0.892 / -0.963 / 1.100 against 1.5 / -1.0 / -1.0 / 1.2).
- Re-running the same design on seeds 0 to 4 gave mean abs errors from 0.02 to 0.22, so the spread came from sampling noise, not from the solver.

Fix (test data, brief criterion kept at < 0.1): `W_true = [[0, 0], [0.5, -1.0], [-0.4, 1.2]]`, features `N(loc=[1, -2], scale=[2, 0.5])`, and `b_true = [0, 0.3, -0.2] - W_true @ loc`, which cancels the mean logit offset so the classes are balanced (counts `[4960, 8634, 6406]` at the brief seed). Across the brief seed and seeds 0 to 7, the worst mean abs error was 0.044 and every run converged in 5 iterations.

Final focused run:

```
wsl.sh timeout 300 python -m pytest tests/test_logreg_temperature.py -v --durations=10
test_gradient_matches_finite_differences PASSED
test_objective_value_and_gradient_literal PASSED
test_recovers_known_logit_differences PASSED
test_separable_data_stays_finite PASSED
test_missing_class_ok PASSED
test_constant_feature_no_nan PASSED
test_temperature_recovers_scale PASSED
test_temperature_bounds PASSED
8 passed in 0.38s
```

Slowest calls: `test_temperature_recovers_scale` 0.03 s, `test_recovers_known_logit_differences` (n = 20,000, 5 Newton iterations) 0.02 s, `test_gradient_matches_finite_differences` 0.01 s. Everything else under 5 ms.

## Test design notes

- `test_gradient_matches_finite_differences`: 25 x 3 random problem, central differences with `h = 1e-6` on all 12 gradient entries (max abs error < 1e-5), and Hessian columns checked on coordinates 0, 3, 5, 8, 11 (weights and biases, three classes) against central differences of the gradient.
- `test_objective_value_and_gradient_literal` (added beyond the brief's list): hand-derived value `4 ln 3 + 3` and gradient for a zero-feature problem, so value, penalty scope, and gradient are pinned by literals and not only by self-consistency with finite differences. It guards the one thing the finite-difference test cannot: a consistently wrong value.
- `test_separable_data_stays_finite`: three well-separated clusters, default `l2 = 1.0`, asserts finite parameters, `converged`, and that the fit classifies the training points.
- `test_missing_class_ok`: labels 0 and 2 only; class-1 probability below 0.05 everywhere. The class-1 bias runs toward minus infinity; the fit returns finite parameters regardless.
- `test_constant_feature_no_nan`: asserts `std == 1` for the zero-variance column, finite parameters, a zero weight column, and that changing the constant column at predict time does not change the logits.
- `test_temperature_bounds` uses logits with margin 1, so the NLL stays strictly decreasing down to the lower bound. With a huge margin the NLL is exactly 0 over a flat stretch of `log T` and golden-section tie-breaking would pick an arbitrary point there. See concern 1.

I tried one extra test (fitted biases have zero mean) and removed it: the bias gradient sums to zero over classes, so from a zero start the bias mean is zero in exact arithmetic and the subtraction only removes round-off. No realistic mutation could fail that test.

Mutation checks run by hand, then reverted:

- Penalizing the bias in the gradient (`grad += l2 * Theta`): 3 tests failed (gradient/Hessian finite differences, recovery, separable).
- Removing the `std == 0` replacement: `test_constant_feature_no_nan` failed.

## Full suite and lint

- Full suite: `wsl.sh timeout 900 python -m pytest -q` gave `1145 passed, 3 deselected, 2 warnings in 151.13s`. The 2 warnings are an unawaited `AsyncMock` coroutine in `tests/test_generator.py` (`backend/core/generator.py:50`), unrelated to this task and present before it.
- Lint: `wsl.sh timeout 120 ruff check .` gave `All checks passed!`.

## Decisions

- The Armijo loop allows 31 trial step sizes (t = 1 down to 2^-30, that is 30 halvings). If none passes, the fit stops and reports `converged = False` rather than taking an unsafe step.
- `n_iter` counts accepted Newton steps. The convergence test runs before each step, so a fit that starts converged reports `n_iter = 0`.
- Population standard deviation (`ddof = 0`) for standardization.
- No input validation for labels outside `[0, n_classes)`; the brief does not ask for it and `np.eye(n_classes)[y]` raises `IndexError` there.

## Concerns

1. `fit_temperature` on perfectly separable logits with a large margin (NLL exactly 0 over a stretch of `log T`, because probabilities saturate at 1.0) has a flat objective, so golden-section tie-breaking (`f_c < f_d` false) moves toward the upper end and the result in the flat zone is arbitrary. Real calibration logits come from a fit with `l2 = 1.0`, so margins stay moderate; the Task 7+ callers should not hit this. Not changed, since the brief fixes the algorithm.
2. `test_recovers_known_logit_differences` data differs from the brief's loose wording only in the (free) choice of `W_true`/`b_true`; the threshold and seed are as specified.
