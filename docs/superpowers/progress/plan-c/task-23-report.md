# Task 23 report: baselines, calibrator training, design selection

Status: DONE_WITH_CONCERNS (see Concerns)

Commit: `f955150 feat: add factuality baselines and calibrator training with design selection`
(branch `feature/evidencegraph`, on top of `9ce33be`; not pushed; the worktree was clean before and after).

## What I built

- `evaluation/baselines.py`
  - `LEGACY_NLI_W = 5/7`, `LEGACY_SIM_W = 2/7`.
  - `legacy_as_shipped_score` (min over signals of `contra + 0.5 * entail`), `legacy_fixed_score`
    (`contra + 0.5 * neutral`), both `5/7 * nli_term + 2/7 * (1 - clip(max dense_cos, 0, 1))`, 1.0 with no
    signals. They use every cached signal (depth 8), so they isolate the scoring method.
  - `legacy_predictions(scores, "binary" | "tiers")` (binary: supported below 0.5 else insufficient;
    tiers: supported below 0.2, contradicted above 0.6, else insufficient; any other mode raises
    `ValueError`).
  - `similarity_support` (clipped max cosine, 0.0 without signals), `nli_support` (max entail, 0.0 without).
  - `choose_threshold(support, gold, grid=i/100)`: best detection F1 (via `detection_prf`) for
    "supported iff support >= t", ties to the smaller t, independent of grid order.
- `evaluation/train_calibrator.py`
  - `FEATURE_SETS` exactly as in the brief.
  - `build_matrix(units, signals, evidence_k)` -> `(X, y, groups)`; a unit missing from `signals`
    raises `KeyError("no cached signals for unit <id>")`.
  - `train(fit, select, risk, ...)`: `fit_logreg(l2=1.0)` on the fit columns of the chosen feature set (mean and
    std come from the fit role), `fit_temperature` on select (1.0 when `use_temperature=False`),
    `select_thresholds` on risk. The `Calibrator` has `feature_names` = the chosen subset, `unit="sentence"`,
    the given `premise_mode`/`evidence_k`/`models`, `candidates=20` (`evaluation.extract.CANDIDATES`).
    The report and the artifact metadata hold split sizes, select metrics (`nll`, `ece`, `brier`, `macro_f1`,
    `macro_f1_decision`), and the risk-split thresholds, source counts and traces.
  - `select_design(splits_by_config, feature_sets=FEATURE_SETS)` -> `{"choice": row, "table": [rows]}`.
  - CLI `python -m evaluation.train_calibrator --cache-key K [--alt-cache-key K2] --manifest P --out P`.
    It reads `config.json` and `units.jsonl` of each cache, builds matrices for `evidence_k` in (1, 2, 4, 8)
    per premise mode, runs `select_design`, trains the chosen configuration, saves the artifact and writes
    `evaluation/results/training_report.json`. I did not run it on real data and did not create
    `backend/calibration/artifacts/default.json`.
- `tests/test_baselines_training.py`: 21 tests (all synthetic, seeded 20261026, whole file about 1 s).

## Test role isolation (hard constraint)

- `TRAIN_ROLES = ("fit", "select", "risk")`. `_read_units` drops test rows from `units.jsonl` before building
  `RTUnit`s, and signals are read through `_CachedSignals`, a lazy mapping over the fit/select/risk unit ids
  only (`SignalCache.get` per unit, never `load_all`).
- The CLI test caches signals for test-role units on purpose and spies on `SignalCache.get`/`load_all`: a
  test-role id or a `load_all` call fails the test. The artifact's `split_sizes` has exactly fit/select/risk.

## RED evidence

1. Before any production file existed:
   ```
   tests/test_baselines_training.py:20: in <module>
       from evaluation import ragtruth, train_calibrator
   E   ImportError: cannot import name 'train_calibrator' from 'evaluation'
   1 error in 1.16s
   ```
2. With behavior-free stubs (every function `raise NotImplementedError`; `FEATURE_SETS` constants only), so
   each test fails on its own:
   ```
   16 failed, 1 passed, 3 errors in 0.58s
   ```
   The 1 pass is `test_feature_sets_group_the_columns_for_the_ablation` (the stub already held the brief's
   constants). The 3 errors are the three `train` tests failing in the module fixture (`build_matrix` stub).
   Every failure was `NotImplementedError` from the stubbed function.

## GREEN evidence

```
$ ... pytest tests/test_baselines_training.py -q
.....................   21 passed in 0.83s
```

Mutation checks (each mutation applied, tests run, file restored; `git status` clean afterwards):

| Mutation | Result |
|---|---|
| `choose_threshold`: `f1 > best` -> `>=` (ties to larger t) | `test_choose_threshold_ties_to_smaller` fails |
| tie-break key: `evidence_k` before `n_features` | `test_select_design_prefers_fewer_features_over_smaller_evidence_k` fails |
| `use_temperature` ignored | `test_train_without_temperature_scales_by_one` fails |
| as-shipped reads `neutral` instead of `entail` | `test_legacy_score_literals` and the min/max test fail |
| `TIE_TOLERANCE = 0` | survived at first; added `test_select_design_treats_rounding_noise_as_a_tie`, which then fails (`assert 8 == 2`) |

Synthetic end-to-end numbers (seed 20261026, 10 percent label noise, 400/150/300 fit/select/risk units):
select NLL 0.395, ECE 0.026, Brier 0.204, argmax macro-F1 0.868, temperature 0.992, Newton converged in 6 steps,
`tau_s = 0.95`, `tau_c = 0.76`. The synthetic risk traces are real LTT output.

## Full suite and lint

- `pytest -q` (full, WSL, timeout 900): `1570 passed, 3 deselected in 284.42s` (the 3 are the `models` marker
  the project's `addopts` deselects). No failures, none pre-existing.
- `ruff check .`: `All checks passed!`. `ruff format --check` on my three files: already formatted. (The repo has
  15 other files that `ruff format --check` would reformat; I left them alone.)
- No em dashes, LF line endings, no `# noqa`, no new dependencies.

## Decisions

- `select_design` values are `(fit, select, risk)` triples, the same three arguments `train` takes, so one
  mapping feeds selection and the final `train(*splits[key], ...)`. Only fit and select are read for selection.
  (The brief only says `tuple`.)
- Select NLL for design selection is measured after the temperature fitted on select, i.e. the NLL of the model
  that would ship; it uses the same `_fit_scaled` helper as `train`.
- Tie rule: NLLs within `TIE_TOLERANCE = 1e-9` tie (last-bit noise between a wide and a narrow fit must not
  decide), then fewer features, then smaller `evidence_k`, then `passage` before `window`.
- The select macro-F1 is reported two ways: `macro_f1` (argmax of calibrated probabilities; the model-quality
  number the brief's "select macro-F1 above 0.8" is asserted on) and `macro_f1_decision` (verdicts under the
  risk-controlled thresholds, i.e. the deployed rule).
- CLI adds no flags beyond the brief. Cache root and report path are the module defaults (relative to cwd), so
  the tests `chdir` into `tmp_path`. `--manifest` is used for `manifest_hash`, and the CLI refuses a cache whose
  `config.json` `manifest_hash` differs, whose `candidates` is not 20, or two caches that share a premise mode
  or differ in anything but it (`ValueError`, raised before any file is written).
- The premise mode of each cache comes from its own `config.json`, not from which flag it was passed to.
- Metadata from the CLI: `manifest_hash`, `cache_key`, `git_commit`, `git_dirty` (reusing
  `run_retrieval.git_info`), `created_at`; `train` adds `feature_set`, `split_sizes`, `select`, `risk`.
- I did not reuse `run_retrieval.collect_env` (it needs a database); only `git_info`.

## Concerns

1. `git_info` returns `(None, None)` when Python runs in WSL on this Windows-created worktree (verified:
   `(None, None)`), so an artifact trained from WSL has `metadata.git_commit = null` and `git_dirty = null`. The
   brief wants the git commit in the metadata. Task 27 should run the CLI where `git rev-parse` works, or add a
   commit override; I did not add a flag the brief does not name.
2. The synthetic `tau_s = 0.95` is a vacuous pass: with 10 percent label noise the calibrated `p_supported` tops
   out near 0.9, so nothing is accepted at 0.95 and the risk is 0 (p = 0.0025). That satisfies the brief's
   "`tau_s` not None", and the wiring is checked against `select_thresholds(calibrator.predict_proba(risk))`.
   With 4 to 6 percent noise the same data gives `tau_s = None` (p = 0.058 and 0.090 at 0.95 on 59 sources), a
   reminder that LTT needs many risk sources; `macro_f1_decision` is about 0.15 to 0.17 on this data because
   the thresholds abstain.
3. Not run at real scale: the CLI re-reads each unit's cache record once per `evidence_k` and premise mode
   (about 8 reads per unit for two caches) to keep memory flat instead of holding all signals. Estimated
   minutes for ~65k units, unmeasured.
4. The legacy baselines score all cached signals (depth 8), as the brief says, not the top `evidence_k`.
