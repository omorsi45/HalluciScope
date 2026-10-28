# Task 24 report: factuality evaluation runner, ablations, atomic subset, report

Status: DONE_WITH_CONCERNS (concerns at the end are about scope edges and what could not be run, not failures)

Worktree: `C:/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/evidencegraph`, branch `feature/evidencegraph`,
base `f955150`. Commit: `79ae886 feat: add factuality evaluation runner, ablations, and report` (3 files, staged by name,
not pushed). Working tree clean afterwards. This report overwrites the notes of the interrupted attempt.

## What was built

- `evaluation/run_factuality.py` (new): `python -m evaluation.run_factuality --cache-key KEY --manifest PATH --artifact PATH
  --split select|test [--final] [--ablations] [--alt-cache-key KEY] [--atomic-subset N] [--out PATH]`.
  - Five systems (`legacy_as_shipped`, `legacy_fixed`, `similarity_only`, `nli_only`, `evidencegraph`) on the cached signals of
    the evaluated role. Per system, overall and per task type: `detection` (P/R/F1/tp/fp/fn/tn), `macro_f1`,
    `false_supported_rate`, `accepted_risk` (`risk`, `k`, `n`, `cp_upper` = `clopper_pearson_upper(k, n, 0.05)`),
    `supported_coverage`, `response_level`, and 95% cluster-bootstrap intervals by source (1,000 resamples, seed 20261026)
    for detection F1, macro-F1, false-supported rate and accepted risk. Legacy systems: binary predictions for everything
    but macro-F1, tiers for macro-F1. `nli_only` and `similarity_only` thresholds come from `choose_threshold` on the select role.
  - Risk-coverage curve with 50 coverage points (coverage `j/50` takes the `ceil(j n / 50)` best-scoring units, integer
    arithmetic) and AURC for every system (`p_supported`, NLI support, similarity support, `1 - legacy score`).
  - `evidencegraph` only: top-label ECE, classwise ECE, Brier, NLL, 15 reliability bins.
  - Outputs: `--out` (default `evaluation/results/factuality_<split>.json`), plus `<stem>_ablations.json` and
    `<stem>_atomic.json` beside it for the report CLI.
  - Test role: `--split test` without `--final` exits 2 in `parse_args`, before any file is touched. The test units have their own
    loader, `_read_test_units`, which only the test split calls; the select split goes through `train_calibrator._open_cache`
    (which drops test rows) and never reaches it. Each test-split run appends `{"at", "git_commit", "artifact_id", "args"}` to
    `evaluation/results/test_log.jsonl` after the inputs are validated and before the first test unit is read (a run that crashes
    while reading is still counted; a run rejected for a bad manifest or artifact is not). The results JSON has
    `test_evaluations: {artifact_id, count (entries for this artifact), log}`.
  - `--ablations`: retrains from the cache(s) per configuration (every `FEATURE_SETS` entry, `evidence_k` in 1, 2, 4, 8, temperature
    off, and the other premise mode when `--alt-cache-key` is given) and scores on the chosen split; rows hold detection F1, macro-F1,
    false-supported rate, accepted risk, NLL, ECE, `tau_s`, `tau_c` plus the configuration.
  - `--atomic-subset N` (needs `--split test --final`): `sample_responses` (stratified by task type with largest-remainder quotas,
    sources shuffled with seed 20261026, one response per source per round), `llm_atomic_claims` with the real `OllamaClient`
    (injectable `client_factory`), `collect_signals` + `compute_features` + `Calibrator` per claim, `align_claims` (claim goes to
    the sentence unit it overlaps most; a sentence is not supported when any aligned claim is not; sentences without claims
    keep their sentence-unit verdict), and the comparison on the same responses (detection and false-supported rate for atomic and
    sentence predictions, claim count, extraction method counts). Model parity with the artifact is checked before any LLM call.
- `evaluation/report.py`: `render_factuality(results, ablations, atomic, *, figures="figures")`, `plot_factuality(results, out_dir)`
  (matplotlib, Agg; `reliability.png`, `risk_coverage.png`), and the CLI `python -m evaluation.report factuality <results.json>
  [--ablations P] [--atomic P] --out docs/results/factuality.md --figures docs/results/figures`. The markdown has the run lines
  (commit, dataset commit, manifest hash, role sizes, models, artifact id, thresholds, baseline thresholds, test evaluation count,
  interval settings), the main table with the header of the brief, per task type tables, calibration, ablations, atomic subset and
  the fixed methodology notes. Every number is read from the JSON, including alpha/delta in the guarantee note.
- `tests/test_run_factuality.py` (new): 32 tests.

## TDD evidence

Stages were run RED then GREEN in turn (command: `.venv\Scripts\python.exe .local\run.py 340 .local\t24.log -- wsl -d Ubuntu --
bash <wt>/.local/wsl.sh timeout 280 python -m pytest tests/test_run_factuality.py -q`).

1. Runner core (guard, log, literal and synthetic metrics, select never reads test):
   RED: `ImportError: cannot import name 'run_factuality' from 'evaluation'` (1 collection error).
   GREEN: `11 passed in 2.41s`.
2. Ablations (7 tests added):
   RED: `4 failed, 14 passed` (`FileNotFoundError: out/results_ablations.json` x3 and `DID NOT RAISE ValueError` for the unknown
   feature set; the three passing ones are guards that only bite once the flag does something).
   One first GREEN attempt failed on my own test data (`assert 1 == 4`: the synthetic units had only two signals, so every
   `evidence_k >= 2` gave the same features); I deepened the synthetic evidence rather than weakening the assertion.
   GREEN: `18 passed in 7.18s`.
3. Report and figures (6 tests added):
   RED: `6 failed, 18 passed` (`AttributeError: module 'evaluation.report' has no attribute 'render_factuality'` /
   `plot_factuality`; CLI `invalid choice: 'factuality'`).
   GREEN: `24 passed in 8.56s`.
4. Atomic subset (6 tests added, 2 of them `pg`):
   RED: `6 failed, 24 passed` (`AttributeError: ... has no attribute 'sample_responses'`, `align_claims`, `_ollama_client`, `run_atomic`).
   GREEN: `30 passed in 7.80s`.
5. Afterwards: a pin test for the confusion-matrix statistics against `classification.py` (written after the code, so it passed at once;
   mutation-checked below) and a tie test for claim alignment (added after mutation M6 survived; verified failing under M6).
   Final file: `32 passed in 7.12s`.

Mutation checks (production line changed, `pytest tests/test_run_factuality.py -x`, file restored after each): thresholds taken from the
evaluated units (caught), macro-F1 from binary predictions (caught), test-log line written after the test read (caught), temperature
switch ignored (caught), premise-mode row reading the artifact cache (caught), "any claim not supported" rule weakened (caught),
log count ignoring the artifact id (caught), test units read for `--split select` (caught), risk-coverage point mapping changed
(caught), guarantee note dropped from the report (caught), alignment tie going to the later unit (survived at first, now caught by
`test_a_claim_between_two_sentences_equally_goes_to_the_earlier_one`).

Hand-derived expectations: `test_literal_metrics_of_the_systems_on_the_select_role` uses eight units whose metrics were worked out by
hand (evidencegraph detection P/R/F1 0.75, macro-F1 109/180, false-supported 0.25, accepted risk 1/4 with its Clopper-Pearson bound,
coverage 0.5, response level; legacy_as_shipped F1 0.4, false-supported 0.75, macro-F1 3/7; legacy_fixed macro-F1 109/180; baseline
thresholds 0.06 and 0.81; curve ends at risk 0.5 at coverage 1.0).

## Full suite and lint

- `python -m pytest -q` (WSL, inner 900 s): `1602 passed, 3 deselected in 365.24s` (the 3 are the `models`-marked tests that
  `addopts` deselects). No failures, no new warnings in the output.
- `ruff check .`: `All checks passed!`. `ruff format` was applied to the two new files; `evaluation/report.py` was already format-clean.
- Em dash scan of the three files: none. Line endings: LF in the working copy; git warns "LF will be replaced by CRLF" as expected.

## Decisions (brief ambiguous or silent)

- Artifact and cache pairing: the artifact is scored on whichever of `--cache-key` / `--alt-cache-key` has its `premise_mode`; the cache's
  `models` must equal `artifact.models`, and `artifact.metadata.manifest_hash` must equal the manifest hash (so an artifact trained on
  other sources cannot be scored on this test role). Both raise `ValueError`; these are rejections before the log line.
- Baselines (legacy, `nli_only`, `similarity_only`) use all cached signals (depth 8); the artifact uses its top `evidence_k`. This is
  the "same evidence" of the methodology note.
- `accepted_risk.cp_upper` is `clopper_pearson_upper(k, n, 0.05)` always, so it is 1.0 when nothing is accepted (`risk` is then null and
  prints as `n/a`).
- `--atomic-subset` needs `--split test`; `--database-url` and `--data-root` (same defaults as `evaluation.extract`) were added because
  the atomic subset has to ingest sources into Postgres and read the response texts. The results JSON also carries `cp_delta` so the
  report text has no literal numbers of its own.
- Intervals are computed from 3x3 confusion matrices (so 1,000 resamples stay fast on ~20k units); point estimates use
  `evaluation/metrics/classification.py` and a test pins the two.
- The "premise mode" ablation row is the other premise mode of `--alt-cache-key` (window when the artifact is passage), named by its mode.
- Atomic verdict of a sentence with a contradicted claim is `contradicted`, else `insufficient`; only supported vs not supported feeds the
  reported metrics.

## Concerns

1. Not run on real data, real models, or a real Ollama (per instructions). The `main` path `asyncio.run(run_atomic(...))` with the real
   `OllamaClient` and `load_models("real", "real", ...)` is only covered through `run_atomic` with injected fakes; the default
   factory is checked to build an `OllamaClient` from `Settings`.
2. Under WSL `git_info` returns `(None, None)` on this Windows-created worktree, so `git_commit` is null in the results and the test log
   (known from Task 23); the report prints "unknown".
3. Real-scale bootstrap time was not measured (synthetic runs use 200 resamples via the module constant `N_RESAMPLES`, which the tests
   patch; the default is 1,000).
4. `count_test_runs` raises on a malformed test-log line (loud on purpose); an append cut off by a crash would need a manual fix.
5. Out of scope and left alone: `evaluation/AGENTS.md` still lists only the retrieval commands.
6. The artifact check on `metadata.manifest_hash` rejects hand-built artifacts without that field (anything made by
   `train_calibrator` has it).
