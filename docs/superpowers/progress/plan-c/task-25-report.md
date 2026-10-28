# Task 25 report: synthetic temporal and provenance benchmark

Status: DONE_WITH_CONCERNS (concerns are about the brief's wording, listed at the end; the work matches the brief's intent and every check is green).

Commit: `97047a0 feat: add synthetic temporal provenance benchmark` on `feature/evidencegraph`, not pushed. Staged by name: `evaluation/synthetic_temporal.py`, `evaluation/run_temporal.py`, `evaluation/report.py`, `tests/test_temporal.py`.

## What was built

- `evaluation/synthetic_temporal.py`: frozen `FactSpec`, `FamilySpec`, `TemporalClaim` exactly as in the brief, plus `SyntheticCorpus`, `generate(seed=20261026, n_families=40)`, `render_document(family, version_index)`, and small helpers (`claim_value`, `document_title`, `fact_sentence`, `DOMAIN_FACTS`, `FILLERS`). Four domains with 12 fact templates each (all five kinds in each), so a family (6 to 10 facts, plus the one v2 adds) always leaves one fact it never mentions for the `absent` claim. Only `random.Random(seed)` and the fixed tables in the module; no network.
  - v1 effective `2026-01-01 + index days`, v2 `+90`, v3 `+180`; v2 changes 2 to 4 facts, removes 1, adds 1; `n_families // 4` families get a v3 that changes 1 to 2 more.
  - Claims per fact: `true_v1`, `true_latest` (only when the latest value differs from v1), `never_true` (a value that never occurs in any version; skipped for a boolean that takes both values), plus one `absent` claim per family. Gold by the three-way rule from the brief, computed from `versions`.
  - One distractor per family (other entity, same templates and keys, other values, one version) and `min(10, n_families)` foreign copies whose every present value conflicts with the original at that version (a value that is constant across versions stays constant in the copy, so the copy changes where the original does).
- `evaluation/run_temporal.py`: the CLI from the brief (`--database-url --families --embedder --nli --artifact --out --sample-out`; same env var and default database as `run_retrieval`), the flow from the brief, and pure metric functions (`version_scores`, `temporal_correct(ness)`, `source_change_scores`, `span_check`, `span_scores`, `gold_changed`, `predicted_changed`, `count_cross_tenant_links`, `render_sample`). Exit code 1 when `cross_tenant_links != 0` (files are written first). Fresh tenants per run (`temporal-<8 hex>-a/-b`), so a rerun never meets old documents.
- `evaluation/report.py`: `render_temporal(results)` and a `temporal` subcommand (`python -m evaluation.report temporal evaluation/results/temporal.json --out docs/results/temporal.md`). The report states that this is an out-of-distribution check labeled by construction, that the calibrator is used unchanged, and that the sample file is for human review; it also lists the metric definitions.
- `tests/test_temporal.py`: 35 tests (33 without Postgres, 2 marked `pg`; counts below).

Not run here, as instructed: the real benchmark (Task 27). No file under `docs/results/` was produced or committed.

## RED

The tests were written first against the not-yet-existing modules. `.local/t25-red.log`:

```
$ pytest tests/test_temporal.py -q -x
ERROR collecting tests/test_temporal.py
tests/test_temporal.py:16: in <module>
    from evaluation import report, run_temporal, synthetic_temporal
E   ImportError: cannot import name 'run_temporal' from 'evaluation' (.../evaluation/__init__.py)
1 error in 1.09s
exit=2
```

Because the whole module is the missing feature, RED is the import failure for every test in the file. After implementing, a mutation check (below) supplied the per-behavior RED: each mutant is a deliberate fault and every one of them makes the new tests fail.

## GREEN

`.local/t25-green4.log` (final code, with the two Postgres tests):

```
ruff format ...: 4 files left unchanged
All checks passed!
...................................                                      [100%]
35 passed in 19.93s
```

`tests/test_temporal.py` has 35 tests: generator determinism/sizes/dates/v2/v3 changes/gold/claim kinds/limits/render/distractors/foreign copies/sentence-extraction round trip, the hand-computed metrics (temporal correctness, version scores, source change, span check, gold/predicted change), `render_sample`, argument parsing, `main` exit code, `render_temporal` golden and CLI, and two `pg` tests: `test_count_cross_tenant_links_sees_a_link_into_the_other_tenant` and `test_run_temporal_smoke_fakes` (about 18 s).

## Full suite and lint

- `pytest -q` (900 s bound, `.local/t25-full.log`): **1637 passed, 3 deselected in 329.63s**, exit 0. No failure.
- `ruff check .`: All checks passed. `ruff format --check` on the four changed files: already formatted.
- LF line endings in all four files (`git ls-files --eol`: `i/lf w/lf`); no U+2014 in any of them. `git diff` of `report.py` touches only the docstring, the new renderer and the new subcommand.
- The pre-commit gitleaks hook ran and found no leaks.

## Mutation check (`.local/t25-mutate.py`, `.local/t25-mut2.log`)

23 mutants over the three source files, each run against `tests/test_temporal.py -x`: **23 killed, 0 survivors**. Examples: swapping contradicted/insufficient in the gold rule (`test_gold_by_construction`), choosing the dropped fact among the changed ones (`test_v2_changes_and_removals`), v3 for half the families (`test_generate_sizes_and_structure`), `true_latest` without the "differs" condition, the foreign copy allowed to keep the original value, `V2_OFFSET` 91 days, an as-of analysis that ignores `as_of` (smoke: `assert 32 == 72`), a bench user disabled after login (smoke: `assert 0 > 0` on `n_checked`), the cross-tenant count made blind (`... and false`, or an empty foreign-id list), a changed `change_points`, accuracy/precision/span/gold-change/predicted-change definitions, and three swapped renderer columns. A first run had 3 survivors (predicted/gold change definitions and a redundant guard); the definitions were extracted into `gold_changed` / `predicted_changed` with unit tests and smoke assertions, and the redundant guard was removed.

## Inspection run (not the real benchmark)

On a scratch database `eg_t25_scratch` (migrated, since dropped), `python -m evaluation.run_temporal --families 8 --embedder fake --nli fake --artifact <test calibrator>` finished in 36 s, exit 0, then `evaluation.report temporal` rendered the markdown. 8 families, 169 claims, 18 versions, 26 analyses, 20 re-verification jobs, 2188 of 2188 evidence links valid, `cross_tenant_links` 0, every claim re-checked at least once, source-change precision 0.93 and recall 0.89 (fake NLI, so only the shape matters). The 30-case sample file had 30 rows. Segmentation sweep: 40 seeds x 40 families, the claims of every family come back unchanged from `sentence_claims`, 15 to 29 claims per family, and every version of every family and foreign copy renders to a distinct document.

## Decisions (where the brief and the code disagreed)

1. **`cross_tenant_links` is not counted as eg_owner.** The brief says to count it as `eg_owner` because `eg_app` cannot bypass RLS. Verified on the cluster: `eg_owner` is `rolsuper f`, `rolbypassrls f`, and RLS is forced (`relforcerowsecurity t`) on `passages` and `evidence_links`; in `evidencegraph_bench`, `select count(*) from passages` returns 26398 as `postgres` and 0 as `eg_owner`. An owner-level count with no tenant setting sees no rows, so it would print a meaningless 0 and could never catch a leak. `count_cross_tenant_links(db, A, B)` instead reads tenant B's passage ids inside B's transaction and counts, inside A's transaction, A's `evidence_links` rows whose `passage_id` is among them (each side is read where it is visible; the role does not matter). A test fabricates a link into the other tenant the way a faulty repair could (superuser, `session_replication_role = replica`) and checks the count goes 0 to 1, and that the mirror direction stays 0; the run exits non-zero when the count is not 0, as asked.
2. **The bench users are not disabled.** The brief says to reuse the CLI approach (a disabled user with an unusable password hash). `recheck_claims` resolves the analysis creator with `principal_for_user`, which returns None for a disabled user, and then writes nothing. The bench users get the CLI's locked hash (`!`, nothing verifies against it, so nobody can log in) but stay enabled. Demonstrated by mutation: disabling the user after the principal exists makes every `claim_checks` count 0 (`assert 0 > 0` in the smoke test).
3. **Predicted source change = the claim's latest verdict (newest re-check, else its own) differs from its original verdict**, not literally "the latest `claim_checks.changed`". The gold change is defined as last version against v1. For a claim with one re-check the two are identical, but for a v3 family the later check's `changed` only compares v3 with v2 (the previous check), so a claim that changed at v2 and is confirmed at v3 would be counted as not changed against a gold of changed. The cumulative form matches the gold definition. The renderer and the module docstring state the definition. If the controller wants the literal reading it is a two-line change in `run` (use `latest[row_id]["changed"]`).
4. **Sequential rounds**: version 2 of every document is uploaded and all `reverify_document` jobs run, then version 3 and its jobs, so each check reflects the corpus as it was at that point. The run fails (`RuntimeError`, after the jobs) unless every queued `reverify_document` job of the two tenants is `succeeded`, so a silent failure cannot turn into an empty source-change measurement.
5. Analyses are created and run one at a time (`max_active_analyses` is 3 per user and `enqueue=False` leaves them `queued` until run). The run raises `analysis_timeout_s` to 3600 s via `Settings(...)` for the real models (default 300 s).
6. `TemporalClaim` keeps the six fields of the brief; the value a claim states is read back from its sentence with `claim_value(claim, family)` (templates have exactly one `{value}`), which the gold test and the span check use.
7. `render_sample` lives in `run_temporal.py` (the brief names only `render_temporal` for `report.py`); both only format. The sample (seed 20261026, 30 of the as-of cases, sorted by claim and version) is also stored in the results JSON under `sample`.
8. Calibrator, models and database defaults reuse the existing helpers (`extract.load_models`, `run_retrieval.DEFAULT_DATABASE_URL` and `collect_env`, `beir.ensure_bench_scope`); the calibrator is loaded with `Calibrator.load` and never modified.

## Concerns / notes for Task 27

- `evidencegraph_bench` has no `evidence_links` table (migration 0004 is not applied there; `pg_class` shows only `passages`): run `migrate` on it before the real run.
- The default `--artifact` is `backend/calibration/artifacts/default.json`. With real models the artifact must have been trained with the same NLI, embedder and reranker ids (parity is enforced by the engine); a mismatch fails every analysis as `calibrator_mismatch`.
- Cost: about 21 claims per family (15 to 29), 40 families, 130 analyses with the real NLI plus the re-verification of claims whose evidence is thin (in the inspection run every claim was re-checked at least once). It is the largest benchmark so far; expect it to be slow on CPU.
- `eg_owner`-level counting (the brief's wording) would have been vacuous; see decision 1. The same blind spot could affect other bench checks that were written assuming owner access.
- A Windows-created worktree has no git for the WSL process, so `env.git_commit` is `null` in a run started from WSL (existing R-B2 behavior); the report prints `unknown`.
