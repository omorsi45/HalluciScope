# Task 22 report: signal cache, extraction runner, throughput-based sizing

Status: DONE_WITH_CONCERNS (one blocking finding for Task 27: at the measured throughput the extraction does not fit the 4 h budget even at the floors; see Concerns).

Commit: `9ce33be feat: add cached evidence extraction for factuality benchmarks` (3 files, 1,421 insertions, LF, gitleaks hook passed, nothing pushed).

## What was built

- `evaluation/cache.py`
  - `config_key(models, *, premise_mode, depth, candidates, manifest_hash)`: first 16 hex chars of the sha256 of the canonical JSON (`sort_keys`, compact separators) of `config_dict(...)`. `config_dict` is the same dict, reused for `config.json`.
  - `SignalCache(root, key, *, config=None)`: `root/<key>/signals.jsonl` + `config.json`; `__contains__`, `get`, `put_many` (append, flush, fsync), `load_all`, `directory`. In memory it keeps unit id to byte offset only, so a long run does not hold the signals; `get` seeks one line.
  - Crash safety: a record counts only when its line ends with `\n`. Load ignores an unterminated tail; `put_many` cuts the tail off before appending (so the next record is never glued onto half a line). A complete line that is not a record raises `ValueError` (no silent drop). `put_many` skips ids already cached, and re-syncs with the file when its size differs from what the instance remembers (another instance appended), so a stale instance can neither duplicate nor truncate records.
  - Line layout exactly as in the dispatch: `{"unit_id": ..., "signals": [{passage_id, text, nli: {entail, contra, neutral}, rerank, dense_cos}]}`.
  - `write_atomic(path, text)` (temp file + `os.replace`) for `config.json` and `units.jsonl`.
- `evaluation/extract.py`
  - `python -m evaluation.extract --roles ... [--final] --premise-mode passage|window [--limit-units N] [--embedder real|fake] [--nli real|fake] [--database-url URL] [--data-root PATH] [--manifest PATH]`. `test` in `--roles` without `--final` is `parser.error` (exit 2, message names `--final`). Unknown roles also exit 2.
  - Flow: verify and load the data (`data.fetch` sha256 check, `load_ragtruth`), read the manifest, `plan_units` (pure; the run order is roles as in `ROLES`, manifest source order, file order of responses; `--limit-units` takes the first N units), load models, `config_key`, open the cache, write `units.jsonl`, and only if some unit is uncached open the database, `ensure_bench_scope` for tenant `ragtruth-bench`, `ingest_ragtruth` for the sources with pending units, then per response `collect_signals(..., scope=SearchScope(tenant_id, (collection_id,)), depth=8, candidates=20, premise_mode=...)` with `Claim(ordinal=i, text, answer_start, answer_end, method="sentence")` and `cache.put_many` after every response. Progress on stderr every 100 units with units/s.
  - `units.jsonl` (in the cache folder): one line per unit with all `RTUnit` fields (unit_id, response_id, source_id, task_type, role, start, end, text, gold). Claim text is included because the features need it. It is rewritten atomically only when new units are added, and always before their signals are cached.
  - `measure --units 200 [--premise-mode passage|window] [--database-url] [--data-root] [--out]`: samples fit units, ingests their sources (untimed), times `collect_signals`, counts NLI pairs and NLI seconds with a wrapping scorer, and writes `evaluation/results/throughput.json`.
  - `plan_sizes(seconds_per_source, population, budget_s=14_400.0)` exactly per the brief (start: all, fit capped at 600; shrink by 10 in the order fit, select, risk, test, floors 200/150/150/150; floors returned with `"over_budget": True` when still over).
- `tests/test_extract_cache.py`: 42 tests.

## Files changed

`evaluation/cache.py`, `evaluation/extract.py`, `tests/test_extract_cache.py` (all new). Nothing else touched.

## RED evidence

RED 1 (nothing implemented), `pytest tests/test_extract_cache.py -q -x`:

```
ERROR collecting tests/test_extract_cache.py
tests/test_extract_cache.py:11: in <module>
    from evaluation import data, extract, ragtruth
E   ImportError: cannot import name 'extract' from 'evaluation'
1 error in 1.03s   exit=2
```

RED 2 (interface stubs only: empty `extract.py`, `SignalCache` with no `__init__`, `config_key` raising `NotImplementedError`), so each test fails for the missing behavior and none can reach a model, a network call, or a long loop:

```
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  [100%]
E   TypeError: SignalCache() takes no arguments          (cache tests)
E   AttributeError: module 'evaluation.extract' has no attribute 'plan_sizes' / 'plan_units' / 'role_profile' / 'sample_units' / 'main' / 'parse_args' / 'parse_measure_args' ...   (the rest)
40 failed in 0.87s   exit=1
```

(That was the 40 tests of the first draft; two cache tests were added after the mutation check, see below.)

## GREEN evidence

- First GREEN: `pytest tests/test_extract_cache.py -q` -> `40 passed in 2.65s`.
- After the mutation check added two tests: `42 passed in 2.63s`.

### Mutation check (`.local/t22-mutate.py`, 36 single-site mutants of cache.py and extract.py)

First round: 5 survivors. Each was analysed:

1. `self._offsets[unit_id] = self._end` -> `0`: no test read a unit through the instance that wrote it. Added `test_cache_serves_units_written_through_the_same_instance`.
2. Dropping the resync in `_sync`: the stale-instance test passed by luck (truncation then re-append gave the same bytes). Rewrote `test_cache_notices_what_another_instance_appended` so the stale instance writes different content for u1; the mutant now wipes u1 and fails.
3. `floors = min(FLOORS, population)` was an equivalent mutant (a role smaller than its floor never enters the shrink loop). Removed the redundant `min` from the code.
4. `units.jsonl` rewritten when nothing is new: bytes were identical, so undetected. The rerun assertion in the smoke test now also compares `st_mtime_ns` of both files.
5. Timing arithmetic in `measure` (adding a constant per response instead of the elapsed time): see Concerns (accepted survivor).

Final round: 35 killed, 1 survivor (item 5). Log: `.local/t22-mut3.log`.

## Full suite and lint

- Full suite (`pytest -q`, inner timeout 900): `1549 passed, 3 deselected in 234.79s` (previous baseline 1507; +42).
- `ruff check .`: `All checks passed!`. `ruff format` applied to the three new files only.
- `git ls-files --eol`: `i/lf w/lf` for all three files. No em dashes in them.

## Real-model verification (not part of the tests)

All runs in WSL with real models (`HF_HUB_OFFLINE=1`, checkpoints already in the HF cache), database `evidencegraph_bench`:

1. `measure --units 10` (probe, output deleted): 0.884 units/s, 2.88 pairs/s.
2. `measure --units 200` -> `evaluation/results/throughput.json` (gitignored), premise mode `passage`, depth 8, candidates 20:

```
units_per_s       0.5887      (200 units in 339.7 s)
nli_pairs_per_s   2.2359      (713 pairs in 318.9 s of NLI time)
mean_units_per_source  fit 44.779, select 44.125, risk 44.100, test 42.078   (full data)
population        fit 1724, select 400, risk 391, test 450
seconds_per_source  fit 76.06, select 74.95, risk 74.91, test 71.47
env: AMD Ryzen 9 7900X (WSL, 24 logical CPUs), 15.2 GB, torch 2.14.1+cpu, Postgres 17.11, pgvector 0.8.7
```

   git_commit/git_dirty are null (WSL git cannot read the Windows worktree, R-B2).
3. `plan_sizes(throughput["seconds_per_source"], throughput["population"])` (the file feeds it directly): default 4 h budget -> `{'fit': 200, 'select': 150, 'risk': 150, 'test': 150, 'over_budget': True}`. The floors alone cost 48,412 s = 13.45 h. 13.5 h -> floors without the flag; 14 h -> test 170.
4. End-to-end real extraction on a 4-source scratch manifest (`ragtruth manifest --sizes fit=2,select=1,risk=1,test=1`, kept in `.local`, deleted): `--limit-units 8` extracted 8; `--limit-units 12` extracted 4 and skipped 8; the same command again extracted 0 and skipped 12. A cached record had real values (entail 0.989, rerank 7.28, dense_cos 0.887, passage text 1,196 characters). The scratch cache folder was deleted.

Side effect to know: the `ragtruth-bench` tenant now exists in `evidencegraph_bench` with about 31 ingested `rt-<source_id>` collections (30 sampled by `measure`, one more from the scratch run). Ingestion is idempotent, so Task 27 reuses them.

## Decisions (ambiguities resolved)

- `--embedder fake` = `HashEmbedder` + `KeywordReranker`, `--nli fake` = `RuleNLIScorer`; real uses `load_embedder` / `load_reranker` / `load_nli` with `Settings()`. The `models` mapping of the config key holds the actual `model_id` values, so fake runs can never share a cache with real runs.
- Tenant `ragtruth-bench`. `ensure_bench_scope` also creates an empty benchmark collection `ragtruth`; only the tenant id is used. `beir.py` is not modified.
- `SignalCache` signature gained an optional keyword `config` (written to `config.json` when absent) because the class cannot derive the config from the key. Call shape `SignalCache(root, key)` from the brief still works.
- `--limit-units N` means the first N units of the run order (a deterministic prefix, like `--limit-queries`), so a limited run followed by a full run resumes.
- `--premise-mode` is required for extraction (the brief lists it unbracketed). `measure` also takes `--premise-mode` (default `passage`) and records it, because window mode costs far more NLI pairs and one number would be wrong for it. This is a small addition to the brief's `measure --units 200`.
- `measure` always uses real models from the command line. `measure()` takes an optional `models` argument only so the test can inject fakes; nothing writes fake numbers to the default path.
- Sampling for `measure`: a seeded shuffle (`SEED = 20261026`) of the fit sources, taking every source's first response before any second response, so the 200 units come from many sources and task types (about 30 sources here).
- `nli_pairs_per_s` = pairs / time spent inside `nli.score`; `units_per_s` = units / wall time of `collect_signals` (retrieval, rerank, embedding, NLI); `seconds_per_source[role] = mean_units_per_source[role] / units_per_s`. `population` (sources per role) is also in the file so `plan_sizes` can be called with the file alone.
- `plan_sizes` adds the `over_budget` key only when true, so a within-budget result can go straight to `build_manifest` as `sizes` (the caller pops the flag otherwise).
- Leakage guard beyond the brief: `plan_units` checks that every manifest source sits in the role its own split gives it, so a manifest cannot smuggle test sources into `fit` and bypass the `--final` rule (test `test_plan_units_refuses_a_test_source_listed_under_another_role`).
- Ingestion is skipped for sources with no uncached unit, and the database is not opened at all when everything is cached.

## Concerns

1. Budget (blocking for Task 27 planning): measured 0.589 units/s (about 75 s per source). The 4 h budget buys roughly 190 sources in total; the plan_sizes floors (650 sources) need 13.45 h, so `plan_sizes` reports `over_budget`. Options for the controller: raise the budget to about 14 h, lower the floors, or cut cost. NLI is 94 % of the time (2.24 pairs/s for deberta-v3-large on CPU, batch 16, up to 512 tokens). The model, `evidence_k`, depth, and candidates are fixed by the spec, so I did not touch them. Also note the 200-unit sample had 3.6 pairs per unit, so `window` premise mode would multiply this several times; run `measure --premise-mode window --out <other path>` before sizing a window run.
2. The estimate counts `collect_signals` only; ingestion (about 0.1 s per source here) is excluded, as the brief says. It is printed to stderr by `measure`, not recorded.
3. One mutant survives: the arithmetic of `collect_seconds` in `measure` is checked for consistency (`units_per_s = n_units / collect_seconds`, `collect_seconds >= nli_seconds >= sleep floor of the spy`) but not against an independent clock; an upper-bound assertion would be flaky on a loaded WSL host.
4. `.local/run.py` crashes printing a log that contains the non-cp1252 characters of the Hugging Face progress bars (the command itself completes; I read the logs directly). Not fixed (outside this task).
5. `evaluation/manifests/ragtruth_splits.json` still does not exist (Task 27 creates it after sizing); `--manifest` defaults to that path.
6. The mixed-writer case is unsupported by design: one process writes a cache at a time (documented in the module docstring).

Logs: `.local/t22-red1.log`, `t22-red2.log`, `t22-green1.log`, `t22-green2.log`, `t22-mut3.log`, `t22-full.log`, `t22-ruff.log`, `t22-measure.log`, `t22-real1..3.log` in the feature worktree.
