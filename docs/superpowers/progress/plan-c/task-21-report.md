# Plan C Task 21: RAGTruth harness (report)

Status: DONE
Commit: `14fd1d6 feat: add RAGTruth loading, span-aligned units, and source-level splits` (branch `feature/evidencegraph`, not pushed)
Files (committed): `evaluation/ragtruth.py` (new), `tests/test_ragtruth.py` (new). Nothing else touched.

## What was built (`evaluation/ragtruth.py`)

- Frozen dataclasses `RTSpan`, `RTSource`, `RTResponse`, `RTUnit`; `CONFLICT` / `BASELESS` (frozensets with the brief's values); `COMMIT`, `SEED = 20261026`, `ROLES = ("fit", "select", "risk", "test")`.
- `load_ragtruth(root)`: reads `source_info.jsonl` (renders each source through `render_source`) and `response.jsonl`; keeps every response and every label (no quality or `implicit_true` / `due_to_null` filtering). `response_id` from `id`, `source_name` from `source`.
- `render_source(task_type, source_info)`: Summary, QA (passages + question), Data2txt (`key: value` per scalar, insertion order, nested keys joined with `.`, list items indexed, None skipped, bools `true`/`false`). Unknown task type raises `ValueError`.
- `split_role(source_id, split)`: `test` for the test split, else `int(sha256(source_id.encode()).hexdigest(), 16) % 100` with `< 70` fit, `70..84` select, `>= 85` risk.
- `unitize(response, source, role)`: units are `segment_sentences(response.response)`; a span labels a unit when the overlap holds a non-whitespace character; conflict beats baseless beats supported; stats `n_spans`, `n_span_text_mismatch`, `n_spans_without_unit`. Unit id `f"{response_id}:{i}"`.
- `allocate(counts, size)`: largest-remainder allocation in integer arithmetic (ties by task name), always sums to `size`.
- `build_manifest(...)` and `manifest_hash(manifest)`: per (role, task type) sampling without replacement with `random.Random(f"{seed}:{role}:{task}")` over sorted ids, so a role's sample does not depend on the other roles' sizes or on input order. Output keys exactly as in the brief. Hash is sha256 of `json.dumps(roles, sort_keys=True, separators=(",", ":"))`.
- `ingest_ragtruth(services, sources, source_ids, *, tenant_id)`: one `benchmark` collection `rt-<source_id>` and one document `rt-<source_id>.txt` per source through `parse_text` and `create_version`, one transaction per source; existing collection and document reused; returns source id to collection id.
- CLI: `python -m evaluation.ragtruth manifest --sizes fit=N,select=N,risk=N,test=N --out PATH [--data-root DIR]` (`all` allowed). Output is `json.dumps(manifest, sort_keys=True, indent=2) + "\n"`; prints the path and `manifest_hash`.

## RED

Tests written first (`tests/test_ragtruth.py`), run before `evaluation/ragtruth.py` existed:

```
$ python -m pytest tests/test_ragtruth.py -q      (WSL wrapper, timeout 280)
ImportError while importing test module '.../tests/test_ragtruth.py'.
tests/test_ragtruth.py:11: in <module>
    from evaluation import data, ragtruth
E   ImportError: cannot import name 'ragtruth' from 'evaluation'
1 error in 1.66s
exit=2
```

## GREEN

```
$ python -m pytest tests/test_ragtruth.py -q
44 passed in 1.64s          (first green run)
```

The pg tests really ran (Postgres container up): `test_ingest_ragtruth_idempotent`, `..._stores_the_rendered_source_text`, `..._refuses_a_collection_that_is_not_a_benchmark`, `..._rejects_a_source_without_text_and_creates_nothing`, plus `test_load_ragtruth_...` and `test_manifest_command_writes_canonical_json` (6 passed in that filtered run).

### Mutation sanity check (production file edited, tests rerun, then reverted)

First mutation set, 5 failures as intended: gold order swapped (both `test_conflict_beats_baseless` cases), bucket `<= 70` (`test_split_roles_deterministic[18-train-select]`, `test_split_role_buckets_follow_the_sha256_rule`, and the CLI test).

The whitespace-filter mutation (drop `not text[low:high].isspace()`) did NOT fail the brief's literal test: sentences exclude surrounding whitespace, so a span on the space between sentences overlaps no unit even without the filter. I parametrized `test_whitespace_only_overlap_ignored` to also cover a space inside a sentence (`(1, 2)`), which does fail under that mutation.

Second mutation set, 9 failures as intended: whitespace filter removed (`[1-2]` case), leftover seats dropped in `allocate` (three allocate tests, stratified manifest, same-seed, input-order, CLI), existence check removed in `ingest_ragtruth` (`test_ingest_ragtruth_idempotent`). All mutations reverted; final full suite below ran on the reverted file.

### After the final edit

- `ruff format --check evaluation/ragtruth.py tests/test_ragtruth.py`: 2 files already formatted (the repo is format-clean, so I ran `ruff format` on my two files).
- `ruff check .`: All checks passed!
- Full suite: `python -m pytest -q`, `1507 passed, 3 deselected in 400.67s (0:06:40)`, exit 0 (the 3 deselected are `models`-marked, deselected by the repo's `addopts`).
- Test file: 45 tests in `tests/test_ragtruth.py` (including parametrized cases). No em dash in either file.
- Pre-commit gitleaks hook: no leaks found. Both files are `i/lf w/lf`.

## Real-data sanity check (one-off script, counts only, not committed, deleted)

Run on `evaluation/data/response.jsonl` and `source_info.jsonl`.

- 2,965 sources (Summary 943, QA 989, Data2txt 1,033), 17,790 responses (train 15,090, test 2,700). Ids load as `str`.
- No source has responses in both splits. No source without responses. No response with an unknown source.
- Empty rendered source texts after `parse_text`: 0. QA sources without a question: 0.
- Label types: Evident Conflict 5,324, Evident Baseless Info 6,237, Subtle Baseless Info 2,527, Subtle Conflict 201 (matches the controller's counts; no other label type exists).
- `unitize` over all responses: `n_spans` 14,289, `n_span_text_mismatch` 0, `n_spans_without_unit` 0. Unit gold counts: supported 117,078, contradicted 5,133, insufficient 8,816 (131,027 sentence units).

### Role populations (sources per role and task type, all sources, no subsampling)

| Role | Sources | Data2txt | QA | Summary | Responses |
|---|---|---|---|---|---|
| fit | 1,724 | 597 | 599 | 528 | 10,344 |
| select | 400 | 141 | 130 | 129 | 2,400 |
| risk | 391 | 145 | 110 | 136 | 2,346 |
| test | 450 | 150 | 150 | 150 | 2,700 |
| total | 2,965 | 1,033 | 989 | 943 | 17,790 |

`manifest_hash` of the full-population manifest (all sizes `all`): `a6558420e73f613d5f1d430324db504926e4e0e654817d1abddbf60cbbb2c8a9`.

CLI smoke on the real files with `--sizes fit=300,select=100,risk=100,test=all` (output written under `.local/`, then deleted, not committed): exit 0, role sizes 300 / 100 / 100 / 450 exactly; per-task counts fit 104 Data2txt / 104 QA / 92 Summary, select 35 / 33 / 32, risk 37 / 28 / 35, test 150 / 150 / 150; `data_sha256` equals the two pinned registry hashes; `manifest_hash=5463ff38b02a6dcaeff2aeabea7c71ab8978a71ecd05d1e339cb2edf7b8caf7f`. No manifest was committed (the brief creates only the module and tests; the real sizes are a later decision).

## Decisions

1. Cross-split guard: `build_manifest` raises `ValueError` if one source's responses map to more than one role (train and test mixed). Leakage control is the point of R1, and a silent mix would put one source in two roles. Real data has none. Covered by a test.
2. `build_manifest` requires `sizes` to name exactly the four roles, and raises on a size above the role's population or below zero. `parse_sizes` rejects anything but `fit|select|risk|test = N or all`.
3. `ingest_ragtruth` raises `ValueError` for a source whose parsed text is empty (before any DB write), instead of silently skipping it like `ingest_beir` does: a benchmark source with no evidence would otherwise vanish from the returned map and skew metrics. Real data has none. Covered by a test.
4. Label types outside `CONFLICT | BASELESS` count as no hallucination (`supported`), exactly as the brief words the rule. The real data has no other type.
5. The CLI calls `data.fetch` for the two pinned files (verifies sha256; downloads only if no verified copy exists) and records the registry's pinned sha256 values keyed by file name (`response.jsonl`, `source_info.jsonl`) as `data_sha256`. `--out` is required (as in the brief's command shape); `--data-root` defaults to `evaluation/data`.
6. Manifest `counts[role].sources` lists every task type present in the role's population (zeros included); `responses` counts the responses of the chosen sources; `sizes` is stored as requested (`null` = all).
7. Ingestion stores only the rendered source text as the document (the QA question is the query, not evidence). Document title is `rt-<source_id>`. The stored text is `parse_text`'s canonical text. The collection lookup is also filtered by `tenant_id` (belt and braces on top of RLS) and refuses an existing non-benchmark collection of the same name, mirroring `ensure_bench_scope`.
8. `_read_jsonl` is a local 5-line copy instead of importing the private helper from `evaluation.beir`; `beir.py` is untouched.

## Concerns

- None blocking. `evaluation/manifests/ragtruth_splits.json` is not generated here; the actual sizes for `fit` / `select` / `risk` still need to be chosen by the controller, then the CLI run once and the result committed.
- One collection per source means up to 2,965 `rt-*` collections in the benchmark tenant if the whole dataset is ingested; ingestion is one transaction per source, so a large run is many small transactions (resumable, since a rerun skips what exists).
- Windows git warns "LF will be replaced by CRLF" on commit (expected with `core.autocrlf=true`); both files are `i/lf w/lf`.
