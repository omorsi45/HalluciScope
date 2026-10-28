# Plan B Task 19 report: pinned dataset downloads and BEIR loading and ingestion

Status: DONE
Commit: `0aea26b feat: add pinned dataset downloads and BEIR ingestion` (on `feature/evidencegraph`, not pushed)

## What was built

- `evaluation/data.py`: frozen `DatasetFile`, `DATA_DIR`, `REGISTRY` (4 entries, all pinned), `DatasetIntegrityError`, `fetch`, `remote_sha256`, and the CLI (`python -m evaluation.data fetch <name>...` and `hash <name>`).
  - Download streams (1 MiB chunks, `follow_redirects=True`) into `tempfile.mkstemp` in `data_dir`, hashing as it goes. Mismatch deletes the temp file and raises `DatasetIntegrityError`. Success renames atomically with `Path.replace`. Any failure (HTTP error, mismatch) removes the temp file.
  - An existing file whose sha256 equals the pin is not downloaded again. An existing file with the wrong hash is re-downloaded and replaced only after the new download verifies.
  - Zips: every member name is checked before anything is extracted (absolute path, leading backslash, Windows drive letter, any `..` component, backslash separators normalised). Extraction goes to a scratch directory in `data_dir` and is renamed to `data_dir/<name>/`; on any failure the scratch directory is removed, so a rejected archive leaves no extracted files.
  - `hash <name>` downloads without verification, prints only the hex digest on stdout and keeps nothing.
- `evaluation/beir.py`: `BeirData`, `IngestSummary`, `load_beir`, `ensure_bench_scope`, `ingest_beir`, constant `BATCH_SIZE = 64`.
- `tests/test_beir.py`: 25 tests (14 on data.py, 11 on beir.py, of which 6 are `pg`).
- `.gitignore`: `evaluation/data/` added (verified with `git check-ignore`; the downloads do not show in `git status`).

## Pinned hashes and sizes

Obtained with `python -m evaluation.data hash <name>` through the WSL wrapper (`timeout 900`), all four exit 0, no 404. Afterwards `python -m evaluation.data fetch` of all four succeeded against the pins, and an independent `sha256sum` on the files gave the same four digests.

| name | sha256 | bytes |
|---|---|---|
| scifact (zip) | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` | 2,816,079 |
| nfcorpus (zip) | `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b` | 2,448,432 |
| ragtruth_response (`response.jsonl`) | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` | 21,458,735 |
| ragtruth_source (`source_info.jsonl`) | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` | 15,117,971 |

The four digests are in the commit message body. Downloads are under `evaluation/data/` (ignored, not committed).

Real-data load check (`load_beir(fetch(name), name)`, not a committed test):
- scifact: 5183 docs, 300 test queries, 339 qrel rows, grades {1}, 0 empty documents.
- nfcorpus: 3633 docs, 323 test queries, 12334 qrel rows, grades {1, 2}, 0 empty documents.

## RED evidence

Stage 1 (download tests written first, no `evaluation/data.py`):

```
tests/test_beir.py:12: in <module>
    from evaluation import data
E   ImportError: cannot import name 'data' from 'evaluation'
1 error in 0.75s   (exit=2)
```

After `data.py` existed with the hashes still blank, `test_registry_hashes_pinned` was the one failure (it went green only once the real digests were pinned):

```
F.............
>           assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256), entry.name
E           AssertionError: scifact   (sha256='')
1 failed, 13 passed in 0.84s
```

Stage 2 (BEIR tests appended, no `evaluation/beir.py`):

```
tests/test_beir.py:13: in <module>
    from evaluation import beir, data
E   ImportError: cannot import name 'beir' from 'evaluation'
1 error in 0.80s   (exit=2)
```

No RED run touched the network (all download tests use `httpx.MockTransport`). Both RED runs are collection errors because the module did not exist; they fail for the right reason (feature missing).

Mutation checks (temporary edits, then restored and re-verified by grep):
- removed the `..` component check: 3 zip-slip cases fail (`../evil.txt`, `tiny/../../evil.txt`, `..\evil.txt`). Python's own `extractall` silently flattens `..`, so only the explicit rejection makes these pass.
- `BATCH_SIZE = 65`: `test_ingest_commits_in_batches_of_64_documents` fails (`2 == 3`).
- hash check disabled: `test_fetch_verifies_hash` fails (DID NOT RAISE).
- existing-document lookup disabled: `test_ingest_idempotent`, `test_ingest_doc_ids_restricts_and_later_runs_add_the_rest` and `test_empty_doc_skipped` fail.

## GREEN evidence

```
tests/test_beir.py -q  ->  25 passed in 1.25s
full suite (timeout 900): 997 passed, 2 deselected, 2 warnings in 136.41s (exit 0)
ruff check .                                   -> All checks passed!
ruff format --check (the 3 new files)          -> 3 files already formatted
```

The 2 warnings are the existing `tests/test_generator.py` `AsyncMock` "never awaited" RuntimeWarnings from legacy code (`backend/core/generator.py:50`), unrelated to this task. The 2 deselected tests are the `models` marker.

Commit hook ran gitleaks: no leaks. Staged by name: `.gitignore evaluation/data.py evaluation/beir.py tests/test_beir.py`. `git ls-files --eol` shows `i/lf w/lf` for the three new files. No em dashes in any new file or in the commit message.

## Tests added (beyond the seven the brief names)

Each names a break it catches:
- `test_fetch_downloads_verifies_and_extracts_zip`, `test_fetch_file_kind_returns_the_file`: happy path and no stray temp files.
- `test_fetch_http_error_leaves_no_file`: temp file removed on a 404.
- `test_fetch_rejects_zip_slip` is parametrized over 5 member names (`../`, nested `..`, absolute, backslash `..`, drive letter), and also asserts nothing was extracted, including the harmless member listed before the bad one.
- `test_fetch_extracts_when_archive_exists_without_directory`, `test_fetch_replaces_existing_file_with_wrong_hash`: the skip rule is "exists with the right hash", not "exists".
- `test_hash_command_prints_digest_and_keeps_no_file`: the digest that gets pinned is checked against an independent `hashlib` value.
- `test_load_beir_tiny` (nested and flat layout), `test_load_beir_other_split`, `test_load_beir_rejects_qrels_without_header`. The qrels fixtures use CRLF because the published SciFact `test.tsv` has CRLF line endings (seen with `cat -A`).
- `test_ensure_bench_scope_*` (2 tests), `test_ingest_doc_ids_restricts_and_later_runs_add_the_rest`, `test_ingest_text_title_and_filename`, `test_ingest_commits_in_batches_of_64_documents` (130 documents, counts `tenant_tx` calls, expects 3).

## Decisions (brief ambiguous or silent)

1. Local file layout. Zip: archive at `data_dir/<URL basename>` (`scifact.zip`), extracted into `data_dir/<name>/`. Plain file: `data_dir/<URL basename>`, so `ragtruth_response` and `ragtruth_source` land side by side as `evaluation/data/response.jsonl` and `evaluation/data/source_info.jsonl`. Plan C Task describes `load_ragtruth(root)` reading `response.jsonl` and `source_info.jsonl` "as fetched by evaluation.data", which needs both in one directory. `fetch` returns the file path for these.
2. BEIR zips contain one top-level folder named after the dataset, so the real corpus is at `evaluation/data/scifact/scifact/corpus.jsonl`. `load_beir(root, name)` therefore uses `root` if it holds `corpus.jsonl`, else `root / name`. `load_beir(fetch("scifact"), "scifact")` works and so does passing the inner folder. Both layouts are tested.
3. `ensure_bench_scope` raises `ValueError` when a collection with that name exists but is not `kind = 'benchmark'` (silently benchmarking into a normal collection would be wrong). Tenants have no unique name, so reuse is the oldest tenant with that name (`order by created_at, id`). No advisory lock: benchmark scope creation is a serial CLI step.
4. Documents with a title but no text are ingested (the text is the title); only documents with no text after `parse_text` normalization are skipped, as the brief says. The empty-title omission is invisible after normalization (the normalizer drops blank pieces), so no test distinguishes it; it only affects the blob hash.
5. Document `title` is the BEIR title, or the BEIR id when the title is empty (`documents.title` is `not null`). `effective_from` is one `datetime.now(UTC)` per `ingest_beir` call. `created_by` is `None`.
6. No `blobs` row is written for benchmark documents (there is no uploaded original); `blob_sha256` is the sha256 of the ingested bytes. `seed_docs` in `tests/pg_fixtures.py` does the same. `document_versions.blob_sha256` has no foreign key to `blobs`.
7. `qrels` keep grade 0 rows (BEIR uses them). A qrels file whose first line is not the `query-id` header raises `ValueError` instead of silently dropping a row.
8. The CLI uses `argparse` (no new dependency). Unknown dataset names are rejected by argparse `choices`. `fetch` prints `<name>: <path>`; an integrity error prints to stderr and exits 1.
9. `ingest_beir` lets `IngestError` from `create_version` (for example `too_many_passages`) propagate and roll back the current batch of 64: a benchmark should fail loudly rather than quietly drop documents.
10. The tool I used to append to the test file wrote CRLF; I converted `tests/test_beir.py` to LF before committing (new file, rules say LF).

## Concerns

- None blocking. Observation for later tasks: `ingest_beir` embeds one document at a time through `create_version`, as the brief specifies. With the real BGE model on CPU, a full nfcorpus ingest will be slow; batching embeds across documents would be a separate optimisation.
- `data_dir`/`DATA_DIR` is relative to the working directory (`evaluation/data`), as the brief specifies, so the CLI must be run from the repository root.
- The sha256 values are those of the files served on 2026-10-27 from the UKP host and the pinned RAGTruth commit. The UKP URL is not commit-pinned, so a future change on that server would surface as a `DatasetIntegrityError`, which is the intended behaviour.
