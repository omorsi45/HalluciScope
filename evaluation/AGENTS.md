# Evaluation instructions

Applies to `evaluation/` and descendants. Follow the root instructions too.

## Layout
- `run_*.py` modules are the benchmark runners; run them as modules from the repository root, for example `python -m evaluation.run_retrieval --dataset scifact`. They are not CLI commands.
- `data.py` downloads the pinned public datasets and verifies their sha256 before use. `beir.py` loads a BEIR dataset and ingests its corpus through the same code path as an upload.
- `metrics/` holds the original metric implementations (no metric libraries), tested with small deterministic cases.
- `report.py` renders a runner's raw JSON results as the Markdown report under `docs/results/`; every number in it is read from the results file.
- `manifests/` records the pinned source of each prepared dataset (URL, sha256, split, and sizes).

## Data and methods
- Benchmark data, caches, and raw results are not committed. Downloads go to `evaluation/data/` and runner output to `evaluation/results/`; both are gitignored.
- The only committed benchmark artifacts are `evaluation/manifests/*.json`, `docs/results/*.md`, and `backend/calibration/artifacts/default.json`.
- Every number in `docs/results/` and in prose comes from a committed script's output. Do not report placeholder or synthetic metrics as measured results, and do not edit a report by hand.
- Fix parameters, thresholds, and splits in the runner before evaluating; never tune on the evaluation labels. Evaluate the RAGTruth test split once, in the final factuality run (`--split test --final`), and never use it to fit or select anything.
- Keep labels, dataset splits, thresholds, and experimental configurations traceable and unchanged unless the task specifically calls for a change.
- Runners use the real models and a Postgres database, and some need Ollama, local model downloads, and substantial computation. State these prerequisites before treating a run as an ordinary check.

## Commands
From the repository root, with Postgres running (see the root instructions) and the evaluation dependencies installed:
```bash
python -m evaluation.data fetch scifact nfcorpus
python -m evaluation.run_retrieval --dataset scifact
python -m evaluation.report retrieval evaluation/results/retrieval_*.json --out docs/results/retrieval.md
```
For changes limited to metric calculations, prefer small deterministic tests over running the full benchmark.
