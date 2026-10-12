# Evaluation instructions

Applies to `evaluation/` and descendants. Follow the root instructions too.

## Data and methods
- Labeled datasets live under `evaluation/datasets/` as JSONL. Each line has `document`, `question`, and a `claims` array of objects with `text` and boolean `hallucinated` labels; `demo.jsonl` shows the format.
- `benchmarks.py` computes precision, recall, F1, and ROC AUC from predicted hallucination scores. Its current loop associates generated claims with reference labels by index. If changing claim generation or alignment, address that limitation explicitly instead of assuming labels still correspond.
- `ablation.py` evaluates several verifier-weight combinations. `calibration.py` reads benchmark output and plots score calibration.
- Keep labels, dataset splits, thresholds, and experimental configurations traceable and unchanged unless the task specifically calls for a change. Do not report placeholder or synthetic metrics as measured results.
- Benchmark and ablation runs use the real pipeline and can require Ollama, local model downloads, and substantial computation; state these prerequisites before treating a run as an ordinary check.
- Generated artifacts belong in `evaluation/results/`, which is gitignored.

## Commands
From the repository root, with the evaluation dependencies and local models available:
```bash
halluciscope evaluate --dataset demo --output evaluation/results/
halluciscope evaluate --ablation --dataset demo --output evaluation/results/
```
For changes limited to metric calculations, prefer small deterministic tests over running the full model pipeline.
