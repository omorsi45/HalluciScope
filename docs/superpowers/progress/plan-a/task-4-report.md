# Task 4 report: pin ruff rules, lint to zero

## Built
- pyproject.toml: `extend-exclude = [".worktrees", ".superpowers", ".local", "docs/superpowers"]`, `[tool.ruff.lint] select = ["E","F","I","B","UP","SIM","ASYNC","RUF"]`, bugbear `extend-immutable-calls` (typer/fastapi list verbatim from brief).
- Lint-only change; no test assertion changed.

## Evidence (lint replaces RED/GREEN)
- Before (new rule set): `ruff check .` = 94 errors (E501 36, I001 25, F401 8, B905 5, ASYNC230 4, ASYNC240 4, SIM117 4, B904 2, RUF002 1, B011 1, UP042 1, UP012 1, RUF059 1, F841 1). Suite: 53 passed.
- After: `All checks passed!`. Suite: 53 passed, 2 warnings.

## Files changed
30 files: pyproject.toml; backend/{api/routes/analysis,api/startup,cli/main,core/chunker,core/decomposer,core/document,core/ensemble,core/generator,core/pipeline,core/verifiers/{base,consistency,nli},db/repository}.py; evaluation/{ablation,benchmarks,calibration}.py; tests/{conftest,test_api,test_chunker,test_cli,test_consistency,test_db,test_decomposer,test_document,test_ensemble,test_generator,test_nli,test_pipeline,test_similarity}.py.

## Decisions
- E501 wrapped, no noqa anywhere. Prompt strings (RAG_SYSTEM_PROMPT, DECOMPOSE_*) wrapped with backslash line continuation inside triple quotes; verified resulting strings identical to HEAD.
- ASYNC230/240: added `read_dataset` and `write_json` helpers in evaluation/benchmarks.py (reused by ablation.py), called via `asyncio.to_thread`; mkdir also via to_thread. Dataset read once up front instead of lazily per config/line (same output, same FileNotFoundError message).
- B904: `raise ... from exc` on ConnectError/TimeoutException (messages unchanged).
- OllamaError timeout message em dash kept byte-for-byte (ledger ruling); ruff does not flag it, so no noqa was needed. It is the only em dash left in .py files.
- Removed em dash in chunker docstring; RUF002 `×` in nli docstring -> "x".
- UP042: ConfidenceTier now `StrEnum` (py311 target OK). No str()/f-string use of the enum found; only `.value` and equality are used.
- B905: strict=True in evaluation/benchmarks.py (equal lengths by construction); strict=False in EnsembleScorer.score to preserve current behavior.
- F841 in test_document: removed unused `page = writer.pages[0]`; B011: `assert False` -> `raise AssertionError(...)` (inside try/except ValueError, same flow).
- SIM117: combined with parenthesized context managers in test_consistency.
- RUF059: `fig` -> `_fig` in calibration.py.

## Concerns
- EnsembleScorer.score silently truncates on mismatched verifier result lengths (strict=False). strict=True would be safer but changes behavior; left for a later task.
