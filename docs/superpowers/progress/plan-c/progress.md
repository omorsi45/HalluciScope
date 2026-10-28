# SDD ledger: plan docs/superpowers/plans/2026-10-26-evidencegraph-c-verify-calibrate.md

Spec: docs/superpowers/specs/2026-10-26-evidencegraph-design.md (reachable; R1-R10 override earlier sections).
Branch: feature/evidencegraph. Start commit: f68c2ac (Plan B complete).
Process (Omar, 2026-10-27): controller plans and verifies; one claude-sonnet-5.5 implementer per task, sequential; no per-task reviewer, no Opus. Controller acceptance per task: RED then GREEN in the report, full suite and ruff pass (controller re-runs), commit with the plan's message exists, diff stat touches only the task's files.
Environment: as in the Plan B ledger (Python in WSL via .local/wsl.sh, Postgres in Docker eg-pg, Windows git). Briefs: feature worktree .superpowers/sdd/plan-c/task-N-brief.md. Reports: this folder.

## Pre-flight scan

| Rows | Produces / consumes | Finding |
|---|---|---|
| T1 -> T2, T3, T13, T18, T19 | LLMClient, LLMFailure codes, ScriptedLLM | Consistent. |
| T2 -> T3, T13 | NO_INFO_ANSWER, CitationRef, parse_citations, generate_answer | Consistent. |
| T3 -> T4, T8 | STOPWORDS (backend.verify.claims) used by RuleNLIScorer and features | Consistent order. |
| T4 self + T11, T13, T14 fixtures | RuleNLIScorer contradiction if "at least half" of hypothesis content tokens occur in the premise | Defect: with the plan's own fixtures ("Refunds take 30 days." vs "Exchanges take 14 days.", shared "take", "days") every claim gets entail 0.9 from its own passage and contra 0.9 from the other, so make_test_calibrator gives p_s about 0.49 < tau_s 0.5 and test_three_verdicts / test_ask_end_to_end would expect supported but get insufficient. Ruling R-C2. |
| T5, T6, T7 literals | ECE 0.4, Brier 0.14, NLL 0.35667, AURC 0.270833; binom_cdf, CP upper, HB p-values | Controller recomputed every literal by hand: all consistent with the stated formulas. |
| T7 -> T23 | select_thresholds, GRID | Consistent. |
| T8 literal | feature vector for s1/s2 | Recomputed: consistent. |
| T9 -> T11, T13 | Calibrator, make_test_calibrator (model ids match fakes: fake-rule-nli, fake-hash-384, fake-keyword) | Consistent. |
| T10 -> T11 | gather_evidence, window_spans, relation_for | Consistent. |
| T11 -> T13, T17, T22 | VerificationEngine, collect_signals | Consistent. |
| T12 FKs vs Global Constraints | evidence_links/citations reference passages(id, version_id); claims etc. single-column | Must include tenant_id. Ruling R-C1. passages already has unique (tenant_id, id, version_id) from 0003. |
| T12 -> T13 | save_results takes CitationRecord, defined in backend/analysis/service.py which T13 creates | T12's roundtrip test needs the type first. Ruling R-C3. |
| T13 self | worker CLI loads OllamaClient, NLI, calibrator; Worker already supports timeouts_s | Existing worker CLI tests must patch the loaders (dispatch note). Failure hook run_analysis required (Global Constraints). |
| T14 -> T15, T16, T17 | readable_subset, require_analysis, schemas | Consistent. |
| T17 self | Principal.auth gains "system" here; Global Constraints say T18 uses it | Consistent (T17 adds, T18 uses). Failure hooks reverify_document and reverify_analysis (log only) required. |
| T18 -> T19 | T18 deletes legacy routes and test_api.py; T19 deletes core and legacy tests | Consistent. build_services moves to backend/services.py in T18. |
| T19 self | removes embedding_model_name and legacy loaders from Plan B's loader | Consistent; Plan B's adapters stay. |
| T21 <- Plan B T19 | load_ragtruth(root) reads response.jsonl and source_info.jsonl from one root | Plan B stored them side by side in evaluation/data/. Consistent. |
| T22 self | plan_sizes budget 14,400 s; throughput measured on this machine | Real NLI runs on CPU here (WSL torch CPU build); sizes follow the measurement and are reported. |
| T25, T26 <- T18 | process_upload, create_analysis_request(enqueue=False) | Consistent. |
| T27 self | dev-postgres.sh, ollama serve, M1 wording | Docker eg-pg and Windows Ollama on this machine; reachability from WSL checked at T27. |
| Review Focus numbering | "Tests in Tasks 2 and 12", "13 and 14", "10 and 12" | Off by one after Task 11 (run_analysis is T13, routes T14, provenance T15). Implementers follow each task's own test list. |
| Plan-mandated defects | none other than R-C2 | Clean. |

Rulings:
- Ruling R-C1: migration 0004 writes every tenant-to-tenant FK with tenant_id: claims and citations -> analyses (tenant_id, analysis_id); evidence_links, claim_checks, feedback -> claims (tenant_id, claim_id); evidence_links and citations -> passages (tenant_id, passage_id, version_id) references passages (tenant_id, id, version_id); referenced tables declare unique (tenant_id, id); users stays single-column. Plus a cross-tenant ForeignKeyViolation test. Why: Global Constraints (Plan A Task 7 ruling). Cost if wrong: wider indexes only.
- Ruling R-C2: tests/fakes.py RuleNLIScorer predicts contradiction only when every non-number content token of the hypothesis occurs in the premise (not "at least half"). Why: the plan's own fixtures share "take" and "days" across documents, so the half rule makes every claim both entailed and contradicted and the planned verdict tests unsatisfiable; the stricter rule keeps the T4 literal cases and the designed conflict in T14. Cost if wrong: only the fake changes; production NLI is unaffected.
- Ruling R-C3: Task 12 creates backend/analysis/__init__.py and backend/analysis/service.py holding only the frozen CitationRecord (CitationRef fields plus passage_id, version_id); Task 13 extends service.py. Why: Task 12's save_results roundtrip test needs the type before Task 13 exists. Cost if wrong: none.

## Progress

- Task 1: complete (commits f68c2ac..11d0db5; acceptance: 1035 passed, ruff clean, 5 files in scope)
- Task 1: minor (deferred): httpx timeout_s applies per phase, not as one total deadline (run_analysis has its own overall timeout); a trailing slash in ollama base_url gives //api/chat.

- Task 2: complete (commits 11d0db5..69e0000; acceptance: 1056 passed, ruff clean, 4 files in scope). parse_citations blanks markers before segmenting so "sentence. [P1]" attaches to the preceding sentence.
- Task 2: minor (deferred): document_title unsanitized in the context header (could carry a fake [P#] or tag); only lowercase </passage> neutralized; generated reply not stripped before comparing with NO_INFO_ANSWER.

- Task 3: complete (commits 69e0000..d028bae; acceptance: 1097 passed, ruff clean, 5 files in scope; controller checked STOPWORDS keeps fixture content words out). claims.py reuses generate/answer.py's private marker regex and sentence spans so claims and citations agree.
- Task 3: minor (deferred): claims.py imports private names (_MARKER, _sentence_spans) from generate/answer.py (make them public in the final fix wave if touched).

- Task 4: complete (commits d028bae..ea282c9; acceptance: 1123 passed, ruff clean, 6 files in scope; real DeBERTa-v3-large models test passed on CPU). Real checkpoint id2label {0: contradiction, 1: entailment, 2: neutral} read through label_indices. RuleNLIScorer follows R-C2.
- Task 4: minor (deferred): real DeBERTa returned contradiction 0.999 for an unrelated pair (expected model behavior; features and calibration handle it, worth a note in docs/evaluation.md); legacy nli_model_name setting remains until Task 19.

- Task 5: complete (commits ea282c9..8beabf4; acceptance: 1137 passed, ruff clean, 3 files in scope)
- Task 5: minor (deferred): empty inputs give NaN/ZeroDivision in ECE and AURC (callers never pass empty splits; guard if touched).

- Task 6: complete (commits 8beabf4..de77137; acceptance: 1145 passed, ruff clean, 3 files in scope). Fit converges in 8 iterations on the recovery test; test chooses balanced W_true/b_true (the plan fixes no values).
- Task 6: minor (deferred): fit_temperature on perfectly separable large-margin logits has a flat NLL so the returned T is arbitrary within bounds (real logits come from an l2 = 1.0 fit).

- Task 7: complete (commits de77137..6eb93a3; acceptance: 1166 passed, ruff clean, 2 files in scope). LTT simulation 0/200 violations (limit 0.08) with a non-vacuous scenario; plug-in mutant gave 48/200. accept_mask restricts only the numerator; denominator = all bad rows of the source (matches spec R1's L_i for CONTRADICTED).

- Task 8: complete (commits 6eb93a3..7306d32; acceptance: 1175 passed, ruff clean, 3 files in scope)

- Task 9: complete (commits 7306d32..cb56d07; acceptance: 1247 passed, ruff clean, 4 files in scope)
- Task 9: carry-forward to Task 13 (worker load): Calibrator.load lets FileNotFoundError and json.JSONDecodeError propagate; the worker must treat them like CalibratorError. predict_proba needs an (n, 12) matrix. Compare calibrators via to_dict().

- Task 10: complete (commits cb56d07..b4a4b58; acceptance: 1266 passed, ruff clean, 2 files in scope). relation_for ties resolve to contradicts (spec: supports needs entail above contra).
- Task 10: carry-forward to Tasks 11 and 13: gather_evidence uses asyncio.gather without cancelling siblings; a retrieval exception propagates raw (run_analysis maps it).
