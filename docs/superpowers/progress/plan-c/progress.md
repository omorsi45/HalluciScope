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

- Task 11: complete (commits b4a4b58..0461788; acceptance: 1280 passed, ruff clean, 3 files in scope). collect makes one NLI call; verify adds one batched localization call.
- Task 11: minor (deferred): when the best-entail and best-contra item is the same passage, the contra window wins its span.

- Task 12: complete (commits 0461788..c4c02e8; acceptance: 1312 passed, ruff clean, 5 files in scope, 0004 is LF, migrate applied [4]). R-C1 composite FKs and R-C3 CitationRecord applied; cross-tenant FK test present.
- Task 12: carry-forward: get_citations returns citation columns only (Task 14 must join for collection_id and titles); set_status does not guard terminal states (failure hook must check); span-inside-passage is checked in code/tests, not the DB.

- Task 13: attempt 1 cancelled by the 3600 s idle deadline (no commit, no report). Left uncommitted RED-phase edits: tests/test_run_analysis.py (24 tests, new), tests/test_worker.py (harness fakes for load_nli, OllamaClient, timeouts_s), tests/pg_fixtures.py (services fixture with ScriptedLLM, RuleNLIScorer, make_test_calibrator). No production code, no stray process in WSL afterwards. Cause unknown (likely one unbounded command or a stalled turn). Re-dispatching fresh: keep the partial tests for review, every command bounded at the Windows level with .local/run.py, report file written early.

- Task 13: attempt 1 root cause (found by attempt 2): the test harness FakeWorker.run waited on stop.wait() forever, so in RED the CLI worker tests hung with no SIGINT. Fixed with a bounded wait (10 s).
- Task 13: attempt 2 implemented and reached GREEN on focused tests (31 + 16), then the session was interrupted before the full suite and commit. ENV: the eg-pg container exited 255 during the controller's first full run (Docker VM restart, not OOM); restarted, orphan eg_test db dropped. Controller acceptance: 1349 passed, ruff clean, LF, no em dashes; read the error mapping (timeout, LLMFailure, mismatch, retrieval_failed, internal_error) and confirmed the JSON formatter logs only exc_type and frames. Committed by the controller as 05433e5 (9 files in scope).
- Task 13: complete (commits c4c02e8..05433e5). Decisions in the report: known_at from the database clock; _GuardedRetriever maps any retrieval exception to retrieval_failed; engine built before any LLM call so calibrator_mismatch fails fast; worker validates the artifact before loading models and exits 1 naming the field.

- Task 14: complete (commits 05433e5..4e7f692; acceptance: 1365 passed, ruff clean, 5 files in scope). Redaction also nulls passage_is_current; collection_ids returned as stored (R8 redacts evidence, citations, passages only).
- Task 14: minor (deferred): three read queries live as private helpers in routes/analyses.py instead of the repo module (brief file list excluded the repo).

- Task 15: complete (commits 4e7f692..4ec6772; acceptance: 1380 passed, ruff clean, 7 files in scope). For a current-version passage, current_equivalent_passage_id is the passage itself.
- Task 15: minor (deferred): validate_analysis does no authorization (internal; any HTTP exposure must check analysis visibility first).

- Task 16: complete (commits 4ec6772..da39823; acceptance: 1394 passed, ruff clean, 4 files in scope). Sentence nodes only for sentences with a valid citation; feedback note rejects NUL (422).
- Task 16: minor (deferred): GET /claims/{id}/checks unpaginated (per brief).

- Ruling: test_unrelated_change_skips_claim, its positive control, and test_new_passage_triggers_recheck use make_test_calibrator(evidence_k=1). Why: with the default evidence_k = 4 a two-document corpus gives every claim fewer than 4 links, so spec R3(b) (thin evidence) correctly flags it and the plan's "no check row" expectation contradicts R3. The spec wins; the trigger logic is unchanged. Cost if wrong: test fixtures only.
- Task 17: complete (commits da39823..587cf57; acceptance: 1434 passed, ruff clean, 8 files in scope; test_ingest_upload now drains only ingest_upload since reverify_document has a handler).
- Task 17: minor (deferred): reverify jobs use the default 300 s worker timeout; a retried reverify_document appends duplicate checks (changed=false) for analyses it already handled; no rate limit on POST /analyses/{id}/reverify; claims_to_recheck holds a tenant transaction across reranker scoring.

- Task 18: complete (commits 587cf57..2ae414f; acceptance: 1469 passed, ruff clean, 19 files in scope; legacy routes and tests/test_api.py removed with git rm). Legacy endpoints now require a session and return string ids (frontend updated in Plan D).
- Task 18: carry-forward to Task 19: cli/main.py keeps _build_pipeline only for the legacy `evaluate` command; remove both with the legacy pipeline.
- Task 18: minor (deferred): an ingest failure inside a quick check (no_text, bad_encoding) surfaces as 500 in POST /api/analyze per the brief's "500 otherwise"; a 422 would be more accurate.

- Ruling: the legacy `halluciscope evaluate` CLI command and cli/main.py's _build_pipeline were removed with the legacy pipeline in Task 19. Why: they only drove the deleted pipeline and evaluation/benchmarks.py, whose metrics spec section 4 found invalid; spec 4's "preserved Typer CLI" names are superseded by the benchmark entry points `python -m evaluation.run_*`. Cost if wrong: a thin `evaluate` alias can be added in Plan D.
- Task 19: complete (commits 2ae414f..e424e8f; acceptance: 1441 passed, 0 warnings, ruff clean; 44 files incl. deletions; no remaining imports of deleted modules; langchain-text-splitters and aiosqlite removed from pyproject and the WSL venv).
- Task 19: minor (deferred): README still documents the old architecture and `evaluate` (Plan D Task 16 rewrites it); Settings.consistency_samples kept but unused (requests use options.consistency_samples); .gitignore keeps legacy lines (evaluation/datasets/*.json, *.db, *.sqlite); tests/conftest.py starts with a blank line.

- Task 20: complete (commits e424e8f..95951b3; acceptance: 1462 passed, ruff clean, 3 files in scope). Bootstrap draws from np.unique(groups) (sorted), so intervals do not depend on row order.
- ENV: full suite 250-370 s; other WSL workloads (a kind Kubernetes cluster and Go race tests from other projects) share the CPU; test_env_isolation's child pytest takes up to 93 s. Inner limit 900 s still has margin.

- Task 21: complete (commits 95951b3..14fd1d6; acceptance: 1507 passed, ruff clean, 2 files in scope). Real-data role populations (counts only): fit 1,724 sources, select 400, risk 391, test 450 (150 per task type); 17,790 responses; 0 span text mismatches; gold units supported 117,078, contradicted 5,133, insufficient 8,816. Manifest is produced in Task 27 after throughput sizing.

- Task 22: complete (commits 14fd1d6..9ce33be; acceptance: 1549 passed, ruff clean, 3 files in scope). Real-model measure (passage mode, under contention from other WSL workloads): 0.589 units/s, 2.24 NLI pairs/s, about 75 s per source; NLI is 94% of the time. The ragtruth-bench tenant in evidencegraph_bench holds 31 sources from the smoke run (reused by Task 27).
- COMPUTE FINDING for Task 27: plan_sizes floors (fit 200, select 150, risk 150, test 150 = 650 sources) cost about 13.5 h at the measured speed versus the planned 4 h budget; full data would be about 62 h. No CUDA GPU (AMD Radeon RX 9060 XT; torch here is the CPU build). Options before Task 27: re-measure on an idle machine; length-sorted NLI batching (numerically equivalent, less padding); then run at the floors with a larger budget and state the scale next to every number (within Omar's "sized to his hardware" agreement). Changing the NLI model to base or dropping fp32 would change spec calls and needs Omar. Decide at Task 27.
- .local/run.py: stdout now uses errors="replace" (it crashed printing non-cp1252 log text).

- Task 23: complete (commits 9ce33be..f955150; acceptance: 1570 passed, ruff clean, 3 files in scope). Test-role rows are dropped before loading; a spy test fails on any test-role cache read.
- Task 23: carry-forward to Task 27: run train_calibrator through .local/wsl-git.sh so metadata.git_commit is recorded; with few risk sources the fixed sequence can abstain (tau_s None or a vacuous 0.95), which is the honest LTT outcome and must be reported, not tuned away.

- Task 24: attempt 1 interrupted by the session before any code (notes only). Attempt 2 complete (commits f955150..79ae886; acceptance: 1602 passed, ruff clean, 3 files in scope). Runner adds --database-url and --data-root (atomic subset); refuses an artifact whose manifest_hash or models differ from the cache.
- Task 24: carry-forward to Task 27: run run_factuality through .local/wsl-git.sh so git_commit is recorded; real-scale bootstrap time unmeasured; evaluation/AGENTS.md command list to update in Plan D docs.

- Ruling: cross_tenant_links in run_temporal is measured by reading tenant B's passage ids inside B's tenant transaction and matching them against tenant A's evidence_links inside A's, not "as eg_owner". Why: RLS is forced and eg_owner has no BYPASSRLS, so an owner-level count sees zero rows and would always print a vacuous 0 (verified: 26,398 passages as postgres, 0 as eg_owner). A test fabricates a cross-tenant link and the count goes 0 to 1. Cost if wrong: none; the measurement is stricter.
- Ruling: predicted source change in run_temporal = latest verdict (newest re-check, else original) differs from the original verdict, matching the gold definition (gold at the last version vs v1). The brief's literal "latest claim_checks.changed" compares only consecutive checks and would mismeasure three-version families. Cost if wrong: a two-line change.
- Task 25: complete (commits 79ae886..97047a0; acceptance: 1637 passed, ruff clean, 4 files in scope). Bench users get the CLI's locked password hash but are not disabled (principal_for_user returns None for disabled users, which would suppress re-verification).
- Task 25: carry-forward to Task 27: migrate evidencegraph_bench to head (0004 not applied there yet) before any Plan C benchmark run.

- Task 26: complete (commits 97047a0..eab71d9; acceptance: 1654 passed, ruff clean, 3 files in scope). Nearest-rank percentiles; failures excluded from stage stats; no warm-up request (first request affects mean only at n = 50).

- Pre-Task 27 spike (controller, scratch .local/nli_spike.py, synthetic pairs, low load): DeBERTa-v3-large CPU fp32 batch 16, 12 threads: unsorted 3.69 pairs/s, length-sorted 5.11 (+38%); batch 32 sorted 4.15; bf16 autocast 9.13 (not adopted: spec 13 says fp32; offer to Omar).
- Fix: c383a19 perf: batch NLI pairs by length to cut padding (DebertaNLIScorer sorts by characters, longest first, restores input order). Acceptance: 1656 passed, ruff clean, models tests 3 passed.

- Task 27: ENV after reboot (2026-10-29 16:43): WSL fresh, Docker restarted, eg-pg up with evidencegraph and evidencegraph_bench intact. Ollama on Windows listens on 127.0.0.1:11434 only; WSL is NAT (gateway 172.25.80.1) and cannot reach it. Needed only for the atomic subset (Step 5) and latency (Step 6).
- Ruling: the controller runs Task 27's long commands itself as detached processes (Start-Process -> wsl -> .local/wsl-git.sh, logs in .local/t27-*.log) and polls them, instead of a blocking Sonnet runner. Why: extraction at the plan floors takes several hours and a blocking subagent stage is cancelled after 3,600 s without observable progress; the runs execute committed scripts only and change no code. Cost if wrong: none to results; publishing still uses the committed renderers.
- Task 27 run plan (resume from the first unchecked step; caches resume by unit):
  1. migrate evidencegraph_bench to head; 2. measure on the idle machine -> plan_sizes -> manifest -> commit `chore: pin RAGTruth benchmark splits`; 3. extract fit,select,risk passage (window ablation only if 40% of budget unused); 4. train -> artifact; check tau_s not None and select macro-F1 vs nli_only; 5. extract test --final; run_factuality --split test --final --ablations --atomic-subset 100 (Ollama); 6. run_temporal; run_latency --requests 50 (Ollama); 7. render; 8. suite + ruff; commit artifact, then docs.

- Task 27 step 1: done (evidencegraph_bench migrated, applied [4]).
- Task 27 step 2 measure (idle after reboot, load 0.35, length-sorted NLI): 0.713 units/s, 2.73 NLI pairs/s, 713 pairs over 200 fit units, NLI 93% of 280.5 s; seconds per source fit 62.8, select 61.9, risk 61.9, test 59.0. plan_sizes(budget 14,400 s) -> floors fit 200, select 150, risk 150, test 150 with over_budget; estimate 11.1 h of extraction.
- Ruling: run at the plan floors (fit 200, select 150, risk 150, test 150) despite the 4 h budget in spec 20.2 and the plan; keep fp32 DeBERTa-v3-large. Why: the floors are the plan's minimum for a usable fit and a meaningful LTT risk split and test intervals; going faster needs bf16 (spec 13 says fp32, about 1.8x in the spike) or the base NLI model (spec call 3), both Omar's decisions; machine time is reversible and Omar agreed to runs sized to the hardware with the scale stated. Window-premise ablation is not run (no budget left). Cost if wrong: about 11 h of CPU time; smaller samples widen every interval, which the report shows.

- Task 27 step 2: manifest committed 54efca2 chore: pin RAGTruth benchmark splits (fit 200, select 150, risk 150, test 150 sources; 1200/900/900/900 responses; manifest hash 3f8e5ae6...bcd6).
- Task 27 step 3: extraction (fit,select,risk, passage) ran detached 2026-10-29 16:52 to about 00:53 at 0.65-0.66 units/s and reached 18,807 of 22,221 units when the machine rebooted (WSL and Docker down on 2026-10-30 14:57). Cache resumes by unit; relaunching. Detached launch needs the wsl-git.sh path quoted inside the Start-Process argument string (the space in "Omar Morsi" split it once).

- Task 27 step 3: done. Second run extracted the remaining 3,369 units (0.73 units/s) and finished before the next reboot; cache 2e010a5395824ae7 holds all 22,221 fit/select/risk units (fit 8,937, select 6,604, risk 6,680), 0 missing, 0 unparsable, manifest hash matches.
- Task 27 step 4: trained (through wsl-git.sh) -> backend/calibration/artifacts/default.json (untracked, not committed) and evaluation/results/training_report.json. Design chosen on select: passage, evidence_k 8, feature set nli_retrieval_lexical_similarity (11 features); T 0.962; Newton converged in 6 iterations; select NLL 0.3456, ECE 0.0099, Brier 0.173. tau_s = None, tau_c = 0.5.
- Task 27: STOP (plan step 4 gate fired twice): tau_s is None, and select macro-F1 under decisions (0.038) is below nli_only's (0.349). Select-role comparison (run_factuality --split select, no test data): legacy_as_shipped detF1 0.042, legacy_fixed 0.144, similarity_only 0.249, nli_only 0.293 (FSR 0.404, accepted risk 0.058), evidencegraph abstains on everything (supported coverage 0, AURC 0.040, the best of the five).
- Task 27 root cause (systematic-debugging phase 1; scratch .local/feat_diag.py, gate_diag.py): no code defect found. NLI pairs are (premise, claim); labels from config; features separate classes (select max_entail AUC 0.717); calibrated p_supported AUC 0.76 (select) / 0.79 (risk), ECE 0.0099. The LTT fixed sequence starts at t = 0.95 (spec R1); there the per-source false-supported rate on risk is 0.1075 > alpha_s 0.10 (n = 134 sources), p = 1.0, so the walk stops at its first step and certifies nothing. On select, p_s >= 0.95 still accepts 10.9% of not-supported units (0.97: 5.1%; 0.99: almost nothing passes, max p_s 0.991). CONTRADICTED: no unit reaches p_c 0.5 (contradicted is 4% of units and max_contra does not separate it: AUC 0.49), so tau_c = 0.5 is vacuous and the system never predicts contradicted either.
- Task 27: test split untouched (no test_log.jsonl). Note for honesty: the controller's diagnostics looked at risk-split acceptance rates after the failed LTT pass, so re-running LTT with a new grid on the same risk split would no longer be clean.
- Task 27: decision for Omar (changes spec R1 or the published claim; not ruled by the controller): (A) keep R1, publish that no supported verdict is certifiable at alpha 0.10 at this scale (product shows insufficient for nearly every claim); (B, recommended) set the grid start to 0.99 (spec section 14's original grid) as a design choice made on the select role, and certify on 150 fresh risk-bucket sources never extracted or inspected (241 unused risk-bucket sources exist; about 2.6 h extraction; manifest gains a fresh risk role); (C) raise alpha_s: not recommended (target chosen after seeing data). Open earlier question bundled: bf16 NLI (about 1.8x faster) vs spec 13 fp32.
