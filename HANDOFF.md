# EvidenceGraph handoff prompt

Paste everything below this line into the next coding agent.

---

You are continuing a large, partly finished engineering project for Omar Morsi. Read this whole message before acting. Everything you need is in the GitHub repository; nothing is on this machine yet.

## 1. What the project is

Repository: `omorsi45/HalluciScope` (public GitHub repo, default branch `master`). HalluciScope was a small claim-level RAG hallucination checker. We are rebuilding it into **EvidenceGraph**: a multi-tenant document intelligence system over large, versioned document collections. A user uploads documents, asks questions (or pastes an answer from any model), and every atomic claim gets a calibrated verdict (supported, contradicted, insufficient) linked to the exact stored passage and immutable document version that justifies it. When a source changes, past verdicts that depended on it are re-checked. It ships with real, reproducible benchmarks (BEIR SciFact and NFCorpus for retrieval, RAGTruth for factuality, a synthetic temporal set, latency) with baselines and ablations. Audience: hiring teams at frontier AI labs and legal-AI companies, so correctness and honest numbers matter more than speed.

Architecture: one PostgreSQL 17 + pgvector database (forced row-level security per tenant), a FastAPI API process at `/api/v1`, a worker process that owns the ML models and runs jobs from a `SKIP LOCKED` job table, Ollama for the LLM, and a React 19 + TypeScript + Vite + Tailwind frontend.

## 2. Where everything is

Two branches on GitHub. `master` is untouched.

- `feature/evidencegraph`: all code. 43 commits ahead of `master`. Plan A (foundation) is complete: 446 tests pass, `ruff check .` is clean (verified 2026-10-27).
- `evidencegraph-notes`: an orphan branch with no code, only working documents:
  - `docs/superpowers/specs/2026-10-26-evidencegraph-design.md`: the design spec. It is the binding authority. Section 26 (rulings R1 to R10) overrides earlier sections.
  - `docs/superpowers/plans/2026-10-26-evidencegraph-{a-foundation,b-ingest-retrieval,c-verify-calibrate,d-product-release}.md`: the four implementation plans. Each task has exact files, interfaces, tests, and commit messages. Plans B, C, and D were updated with rulings from Plan A's reviews (see their Global Constraints).
  - `docs/superpowers/progress/plan-a/progress.md`: the Plan A ledger. It records every task's outcome, every `Ruling:` line (decisions made on Omar's behalf), and deferred minor findings.
  - `docs/superpowers/progress/plan-a/`: per-task implementer reports and findings, plus `final-findings.md` and `final-fixwave-report.md` (the last fix wave).
  - `docs/superpowers/progress/implementer-rules.md`: the rules every implementer followed. Its absolute paths are from Omar's Mac, so rewrite them for this machine before use.
  - `HANDOFF.md`: this prompt.

## 3. Set up on this machine

1. Find Omar's existing GitHub workspace folder and use it. Do not create repos elsewhere. Run `gh auth status` first.
2. Clone the repo there. Check out `feature/evidencegraph` in a worktree for code work (for example `.worktrees/evidencegraph`). Check out `evidencegraph-notes` in a second worktree (for example `.worktrees/notes`). Add `.worktrees/` to `.git/info/exclude`.
3. In the feature worktree, add `docs/superpowers/`, `.superpowers/`, `.local/`, and `.venv/` to `.git/info/exclude`. Specs, plans, and ledgers must never be committed to `feature/evidencegraph`. They live only on `evidencegraph-notes`.
4. Toolchain the code assumes:
   - Python 3.14 (pinned deps include torch 2.14.1).
   - Node 24 or newer with npm.
   - PostgreSQL 17 with pgvector 0.8 and citext.
   - Ollama with `llama3.2:3b` (needed only for Plan C real runs).
   - Create `.venv` in the feature worktree and run `pip install -e ".[dev,eval]"`.
5. Postgres for tests: the suite needs a superuser `postgres` with trust auth on `localhost:54329` (override with `HALLUCISCOPE_TEST_ADMIN_URL`). The test fixtures create the roles `eg_owner` and `eg_app` and temporary databases themselves. `scripts/dev-postgres.sh start` sets up a dev cluster on macOS with Homebrew (`PG_BIN` overrides the binary path). On Windows, use WSL2 Ubuntu with `postgresql-17` and `postgresql-17-pgvector`, or Docker: `docker run -d --name eg-pg -p 54329:5432 -e POSTGRES_HOST_AUTH_METHOD=trust pgvector/pgvector:pg17`. For the dev database `evidencegraph`, replicate what `scripts/dev-postgres.sh` does: create roles `eg_owner` and `eg_app` with `LOGIN NOSUPERUSER NOBYPASSRLS`, a database owned by `eg_owner`, and the extensions `vector` and `citext` created by `postgres`.
6. Verify the baseline before changing anything:
   - `python -m pytest -q`: 446 passed.
   - `ruff check .`: clean.
   - `python -m backend.cli.main migrate`: applies `[1, 2]` on a fresh dev database.

   Run the CLI as `python -m backend.cli.main <command>` from the worktree root. On Omar's Mac the installed `halluciscope` entry point broke because macOS hidden flags made Python skip `.pth` files. It may work on this machine, but the plans assume the module form.

## 4. How to work (Omar's instructions, binding)

- **Roles:** You plan, coordinate, and verify. A Sonnet model implements: use `claude-sonnet-5.5` or this client's current Sonnet ID; check the client's model list and never guess a name. Run one implementer subagent per task, sequentially, never in parallel.
- **No per-task review:** do not dispatch reviewer subagents and do not use Opus. After each task, run a fast acceptance check yourself:
  - the implementer's report shows a failing test before the change and passing tests after;
  - the full suite and `ruff check .` pass;
  - the commit exists with the plan's message;
  - the diff stat touches only the task's files.

  If a check fails, send the implementer back with the concrete failure. There is no review loop.
- **One review at the end:** when every task in Plans B, C, and D is done, review the whole branch yourself once. Fix what matters in one fix wave, then use superpowers `finishing-a-development-branch`: push and open one PR from `feature/evidencegraph` into `master`. Ask Omar before merging.
- **One milestone:** all work stays on `feature/evidencegraph`. No per-plan PRs, branches, or GitHub milestones.
- **Move fast:**
  - Keep dispatch prompts short and point to a task brief file instead of pasting plan text.
  - Extract each brief with superpowers' `task-brief` script, or copy the `### Task N` section into a file.
  - Batch two or three tiny same-file tasks into one dispatch when the plan allows it.
  - Bound every test or server command with a timeout. A runaway scrypt test once stalled for an hour.
- **Superpowers skills:** read `test-driven-development`, `subagent-driven-development` (skip its task-review steps per Omar), `systematic-debugging`, `verification-before-completion`, and `finishing-a-development-branch` from this machine's superpowers install. Forward the rules a subagent needs inside its prompt: TDD, no em dashes, the commit rules, and timeouts.
- **Progress must stay in the repo:** keep the ledger for each plan at `docs/superpowers/progress/plan-<b|c|d>/progress.md` in the notes worktree. Append one line per task: `Task N: complete (commits a..b)`, plus any `Ruling:` with what it costs if wrong. Commit and push the notes branch at least after every five tasks and at every stop, so work can resume from any machine. Push `feature/evidencegraph` at the same points. Use plain `git push`; never force.

## 5. Rules that bind every change

- **Spec over plan:** the spec is the authority and the plan is its argument. When they conflict, follow the spec and record a `Ruling:`.
- **TDD is mandatory:** a failing test first, then the minimal code, then the full suite. Never skip, weaken, or bypass tests or hooks. Never use `--no-verify` or `KMP_DUPLICATE_LIB_OK`, and never swallow exceptions.
- **No fabricated results:** never fabricate or estimate benchmark numbers. Every number in `docs/results/` and the README must come from a committed script run on stated data (Plan B Task 21, Plan C Task 27, Plan D Task 16). Omar agreed to real runs sized to his hardware, with the scale stated next to every number. The RAGTruth test split is evaluated once, with `--final`, and logged.
- **Git:**
  - Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `build:`, `ci:`, `refactor:`), using the plan's exact messages.
  - Stage files by name; no `git add .`.
  - Never mention AI, assistants, Claude, Kiro, or agents in commits, PR text, code comments, or product docs.
  - Never push to `master`, force-push, amend pushed commits, or merge without Omar's approval.
- **No em dashes anywhere:** code, comments, docs, commit messages, replies to Omar.
- **Code:** dependencies pinned with `==`. Parameterized SQL only. Never log document text, questions, answers, tokens, or exception messages; log constant strings plus whitelisted fields. Context-var values must be `str`.
- **Rulings carried into Plans B and C** (already written into their Global Constraints):
  - Every foreign key between tenant tables includes `tenant_id`.
  - Job payloads carry only ids, counts, and codes.
  - Register failure hooks with `build_failure_handlers` for `ingest_upload`, `search`, and `run_analysis`.
  - `require_collection` and `intake_upload` take collection `kinds`; quick-check paths pass `("quick",)`.
  - Error codes are `api_key_not_found` and `grantee_not_member`.
  - Plan B Task 7's migration also adds the indexes deferred from Plan A.
- **Deferred items still owed before the final PR** (from the Plan A ledger):
  - make logout idempotent before the frontend work in Plan D Task 2;
  - purge old throttle, session, job, and worker rows;
  - a registration throttle and an API key cap;
  - uvicorn `--proxy-headers` behind nginx (already in Plan D Task 12).

  Also triage the remaining `minor (deferred)` lines in the Plan A ledger during the final review.

## 6. Exactly where to resume

1. Plan A is complete. Its final fix wave (commits `15bc7d6..536b023`) was never independently re-reviewed, so cover that range in the final whole-branch review.
2. Start **Plan B Task 1** (ingestion types, normalization, text assembler) and run Plan B through Task 21. Task 21 is the real BEIR run, which needs about 220 MB of models and tens of minutes per dataset.
3. Then run Plan C Tasks 1 to 27. Task 27 is the real RAGTruth, temporal, and latency runs, which need the NLI model (about 1.7 GB) and Ollama.
4. Then run Plan D Tasks 1 to 16. Put `frontend/node_modules` wherever the machine needs it. Docker verification is in Task 12; report honestly if this machine cannot run it.
5. Do the final review, one fix wave, and the PR as described in section 4.

When you report to Omar, be brief and direct, lead with the outcome, and list every `Ruling:` you made with what it costs if wrong.
