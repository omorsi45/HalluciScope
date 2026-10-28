# Implementer rules (EvidenceGraph, all plans)

Read this whole file before starting. These rules bind every task.
Machine: Omar's Windows PC (PowerShell 5.1 shell, WSL2 Ubuntu 24.04 for Python). The Mac variant of these rules is in git history of this file.

## Workspace
- Repo worktree: `C:/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/evidencegraph` (branch `feature/evidencegraph`). Work only here. WSL path of the same folder: `/mnt/c/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/evidencegraph`.
- Shell is Windows PowerShell 5.1: no `&&`, `||`, or ternaries. Run commands one per line or separate with `;` and check `$LASTEXITCODE`.
- Git: use Windows git from PowerShell only, e.g. `git -C "C:/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/evidencegraph" status`. Never run git inside WSL (the worktree's gitdir is a Windows path and WSL git would misread it). Never push, never commit to master, never force, never amend, never rebase, never `git add .` or `-A`, never `git stash`, never `git clean`, never `git worktree prune`, never `--no-verify`. Stage files by name. Use `git rm` for deletions.
- Line endings: git has `core.autocrlf=true`, so existing files are CRLF in the working copy and LF in the index. Write new files with LF. Do not convert existing files. The "LF will be replaced by CRLF" warning is expected.
- Never commit or modify: `docs/superpowers/`, `.superpowers/` (except reading your brief), `.local/`, `.venv/`.

## Running Python (WSL only)
- Native Windows Python cannot run this suite (psycopg async refuses the Proactor event loop; the worker uses `loop.add_signal_handler`). Every Python, pytest, ruff, or CLI command runs in WSL through the wrapper, which cds into the worktree and uses the Linux venv (`~/venvs/evidencegraph`, Python 3.14.9, torch 2.14.1+cpu):
  - `wsl -d Ubuntu -- bash "/mnt/c/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/evidencegraph/.local/wsl.sh" timeout 300 python -m pytest tests/test_x.py -q`
  - Full suite: `... wsl.sh timeout 900 python -m pytest -q`
  - Lint: `... wsl.sh timeout 120 ruff check .`
  - CLI: `... wsl.sh timeout 120 python -m backend.cli.main <command>`
  - pip: `... wsl.sh timeout 900 uv pip install --python /home/omar_morsi/venvs/evidencegraph/bin/python <pkg>==<ver>` (only for dependencies the brief names).
- Do not use `.venv\Scripts\python.exe` for tests; it exists only for small Windows helpers.
- Postgres: Docker container `eg-pg` (pgvector/pgvector:pg17, PostgreSQL 17.11, pgvector 0.8.7), superuser `postgres`, trust auth, `localhost:54329`, reachable from WSL. If it is down: `docker start eg-pg`. Do not run `scripts/dev-postgres.sh` (macOS Homebrew). For admin SQL: `docker exec eg-pg psql -U postgres -h localhost -d <db> -v ON_ERROR_STOP=1 -c "<sql>"`. Dev database `evidencegraph` exists with roles `eg_owner` and `eg_app`.

## Time limits (mandatory)
- Wrap every WSL command in the Windows-level killer as well, so a stuck WSL session cannot hang your turn: from the worktree root in PowerShell, `.venv\Scripts\python.exe .local\run.py <outer_s> .local\<name>.log -- wsl -d Ubuntu -- bash "<wt-wsl-path>/.local/wsl.sh" timeout <inner_s> <command>` with outer = inner + 60. It prints the log tail and `exit=<code>`. Focused tests: inner 280. Full suite: inner 900.
- Every pytest, script, or server command runs under Linux `timeout` inside the wrapper (300 s focused, 900 s full suite). Exit code 124 means the limit fired: treat it as a failure, find the cause, never just raise the limit.
- Test fakes that wait on an event (for example a fake worker waiting on `stop`) must use a bounded wait, so a RED run fails instead of hanging.
- Write your report file early and append as you go.
- A RED test must never trigger the expensive or dangerous behavior it guards against (for example a real scrypt call with huge parameters, an unbounded loop, a network call, a model download). Replace the expensive dependency with a spy that records the call and returns immediately or raises, so the RED run fails fast.
- A previous attempt stalled for an hour this way. Do not repeat it.

## Process
- TDD is mandatory: write the failing tests from the brief, run them and capture the RED output, implement minimally, run them GREEN, then run the full suite and ruff once before committing. Record RED and GREEN evidence in your report.
- Read the TDD skill first: `C:/dev/superpowers/skills/test-driven-development/SKILL.md` (and `writing-good-tests.md` in the same directory when writing tests).
- On any unexpected failure, follow `C:/dev/superpowers/skills/systematic-debugging/SKILL.md`: find the root cause. Never bypass: no skipped or weakened tests, no `try/except: pass`, no `KMP_DUPLICATE_LIB_OK`, no `# noqa` without a reason comment, no disabling hooks.
- Implement exactly the brief. No extra features, no unrelated refactors. If the brief is ambiguous or wrong, prefer the spec (`C:/Users/Omar Morsi/Desktop/Projects/HalluciScope/.worktrees/notes/docs/superpowers/specs/2026-10-26-evidencegraph-design.md`; section 26 rulings override earlier sections) and record the decision in your report; if it is genuinely blocking, report NEEDS_CONTEXT.
- Do not dispatch subagents.

## Code rules
- Dependencies pinned with `==`. Do not add dependencies the brief does not name.
- Parameterized SQL only (`%s`). Never interpolate user values.
- Never log document text, questions, answers, passwords, tokens, or exception messages from handlers; log constant messages with whitelisted fields. Context-var values are `str`.
- No em dashes (the long dash character U+2014) anywhere: code, comments, docs, commit messages.
- Commit messages: conventional `type: summary`, imperative, lowercase type (feat, fix, refactor, test, docs, chore, build, ci). Never mention AI, assistants, Claude, Kiro, agents, or tools that wrote the code. Use the exact message the brief gives.
- Match existing style; keep files focused.

## Report
- Write the full report to the report file path given in the dispatch (create it; it lives in the notes worktree; do not commit it). Include: what you built, files changed, RED and GREEN evidence (commands and relevant output), full-suite result, lint result, decisions you made, concerns.
- Final reply under 15 lines: Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary, concerns, report path.
