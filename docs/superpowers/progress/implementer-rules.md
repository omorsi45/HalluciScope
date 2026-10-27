# Implementer rules (EvidenceGraph, all plans)

Read this whole file before starting. These rules bind every task.

## Workspace
- Repo worktree: /Users/omarmorsi/Desktop/Github/HalluciScope/.worktrees/evidencegraph (branch feature/evidencegraph). Work only here.
- Git: always prefix with `PATH=/usr/bin:$PATH` (the /usr/local/bin/git binary is x86 and fails with "Bad CPU type"). Never push, never commit to master, never force, never amend, never rebase, never `git add .` or `-A`, never `git stash`, never `git clean`, never `--no-verify`. Stage files by name. Use `git rm` for deletions.
- Never commit or modify: docs/superpowers/, .superpowers/ (except writing your report file where told), .local/, .venv/.
- Python: `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check .`, `.venv/bin/python -m pip ...`. Python is 3.14.7.
- Postgres (from Plan A Task 5 on): `bash scripts/dev-postgres.sh start`. Before Task 5 exists: `PATH=/opt/homebrew/opt/postgresql@17/bin:$PATH pg_ctl -D .local/pgdata -o "-p 54329 -k $PWD/.local" -l .local/pg.log start`. Superuser `postgres`, trust auth, port 54329. Leave it running when you finish.

## Time limits (mandatory)
- macOS has no `timeout` binary. Run every pytest, script, or server command under a hard limit with perl: `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q` (600 s for the full suite, 300 s for a focused file). Exit code 142 means the limit fired: treat it as a failure, find the cause, never just raise the limit.
- A RED test must never trigger the expensive or dangerous behavior it guards against (for example a real scrypt call with huge parameters, an unbounded loop, a network call). Replace the expensive dependency with a spy that records the call and returns immediately or raises, so the RED run fails fast.
- A previous attempt stalled for an hour this way. Do not repeat it.

## Process
- TDD is mandatory: write the failing tests from the brief, run them and capture the RED output, implement minimally, run them GREEN, then run the full suite once before committing. Record RED and GREEN evidence in your report.
- Read the TDD skill first: /Users/omarmorsi/.kiro/skills/sp-tdd/SKILL.md (and writing-good-tests.md in the same directory when writing tests).
- On any unexpected failure, follow /Users/omarmorsi/.kiro/skills/sp-systematic-debugging/SKILL.md: find the root cause. Never bypass: no skipped or weakened tests, no `try/except: pass`, no `KMP_DUPLICATE_LIB_OK`, no `# noqa` without a reason comment, no disabling hooks.
- Implement exactly the brief. No extra features, no unrelated refactors. If the brief is ambiguous or wrong, prefer the spec (docs/superpowers/specs/2026-10-26-evidencegraph-design.md; section 26 rulings override earlier sections) and record the decision in your report; if it is genuinely blocking, report NEEDS_CONTEXT.
- Do not dispatch subagents.

## Code rules
- Dependencies pinned with `==`. Do not add dependencies the brief does not name.
- Parameterized SQL only (`%s`). Never interpolate user values.
- Never log document text, questions, answers, passwords, tokens, or exception messages from handlers; log constant messages with whitelisted fields.
- No em dashes (the long dash character) anywhere: code, comments, docs, commit messages.
- Commit messages: conventional `type: summary`, imperative, lowercase type (feat, fix, refactor, test, docs, chore, build, ci). Never mention AI, assistants, Claude, Kiro, or tools that wrote the code. Use the exact message the brief gives.
- Match existing style; keep files focused.

## Report
- Write the full report to the report file path given in the dispatch (create it). Include: what you built, files changed, RED and GREEN evidence (commands and relevant output), full-suite result, lint result, decisions you made, concerns.
- Final reply under 15 lines: Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary, concerns, report path.
