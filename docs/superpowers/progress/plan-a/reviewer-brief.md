# Reviewer brief (EvidenceGraph Plan A)

You review one task's diff: spec compliance first, then code quality. Read-only: never modify files, the index, HEAD, or branches. Do not dispatch subagents. Do not re-run the full suite; run a focused test only for a specific doubt. Git commands must be prefixed with `PATH=/usr/bin:$PATH`.

## Binding global constraints (verbatim from Plan A)
- Work only in /Users/omarmorsi/Desktop/Github/HalluciScope/.worktrees/evidencegraph on branch feature/evidencegraph.
- Never commit docs/superpowers/, .superpowers/, .local/, .venv/.
- Commit messages: conventional `type: summary`, imperative, no mention of AI, assistants, or tools. No em dashes anywhere (code comments, docs, messages).
- Dependencies pinned with `==` in pyproject.toml. Versions: fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.1, pydantic-settings 2.16.1, sentence-transformers 6.2.0, transformers 5.20.2, torch 2.14.1, httpx 0.28.1, typer 0.27.3, rich 15.0.0, pypdf 6.21.2, python-multipart 0.0.32, numpy 2.5.4, psycopg[binary] 3.3.6, psycopg-pool 3.3.3, pgvector 0.5.1, pytest 9.1.1, pytest-asyncio 1.4.1, pytest-cov 7.1.0, ruff 0.17.2, matplotlib 3.11.2. Keep langchain-text-splitters and aiosqlite pinned at installed versions until Plan C; remove faiss-cpu, datasets, scikit-learn.
- SQL: parameterized queries only; dynamic identifiers only from code constants.
- Never log document text, questions, answers, passwords, tokens, or raw exception messages from job handlers. Log messages are constant strings; data goes in whitelisted fields.
- API base path /api/v1. Errors are application/problem+json `{"type": "/errors/<code>", "title", "status", "detail", "code"}`.
- Cookies eg_session (HttpOnly) and eg_csrf (readable), SameSite=Lax, Path=/, Max-Age=604800, Secure unless HALLUCISCOPE_DEV_INSECURE_COOKIES=true. CSRF header X-CSRF-Token.
- Postgres roles: eg_owner owns schema and runs migrations; eg_app runtime role (not owner, no BYPASSRLS). Tenant context is transaction-local setting app.tenant_id.
- Tests needing Postgres use the `pg` marker and fail (never skip) with setup instructions when unreachable. Unit tests never download models or call Ollama.
- Bypasses are defects: skipped or weakened tests, swallowed exceptions, KMP_DUPLICATE_LIB_OK, unexplained noqa.

## Spec
docs/superpowers/specs/2026-10-26-evidencegraph-design.md is the authority (section 26 rulings override earlier sections). Ledger rulings: R-A1 moves pytest-asyncio session loop-scope settings into Task 5.

## Output
Use the format in your dispatch: Spec Compliance (✅/❌, ⚠️ cannot verify), Strengths, Issues (Critical/Important/Minor with file:line), Assessment (Approved | Needs fixes). Begin directly with the verdict. No em dashes.
