# EvidenceGraph Plan A: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the legacy baseline and build the secure multi-tenant backend foundation (Postgres schema with RLS, auth, authorization, API skeleton, observability, job queue and worker) that Plans B to D build on.

**Architecture:** PostgreSQL-centric modular monolith. A FastAPI process (model-free except the embedder added in Plan B) and a worker process share one Postgres 17 + pgvector database. Tenant isolation is enforced twice: application scoping through `Database.tenant_tx()` and forced row-level security. Background work goes through a `SKIP LOCKED` job table written in the same transaction as the request.

**Tech Stack:** Python 3.14.7, FastAPI 0.162.0, psycopg 3.3.6 (async) + psycopg-pool 3.3.3, PostgreSQL 17.11 + pgvector 0.8.7 (Homebrew, local cluster on port 54329), pytest 9.1.1 + pytest-asyncio 1.4.1, ruff 0.17.2.

**Spec:** `docs/superpowers/specs/2026-10-26-evidencegraph-design.md` (sections 8, 16, 17, 18, 21, 22, and rulings R1 to R10 in section 26, which override earlier sections).

## Global Constraints

- Work only in `/Users/omarmorsi/Desktop/Github/HalluciScope/.worktrees/evidencegraph` on branch `feature/evidencegraph`.
- Git: always `PATH=/usr/bin:$PATH git ...` (`/usr/local/bin/git` is an x86 binary that cannot run). Never push, never commit to `master`, never force anything.
- Never commit `docs/superpowers/`, `.superpowers/`, `.local/`, `.venv/` (all excluded via `.git/info/exclude`). Stage files by name, never `git add .` or `-A`.
- Commit messages: conventional `type: summary` (feat, fix, refactor, test, docs, chore, build), imperative, no mention of AI, assistants, or tools that wrote the code. No em dashes anywhere (code comments, docs, messages).
- Python: run everything with `.venv/bin/python -m ...` from the worktree root. Tests: `.venv/bin/python -m pytest -q`. Lint: `.venv/bin/ruff check .`.
- Dependencies pinned with `==` in `pyproject.toml`. Versions to use: fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.1, pydantic-settings 2.16.1, sentence-transformers 6.2.0, transformers 5.20.2, torch 2.14.1, httpx 0.28.1, typer 0.27.3, rich 15.0.0, pypdf 6.21.2, python-multipart 0.0.32, numpy 2.5.4, psycopg[binary] 3.3.6, psycopg-pool 3.3.3, pgvector 0.5.1, pytest 9.1.1, pytest-asyncio 1.4.1, pytest-cov 7.1.0, ruff 0.17.2, matplotlib 3.11.2. Keep `langchain-text-splitters` and `aiosqlite` pinned at their installed versions until Plan C removes the legacy pipeline; remove `faiss-cpu`, `datasets`, `scikit-learn`.
- SQL: parameterized queries only (`%s` placeholders); never interpolate user values. Identifiers that must be dynamic come from code constants only.
- Never log document text, questions, answers, passwords, tokens, or raw exception messages from job handlers. Log messages are constant strings; data goes in whitelisted fields.
- API base path `/api/v1`. Errors are `application/problem+json` bodies `{"type": "/errors/<code>", "title": str, "status": int, "detail": str | null, "code": str}`.
- Cookies: `eg_session` (HttpOnly) and `eg_csrf` (readable), `SameSite=Lax`, `Path=/`, `Max-Age=604800`, `Secure` unless `HALLUCISCOPE_DEV_INSECURE_COOKIES=true`. CSRF header `X-CSRF-Token`.
- Postgres roles: `eg_owner` owns schema objects and runs migrations; `eg_app` is the runtime role (not owner, no `BYPASSRLS`). Tenant context is the transaction-local setting `app.tenant_id`.
- Local Postgres: `scripts/dev-postgres.sh start` (cluster `.local/pgdata`, port 54329, socket dir `.local`, trust auth). Test admin URL env `HALLUCISCOPE_TEST_ADMIN_URL`, default `postgresql://postgres@localhost:54329/postgres`. Tests that need Postgres use the `pg` marker and fail (never skip) with setup instructions when it is unreachable.
- Unit tests never download models or call Ollama.

## Review Focus

1. Plain-http development login: a session cookie marked `Secure` is silently dropped over `http://localhost`, so login "works" and every later request is 401. Expected: with `dev_insecure_cookies=True` the cookies carry no `Secure` attribute, otherwise they do. Test in Task 12.
2. Tenant context surviving on a pooled connection: after `tenant_tx(A)` returns its connection to the pool, a following `system_tx()` or `tenant_tx(B)` on the same physical connection must not see tenant A's rows. Test in Task 6.
3. Email case and whitespace: registering `" Alice@Example.com "` then logging in as `alice@example.com` reaches the same account; registering `ALICE@example.com` again returns 409 `email_taken`. Test in Task 11.
4. Two workers polling at once never run the same job, and a job whose worker died mid-run is re-queued after the stale timeout with its attempt counted. Tests in Task 16.
5. Oversized request bodies sent without `Content-Length` (chunked) are rejected with 413 `payload_too_large` after the limit is crossed, not buffered. Test in Task 10.

## File Map

Created
- `scripts/dev-postgres.sh`: start/stop/init the local cluster, roles, dev database, extensions.
- `backend/db/pool.py`: `Database` (async pool, `tenant_tx`, `system_tx`).
- `backend/db/migrate.py`, `backend/db/migrations/0001_global.sql`, `backend/db/migrations/0002_collections.sql`.
- `backend/db/repos/{users,tenants,sessions,api_keys,audit,collections,members}.py`.
- `backend/auth/{passwords,tokens,service,deps}.py`, `backend/authz/policy.py`.
- `backend/obs/{logging,metrics,metrics_server}.py`.
- `backend/api/{errors,middleware,pagination}.py`, `backend/api/routes/{auth,api_keys,collections,members,health}.py`, `backend/api/schemas_v1.py`.
- `backend/jobs/{queue,worker,registry}.py`.
- Tests: `tests/pg_fixtures.py` (imported by `conftest.py`), `tests/helpers/auth.py`, `tests/test_migrate.py`, `tests/test_schema_global.py`, `tests/test_rls.py`, `tests/test_passwords.py`, `tests/test_tokens.py`, `tests/test_obs_logging.py`, `tests/test_obs_metrics.py`, `tests/test_api_skeleton.py`, `tests/test_auth_service.py`, `tests/test_auth_routes.py`, `tests/test_api_keys.py`, `tests/test_collections.py`, `tests/test_members.py`, `tests/test_jobs_queue.py`, `tests/test_worker.py`, `tests/test_health_metrics.py`.

Modified
- `pyproject.toml`, `backend/config.py`, `backend/core/chunker.py`, `backend/core/verifiers/nli.py`, `backend/core/generator.py`, `backend/core/decomposer.py`, `backend/api/app.py`, `backend/api/startup.py`, `backend/cli/main.py`, `tests/conftest.py`, `tests/test_chunker.py`, `tests/test_nli.py`, `tests/test_decomposer.py`, `backend/core/verifiers/AGENTS.md`, lint-only edits across `backend/`, `evaluation/`, `tests/`.

---

### Task 1: Remove FAISS and pin dependencies so the full suite runs in one process

Root cause (verified): `torch` and `faiss-cpu` each load their own `libomp.dylib`; the second initialization aborts the process (`OMP: Error #15`). `KMP_DUPLICATE_LIB_OK` is forbidden.

**Files:**
- Modify: `backend/core/chunker.py`, `pyproject.toml`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Produces: `DocumentIndex(chunks: list[str], embeddings: np.ndarray)` (no `faiss_index` field); `Chunker.retrieve(query, doc_index, top_k=None) -> list[str]` ordered by descending dot product of normalized embeddings, ties by lower chunk index.

- [ ] **Step 1: Write the failing tests** in `tests/test_chunker.py`: replace `assert doc_index.faiss_index is not None` with `assert "faiss_index" not in DocumentIndex.__dataclass_fields__`; add `test_retrieve_orders_by_cosine` with a fake embedder mapping `"q" -> [1,0]`, chunks `"a" -> [0,1]`, `"b" -> [0.8,0.6]`, `"c" -> [1,0]` (build a `DocumentIndex` directly) and assert `chunker.retrieve("q", idx, top_k=2) == ["c", "b"]`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_chunker.py -q`. Expected: FAIL (`faiss_index` still a field; ordering test fails or errors).
- [ ] **Step 3: Implement** numpy retrieval in `backend/core/chunker.py` (`scores = doc_index.embeddings @ q`, stable argsort descending, `np.argsort(-scores, kind="stable")[:k]`); remove the `faiss` import. In `pyproject.toml`: pin every dependency to the Global Constraints versions, drop `faiss-cpu`, `datasets`, `scikit-learn`; add `psycopg[binary]==3.3.6`, `psycopg-pool==3.3.3`, `pgvector==0.5.1`; under `[tool.pytest.ini_options]` add `markers = ["pg: needs local Postgres", "models: loads real ML models"]` and `addopts = "-m 'not models'"`. `evaluation/benchmarks.py` imports scikit-learn metrics: replace them with a local `compute_metrics` using plain Python (precision, recall, F1 with zero_division 0; `auc_roc` via the rank-sum formula, `None` if one class) so the module still imports.
- [ ] **Step 4: Reinstall and run the whole suite in one process:** `.venv/bin/python -m pip uninstall -y faiss-cpu scikit-learn datasets && .venv/bin/python -m pip install -e ".[dev,eval]" && .venv/bin/python -m pytest -q`. Expected: all tests pass in one run (no abort), count at least 48.
- [ ] **Step 5: Commit** `pyproject.toml backend/core/chunker.py evaluation/benchmarks.py tests/test_chunker.py` with `fix: drop faiss to stop duplicate OpenMP runtime abort`.

### Task 2: Read NLI label order from the model config

Verified on 2026-10-26: `cross-encoder/nli-deberta-v3-large` has `id2label = {0: contradiction, 1: entailment, 2: neutral}`; the code assumes `[contradiction, neutral, entailment]`, so neutral claims score as supported.

**Files:**
- Modify: `backend/core/verifiers/nli.py`, `backend/core/verifiers/AGENTS.md`
- Test: `tests/test_nli.py`

**Interfaces:**
- Produces: `NLIVerifier(tokenizer, model)` reads `model.config.id2label` (keys may be int or str, names compared lowercased) into `self.label_index: dict[str, int]` with keys `contradiction`, `entailment`, `neutral`; raises `ValueError("NLI model labels must include contradiction, entailment, neutral")` otherwise. `_compute_hallucination_score(self, contradiction: float, neutral: float, entailment: float) -> float` returns `contradiction + 0.5 * neutral` (legacy semantics kept). Details keys unchanged.

- [ ] **Step 1: Write failing tests** in `tests/test_nli.py`: helper builds a fake model whose `config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}` and whose logits put 6.0 at a chosen index. `test_entailment_index_from_config`: logits at 1 -> `hallucination_score < 0.05` and `details["entailment"] > 0.95`. `test_neutral_scores_half`: logits at 2 -> `abs(score - 0.5) < 0.05` and `details["neutral"] > 0.95`. `test_missing_labels_raise`: `id2label = {0: "a", 1: "b", 2: "c"}` -> `pytest.raises(ValueError)`. `test_truncates_premise_only`: tokenizer called once with `truncation="only_first"`. Update `test_nli_score_mapping` to the new keyword signature and `test_nli_verify_returns_correct_count` to set `config.id2label` and put the strong logit at index 1.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_nli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** in `backend/core/verifiers/nli.py`; tokenizer call uses `truncation="only_first"`. Update the AGENTS.md NLI bullet to say labels are read from `id2label`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `backend/core/verifiers/nli.py backend/core/verifiers/AGENTS.md tests/test_nli.py` with `fix: read NLI label order from model config`.

### Task 3: Send claim decomposition through its own system prompt

**Files:**
- Modify: `backend/core/generator.py`, `backend/core/decomposer.py`
- Test: `tests/test_decomposer.py`, `tests/test_generator.py` (must keep passing)

**Interfaces:**
- Produces: `async def chat_completion(messages: list[dict[str, str]], base_url: str, model: str, temperature: float = 0.0, client: httpx.AsyncClient | None = None) -> str` in `backend/core/generator.py`, carrying today's error mapping to `OllamaError`; `generate_answer` keeps its signature and calls it. `decompose_claims(answer, base_url, model, client=None)` sends `[{"role": "system", "content": DECOMPOSE_SYSTEM}, {"role": "user", "content": DECOMPOSE_USER.format(answer=answer)}]` at temperature 0.0.

- [ ] **Step 1: Write failing test** `test_decompose_uses_decompose_system_prompt`: patch `backend.core.decomposer.chat_completion` (AsyncMock returning `"1. A.\n2. B."`), call `decompose_claims`, assert `messages[0] == {"role": "system", "content": DECOMPOSE_SYSTEM}`, `"Context:" not in messages[1]["content"]`, the answer text is in `messages[1]["content"]`, and the result is `["A.", "B."]`. Change `test_decompose_claims` to patch `chat_completion` instead of `generate_answer`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_decomposer.py -q`. Expected: FAIL (`chat_completion` not found in decomposer).
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS (including `tests/test_generator.py` and `tests/test_consistency.py`).
- [ ] **Step 5: Commit** `backend/core/generator.py backend/core/decomposer.py tests/test_decomposer.py` with `fix: use the decomposition system prompt for claim extraction`.

### Task 4: Lint to zero with an explicit rule set

**Files:**
- Modify: `pyproject.toml`; lint fixes in any `backend/`, `evaluation/`, `tests/` file ruff reports (47 findings on 2026-10-26: I001, F401, ASYNC230, SIM117, B008, UP012, RUF059, F841).

- [ ] **Step 1: Pin the rule set** in `pyproject.toml`: `[tool.ruff.lint] select = ["E", "F", "I", "B", "UP", "SIM", "ASYNC", "RUF"]`; `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["typer.Option", "typer.Argument", "fastapi.Depends", "fastapi.Query", "fastapi.File", "fastapi.Form", "fastapi.Header", "fastapi.Cookie"]`. Run `.venv/bin/ruff check .` and record the count (expected non-zero).
- [ ] **Step 2: Fix** with `ruff check . --fix` for safe fixes, then by hand: SIM117 combine `with` statements; ASYNC230 in `evaluation/` read files with `await asyncio.to_thread(path.read_text)` or move reads out of the coroutine; RUF059/F841 rename to `_`. No `noqa` unless the rule is wrong for the line, with a reason comment.
- [ ] **Step 3: Verify** `.venv/bin/ruff check .` prints `All checks passed!` and `.venv/bin/python -m pytest -q` passes.
- [ ] **Step 4: Commit** the changed files by name with `chore: pin ruff rules and fix lint findings`.

### Task 5: Settings, local Postgres script, migration runner, `halluciscope migrate`

**Files:**
- Create: `scripts/dev-postgres.sh`, `backend/db/migrate.py`, `backend/db/migrations/` (empty dir with `.gitkeep` until Task 6), `tests/pg_fixtures.py`, `tests/test_migrate.py`
- Modify: `backend/config.py`, `backend/cli/main.py`, `tests/conftest.py` (`from tests.pg_fixtures import *` with `# noqa: F403` and a reason comment)

**Interfaces:**
- Produces in `Settings`: `database_url: str = "postgresql://eg_app@localhost:54329/evidencegraph"`, `migration_database_url: str = "postgresql://eg_owner@localhost:54329/evidencegraph"`, `cors_origins: list[str] = ["http://localhost:5173"]`, `allow_registration: bool = True`, `dev_insecure_cookies: bool = False`, `metrics_token: str | None = None`, `worker_concurrency: int = 1`, `worker_metrics_port: int = 9101`, `log_level: str = "INFO"`, `session_ttl_days: int = 7`. Existing fields stay.
- Produces in `backend/db/migrate.py`: `MIGRATIONS_DIR: Path` (package dir `migrations`); `@dataclass(frozen=True) class Migration(version: int, name: str, path: Path, checksum: str)` (checksum = sha256 hex of file bytes); `class MigrationError(RuntimeError)`, `class MigrationChecksumError(MigrationError)`; `def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]` (files matching `^(\d{4})_([a-z0-9_]+)\.sql$`, sorted by version, other files ignored, duplicate version raises `MigrationError`); `async def migrate(url: str, directory: Path = MIGRATIONS_DIR) -> list[int]` (creates `schema_migrations(version int primary key, name text not null, checksum text not null, applied_at timestamptz not null default now())` if absent; verifies every applied checksum first and raises `MigrationChecksumError` on mismatch; applies each pending file in its own transaction after `select pg_advisory_xact_lock(727274)`; returns newly applied versions); `async def pending(url: str, directory: Path = MIGRATIONS_DIR) -> list[int]`.
- Produces in `tests/pg_fixtures.py`: `ADMIN_URL` (env `HALLUCISCOPE_TEST_ADMIN_URL`, default `postgresql://postgres@localhost:54329/postgres`); `def role_url(role: str, dbname: str) -> str` (via `psycopg.conninfo.make_conninfo(ADMIN_URL, user=role, dbname=dbname)`); `async def ensure_roles() -> None` (creates `eg_owner` and `eg_app` with `LOGIN NOSUPERUSER NOBYPASSRLS` if missing); `async def create_database(name: str) -> None` (owner `eg_owner`, then as admin in that db `create extension if not exists vector; create extension if not exists citext`); `async def drop_database(name: str) -> None` (`drop database if exists ... with (force)`); fixture `fresh_db_url` (function scope, database `eg_tmp_<8 hex>`, yields the `eg_owner` URL, drops after). Any connection failure calls `pytest.fail("Postgres not reachable at <ADMIN_URL>. Run scripts/dev-postgres.sh start", pytrace=False)`.
- `pyproject.toml` `[tool.pytest.ini_options]` gains `asyncio_default_fixture_loop_scope = "session"` and `asyncio_default_test_loop_scope = "session"` in this task (ledger ruling R-A1), so the async pg tests here and later share one event loop with session fixtures.
- CLI: `halluciscope migrate` runs `migrate(Settings().migration_database_url)` and prints `applied: [..]` or `up to date`.

- [ ] **Step 1: Write `scripts/dev-postgres.sh`** (`start` and `stop`). `PG_BIN=${PG_BIN:-/opt/homebrew/opt/postgresql@17/bin}`; data dir `.local/pgdata` relative to the repo root (`$(git rev-parse --show-toplevel)` is not reliable inside worktrees with x86 git, so use the script's own directory: `ROOT="$(cd "$(dirname "$0")/.." && pwd)"`); `start` runs `initdb -U postgres --auth=trust -E UTF8 --locale=en_US.UTF-8` only when the dir is missing, `pg_ctl -o "-p 54329 -k $ROOT/.local" -l $ROOT/.local/pg.log start`, waits with `pg_isready`, creates roles and database `evidencegraph` (owner `eg_owner`) idempotently, creates extensions `vector` and `citext` as `postgres`, prints both URLs. `stop` runs `pg_ctl stop`. `set -euo pipefail`. Run `bash scripts/dev-postgres.sh start` and verify `psql -h localhost -p 54329 -U eg_owner -d evidencegraph -Atc "select extname from pg_extension order by 1"` lists `citext`, `plpgsql`, `vector`.
- [ ] **Step 2: Write failing tests** in `tests/test_migrate.py`: `test_discover_orders_and_ignores_other_files` (no pg; tmp files `0002_b.sql`, `0001_a.sql`, `notes.txt` -> versions `[1, 2]`, names `["a", "b"]`); `test_discover_rejects_duplicate_versions` (`0001_a.sql`, `0001_b.sql` -> `MigrationError`); `@pytest.mark.pg test_migrate_applies_once` (two files creating tables `t1`, `t2` -> first call `[1, 2]`, second call `[]`, both tables exist); `@pytest.mark.pg test_changed_checksum_refuses` (apply, edit file 1, call again -> `MigrationChecksumError`); `@pytest.mark.pg test_failed_migration_rolls_back` (file 2 is `create table t2(id int); select from nope;` -> raises, `schema_migrations` holds only version 1, table `t2` does not exist); `test_settings_defaults` asserts the new `Settings` defaults above.
- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_migrate.py -q`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement** `backend/db/migrate.py`, the settings fields, `tests/pg_fixtures.py`, and the CLI command. Use `psycopg.AsyncConnection.connect(url, autocommit=True)` and `conn.transaction()` blocks.
- [ ] **Step 5: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 6: Commit** `scripts/dev-postgres.sh backend/db/migrate.py backend/db/migrations/.gitkeep backend/config.py backend/cli/main.py tests/pg_fixtures.py tests/conftest.py tests/test_migrate.py` with `feat: add migration runner and local Postgres setup`.

### Task 6: Global schema and the `Database` pool with tenant transactions

**Files:**
- Create: `backend/db/migrations/0001_global.sql`, `backend/db/pool.py`, `tests/test_schema_global.py`
- Modify: `tests/pg_fixtures.py`

**Interfaces:**
- Schema `0001_global.sql`: `create extension if not exists citext; create extension if not exists vector;` then the global tables of spec section 8 exactly: `tenants`, `users` (`email citext not null unique`), `memberships`, `sessions`, `api_keys`, `auth_throttle`, `jobs` (columns as in section 8, `payload jsonb not null default '{}'`, `status text not null default 'queued'`, `attempts int not null default 0`, `max_attempts int not null default 3`, `priority int not null default 0`, `run_after timestamptz not null default now()`, index `jobs_claim_idx on jobs (status, priority desc, run_after, id)`), plus `workers(id text primary key, kinds text[] not null, started_at timestamptz not null default now(), heartbeat_at timestamptz not null default now())` and `audit_log` as global per R10b (`tenant_id uuid null references tenants(id)`, index `(tenant_id, at desc, id desc)`). Grants: `grant usage on schema public to eg_app`; `grant select, insert, update, delete on all tables in schema public to eg_app`; `grant usage, select on all sequences in schema public to eg_app`; `alter default privileges for role eg_owner in schema public grant select, insert, update, delete on tables to eg_app`; same for sequences; `revoke insert, update, delete on schema_migrations from eg_app`.
- Ruling (deviates from spec section 18): roles are created outside migrations (dev script, test fixture, Docker init script) because roles are cluster-global; migrations only grant. Extensions are created by the superuser before migrations; the migration's `create extension if not exists` is then a no-op.
- `backend/db/pool.py`: `class Database` with `__init__(self, url: str, *, min_size: int = 1, max_size: int = 10)`, `async open()`, `async close()`, `tenant_tx(self, tenant_id: UUID) -> AsyncContextManager[AsyncConnection]` (pool connection, `conn.transaction()`, first statement `select set_config('app.tenant_id', %s, true)` with `str(tenant_id)`), `system_tx(self) -> AsyncContextManager[AsyncConnection]` (same with `''`). Pool: `psycopg_pool.AsyncConnectionPool(url, min_size=..., max_size=..., open=False, kwargs={"autocommit": True, "row_factory": dict_row})`.
- Fixtures added to `tests/pg_fixtures.py`: session-scoped `pg_urls` (creates database `eg_test_<8 hex>`, runs `migrate()` as `eg_owner`, yields `{"owner": url, "app": url}`, drops at session end); function-scoped `db` (opens `Database(pg_urls["app"], max_size=4)`, yields, closes, then as `eg_owner` truncates every table in `public` except `schema_migrations` with `restart identity cascade`). Session-scoped async fixtures need `@pytest_asyncio.fixture(scope="session", loop_scope="session")`; set `asyncio_default_fixture_loop_scope = "session"` and `asyncio_default_test_loop_scope = "session"` in `pyproject.toml` so pools and tests share one loop.

- [ ] **Step 1: Write failing tests** in `tests/test_schema_global.py` (all `pg`): `test_global_tables_exist` (the ten names above plus `schema_migrations` in `information_schema.tables`); `test_email_is_case_insensitive_unique` (insert `a@x.com`, then `A@X.COM` -> `psycopg.errors.UniqueViolation`); `test_job_status_check` (status `bogus` -> `CheckViolation`); `test_app_role_cannot_write_migrations` (as app: insert into `schema_migrations` -> `InsufficientPrivilege`); `test_app_role_has_no_bypassrls` (`select current_user, rolbypassrls from pg_roles where rolname = current_user` -> `("eg_app", False)`); `test_tenant_setting_does_not_leak_across_pooled_connections` (Review Focus 2: `Database(app_url, max_size=1)`; inside `tenant_tx(A)` `current_setting('app.tenant_id', true) == str(A)`; afterwards inside `system_tx()` it is `''` or `None`; inside `tenant_tx(B)` it is `str(B)`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_schema_global.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the migration, `pool.py`, and fixtures.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `backend/db/migrations/0001_global.sql backend/db/pool.py tests/pg_fixtures.py tests/test_schema_global.py pyproject.toml` (remove `backend/db/migrations/.gitkeep` with `git rm`) with `feat: add global schema and tenant-scoped transactions`.

### Task 7: Collections schema with forced row-level security

**Files:**
- Create: `backend/db/migrations/0002_collections.sql`, `tests/test_rls.py`

**Interfaces:**
- Tables: `collections(id uuid primary key default gen_random_uuid(), tenant_id uuid not null references tenants(id), name text not null, description text not null default '', kind text not null default 'normal' check (kind in ('normal','quick','benchmark')), created_by uuid references users(id), created_at timestamptz not null default now(), unique (tenant_id, name))`; `collection_grants(tenant_id uuid not null references tenants(id), collection_id uuid not null references collections(id) on delete cascade, user_id uuid not null references users(id), permission text not null check (permission in ('read','write','admin')), created_at timestamptz not null default now(), primary key (collection_id, user_id))`.
- RLS pattern (every tenant table in every later migration copies it): `alter table X enable row level security; alter table X force row level security; create policy tenant_isolation on X using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid) with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);`
- `tests/test_rls.py` defines `GLOBAL_TABLES = {"tenants", "users", "memberships", "sessions", "api_keys", "auth_throttle", "jobs", "workers", "audit_log", "schema_migrations"}`; every other table in `public` must have RLS enabled and forced. Later plans extend `GLOBAL_TABLES` only with a spec reason.

- [ ] **Step 1: Write failing tests** (all `pg`): `test_collection_visible_only_in_own_tenant` (create tenants A, B via `system_tx`; insert a collection in `tenant_tx(A)`; count is 1 in A and 0 in B); `test_no_tenant_context_sees_no_rows` (`system_tx` count is 0); `test_insert_for_other_tenant_rejected` (in `tenant_tx(A)` insert with `tenant_id = B` -> `psycopg.errors.InsufficientPrivilege`); `test_every_tenant_table_has_forced_rls` (query `pg_class` for `relkind = 'r'` in `public`; for names not in `GLOBAL_TABLES`, `relrowsecurity and relforcerowsecurity`; also assert at least `collections` and `collection_grants` were checked).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_rls.py -q`. Expected: FAIL (tables missing).
- [ ] **Step 3: Implement** `0002_collections.sql`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `backend/db/migrations/0002_collections.sql tests/test_rls.py` with `feat: add collections with forced row-level security`.

### Task 8: Password hashing and token primitives

**Files:**
- Create: `backend/auth/__init__.py`, `backend/auth/passwords.py`, `backend/auth/tokens.py`, `tests/test_passwords.py`, `tests/test_tokens.py`

**Interfaces:**
- `backend/auth/passwords.py`: `PASSWORD_MIN = 12`, `PASSWORD_MAX = 256`; `class WeakPassword(ValueError)`; `def validate_password(password: str) -> None`; `def hash_password(password: str) -> str` (scrypt `n=2**14, r=8, p=1, dklen=32`, 16-byte salt, format `scrypt$16384$8$1$<salt_b64>$<hash_b64>` with standard base64 without padding); `def verify_password(password: str, stored: str) -> bool` (constant-time compare; malformed `stored` returns False); `def verify_dummy(password: str) -> None` (runs a verify against a lazily computed module-level hash, for timing equalization). scrypt needs `maxmem=64 * 1024 * 1024`.
- `backend/auth/tokens.py`: `def new_session_token() -> str` and `def new_csrf_token() -> str` (`secrets.token_urlsafe(32)`); `def token_hash(token: str) -> bytes` (sha256 digest, 32 bytes); `def new_api_key() -> tuple[str, str]` returns `(token, prefix)` with `prefix = secrets.token_hex(4)` and `token = f"egk_{prefix}_{secrets.token_urlsafe(32)}"`; `def parse_api_key(token: str) -> str | None` returns the prefix when `token` matches `^egk_[0-9a-f]{8}_[A-Za-z0-9_-]{43}$`, else None.

- [ ] **Step 1: Write failing tests**: `test_hash_format_and_roundtrip` (starts with `scrypt$16384$8$1$`, `verify_password(pw, h)` True, wrong password False); `test_hashes_are_salted` (two hashes differ); `test_malformed_hash_is_false` (`verify_password("x" * 12, "garbage")` False, no exception); `test_password_length_bounds` (11 chars raises `WeakPassword`, 12 ok, 256 ok, 257 raises); `test_token_hash_is_sha256` (equals `hashlib.sha256(b"abc").digest()` for `"abc"`); `test_api_key_shape_and_parse` (token matches the regex, `parse_api_key(token) == prefix`); `test_parse_api_key_rejects_malformed` (`"egk_zz_short"`, `"Bearer x"`, `""` -> None); `test_session_tokens_unique` (100 tokens, 100 distinct).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_passwords.py tests/test_tokens.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the five files with `feat: add password hashing and token primitives`.

### Task 9: Structured JSON logging, metrics registry, worker metrics server

**Files:**
- Create: `backend/obs/__init__.py`, `backend/obs/logging.py`, `backend/obs/metrics.py`, `backend/obs/metrics_server.py`, `tests/test_obs_logging.py`, `tests/test_obs_metrics.py`

**Interfaces:**
- `backend/obs/logging.py`: context vars `request_id_var`, `tenant_id_var`, `user_id_var`, `job_id_var` (default None); `@contextmanager def bind(**values) -> Iterator[None]` (sets and resets the matching vars; unknown key raises `KeyError`); `ALLOWED_EXTRA: frozenset[str] = {"route", "method", "status", "duration_ms", "kind", "outcome", "error_code", "attempt", "worker_id", "n_passages", "n_claims", "n_reused", "n_new", "bytes", "version_number", "stage"}`; `class JsonFormatter(logging.Formatter)` emitting one JSON object per line with `ts` (ISO 8601 UTC), `level`, `logger`, `msg` (`record.getMessage()`), non-None context vars, and only `ALLOWED_EXTRA` attributes; on exceptions it adds `exc_type` (class name) and `exc_frames` (list of `"file:line in func"`), never the exception message; `def configure_logging(level: str = "INFO") -> None` installs the formatter on the root logger and on `uvicorn`, `uvicorn.access` (access logger disabled, we log requests ourselves).
- `backend/obs/metrics.py`: `class Counter(name, help, labelnames=())` with `.labels(**kw) -> child` and `.inc(amount: float = 1.0)`; `class Gauge` with `.labels(...)` and `.set(value)`; `class Histogram(name, help, labelnames=(), buckets=DEFAULT_BUCKETS)` with `.observe(value)`; `DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)`; `class Registry` with `register(metric)` and `render() -> str` in Prometheus text format 0.0.4 (`# HELP`, `# TYPE`, labels in declared order, values escaped for `\`, `"`, newline; histogram `_bucket{le=...}` cumulative including `+Inf`, `_sum`, `_count`); wrong or missing label names raise `ValueError`; all mutation is thread-safe (one `threading.Lock` per metric). Module-level `REGISTRY` and metrics: `HTTP_REQUESTS` (`http_requests_total`, labels route, method, status), `HTTP_DURATION` (`http_request_duration_seconds`, route, method), `JOBS` (`jobs_total`, kind, outcome), `JOB_DURATION` (`job_duration_seconds`, kind), `STAGE_SECONDS` (`pipeline_stage_seconds`, stage), `INGEST_PASSAGES` (`ingest_passages_total`, outcome), `JOBS_QUEUED` (gauge `jobs_queued`).
- `backend/obs/metrics_server.py`: `def start_metrics_server(host: str, port: int, token: str | None, registry: Registry = REGISTRY) -> ThreadingHTTPServer` serving `GET /metrics` in a daemon thread: 404 when `token` is None, 401 unless `Authorization: Bearer <token>` matches (compare with `hmac.compare_digest`), else 200 `text/plain; version=0.0.4`.

- [ ] **Step 1: Write failing tests**: `test_formatter_drops_unlisted_extra` (log `"analysis finished"` with `extra={"n_claims": 3, "question": "SECRET-Q"}` -> JSON has `n_claims == 3`, no `question` key, `"SECRET-Q"` absent from the line); `test_bound_context_appears` (inside `bind(request_id="r1")` the line has `request_id == "r1"`, after exit it is absent); `test_exception_message_not_logged` (`logger.exception("job failed")` on `ValueError("SECRET-DOC")` -> `exc_type == "ValueError"`, `"SECRET-DOC"` absent); `test_counter_render` (inc twice with `route="/x", method="GET", status="200"` -> render contains `http_requests_total{route="/x",method="GET",status="200"} 2.0`); `test_histogram_buckets_cumulative` (observe 0.003 and 0.2 -> `le="0.005"} 1.0`, `le="0.25"} 2.0`, `le="+Inf"} 2.0`, `_count 2.0`); `test_label_escaping` (value `a"b\nc` renders as `a\"b\nc` with a literal backslash-n); `test_unknown_label_raises`; `test_metrics_server_auth` (port 0: no token configured -> 404; wrong token -> 401; right token -> 200 and body contains `# TYPE`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_obs_logging.py tests/test_obs_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the six files with `feat: add structured logging and metrics registry`.

### Task 10: API skeleton: problem+json errors, request context, security headers, CORS allowlist, body limits

**Files:**
- Create: `backend/api/errors.py`, `backend/api/middleware.py`, `tests/test_api_skeleton.py`
- Modify: `backend/api/app.py`, `backend/api/startup.py`

**Interfaces:**
- `backend/api/errors.py`: `class ApiError(Exception)` with `__init__(self, status: int, code: str, title: str, detail: str | None = None, headers: dict[str, str] | None = None)`; `def problem_response(status: int, code: str, title: str, detail: str | None = None, headers=None) -> JSONResponse` (media type `application/problem+json`, body per Global Constraints); `def install_error_handlers(app: FastAPI) -> None` mapping `ApiError` to its problem; `RequestValidationError` to 422 `validation_error` with `detail` built only from `loc` and `msg` of each error (never `input`); Starlette `HTTPException` to its status with code from `{401: "unauthenticated", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}` else `http_error`; any other `Exception` to 500 `internal_error`, title `Internal server error`, logged with `logger.exception("unhandled error")`. Task 11 adds the `AuthError` mapping here.
- `backend/api/middleware.py` (pure ASGI classes): `RequestContextMiddleware` (request id from `X-Request-ID` when it matches `^[A-Za-z0-9._-]{1,64}$`, else `uuid4().hex`; binds `request_id_var`; sets response header `X-Request-ID`; logs `"request"` with route template (`scope["route"].path` when matched, else `"unmatched"`), method, status, duration_ms; updates `HTTP_REQUESTS` and `HTTP_DURATION`); `SecurityHeadersMiddleware` (always `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`; for paths starting `/api/` also `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`); `BodySizeLimitMiddleware(app, json_limit: int = 1_048_576, upload_limit: int = 20_971_520 + 65_536)` (`multipart/form-data` uses `upload_limit`; rejects early on `Content-Length` over the limit; otherwise counts streamed bytes in a wrapped `receive` and, when the count passes the limit before the response starts, returns 413 `payload_too_large`).
- `create_app(settings: Settings | None = None, lifespan=None) -> FastAPI`: `app.state.settings = settings or Settings()`; middleware order from outermost: RequestContext, SecurityHeaders, CORS (`allow_origins=settings.cors_origins`, `allow_credentials=True`, methods GET/POST/PUT/PATCH/DELETE, headers `Content-Type`, `X-CSRF-Token`, `X-Request-ID`, `Authorization`, `expose_headers=["X-Request-ID"]`), BodySizeLimit. Mount an `APIRouter` at `/api/v1` (`v1 = APIRouter()`, later tasks include their routers into it). Legacy `/health`, `/api/analyze`, `/api/analyses`, `/api/feedback` keep working unchanged (Plan C replaces them).
- `create_configured_app()` lifespan additionally calls `configure_logging(settings.log_level)`, opens `Database(settings.database_url)` into `app.state.db`, and closes it on shutdown.

- [ ] **Step 1: Write failing tests** in `tests/test_api_skeleton.py` (no Postgres). Fixture builds `create_app(Settings(cors_origins=["http://localhost:5173"]))` and adds test routes: `GET /api/v1/_apierr` raising `ApiError(409, "conflict_x", "Conflict")`, `GET /api/v1/_boom` raising `RuntimeError("SECRET-BOOM")`, `POST /api/v1/_typed` taking a Pydantic body `{n: int}`, `POST /api/v1/_echo` returning `{"len": len(await request.body())}`. Tests: `test_problem_json_shape` (exact body `{"type": "/errors/conflict_x", "title": "Conflict", "status": 409, "detail": None, "code": "conflict_x"}`, content type `application/problem+json`); `test_unhandled_error_is_generic` (500, code `internal_error`, `"SECRET-BOOM"` not in the body and not in captured log output); `test_validation_error_hides_input` (`{"n": "SECRET-INPUT"}` -> 422 `validation_error`, `"SECRET-INPUT"` not in body); `test_request_id_echo_and_replace` (`abc-1` echoed; `bad id!` replaced by 32 hex chars); `test_security_headers_on_api`; `test_cors_allows_listed_origin_only` (preflight from `http://localhost:5173` gets that `access-control-allow-origin` and `access-control-allow-credentials: true`; `http://evil.example` gets no `access-control-allow-origin`); `test_body_limit_by_content_length` (1,048,577-byte JSON body to `_echo` -> 413 `payload_too_large`); `test_body_limit_chunked` (Review Focus 5: body from an async generator yielding 64 KB chunks totalling 2 MB, no `Content-Length` -> 413); `test_small_body_passes` (`_echo` with 10 bytes -> `{"len": 10}`); `test_legacy_health_ok`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_api_skeleton.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** errors, middleware, `create_app`, lifespan changes.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS (legacy `tests/test_api.py` still green).
- [ ] **Step 5: Commit** `backend/api/errors.py backend/api/middleware.py backend/api/app.py backend/api/startup.py tests/test_api_skeleton.py` with `feat: add problem errors, request context, security headers, body limits`.

### Task 11: Auth service, global repositories, audit log

**Files:**
- Create: `backend/db/repos/__init__.py`, `backend/db/repos/users.py`, `backend/db/repos/tenants.py`, `backend/db/repos/sessions.py`, `backend/db/repos/api_keys.py`, `backend/db/repos/audit.py`, `backend/auth/service.py`, `tests/test_auth_service.py`
- Modify: `backend/api/errors.py` (map `AuthError`)

**Interfaces:**
- Repos take an open `AsyncConnection` (from `system_tx`) as first argument and return `dict` rows: `users.insert_user(conn, email, password_hash, display_name)`, `users.get_user_by_email(conn, email)`, `users.get_user(conn, user_id)`; `tenants.insert_tenant(conn, name)`, `tenants.add_membership(conn, tenant_id, user_id, role)`, `tenants.get_membership(conn, tenant_id, user_id)`, `tenants.list_memberships(conn, user_id)` (with tenant name, oldest first); `sessions.insert_session(conn, token_hash, csrf_hash, user_id, tenant_id, expires_at)`, `sessions.get_active_session(conn, token_hash, now)` (not revoked, not expired, user not disabled), `sessions.revoke_session(conn, token_hash)`, `sessions.set_session_tenant(conn, session_id, tenant_id)`; `api_keys.insert_api_key(conn, tenant_id, user_id, name, prefix, key_hash)`, `api_keys.get_active_by_hash(conn, key_hash)`, `api_keys.list_for_user(conn, tenant_id, user_id)`, `api_keys.revoke(conn, key_id, tenant_id)`, `api_keys.touch(conn, key_id)`; `audit.record(conn, *, action: str, tenant_id: UUID | None, user_id: UUID | None, target_type: str | None = None, target_id: str | None = None, details: dict | None = None) -> None` (request id taken from `request_id_var`).
- `backend/auth/service.py`:

```python
@dataclass(frozen=True)
class Principal:
    user_id: UUID; email: str; display_name: str
    tenant_id: UUID; role: Literal["owner", "admin", "member"]
    auth: Literal["session", "api_key"]
    session_id: UUID | None = None; csrf_hash: bytes | None = None; api_key_id: UUID | None = None

@dataclass(frozen=True)
class LoginResult:
    session_token: str; csrf_token: str; principal: Principal

class AuthError(Exception):
    def __init__(self, code: str, status: int): ...
# codes: invalid_credentials 401, too_many_attempts 429, email_taken 409, weak_password 422,
# invalid_email 422, registration_disabled 403, not_a_member 404, api_key_tenant_fixed 403, not_found 404

class AuthService:
    def __init__(self, db: Database, settings: Settings, *, clock: Callable[[], datetime] = utcnow): ...
    async def register(self, email: str, password: str, display_name: str) -> tuple[UUID, UUID]  # (user_id, tenant_id)
    async def login(self, email: str, password: str, *, ip: str) -> LoginResult
    async def logout(self, session_token: str) -> None
    async def resolve_session(self, session_token: str) -> Principal | None
    async def resolve_api_key(self, token: str) -> Principal | None
    async def switch_tenant(self, principal: Principal, tenant_id: UUID) -> Principal
    async def create_api_key(self, principal: Principal, name: str) -> tuple[dict, str]  # (info, token)
    async def list_api_keys(self, principal: Principal) -> list[dict]
    async def revoke_api_key(self, principal: Principal, key_id: UUID) -> None
```

- Rules: emails stored `strip().lower()`, valid when they match `^[^@\s]{1,64}@[^@\s]{1,255}$`; display name 1 to 100 chars after strip. `register` creates the user, a tenant named `"<display_name>'s workspace"`, an owner membership, and audit `register`. Throttle: window start = now floored to 15 minutes; keys `email:<normalized>` and `ip:<ip>`; before checking credentials, if the email key has 5 or more failures or the ip key 20 or more, record audit `login_throttled` and raise `too_many_attempts`; unknown email runs `verify_dummy`; any failure upserts both keys `failures + 1`, records `login_failed`, and raises `invalid_credentials`; success deletes the email key for the window, picks the oldest membership as the session tenant, stores only `token_hash(session)` and `token_hash(csrf)`, `expires_at = now + session_ttl_days`, and records `login`. A session or API key resolves to None when its membership in its tenant no longer exists. API key principals cannot switch tenants (`api_key_tenant_fixed`). Members revoke only their own keys; owners and admins revoke any key in the tenant; otherwise `not_found`. Audit `details` never contain passwords or tokens.

- [ ] **Step 1: Write failing tests** in `tests/test_auth_service.py` (all `pg`, a fake clock): `test_register_creates_owner_tenant`; `test_email_normalized_and_unique` (Review Focus 3: register `" Alice@Example.com "`, login as `alice@example.com` succeeds; registering `ALICE@example.com` raises `AuthError` `email_taken`); `test_only_token_hashes_stored` (session row `token_hash == token_hash(result.session_token)` and `csrf_hash == token_hash(result.csrf_token)`); `test_wrong_password_and_unknown_email_same_error` (both `invalid_credentials`, status 401); `test_email_throttle_and_window_reset` (5 wrong attempts, then the correct password raises `too_many_attempts`; advance the clock 16 minutes, correct password succeeds); `test_ip_throttle_across_emails` (20 failures over different emails from one ip, 21st attempt `too_many_attempts`); `test_logout_revokes`; `test_expired_session_rejected` (clock past 7 days -> `resolve_session` None); `test_removed_membership_invalidates_session`; `test_api_key_resolve_touch_and_revoke` (resolves to `auth == "api_key"`, `last_used_at` set; after revoke None); `test_member_cannot_revoke_others_key`; `test_audit_rows_have_no_secrets` (actions `register`, `login_failed`, `login` present; no audit `details` text contains the password or session token).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_auth_service.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** repos, service, and the `AuthError` handler (problem with the error's code and status; titles from a code-to-title dict in `errors.py`).
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the new and modified files with `feat: add auth service with sessions, throttling, and audit log`.

### Task 12: Auth dependencies, CSRF, and auth routes

**Files:**
- Create: `backend/auth/deps.py`, `backend/api/schemas_v1.py`, `backend/api/routes/auth.py`, `tests/helpers/__init__.py`, `tests/helpers/auth.py`, `tests/test_auth_routes.py`
- Modify: `backend/api/app.py` (include router), `backend/api/startup.py` (`app.state.auth = AuthService(db, settings)`)

**Interfaces:**
- `backend/auth/deps.py`: `def get_db(request: Request) -> Database` (`request.app.state.db`); `def get_auth(request: Request) -> AuthService`; `async def current_principal(request: Request) -> Principal` (an `Authorization: Bearer egk_...` header resolves an API key and skips CSRF; otherwise cookie `eg_session`; no credential or unresolvable credential raises `ApiError(401, "unauthenticated", "Authentication required")`; session principals on methods other than GET, HEAD, OPTIONS need `X-CSRF-Token` whose `token_hash` equals `csrf_hash` via `hmac.compare_digest`, else `ApiError(403, "csrf_failed", "CSRF check failed")`; binds `tenant_id_var` and `user_id_var`); `def require_role(*roles: str) -> Callable` (dependency returning the principal or raising 403 `forbidden`).
- `backend/api/schemas_v1.py`: `RegisterIn(email: str, password: str, display_name: str)`, `LoginIn(email: str, password: str)`, `UserOut(id: UUID, email: str, display_name: str)`, `TenantOut(id: UUID, name: str, role: str)`, `SessionOut(user: UserOut, tenant: TenantOut, csrf_token: str)`, `MeOut(user: UserOut, tenant: TenantOut, tenants: list[TenantOut])`, `SwitchTenantIn(tenant_id: UUID)`. Later tasks add their schemas to this module.
- Routes under `/api/v1/auth`: `POST /register` (403 `registration_disabled` when `allow_registration` is false; registers, then logs in; 201 `SessionOut` and cookies); `POST /login` (200 `SessionOut` and cookies; client ip from `request.client.host`); `POST /logout` (204, revokes, deletes both cookies); `GET /me` (`MeOut`); `POST /switch-tenant` (`MeOut`). Cookie attributes per Global Constraints.
- `tests/helpers/auth.py`: fixture-friendly `async def signup(client, email="alice@example.com", password="correct horse battery", display_name="Alice") -> dict` (returns the `SessionOut` JSON; the client keeps cookies) and `def csrf(session: dict) -> dict[str, str]` returning `{"X-CSRF-Token": session["csrf_token"]}`. `tests/pg_fixtures.py` gains `api_settings` (`Settings(dev_insecure_cookies=True, cors_origins=["http://localhost:5173"])`), `api_app` (`create_app(api_settings)` with `state.db = db`, `state.auth = AuthService(db, api_settings)`), and `client` (`httpx.AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test")`). A second independent client fixture `client2` exists for two-user tests.

- [ ] **Step 1: Write failing tests** in `tests/test_auth_routes.py` (all `pg`): `test_register_sets_session_and_me_works`; `test_login_wrong_password_problem` (401, code `invalid_credentials`); `test_unsafe_request_without_csrf_rejected` (`POST /auth/switch-tenant` without header -> 403 `csrf_failed`; with header -> 200); `test_logout_then_me_unauthenticated`; `test_cookie_flags_dev_insecure` (Review Focus 1: `Set-Cookie` for `eg_session` has `HttpOnly`, `SameSite=lax` (case-insensitive), `Path=/`, no `Secure`; `eg_csrf` has no `HttpOnly`); `test_cookie_flags_secure_by_default` (app built with `dev_insecure_cookies=False`: both cookies have `Secure`); `test_registration_disabled`; `test_switch_to_non_member_tenant_404` (code `not_a_member`); `test_bearer_api_key_needs_no_csrf` (key created through `AuthService.create_api_key`; a fresh client without cookies calls `POST /auth/switch-tenant` to its own tenant with `Authorization: Bearer <token>` and no CSRF -> 403 `api_key_tenant_fixed`, proving it passed the CSRF stage); `test_login_throttled_route` (6th attempt -> 429 `too_many_attempts`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_auth_routes.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** deps, schemas, routes, fixtures.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add auth routes with session cookies and CSRF`.

### Task 13: API key routes

**Files:**
- Create: `backend/api/routes/api_keys.py`, `tests/test_api_keys.py`
- Modify: `backend/api/schemas_v1.py`, `backend/api/app.py`

**Interfaces:**
- Schemas: `ApiKeyCreateIn(name: str)` (1 to 100 chars), `ApiKeyOut(id: UUID, name: str, prefix: str, created_at: datetime, last_used_at: datetime | None)`, `ApiKeyCreatedOut(ApiKeyOut + token: str)`.
- Routes under `/api/v1/api-keys`: `GET` (the caller's active keys in the active tenant, `list[ApiKeyOut]`), `POST` (201 `ApiKeyCreatedOut`; audit `api_key_created`), `DELETE /{key_id}` (204; audit `api_key_revoked`; unknown or not permitted -> 404 `not_found`).

- [ ] **Step 1: Write failing tests** (all `pg`): `test_create_returns_token_once` (POST has `token`; GET items have no `token` key); `test_bearer_key_authenticates_me`; `test_revoked_key_rejected` (401 after DELETE); `test_member_cannot_revoke_other_users_key` (owner adds a second user as member through the repos, member's DELETE of the owner's key -> 404, key still works); `test_key_name_validation` (empty name -> 422 `validation_error`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_api_keys.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the router and schemas.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add API key management routes`.

### Task 14: Authorization policy, pagination, collections and grants routes

**Files:**
- Create: `backend/authz/__init__.py`, `backend/authz/policy.py`, `backend/api/pagination.py`, `backend/db/repos/collections.py`, `backend/api/routes/collections.py`, `tests/test_collections.py`
- Modify: `backend/api/schemas_v1.py`, `backend/api/app.py`

**Interfaces:**
- `backend/authz/policy.py` (connections come from `tenant_tx(principal.tenant_id)`): `PERMISSION_RANK = {"read": 1, "write": 2, "admin": 3}`; `async def effective_permission(conn, principal: Principal, collection_id: UUID) -> str | None` (tenant owner or admin -> `"admin"` when the collection exists; else the grant's permission; else None); `async def require_collection(conn, principal, collection_id, needed: Literal["read", "write", "admin"]) -> dict` (no row or no permission -> `ApiError(404, "collection_not_found", ...)`; permission below `needed` -> `ApiError(403, "insufficient_permission", ...)`); `async def readable_collection_ids(conn, principal, requested: Sequence[UUID] | None, *, kinds: Sequence[str] = ("normal",)) -> tuple[UUID, ...]` (None -> every readable collection of those kinds, ordered by created_at; any requested id that is missing or unreadable -> 404 `collection_not_found`). Plans B and C use these signatures unchanged.
- `backend/api/pagination.py`: `def encode_cursor(ts: datetime, key: str) -> str` (urlsafe base64 of JSON `[ts.isoformat(), key]`); `def decode_cursor(cursor: str) -> tuple[datetime, str]` (anything malformed -> `ApiError(400, "bad_cursor", ...)`); dependency `page_params(limit: int = Query(20, ge=1, le=100), cursor: str | None = None) -> PageParams(limit, after: tuple[datetime, str] | None)`; generic `Page[T]` schema `{items: list[T], next_cursor: str | None}`. Ordering everywhere: `created_at desc, id desc`, cursor condition `(created_at, id) < (%s, %s)`.
- Schemas: `CollectionIn(name: str 1..100, description: str = "" max 1000)`, `CollectionPatch(name: str | None, description: str | None)`, `CollectionOut(id, name, description, kind, created_at, permission)`, `GrantIn(user_id: UUID, permission: Literal["read", "write", "admin"])`, `GrantsIn(grants: list[GrantIn])` (max 500), `GrantOut(user_id, email, display_name, permission)`.
- Routes under `/api/v1/collections`: `GET` (`Page[CollectionOut]`, readable `normal` collections only); `POST` (any member; 201; creator gets an `admin` grant; duplicate name -> 409 `collection_name_taken`; audit `collection_created`); `GET /{id}` (read); `PATCH /{id}` (admin; audit `collection_updated`); `DELETE /{id}` (admin, 204; audit `collection_deleted`; Plan B adds the non-empty check); `GET /{id}/grants` (admin, `list[GrantOut]`); `PUT /{id}/grants` (admin; replaces all explicit grants; every `user_id` must be a tenant member else 422 `not_a_member`; audit `grants_changed` with counts only).

- [ ] **Step 1: Write failing tests** in `tests/test_collections.py` (all `pg`, owner Alice and member Bob in one tenant, Carol in another tenant): `test_owner_creates_and_lists`; `test_member_without_grant_cannot_see` (Bob: GET 404 `collection_not_found`, absent from list); `test_read_grant_allows_read_not_write` (after `PUT grants` read for Bob: GET 200, PATCH 403 `insufficient_permission`); `test_cross_tenant_is_404` (Carol GET, PATCH, DELETE, grants on Alice's collection all 404); `test_duplicate_name_409`; `test_grant_to_non_member_422`; `test_pagination_walks_all_pages` (25 collections, `limit=10` -> pages of 10, 10, 5, no duplicates, last `next_cursor` null); `test_bad_cursor_400`; `test_readable_collection_ids_rejects_unreadable` (policy-level test: Bob requesting an ungranted id raises `ApiError` 404; `None` returns only granted ids; `quick` kind collections excluded by default).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_collections.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** policy, pagination, repo, routes, schemas.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add collection authorization and collection routes`.

### Task 15: Tenant members and audit routes

**Files:**
- Create: `backend/db/repos/members.py`, `backend/api/routes/members.py`, `tests/test_members.py`
- Modify: `backend/api/schemas_v1.py`, `backend/api/app.py`

**Interfaces:**
- Schemas: `MemberOut(user_id, email, display_name, role, created_at)`, `MemberAddIn(email: str, role: Literal["owner", "admin", "member"])`, `AuditOut(id: int, action: str, user_id: UUID | None, target_type: str | None, target_id: str | None, at: datetime, details: dict | None)`.
- Routes under `/api/v1/tenants/current`: `GET /members` (any member); `POST /members` (owner or admin; only owners may add `owner`; unknown email -> 404 `user_not_found`; existing member -> 409 `already_member`; 201 `MemberOut`; audit `member_added`); `DELETE /members/{user_id}` (owner or admin; admins cannot remove owners -> 403 `insufficient_permission`; removing the last owner -> 409 `last_owner`; also deletes the user's `collection_grants` in the tenant; 204; audit `member_removed`); `GET /audit` (owner or admin, `Page[AuditOut]` ordered `at desc, id desc`, only this tenant's rows).

- [ ] **Step 1: Write failing tests** (all `pg`): `test_owner_adds_and_lists_member`; `test_admin_cannot_add_owner` (403); `test_member_cannot_add` (403 `forbidden`); `test_unknown_email_404`; `test_already_member_409`; `test_last_owner_cannot_be_removed` (409 `last_owner`); `test_removed_member_loses_session_and_grants` (Bob's existing session gets 401 on `/auth/me`, his grants are gone); `test_audit_lists_only_own_tenant` (Carol's tenant actions absent; includes `member_added`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_members.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add tenant member management and audit routes`.

### Task 16: Postgres job queue

**Files:**
- Create: `backend/jobs/__init__.py`, `backend/jobs/queue.py`, `tests/test_jobs_queue.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Job:
    id: int; tenant_id: UUID | None; kind: str; payload: dict
    priority: int; attempts: int; max_attempts: int; request_id: str | None

async def enqueue(conn, kind: str, payload: Mapping[str, Any], *, tenant_id: UUID | None,
                  priority: int = 0, idempotency_key: str | None = None, request_id: str | None = None,
                  max_attempts: int = 3, run_after: datetime | None = None) -> int
async def claim(conn, worker_id: str, kinds: Sequence[str]) -> Job | None
async def heartbeat(conn, job_id: int, worker_id: str) -> bool
async def complete(conn, job_id: int, worker_id: str) -> bool
async def fail(conn, job_id: int, worker_id: str, error_code: str, *, permanent: bool = False) -> str | None
async def release(conn, job_id: int, worker_id: str) -> bool
async def requeue_stale(conn, stale_after_s: float = 120.0) -> int
async def count_queued(conn) -> int
```

- Rules: `enqueue` works on any connection (including inside a caller's `tenant_tx`, which makes it a transactional outbox); payload values must be `None`, `bool`, `int`, `float`, `str` of at most 100 characters, or lists of those (max 1,000 items); anything else raises `ValueError` (spec R4: ids and counts only). Same `idempotency_key` returns the existing job id. `claim` is one statement: `update jobs set status='running', locked_by=%s, locked_at=now(), heartbeat_at=now(), attempts=attempts+1, updated_at=now() where id = (select id from jobs where status='queued' and run_after <= now() and kind = any(%s) order by priority desc, id for update skip locked limit 1) returning *`. `fail` with `permanent` sets `failed`; otherwise `queued` with `run_after = now() + 5s * 4^(attempts-1)` while `attempts < max_attempts`, else `dead`; `last_error` stores only `error_code`, which must match `^[a-z0-9_]{1,64}$`. `requeue_stale` moves `running` jobs with `heartbeat_at < now() - stale_after_s` to `queued` (run now) or to `dead` with `last_error = 'lease_expired'` when attempts are exhausted. `release` sets `queued`, `attempts - 1`, clears the lock. Mutations only succeed when `locked_by` matches and status is `running` (return False / None otherwise).

- [ ] **Step 1: Write failing tests** (all `pg`): `test_enqueue_in_rolled_back_tx_leaves_no_job`; `test_idempotency_key_returns_existing_id`; `test_payload_rejects_text_and_nesting` (`{"text": "x" * 101}` and `{"a": {"b": 1}}` raise `ValueError`); `test_claim_priority_then_fifo`; `test_claim_respects_run_after_and_kinds`; `test_concurrent_claims_never_duplicate` (Review Focus 4: 20 jobs, 5 claimers on separate connections via `asyncio.gather`, each looping until None; 20 distinct ids total); `test_retry_backoff_then_dead` (attempt 1 fail -> `queued`, `run_after - now` within 4 to 6 s; attempt 2 -> 19 to 21 s; attempt 3 -> `dead`); `test_permanent_failure` (`failed`, `last_error == "bad_file"`); `test_invalid_error_code_rejected` (`"Has Spaces"` -> `ValueError`); `test_requeue_stale_counts_attempt` (Review Focus 4: claimed job with `heartbeat_at` moved back 121 s -> `requeue_stale` returns 1, status `queued`, attempts 1; on a job at max attempts -> `dead`, `lease_expired`); `test_heartbeat_wrong_worker_false`; `test_release_restores_attempt`; `test_count_queued`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_jobs_queue.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `backend/jobs/queue.py`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add Postgres job queue with leases and retries`.

### Task 17: Worker loop, handler registry, `halluciscope worker`

**Files:**
- Create: `backend/jobs/worker.py`, `backend/jobs/registry.py`, `tests/test_worker.py`
- Modify: `backend/cli/main.py`

**Interfaces:**

```python
class PermanentJobError(Exception):
    def __init__(self, code: str): ...
class RetryableJobError(Exception):
    def __init__(self, code: str): ...

@dataclass(frozen=True)
class JobContext:
    job: Job; db: Database; worker_id: str; services: Any

Handler = Callable[[JobContext], Awaitable[None]]

class Worker:
    def __init__(self, db: Database, handlers: Mapping[str, Handler], *, services: Any = None,
                 worker_id: str | None = None, poll_interval_s: float = 1.0, heartbeat_s: float = 15.0,
                 stale_after_s: float = 120.0, timeouts_s: Mapping[str, float] | None = None,
                 default_timeout_s: float = 300.0, shutdown_grace_s: float = 30.0): ...
    async def run_once(self) -> bool   # claims at most one job of a registered kind; True if one ran
    async def run(self, stop: asyncio.Event) -> None
```

- `backend/jobs/registry.py`: `def build_handlers(services: Any) -> dict[str, Handler]` (empty in Plan A; Plans B and C register `ingest_upload`, `search`, `run_analysis`, `reverify_document`).
- Behavior: default `worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"`. Each queue call uses its own `db.system_tx()`. While a handler runs, a background task calls `heartbeat` every `heartbeat_s`. Outcomes: success -> `complete`, `JOBS{kind, outcome="succeeded"}`; `PermanentJobError` -> permanent `fail`, outcome `failed`; `RetryableJobError(code)`, `asyncio.TimeoutError` (code `job_timeout`), or any other exception (code `internal_error`, logged with `logger.exception("job handler error")`) -> retry `fail`, outcome `retry` or `dead` from the returned status. `JOB_DURATION{kind}` observed. Log lines bind `job_id` and carry `kind`, `attempt`, `outcome`. `run()` upserts a `workers` row (`kinds` = handler names), refreshes `heartbeat_at` every `heartbeat_s`, calls `requeue_stale` every 30 s, sleeps `poll_interval_s` when idle (wakes early on `stop`), and on `stop` gives a running job `shutdown_grace_s` to finish, then cancels it and calls `release`; the `workers` row is deleted on exit.
- CLI `halluciscope worker`: `configure_logging`, opens `Database(settings.database_url)`, starts `start_metrics_server("127.0.0.1", settings.worker_metrics_port, settings.metrics_token)`, runs `Worker(db, build_handlers(services)).run(stop)` with `SIGTERM` and `SIGINT` wired to `stop` via `loop.add_signal_handler`. `services` is None in Plan A; Plan B introduces the services container.

- [ ] **Step 1: Write failing tests** (all `pg`, small intervals): `test_run_once_success` (handler records `ctx.job.payload`; job `succeeded`; `run_once` returns True, then False when empty); `test_permanent_error_marks_failed`; `test_unexpected_error_retries_without_leaking_message` (handler raises `ValueError("SECRET-DOC")` -> `queued`, `last_error == "internal_error"`, `"SECRET-DOC"` absent from captured logs); `test_timeout_retries` (`timeouts_s={"slow": 0.05}`, handler sleeps 1 s -> `last_error == "job_timeout"`); `test_heartbeat_advances_during_job` (`heartbeat_s=0.05`, handler sleeps 0.3 s; observed `heartbeat_at` greater than `locked_at`); `test_unregistered_kind_not_claimed`; `test_graceful_stop_releases_job` (handler waits forever, `shutdown_grace_s=0.1`; after `stop` the job is `queued` with attempts 0 and `run()` returns); `test_worker_row_lifecycle` (row exists during `run()`, gone after); `test_jobs_metric_incremented`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_worker.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** worker, registry, CLI command.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` and `.venv/bin/halluciscope worker --help` (exit 0). Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add background worker with leases and graceful shutdown`.

### Task 18: Health, readiness, and the metrics endpoint

**Files:**
- Create: `backend/api/routes/health.py`, `tests/test_health_metrics.py`
- Modify: `backend/db/migrate.py` (add `async def applied_versions(conn) -> set[int]`), `backend/api/app.py`, `backend/api/startup.py` (`app.state.http = httpx.AsyncClient(timeout=120.0)`, closed on shutdown)

**Interfaces:**
- `GET /api/v1/health/live` -> 200 `{"status": "ok"}`.
- `GET /api/v1/health/ready` -> `{"status": "ok" | "unavailable", "db": "ok" | "error", "migrations": "ok" | "pending", "worker": "ok" | "stale", "llm": "ok" | "unavailable"}`; 200 when db and migrations are ok, else 503. Migrations are ok when every `discover()` version is in `applied_versions` (read as `eg_app`). Worker is ok when `max(workers.heartbeat_at)` is within 60 s. LLM is ok when `GET {ollama_base_url}/api/version` through `app.state.http` returns 200 within 2 s. No authentication; no error text in the body.
- `GET /api/v1/metrics`: 404 `not_found` when `metrics_token` is unset; 401 `unauthenticated` unless `Authorization: Bearer <metrics_token>` (constant-time compare); else 200 `text/plain; version=0.0.4` with `REGISTRY.render()` after setting `JOBS_QUEUED` from `count_queued`.

- [ ] **Step 1: Write failing tests** (all `pg`; `app.state.http` is an `httpx.AsyncClient(transport=httpx.MockTransport(...))`): `test_live`; `test_ready_all_ok` (fresh `workers` row, mock LLM 200 -> 200 and all fields ok); `test_ready_reports_stale_worker_and_llm_down` (no workers row, mock transport raises `httpx.ConnectError` -> 200, `worker == "stale"`, `llm == "unavailable"`); `test_ready_pending_migrations_503` (monkeypatch `discover` to add version 9999 -> 503, `migrations == "pending"`); `test_metrics_hidden_without_token` (404); `test_metrics_requires_token` (wrong -> 401); `test_metrics_reports_queue_depth` (3 enqueued jobs -> body has `jobs_queued 3.0` and `http_requests_total`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_health_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run the plan's full gate:** `.venv/bin/ruff check .` (`All checks passed!`) and `.venv/bin/python -m pytest -q` (all pass, Postgres running) and `.venv/bin/halluciscope migrate` against the dev database (prints applied versions `[1, 2]` or `up to date`).
- [ ] **Step 5: Commit** the files with `feat: add health, readiness, and metrics endpoints`.
