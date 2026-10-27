# SDD ledger — plan: docs/superpowers/plans/2026-10-26-evidencegraph-a-foundation.md

Spec: docs/superpowers/specs/2026-10-26-evidencegraph-design.md (reachable; R1-R10 override earlier sections).
Branch: feature/evidencegraph. Start commit: 697f8d5. Git: PATH=/usr/bin:$PATH.
Models: implementers claude-sonnet-5.5 (stage suffix -sonnet5.5); reviewers and all other subagents claude-opus-5.5 (suffix -opus5.5). Omar's instruction 2026-10-26 14:59.
Postgres: bash scripts/dev-postgres.sh start (after Task 5); before that, cluster at .local/pgdata port 54329 via pg_ctl.

## Pre-flight scan

| Rows | Produces / consumes | Finding |
|---|---|---|
| T1 self | numpy retrieval; pins deps; evaluation/benchmarks.py sklearn removal | Consistent. T1 also adds pytest markers + addopts; `pg` marker defined before use (T5). |
| T1 -> T4 | T4 lint-fixes files T1 touched | No conflict (T4 runs after). |
| T2 self | tests use new keyword signature `_compute_hallucination_score(contradiction=, neutral=, entailment=)` | Consistent. |
| T3 self | chat_completion in generator; decomposer patched at backend.core.decomposer.chat_completion | Consistent; test_generator/test_consistency must stay green. |
| T5 -> T6 | T5 fixture fresh_db_url, ensure_roles, create_database; T6 adds pg_urls/db fixtures + loop scope settings | T5 tests are async pg tests; they need the session loop scope T6 sets. Ruling R-A1 below. |
| T5 self | Settings defaults test lists new fields | Consistent. |
| T6 -> T7 | Database.tenant_tx/system_tx; GLOBAL_TABLES in T7 includes `workers` created in T6 | Consistent. |
| T6 self | removes migrations/.gitkeep via git rm; ruling roles outside migrations | Consistent. |
| T8 self | primitives only | Consistent. |
| T9 -> T10, T17 | REGISTRY, HTTP_* metrics used by middleware; JOBS/JOB_DURATION by worker; start_metrics_server by CLI worker | Consistent. |
| T10 -> T11 | errors.py gets AuthError mapping in T11 | Consistent. |
| T10 self | create_app(settings) signature change; legacy tests/test_api.py calls create_app() with no args | Default `settings=None` keeps legacy call valid. Consistent. |
| T11 -> T12 | AuthService, Principal, repos; T12 deps/routes/fixtures client, client2, api_app | Consistent. |
| T11 self | throttle "5 or more failures" then the 6th attempt rejected; T12 test "6th attempt -> 429" | Consistent (5 failures recorded, check before 6th). |
| T12 -> T13, T14, T15 | current_principal, require_role, schemas_v1, helpers signup/csrf | Consistent. |
| T14 -> Plan B/C | readable_collection_ids(conn, principal, requested, *, kinds) | Consistent with Plan B/C signatures. |
| T15 self | removing member deletes grants; existing session 401 because resolve_session checks membership (T11) | Consistent. |
| T16 -> T17 | queue functions; Job dataclass | Consistent. |
| T16 self | payload str max 100 chars; idempotency_key text | Consistent. |
| T17 self | release sets attempts-1 (T16) so graceful stop leaves attempts 0 | Consistent. |
| T18 self | applied_versions added to migrate.py; readiness reads as eg_app (schema_migrations select granted, writes revoked in T6) | Consistent. |
| Plan-mandated defects | none found (no assertion-free tests, no mandated duplication) | Clean. |

Rulings:
- R-A1: Task 5's async pg tests (test_migrate) run before Task 6 introduces session loop-scope settings. Ruling: Task 5 sets `asyncio_default_fixture_loop_scope = "session"` and `asyncio_default_test_loop_scope = "session"` in pyproject.toml itself (moved earlier from Task 6); Task 6 keeps them. Why: function-scoped fixtures work either way, and setting it once early avoids a mid-plan behavior change. Cost if wrong: one config line moves.
- R-A2: `scripts/dev-postgres.sh` must work with the existing cluster at .local/pgdata (created during feasibility, has a scratch db `feas`). Ruling: the script reuses an existing data dir; the `feas` database is harmless scratch and may be dropped by the controller. Cost if wrong: none.

## Progress
- Task 1: implemented ba21f41 (BASE 697f8d5), awaiting review
- Task 1: complete (commits 697f8d5..ba21f41, review clean; 5 Minor parked: tie case untested in test_retrieve_orders_by_cosine; evaluation/benchmarks.py metrics untested and zip not strict (module deleted in Plan C Task 19); README FAISS row (README rewritten in Plan D Task 16); em dash in legacy chunker docstring (module deleted in Plan C Task 19))
- Task 2: complete (commits ba21f41..80f4a98, review clean; 5 Minor parked: permuted/str-key id2label test, match= on ValueError, premise positional check, loop-hoist, only_first raises on >510-token claims. Legacy verifier is replaced by backend/verify/nli.py in Plan C Task 4, whose brief requires string-key tests; carry the permuted-map and only_first notes there)
- Ruling: Task 3 keeps the pre-existing em dash in the legacy OllamaError timeout message (byte-for-byte instruction) because backend/core/generator.py is deleted in Plan C Task 19 and the replacement OllamaClient (Plan C Task 1) must be em-dash free. Cost if wrong: one string edit.
- Task 3: complete (commits 80f4a98..9cb6422, review clean; Minor parked: temperature/pass-through args not asserted, positional call coupling, ConnectError without from exc (legacy module removed in Plan C Task 19))
- Task 4: complete (commits 9cb6422..9eab912, review clean; Minor parked: conftest blank first line, write_json payload untyped, dataset parsed before mkdir (disclosed), compute_metrics now raises on length mismatch. EnsembleScorer zip strict=False kept (legacy, removed Plan C Task 19))
- Task 5: minor (deferred): pending() runs DDL (make it read-only with to_regclass); TOCTOU between discover bytes and apply-time re-read; _connect maps every OperationalError to "not reachable"; changed-checksum test does not prove nothing else applied; settings defaults test sensitive to HALLUCISCOPE_* env; pg_isready loop has no timeout
- Task 5: fix round 1/5 (3 addressed, 0 open; commits e76ec7f..18934ef)
- Task 5: minor (deferred): gather test asserts one runner applies all ([[],[1,2]]) though code allows a split; pending() now waits on the advisory lock; single quote in repo path still breaks -o
- Task 5: complete (commits 9eab912..18934ef, review clean after 1 fix round)
- Task 6: complete (commits 18934ef..d16fc58, review clean)
- Task 6: minor (deferred): leak test does not assert same backend pid; only INSERT on schema_migrations tested; 0001 comment ambiguity about which lines are no-ops; db teardown skips truncate if close raises
- Task 7: Ruling: every foreign key between tenant-scoped tables includes tenant_id (referenced tenant tables declare unique (tenant_id, id); references are (tenant_id, x_id)); collection_grants additionally references memberships (tenant_id, user_id) on delete cascade, so member removal (Task 15) drops grants by cascade. Why: reviewer proved cross-tenant grant rows, an FK existence oracle, and cross-tenant cascade deletes; spec section 17 forbids cross-tenant exposure through any path and spec 15 already uses composite FKs. Plan text deviated from that. Applied to Plans B and C Global Constraints. Cost if wrong: extra unique indexes and wider FKs.
- Task 7: minor (deferred): RLS coverage test does not check a tenant_isolation policy exists or partitioned tables; no behavioral WITH CHECK test for collection_grants; GLOBAL_TABLES duplicated in two tests
- Task 7: fix round 1/5 (1 addressed, 0 open; commits a9f9dff..9d1e40e)
- Task 7: minor (deferred): no index on collection_grants (tenant_id, user_id) for the membership FK; created_by not tied to membership; redundant tenants FK on collection_grants
- Task 7: complete (commits d16fc58..9d1e40e, review clean after 1 fix round). Note for Task 15: member removal now cascades grants via FK; the route need not delete grants explicitly.
- Task 8: minor (deferred): non-canonical base64 and leading-zero integers accepted; first verify_dummy call is about 2x slower (warm at startup); verify_password(pw, None) raises AttributeError if NULL hashes ever exist; n > 2**20 guard unreachable
- Task 8: fix round 1/5 cancelled by idle deadline (no commit). Root cause: RED run called real scrypt on the 131072/2/131070 hash before the guard existed (pytest pid ran 56 min). Killed. Added mandatory perl alarm time limits and "RED must not trigger the expensive behavior" to implementer-rules.md. Uncommitted partial edits left in tests/test_passwords.py for the next attempt. Re-dispatching round 1 fresh.
- Task 8: fix round 1/5 (2 addressed, 0 open; commits f979bdd..2d774e2)
- Task 8: minor (deferred): add p <= 16 cap (tiny n with huge p still costs seconds of PBKDF2); duplicate padded-digest mutation case; field-count label; non-canonical base64 round-trip check
- Task 8: complete (commits 9d1e40e..2d774e2, review clean after 1 fix round)
- Task 9: minor (deferred): %s args and third-party dynamic messages pass through msg by design (worker must retrieve task exceptions); frozenset iteration order varies (sort keys); metrics server Handler has no timeout; labels()/register lose static types; Histogram does not reject a user label named le; no concurrent labels()+render test
- Task 9: carry-forward note for Tasks 10 and 17: uvicorn's default log_config re-enables uvicorn.access and replaces handlers; startup must run uvicorn with log_config=None or configure_logging must run after uvicorn config. Put this in the Task 10 dispatch.
- Task 9: fix round 1/5 (2 addressed + 2 ruling items, 0 open; commits f608866..cc8606c)
- Task 9: minor (deferred): whitespace-only metrics token counts as configured
- Task 9: Ruling (load-bearing for Tasks 12, 17): context vars are typed str | None and the formatter no longer stringifies objects, so every bind() caller passes str(...) for UUIDs (Principal.tenant_id, user_id, job ids). Carry into the Task 12 and Task 17 dispatches. Cost if wrong: a dropped log line plus a stderr traceback.
- Task 9: complete (commits 2d774e2..cc8606c, review clean after 1 fix round)
- Task 10: minor (deferred): validation detail can echo client-supplied dict keys or extra field names via loc (watch in later tasks using dict bodies or extra=forbid)
- Task 10: fix round 1/5 (2 addressed + 4 ruling items, 0 open; commits f204849..14ff84b)
- Task 10: minor (deferred): catch-and-rewrap after body abort logs a spurious ERROR; startup close exceptions mask the startup error; http_client close path verified by reading only; FastAPI exception telemetry never sees swallowed exceptions
- Task 10: complete (commits cc8606c..14ff84b, review clean after 1 fix round)
- Task 11: minor (deferred): switch_tenant for a session principal without session_id reports api_key_tenant_fixed (wrong cause); login audit stores client IP (review retention); fixed-window throttle allows up to 2x across a boundary (by design)
- Task 11: fix round 1/5 (2 addressed + 4 ruling items, 0 open; commits fed9518..34baab0)
- Task 11: minor (deferred): in-flight reservations count (burst of correct logins from one IP can briefly 429); cancelled requests keep reservations (fails closed); auth_throttle old windows never purged; disabled during scrypt yields an unusable session row
- Task 11: complete (commits 14ff84b..34baab0, review clean after 1 fix round)
- Task 12: complete (commits 34baab0..2ec337b, review clean)
- Task 12: minor (deferred): logout with a dead session returns 401 without clearing cookies (make idempotent); logout test asserts only max-age=0; eg_csrf Path/Max-Age and empty X-CSRF-Token not tested; register runs scrypt twice and can 429 after creating the account; comment on contextvar set() relying on per-request tasks
- Task 13: fix round 1/5 (1 addressed + ruling items, 0 open; commits 6460a5c..6eaaad8)
- Task 13: minor (deferred): API key count uncapped per session user; admin revoke branch untested at route level; cross-user revoke audit actor not asserted
- Task 13: complete (commits 2ec337b..6eaaad8, review clean after 1 fix round)
- Task 14: Ruling: collection names are workspace-visible metadata within a tenant; the 409 collection_name_taken on a hidden collection's name is accepted and documented (same-tenant members only; no cross-tenant signal). Why: per-tenant unique names are a usability requirement in spec 8; the leak is a name within the caller's own workspace. Cost if wrong: switch uniqueness to (tenant_id, created_by, name) later. Added to Plan D Task 14 security.md known gaps.
- Task 14: minor (deferred): by-id routes do not filter kind (owners/admins can address quick and benchmark collections by id); PATCH explicit null counts as changed in audit; missing tests: no-grant member 404 on PATCH/DELETE/grants, write grantee cannot PUT grants, owner readable_collection_ids, nonexistent requested id; duplicated permission join between effective_permission and require_collection; membership deleted mid-PUT gives 500
- Task 14: complete (commits 6eaaad8..be870dd, review clean)
- Task 15: Ruling: add-member by email reveals account existence and display name to any workspace admin (brief-required user_not_found). Accepted and documented in Plan D security.md known gaps alongside the Task 14 name oracle. Cost if wrong: switch to invitations later.
- Task 15: minor (deferred): no concurrency test for the last-owner race
- Task 15: fix round 1/5 (1 addressed + 5 ruling items, 0 open; commits a1b6c99..0691b0d)
- Task 15: minor (deferred): no (tenant_id, user_id) index on sessions/api_keys for revocation scans; only owners re-checked inside the removal transaction; session_required implemented twice with duplicated title; disabled users can be added as members
- Task 15: complete (commits be870dd..0691b0d, review clean after 1 fix round)
- Task 16: complete (commits 0691b0d..e44f8a4, review clean)
- Task 16: minor (deferred): payload key length and key count unbounded (and keys interpolated into ValueError text); non-str error_code raises TypeError not ValueError; concurrent same-key enqueue test does not force overlap; claim test cannot detect SKIP LOCKED removal; idempotency hit does not compare tenant or kind (keys must be code-generated and globally unique)
- Task 16: Ruling (load-bearing for Task 17): lease fencing is (job_id, worker_id). The worker uses a distinct worker id per concurrency slot (`<host>:<pid>:<uuid6>:<slot>`), each slot runs one job at a time, so a slot can never reclaim a job it is still running. Carry into the Task 17 dispatch.
- ENV: ~/Desktop is iCloud-synced (brctl shows active sync). iCloud set the UF_HIDDEN flag on every file in the worktree and created " 2" conflict duplicates (1691 in .venv, 2 in pgdata, 1 git index copy at .git/worktrees/evidencegraph/index 2). Python 3.14 site.py skips hidden .pth files, so the editable install's .pth was ignored and the `halluciscope` entry point could not import backend (tests worked only because python -m pytest puts cwd on sys.path). Fix (controller, local and reversible): moved .venv -> .venv.nosync and .local -> .local.nosync with symlinks .venv and .local (iCloud ignores *.nosync), deleted the venv and pgdata conflict duplicates, chflags -R nohidden, excluded the new names in .git/info/exclude. Verified: halluciscope --help works from /tmp; full suite 358 passed. Remaining risk to report to Omar: the git repo itself is still iCloud-synced (repo must stay on Desktop per his rule). frontend/node_modules (Plan D) must also be .nosync with a symlink.
- Task 17: minor (deferred): heartbeat lease loss only logs (handler keeps running; acceptable with idempotent handlers); requeue_stale runs per slot; workers rows from hard crashes never pruned (readiness must filter by heartbeat age, Task 18 does); request_id from the job is not bound into worker logs (spec 18 asks for propagation; carry to Plan B/C handlers)
- Task 17: request_id propagation moved into fix round 1 (spec 18), not deferred
- Task 17: fix round 1/5 (1 addressed + 7 ruling items, 0 open; commits 685c40e..a027cfc)
- Task 17: minor (deferred): asyncio.run teardown can hang on a handler that swallows every cancel (supervisor SIGKILL is the backstop); outer cancellation plus stuck handler unbounded; handler swallowing the timeout cancel keeps heartbeating; shutdown releases a job whose to_thread work keeps running (handlers must be idempotent, Plan B/C); self-cancel retry has no error log line; slot construction failure orphans earlier slots
- Task 17: complete (commits e44f8a4..a027cfc, review clean after 1 fix round)
- ENV update: the UF_HIDDEN flag is reapplied to files as processes touch them (python3 binary on exec, .pth files on site import, migrate.py on write, pg.log on write), even inside .nosync dirs, while new files created by editors stay unhidden; the agent that sets it is unidentified (not iCloud sync: no new conflict copies after .nosync). Ruling: do not rely on .pth processing locally. Run the CLI as `.venv/bin/python -m backend.cli.main <cmd>` from the worktree root (cwd on sys.path); add an `if __name__ == "__main__": app()` guard (Task 18 fix round). Plans B, C, D: replace `.venv/bin/halluciscope` with `.venv/bin/python -m backend.cli.main`; Docker and normal installs keep the `halluciscope` entry point. Report to Omar.
- Task 18: minor (deferred): Bearer scheme match is case-sensitive; applied_versions test and startup assertion never observed RED
- Task 18: fix round 1/5 (1 addressed + 4 ruling items, 0 open; commits f11d939..15bc7d6)
- Task 18: minor (deferred): on an unmigrated DB readiness reports db error (workers table missing) instead of migrations pending; discover() failure reported as db error; statement_timeout test does not distinguish set from set local; /metrics queue count has no deadline; LLM and DB checks sequential (about 4 s worst case)
- Task 18: complete (commits a027cfc..15bc7d6, review clean after 1 fix round)
- ALL 18 TASKS COMPLETE. Plan A range 697f8d5..15bc7d6. Next: final whole-plan review per references/final-review.md.
- Final review (Opus): Ready for Plan B with fixes. No Critical. Important I1 input limits, I2 dead jobs strand domain rows, I3 require_collection ignores kind, I4 throttle IP behind proxy.
- Final review: Ruling: I2 resolved with per-kind failure hooks (Worker failure_handlers, build_failure_handlers, requeue_stale_detailed); Plans B and C must register hooks for ingest_upload (upload failed with error_code), search (search_requests failed), run_analysis (analysis failed), reverify_document and reverify_analysis (log only). Cost if wrong: one more hook API change.
- Final review: Ruling: I3 require_collection gets kinds=("normal",); Plan B intake_upload gains collection_kinds and Plan C quick paths pass ("quick",). Cost if wrong: extra keyword threading.
- Final review: Ruling: I4 deferred to Plan D Task 12 (uvicorn --proxy-headers --forwarded-allow-ips for the nginx network). Cost if wrong: until Plan D, local dev has no proxy so the issue cannot occur.
- Final review: Ruling: deferred indexes (memberships(user_id), collection_grants(tenant_id,user_id), sessions/api_keys(tenant_id,user_id)) go into Plan B Task 7's migration instead of a new Plan A migration, to keep migration numbering in Plans B and C. Cost if wrong: none functionally.
- Final review: deferred to release: purge of old throttle/session/job/worker rows, registration throttle, API key cap, logout idempotency before the frontend (Plan D Task 2 dispatch).
