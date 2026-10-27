# EvidenceGraph Plan D: Product Surface, Deployment, and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A polished React and TypeScript product for upload, ask and verify, evidence inspection, contradictions, document history, and the evidence graph; Docker deployment; CI; demo data; architecture, security, operations, and evaluation documentation; a README whose results block is generated from measured results; and a recorded end-to-end walkthrough.

**Architecture:** The existing Vite app keeps its stack and "forensic microscope" theme. One typed API layer (`src/api/`) talks to `/api/v1` with cookie sessions and the CSRF header; pages are thin and compose tested hooks and pure utilities (span highlighting, polling, graph layout). Deployment is a Compose stack (Postgres with pgvector, one-shot migrate, API, worker, nginx serving the build, optional Ollama) built from pinned images.

**Tech Stack:** React 19.2.4, react-router-dom 7.13.2, axios 1.14.0, motion 12.38.0, Tailwind 4.2.2, Vite 8.0.3, TypeScript 5.9.3; dev: vitest 5.0.5, @testing-library/react 16.3.3, @testing-library/user-event 14.6.8, @testing-library/jest-dom 7.0.1, jsdom 30.1.3. Docker images `pgvector/pgvector:pg17`, `python:3.14-slim`, `node:24-alpine`, `nginx:1.29-alpine`, `ollama/ollama:0.40.11`.

**Spec:** `docs/superpowers/specs/2026-10-26-evidencegraph-design.md` (sections 16, 17, 19, 22, 23 Plan D, 25, and rulings R7, R8, R9). Plans A, B, and C are complete before this plan starts, including Plan C Task 27 (measured results exist under `evaluation/results/` and `docs/results/`).

## Global Constraints

- Every Plan A, B, and C Global Constraint applies unchanged.
- Frontend commands run from `frontend/`: `npm ci`, `npm run lint`, `npm run build`, `npm test` (`vitest run`). All three must pass at the end of every frontend task.
- No new runtime npm dependencies. Dev dependencies added with exact versions only (the five listed above). Existing `package.json` ranges are replaced with the exact installed versions from `package-lock.json` (react 19.2.4, react-dom 19.2.4, react-router-dom 7.13.2, axios 1.14.0, motion 12.38.0, recharts 3.8.1, vite 8.0.3, typescript 5.9.3, tailwindcss 4.2.2, and the rest of the lock's resolved versions).
- All HTTP goes through `frontend/src/api/`; base path `/api/v1` (legacy `/api` only for the quick-check page). Never hardcode a host.
- API character offsets are Unicode code points (Python string indices). The frontend converts them with `cpToUtf16` before slicing JavaScript strings.
- Document text, questions, answers, and passages are untrusted: render them only as React text children. `dangerouslySetInnerHTML`, `innerHTML`, and `eval` are forbidden in `frontend/src` (enforced by a test).
- Verdict presentation is fixed: supported uses `--color-verdict-safe` with a check icon and the text `Supported`; contradicted uses `--color-verdict-danger` with a cross icon and `Contradicted`; insufficient uses `--color-verdict-warn` with a dash icon and `Not enough evidence`. Color is never the only signal.
- Accessibility: every control has an accessible name; side panels trap and restore focus and close on Escape; job progress is announced through `aria-live="polite"`; informative text uses `text-text-secondary` or brighter (`text-text-muted` is for decoration only, since it fails WCAG AA on the dark background).
- Every async view shows a loading state, an empty state, and an error state with the mapped problem message.
- Docker images run as a non-root user and pin base image tags. Compose never publishes Postgres or the worker metrics port outside `127.0.0.1`.
- README and docs numbers come only from `docs/results/summary.json` or the other committed `docs/results/*.md` files, which committed scripts produced.

## Review Focus

1. Text with emoji or combining marks before a highlighted span (for example `"Café 😀 opens at 9."` with a span on `opens at 9`): the highlight covers exactly those characters, not shifted by the UTF-16 surrogate pair. Test in Task 3.
2. A stored document containing `<img src=x onerror=alert(1)>` or `<script>` renders as visible literal text in the version viewer, evidence inspector, and diff view. Tests in Tasks 3, 6, and 8.
3. A session that expires mid-use: the next API call returning 401 sends the user to `/login?next=<current path>`, and signing in returns them to that path. Test in Task 2.
4. Navigating away while an analysis or upload is polling stops all polling requests (no request after unmount, no state update warnings). Test in Task 3.
5. Evidence redacted after a revoked grant (`redacted: true`, text null) shows a labeled "Hidden: you no longer have access to this source" placeholder in the inspector and the graph, never a blank panel or a crash. Tests in Tasks 8 and 9.

## File Map

Created (frontend, under `frontend/src/`)
- `api/http.ts`, `api/types.ts`, `api/endpoints.ts`, `api/problems.ts`
- `auth/AuthContext.tsx`, `auth/RequireAuth.tsx`
- `lib/spans.ts`, `lib/format.ts`, `lib/graphLayout.ts`
- `hooks/usePolling.ts`, `hooks/useAsync.ts`
- `components/VerdictBadge.tsx`, `components/AsyncState.tsx`, `components/HighlightedText.tsx`, `components/SidePanel.tsx`, `components/ProgressStepper.tsx`, `components/AnswerView.tsx`, `components/ClaimList.tsx`, `components/EvidenceInspector.tsx`, `components/UploadDropzone.tsx`, `components/DiffView.tsx`, `components/GraphView.tsx`, `components/CollectionPicker.tsx`, `components/AppShell.tsx`
- `pages/LoginPage.tsx`, `pages/RegisterPage.tsx`, `pages/AskPage.tsx`, `pages/CollectionsPage.tsx`, `pages/CollectionDetailPage.tsx`, `pages/DocumentPage.tsx`, `pages/AnalysesPage.tsx`, `pages/AnalysisPage.tsx` (rewritten: stored analysis view), `pages/GraphPage.tsx`, `pages/QuickCheckPage.tsx` (the former paste flow), `pages/SettingsPage.tsx`
- `test/setup.ts`, `test/server.ts` (a tiny fetch-level fake for axios via a custom adapter), and `*.test.ts(x)` files next to the code they test
- `frontend/vitest.config.ts`

Created (repository)
- `deploy/Dockerfile.api`, `deploy/Dockerfile.frontend`, `deploy/nginx.conf`, `deploy/docker-compose.yml`, `deploy/initdb/01-roles.sh`, `deploy/.env.example`, `.dockerignore`
- `.github/workflows/ci.yml`
- `demo/northwind/*.md` (authored demo corpus), `backend/cli/seed_demo.py`
- `scripts/dev-up.sh`
- `docs/architecture.md`, `docs/decisions/0001-postgres-centric-monolith.md` through `0007-evaluation-protocol.md`, `docs/security.md`, `docs/operations.md`, `docs/evaluation.md`, `docs/results/summary.json`, `docs/images/*.png`
- `LICENSE` (MIT, ruling: Omar asked for an open-source product; MIT is the least restrictive common choice and can be changed before release)
- Tests: `tests/test_deploy_files.py`, `tests/test_seed_demo.py`, `tests/test_docs.py`, `tests/test_readme_summary.py`

Modified
- `frontend/package.json`, `frontend/package-lock.json`, `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/src/components/ClaimCard.tsx`, `frontend/src/components/ScoreGauge.tsx` (kept for the quick-check page), `frontend/eslint.config.js`, `frontend/AGENTS.md`
- `README.md`, `AGENTS.md`, `evaluation/report.py`, `backend/cli/main.py`
- Deleted: `frontend/src/pages/DashboardPage.tsx` (Task 9) and `frontend/src/api/client.ts` (Task 10); `DocumentInput.tsx`, `ChunkViewer.tsx`, `ScoreGauge.tsx`, and `ClaimCard.tsx` stay for the quick-check page

---

### Task 1: Frontend test setup, pinned dependencies, and the typed API layer

**Files:**
- Create: `frontend/vitest.config.ts`, `frontend/src/test/setup.ts`, `frontend/src/test/server.ts`, `frontend/src/api/http.ts`, `frontend/src/api/problems.ts`, `frontend/src/api/types.ts`, `frontend/src/api/endpoints.ts`, `frontend/src/api/http.test.ts`, `frontend/src/security.test.ts`
- Modify: `frontend/package.json` (exact versions; scripts `"test": "vitest run"`, `"test:watch": "vitest"`), `frontend/package-lock.json`, `frontend/tsconfig.app.json` (add `"vitest/globals"` and `"@testing-library/jest-dom"` to `types` only in a separate `tsconfig.test.json` referenced by vitest, so app types stay clean)

**Interfaces:**
- `vitest.config.ts`: `environment: "jsdom"`, `setupFiles: ["src/test/setup.ts"]`, `globals: true`, `include: ["src/**/*.test.{ts,tsx}"]`, `restoreMocks: true`.
- `src/test/server.ts`: `type Route = { method: string; path: string | RegExp; reply: (req: FakeRequest) => FakeReply | Promise<FakeReply> }`; `function installFakeServer(routes: Route[]): { requests: FakeRequest[]; reset(routes?: Route[]): void }` that sets `http.defaults.adapter` to an adapter matching method and path (pathname without `/api/v1`), returning `{status, data, headers}`; unmatched requests reject with a 599 error naming the request so tests fail loudly. `FakeRequest = { method, path, params, data, headers }`.
- `src/api/problems.ts`: `class ApiProblem extends Error` with `status: number`, `code: string`, `title: string`, `detail: string | null`; `function toProblem(err: unknown): ApiProblem` (axios errors with a problem body use it; network errors become code `network_error`, status 0); `const MESSAGES: Record<string, string>` with human messages for at least every code named in Plans A to C (`invalid_credentials`, `too_many_attempts`, `email_taken`, `weak_password`, `registration_disabled`, `csrf_failed`, `unauthenticated`, `forbidden`, `insufficient_permission`, `collection_not_found`, `collection_name_taken`, `collection_not_empty`, `document_not_found`, `upload_not_found`, `version_not_found`, `analysis_not_found`, `passage_not_found`, `claim_not_found`, `unsupported_type`, `too_large`, `payload_too_large`, `effective_from_in_future`, `too_many_analyses`, `search_timeout`, `llm_unavailable`, `llm_timeout`, `analysis_timeout`, `calibrator_missing`, `validation_error`, `network_error`, `internal_error`, and every ingest error code); `function messageFor(p: ApiProblem): string` (mapped message, else `p.title`, else `"Something went wrong."`).
- `src/api/http.ts`: `export const http = axios.create({ baseURL: "/api/v1", withCredentials: true, timeout: 30000 })`; request interceptor adds `X-CSRF-Token` from the in-memory token set by `setCsrfToken(token: string | null)`, falling back to the `eg_csrf` cookie, on POST, PUT, PATCH, DELETE only; response interceptor converts errors with `toProblem` and, for status 401 on any request except `/auth/me` and `/auth/login`, calls the handler registered with `onUnauthenticated(handler: () => void): () => void`. `export const legacyHttp` is the same with `baseURL: "/api"`.
- `src/api/types.ts`: TypeScript types mirroring `backend/api/schemas_v1.py` field for field (`snake_case` kept): `Verdict = "supported" | "contradicted" | "insufficient"`, `Page<T>`, `User`, `Tenant`, `SessionOut`, `Me`, `Collection`, `Grant`, `Member`, `AuditEntry`, `ApiKey`, `ApiKeyCreated`, `DocumentSummary`, `DocumentDetail`, `VersionSummary`, `Version`, `VersionText`, `PassageBoundary`, `Upload`, `UploadAccepted`, `Diff`, `DiffHunk`, `AnalysisAccepted`, `AnalysisSummary`, `Analysis`, `AnalysisOptions`, `Claim`, `Evidence`, `Citation`, `ClaimCheck`, `Passage`, `Graph`, `GraphNode`, `GraphEdge`, and the legacy `LegacyAnalyzeResponse`.
- `src/api/endpoints.ts`: one async function per route used by the UI, returning typed data: `register`, `login`, `logout`, `me`, `switchTenant`, `listCollections`, `createCollection`, `updateCollection`, `deleteCollection`, `getGrants`, `putGrants`, `listMembers`, `addMember`, `removeMember`, `listAudit`, `listApiKeys`, `createApiKey`, `revokeApiKey`, `listDocuments`, `getDocument`, `deleteDocument`, `uploadDocument(collectionId, file, opts?: { title?: string; effectiveFrom?: string })`, `uploadVersion(documentId, file, opts?)`, `getUpload`, `getVersion`, `getVersionText(documentId, n, start?, end?)`, `getPassageBoundaries`, `getDiff(documentId, from, to)`, `ask(body)`, `verify(body)`, `listAnalyses(params)`, `getAnalysis`, `getGraph`, `reverifyAnalysis`, `getPassage`, `sendFeedback(claimId, label, note?)`, `getClaimChecks`, `legacyAnalyze(documentText, question)`. `login` and `register` call `setCsrfToken(session.csrf_token)`; `logout` clears it.

- [ ] **Step 1: Pin and install.** Edit `package.json` to exact versions and add the five dev dependencies; run `npm install` then `npm ci`; confirm `npm run build` and `npm run lint` still pass.
- [ ] **Step 2: Write failing tests** in `src/api/http.test.ts`: `adds CSRF header on unsafe methods only` (POST carries `X-CSRF-Token: t1` after `setCsrfToken("t1")`; GET does not); `falls back to the eg_csrf cookie` (`document.cookie = "eg_csrf=c9"`, token unset -> header `c9`); `maps problem+json to ApiProblem` (409 body with code `collection_name_taken` -> `ApiProblem` with that code and `messageFor` returns the mapped message); `maps network failures` (adapter throws -> code `network_error`, status 0); `calls the unauthenticated handler on 401 except for me and login`; `login stores the csrf token` (fake `/auth/login` returns `csrf_token: "abc"`; next POST carries `abc`). In `src/security.test.ts`: `no raw HTML sinks in source` (reads every `src/**/*.{ts,tsx}` file except test files with `node:fs` and asserts none contains `dangerouslySetInnerHTML`, `.innerHTML`, or `eval(`).
- [ ] **Step 3: Run** `npm test`. Expected: FAIL (modules missing).
- [ ] **Step 4: Implement** the four API modules and test helpers.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 6: Commit** the frontend files with `feat: add typed API client with CSRF and problem mapping`.

### Task 2: Authentication context, login and register pages, route guard, app shell

**Files:**
- Create: `frontend/src/auth/AuthContext.tsx`, `frontend/src/auth/RequireAuth.tsx`, `frontend/src/components/AppShell.tsx`, `frontend/src/pages/LoginPage.tsx`, `frontend/src/pages/RegisterPage.tsx`, `frontend/src/auth/auth.test.tsx`
- Modify: `frontend/src/App.tsx` (routes per spec section 19; placeholder route elements for pages built in later tasks render a short "Coming in this build" panel only until their task replaces them), `frontend/src/main.tsx`, `frontend/index.html` (title becomes `EvidenceGraph: claim-level verification`; the old title contains an em dash)

**Interfaces:**
- `AuthProvider` loads `me()` on mount; context value `{ status: "loading" | "anonymous" | "authenticated", me: Me | null, signIn(email, password): Promise<void>, signUp(email, password, displayName): Promise<void>, signOut(): Promise<void>, switchTenant(id): Promise<void>, refresh(): Promise<void> }`; `useAuth()` throws outside the provider. It registers `onUnauthenticated` to set `anonymous` and navigate to `/login?next=<encoded current path and search>`.
- `RequireAuth` renders children when authenticated, a full-page loading state while loading, and `<Navigate to="/login?next=...">` when anonymous.
- `LoginPage`: email and password fields with labels, submit button `Sign in`, link to register; on success navigates to `next` when it is a same-origin path starting with `/` and not `//`, else `/`; shows `messageFor(problem)` in a `role="alert"` region. `RegisterPage`: display name, email, password (hint "At least 12 characters"), `Create account`; 403 `registration_disabled` shows its message.
- `AppShell`: the existing logo and pill navigation restyled to links `Ask`, `Collections`, `History`, `Quick check`, `Settings`; a tenant switcher (`select` labeled `Workspace`) when the user has more than one tenant; user menu with `Sign out`; skip link `Skip to content` targeting `<main id="main">`; the existing footer text updated to `EVIDENCEGRAPH · CLAIM-LEVEL VERIFICATION`.

- [ ] **Step 1: Write failing tests** in `src/auth/auth.test.tsx` (fake server, `MemoryRouter`): `redirects anonymous users to login with next` (`/collections` with `me` 401 -> login page shown, location `/login?next=%2Fcollections`); `signs in and returns to next` (submit valid credentials -> location `/collections`); `rejects off-site next` (`next=//evil.example` -> lands on `/`); `shows mapped error for invalid credentials` (alert text equals `MESSAGES.invalid_credentials`); `expired session mid-use redirects` (Review Focus 3: authenticated on `/analyses`, a later request returns 401 -> location `/login?next=%2Fanalyses`); `sign out clears session` (`logout` called, login page shown); `form controls have accessible names` (`getByLabelText("Email")`, `getByLabelText("Password")`).
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add sign-in flow, route guard, and app shell`.

### Task 3: Span utilities, polling hook, and shared UI primitives

**Files:**
- Create: `frontend/src/lib/spans.ts`, `frontend/src/lib/format.ts`, `frontend/src/hooks/usePolling.ts`, `frontend/src/hooks/useAsync.ts`, `frontend/src/components/VerdictBadge.tsx`, `frontend/src/components/AsyncState.tsx`, `frontend/src/components/HighlightedText.tsx`, `frontend/src/components/SidePanel.tsx`, `frontend/src/lib/spans.test.ts`, `frontend/src/hooks/usePolling.test.tsx`, `frontend/src/components/primitives.test.tsx`

**Interfaces:**
- `lib/spans.ts`: `function cpToUtf16(text: string, cp: number): number` (walks code points with `codePointAt`; `cp` beyond the end returns `text.length`); `function sliceCp(text: string, start: number, end: number): string`; `type Mark = { start: number; end: number; id: string; kind: string }` (code point offsets relative to `text`); `type Segment = { text: string; marks: Mark[] }`; `function segmentByMarks(text: string, marks: Mark[]): Segment[]` (splits at every mark boundary after clamping to `[0, length]` and dropping empty marks; each segment lists the marks covering it, sorted by start then id; concatenating segment texts returns `text` exactly).
- `lib/format.ts`: `formatDate(iso: string): string` (`Intl.DateTimeFormat` medium date and short time, local zone), `formatPercent(p: number): string` (one decimal, e.g. `"96.2%"`), `formatVersion(n: number): string` (`"v3"`).
- `hooks/usePolling.ts`: `function usePolling<T>(fetcher: (signal: AbortSignal) => Promise<T>, opts: { enabled: boolean; isDone: (value: T) => boolean; initialMs?: number; maxMs?: number }): { data: T | null; error: ApiProblem | null; done: boolean }` (first call immediately; delay starts at `initialMs` (default 500) and multiplies by 1.5 up to `maxMs` (default 4000); stops when `isDone` is true or on error; aborts the in-flight request and clears timers on unmount or when `enabled` turns false).
- `hooks/useAsync.ts`: `function useAsync<T>(fn: (signal: AbortSignal) => Promise<T>, deps: unknown[]): { data: T | null; error: ApiProblem | null; loading: boolean; reload(): void }`.
- `components/VerdictBadge.tsx`: `<VerdictBadge verdict={v} size?="sm" | "md" />` per Global Constraints (icon `aria-hidden`, visible text label).
- `components/AsyncState.tsx`: `<Loading label?>` (`role="status"`), `<EmptyState title body? action?>`, `<ErrorState problem onRetry?>` (`role="alert"` with `messageFor`).
- `components/HighlightedText.tsx`: `<HighlightedText text marks renderMark?={(segment) => className} onMarkClick?={(mark) => void} activeId? />` rendering `segmentByMarks` output as text children inside `<mark>` elements (`tabIndex=0` and `role="button"` only when `onMarkClick` is given; Enter and Space activate).
- `components/SidePanel.tsx`: `<SidePanel open title onClose>`: dialog semantics (`role="dialog"`, `aria-modal="true"`, labelled by the title), focus moves to the panel on open and returns to the opener on close, Tab cycles inside, Escape closes.

- [ ] **Step 1: Write failing tests**: `spans.test.ts`: `cpToUtf16 handles surrogate pairs` (`"a😀b"`: cp 2 -> 3); `segmentByMarks covers exact characters after emoji` (Review Focus 1: text `"Café 😀 opens at 9."`, mark start 7 end 17 -> the marked segment text is `"opens at 9"`); `combining marks counted as code points` (text `"e\u0301 x"` with mark 3..4 -> `"x"`); `overlapping and adjacent marks` (marks 0..5 and 3..8 on `"0123456789"` -> segments `"012"`, `"34"` (two marks), `"567"`, `"89"`; adjacent 0..3 and 3..6 -> no shared segment); `concatenation equals input` (seeded random marks over random text with emoji -> joined segments equal the text). `usePolling.test.tsx` (fake timers): `polls with backoff until done`; `stops on error`; `stops after unmount` (Review Focus 4: unmount after the first response; advancing timers 30 s makes no further fetch and logs no React warning); `aborts in-flight request on unmount` (the fetcher's `signal.aborted` becomes true). `primitives.test.tsx`: `verdict badge always shows a text label` (three verdicts -> `Supported`, `Contradicted`, `Not enough evidence`); `highlighted text renders HTML as literal text` (Review Focus 2: text `"<img src=x onerror=alert(1)>"` -> no `img` element in the container, the literal string visible); `side panel focus trap and escape` (focus starts inside; Tab from the last control returns to the first; Escape calls `onClose` and focus returns to the opener); `marks are keyboard activatable`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add span highlighting, polling, and accessible UI primitives`.

### Task 4: Collections, grants, and the collection picker

**Files:**
- Create: `frontend/src/pages/CollectionsPage.tsx`, `frontend/src/components/CollectionPicker.tsx`, `frontend/src/pages/CollectionsPage.test.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- `CollectionsPage` (`/collections`): paginated list (`Load more` while `next_cursor`), each row shows name, description, the viewer's permission as text, and links to `/collections/:id`; a create form (`Name` required, `Description` optional) that prepends the new collection; for collections where the viewer has `admin`, a `Manage` button opens a `SidePanel` with rename and description fields (`updateCollection`), a grants table listing every tenant member (`listMembers`) with a `select` labeled `Access for <display name>` offering `No access`, `Read`, `Write`, `Admin` (owners and admins of the tenant show `Admin (workspace role)` disabled), `Save access` (`putGrants` with the non-`No access` rows), and `Delete collection` behind a confirm step (409 `collection_not_empty` shows its mapped message).
- `CollectionPicker`: `<CollectionPicker value: string[] | null onChange />` loading all readable collections (following cursors up to 500); `null` means "All my collections"; renders a fieldset with legend `Search in` and one checkbox per collection plus `All collections`.

- [ ] **Step 1: Write failing tests**: `lists collections and loads more`; `creates a collection and shows it first`; `duplicate name shows mapped error`; `admin can edit grants` (changing Bob to `Read` and saving sends `putGrants` with `[{user_id, permission: "read"}]`); `non-admin sees no manage button`; `delete non-empty collection shows conflict message`; `picker toggles between all and specific collections` (checking one collection after `All` emits `[id]`; checking `All` emits `null`).
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add collection management and access editor`.

### Task 5: Collection detail with multi-file upload and new versions

**Files:**
- Create: `frontend/src/pages/CollectionDetailPage.tsx`, `frontend/src/components/UploadDropzone.tsx`, `frontend/src/pages/CollectionDetailPage.test.tsx`

**Interfaces:**
- `UploadDropzone`: `<UploadDropzone onUpload: (file: File) => Promise<UploadAccepted> onSettled?: (upload: Upload) => void />`; drop target plus a visible `Choose files` button wired to a hidden `input type="file" multiple accept=".pdf,.txt,.md,.markdown,.html,.htm,.csv"`; client-side checks before sending: extension in that list (else `MESSAGES.unsupported_type`) and size at most 20,971,520 bytes (else `MESSAGES.too_large`); one row per file with name, size, and status text in an `aria-live="polite"` list: `Uploading`, `Queued`, `Processing`, `Ready (v<n>, <passages> passages, <reused> reused)`, `Unchanged (duplicate of v<n>)`, `Failed: <mapped error_code message>`; statuses after acceptance come from `usePolling(getUpload)` until `created_version`, `duplicate`, or `failed`. Files upload one at a time in selection order.
- `CollectionDetailPage` (`/collections/:id`): collection name and permission; documents table (`Title`, `Type` from the file extension, `Versions`, `Current`, `Added`) linking to `/documents/:id`; empty state "No documents yet. Upload PDF, text, Markdown, HTML, or CSV files."; the dropzone shown only with `write`; per document (write) `Upload new version` opening a `SidePanel` with a file input, an optional `Effective from` `datetime-local` (converted to ISO 8601 with the local offset), and a submit that uses the same row-status display; `Delete` behind a confirm step. The table reloads when any upload settles.

- [ ] **Step 1: Write failing tests**: `uploads two files sequentially and shows ready status` (fake server returns accepted, then `getUpload` sequences `queued`, `processing`, `created_version`; final rows read `Ready (v1, 12 passages, 0 reused)`); `rejects unsupported and oversized files before sending` (`.docx` and a 21 MB `File` -> mapped messages, no request); `duplicate upload shows unchanged status`; `failed ingestion shows mapped error` (`no_text` -> its message); `read-only users see no dropzone`; `new version with effective date sends ISO timestamp` (the multipart `effective_from` field parses as the chosen local time); `empty collection shows guidance`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add document upload with live ingestion status`.

### Task 6: Document history, version viewer, and diff

**Files:**
- Create: `frontend/src/pages/DocumentPage.tsx`, `frontend/src/components/DiffView.tsx`, `frontend/src/pages/DocumentPage.test.tsx`

**Interfaces:**
- `DocumentPage` (`/documents/:id`, query `v`, `start`, `end`, `from`, `to`): header with title, filename, collection link; version timeline (newest first: `v<n>`, `Effective <date>`, `Added <date>`, characters, passages, a `Current` label); selecting a version shows the viewer; a `Compare` control with two selects labeled `From version` and `To version` (defaults: previous and current) switches to the diff view and updates `from` and `to` in the URL.
- Version viewer: loads `getVersionText` in windows of at most 100,000 code points (`Load more` appends the next window); passage boundaries from `getPassageBoundaries` drawn as alternating subtle backgrounds through `HighlightedText` marks (kind `passage`), with each passage's page and section shown in a tooltip; a deep link (`v`, `start`, `end`) loads the window starting at `max(0, start - 2000)`, highlights the span (kind `target`, strong outline), scrolls it into view, and focuses it. Offsets in marks are relative to the window start.
- `DiffView`: `<DiffView diff fromText toText />`: two columns labeled `v<from>` and `v<to>`; for each hunk, the left column shows `fromText` slices and the right column `toText` slices (`delete` red background on the left, `insert` green on the right, `replace` both); `equal` hunks longer than 400 code points collapse to their first and last 120 with a `Show <n> unchanged characters` button; a summary line counts changed hunks. Hunk texts come from slicing with `sliceCp`.

- [ ] **Step 1: Write failing tests**: `timeline lists versions newest first with current label`; `deep link highlights the exact span` (version text `"Intro. Café 😀 refunds take 30 days. End."`, `start`/`end` for `refunds take 30 days` -> the focused `mark` has exactly that text); `document HTML renders literally` (Review Focus 2: version text containing `<script>alert(1)</script>` -> no `script` element, literal text visible); `loads more windows` (text of 150,000 code points -> first request `start=0,end=100000`, `Load more` requests `100000..150000`); `compare renders replace hunk` (diff with one `replace` hunk -> left shows the old sentence with the deletion style, right the new sentence); `collapses long equal hunks`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add document history, version viewer, and diff`.

### Task 7: Ask workspace and the analysis result view

Ruling: the Ask workspace submits and then navigates to `/analyses/:id`, which shows live progress and the result; history entries open the same view, so one tested component serves both (the spec places the result on `/`; this keeps one implementation and gives every result a shareable URL).

**Files:**
- Create: `frontend/src/pages/AskPage.tsx`, `frontend/src/components/ProgressStepper.tsx`, `frontend/src/components/AnswerView.tsx`, `frontend/src/components/ClaimList.tsx`, `frontend/src/pages/AnalysisPage.tsx` (rewritten), `frontend/src/pages/AskAnalysis.test.tsx`

**Interfaces:**
- `AskPage` (`/`): a radio group labeled `Mode` (`Ask a question`, `Verify an answer`); `Question` textarea (required in Ask, optional in Verify, max 2,000 characters with a live counter); `Answer to verify` textarea in Verify mode (max 20,000); `CollectionPicker`; a `Live (current versions)` checkbox checked by default, unchecking reveals a `datetime-local` labeled `As of`; options: `Claim extraction` select (`Atomic claims`, `One claim per sentence`) and, in Ask mode, `Self-consistency samples` number input 0 to 5 labeled as a diagnostic; submit `Ask` or `Verify` disabled while invalid; on 202 navigate to `/analyses/<id>`; 429 `too_many_analyses` and other problems show mapped messages. Below the form, the five most recent analyses (`listAnalyses({ mine: true, limit: 5 })`).
- `ProgressStepper`: `<ProgressStepper kind progress status />` with steps `Retrieving`, `Generating`, `Extracting claims`, `Verifying`, `Saving` (Verify and quick analyses skip `Generating`), each marked done, current, or pending from `progress.stages_done` and `progress.stage`; wrapped in `aria-live="polite"` announcing the current step.
- `AnswerView`: `<AnswerView answer claims citations selectedClaimId onSelectClaim />` renders the answer through `HighlightedText` with one mark per claim span (`answer_start..answer_end`, kind = verdict, styled with the verdict colors at low opacity plus an underline style per verdict so color is not the only cue) and one mark per citation marker (kind `citation` for valid, `citation_invalid` struck through with a `title` of "This citation does not match any retrieved passage"); clicking or pressing Enter on a claim mark calls `onSelectClaim`. Several claims sharing one sentence span are one mark listing all their ids; selecting it selects the first unselected one.
- `ClaimList`: `<ClaimList claims selectedClaimId onSelectClaim />`: one row per claim with `VerdictBadge`, the claim text, `P(supported)` as a percent, a `Source updated` label when `latest_check?.changed`, a `Conflicting evidence` label when `conflict`, and the extraction method when it is `sentence_fallback` ("Extracted without the language model"); rows are buttons.
- `AnalysisPage` (`/analyses/:id`): polls `getAnalysis` with `usePolling` until `succeeded` or `failed`; shows question, mode, scope (collection names), `Live` or `As of <date>`; while running, the stepper; on failure an `ErrorState` with the mapped `error_code`; on success a summary bar (counts per verdict with badges, citation count and invalid count, `Extracted <n> claims`), `AnswerView` (Verify analyses show the supplied answer), `ClaimList`, stage timings in a small table, a `Re-verify now` button for live analyses (`reverifyAnalysis`, then resumes polling until the claims' `latest_check` timestamps change or 60 s pass), and a link `View evidence graph`. Claim selection is kept in the URL as `?claim=<id>` (Task 8 opens the inspector from it).

- [ ] **Step 1: Write failing tests**: `ask submits and navigates` (fill question, choose a collection, submit -> request body `{question, collection_ids: [id], as_of: null, options: {extraction: "llm_atomic", consistency_samples: 0}}` and location `/analyses/a1`); `verify mode requires answer and hides consistency`; `as-of sends ISO timestamp`; `progress stepper announces stages` (polled analysis goes `queued` -> `running` with `stage: "verifying"` -> `succeeded`; the live region text includes `Verifying`); `answer highlights each claim span exactly` (answer with an emoji before the second claim -> each claim mark's text equals the claim's sentence); `invalid citation is struck through with explanation`; `selecting a claim mark updates the URL`; `failed analysis shows mapped error` (`llm_unavailable`); `summary counts match claims`; `reverify button only for live analyses`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add ask workspace and analysis result view`.

### Task 8: Evidence inspector

**Files:**
- Create: `frontend/src/components/EvidenceInspector.tsx`, `frontend/src/components/EvidenceInspector.test.tsx`
- Modify: `frontend/src/pages/AnalysisPage.tsx` (opens the inspector in a `SidePanel` when `?claim=` is set; closing removes the parameter)

**Interfaces:**
- `<EvidenceInspector claim analysis onFeedback? />`, sections in order:
  1. Claim text, `VerdictBadge`, and three labeled horizontal bars (`Supported`, `Contradicted`, `Not enough evidence`) with percent text from `claim.probs`.
  2. `Why this verdict`: one sentence built from the data, never from the LLM: supported -> "Entailed by <document title> v<n> (<entail percent>)." from the item with the highest `nli.entail`; contradicted -> "Contradicted by <title> v<n> (<contra percent>)."; insufficient -> "No source passage entails or contradicts this claim strongly enough." (redacted sources are named "a source you can no longer access").
  3. `Source updated` banner when `claim.latest_check?.changed`: "Re-checked on <date> after a source changed: now <verdict label> (was <original verdict label>)", with `View change` linking to `/documents/<document_id>?from=<evidence version_number>&to=<superseded_by_version>` for the first evidence item whose resolved passage has `superseded_by_version`.
  4. `Conflicting evidence` (when `claim.conflict`): the top supporting and top contradicting items side by side.
  5. Evidence groups `Supporting`, `Contradicting`, `Related`, each item showing document title, `v<n>`, `Effective <date>`, page (when known), the NLI numbers, the rerank score, a `Superseded by v<m>` label when the resolved passage says so, and the passage in context: `getPassage(passage_id)` (one request per non-redacted item, fetched when the panel opens, cached per passage id) supplies `context_before`, `text`, `context_after`, and `start_char`; the span (`span_start - start_char`, `span_end - start_char`, offsets into `text`) is highlighted with `HighlightedText`; `Open in document` links to `/documents/<document_id>?v=<n>&start=<span_start>&end=<span_end>`.
  6. Redacted items (Review Focus 5) render a bordered placeholder "Hidden: you no longer have access to this source" with the relation and NLI numbers only, and no fetch.
  7. `Self-consistency` (when `claim.diagnostics.consistency`): "Appeared in <a> of <n> resampled answers. Diagnostic only; not used in the verdict."
  8. Feedback: `Verdict is correct` and `Verdict is wrong` buttons, an optional `Note` field (max 1,000 characters), `sendFeedback`, then "Thanks, your feedback was recorded."
- Passage fetch failures show an inline `ErrorState` for that item only.

- [ ] **Step 1: Write failing tests**: `shows probabilities and data-driven explanation`; `highlights the evidence span inside its context` (resolved passage `text` = `"Refunds take 30 days. Shipping is free."`, `start_char = 100`, span `100..121` -> the mark text is `"Refunds take 30 days."`); `redacted evidence shows placeholder without fetching` (Review Focus 5: no `getPassage` request; placeholder text visible); `passage HTML renders literally` (Review Focus 2); `source updated banner links to the diff` (`latest_check.changed` and a resolved passage with `superseded_by_version: 2` -> link `/documents/d1?from=1&to=2`); `conflict view shows both sides`; `consistency diagnostic labeled as not used`; `feedback sends label and note`; `escape closes the panel and returns focus to the claim row`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add evidence inspector with provenance and conflicts`.

### Task 9: Analysis history and the evidence graph

**Files:**
- Create: `frontend/src/pages/AnalysesPage.tsx`, `frontend/src/lib/graphLayout.ts`, `frontend/src/components/GraphView.tsx`, `frontend/src/pages/GraphPage.tsx`, `frontend/src/lib/graphLayout.test.ts`, `frontend/src/pages/HistoryGraph.test.tsx`
- Modify: `frontend/src/App.tsx`
- Delete: `frontend/src/pages/DashboardPage.tsx`

**Interfaces:**
- `AnalysesPage` (`/analyses`): rows with question preview (or "Verified answer" when null), kind, status, verdict counts as small badges, `Live` or `As of <date>`, a `Changed since` label from `changed_since`, created date; `Load more`; owners and admins get a `Show` select (`My analyses`, `Everyone in this workspace`) mapped to `mine`; empty state links to `/`.
- `lib/graphLayout.ts`: `const COLUMN_ORDER = ["sentence", "claim", "passage", "version", "document"]`; `function layoutGraph(graph: Graph, opts?: { nodeW?: number; nodeH?: number; colGap?: number; rowGap?: number }): { nodes: Array<GraphNode & { x: number; y: number; w: number; h: number }>; edges: Array<GraphEdge & { d: string }>; width: number; height: number }` with defaults 220, 56, 96, 16. Columns follow `COLUMN_ORDER` (unknown types go last); the first column is ordered by node id; each later column is ordered by the mean row index of its neighbors in earlier columns (nodes without such neighbors last), ties by id. Edge path `d` is a cubic Bezier from the right middle of the left node to the left middle of the right node (edges between nodes of the same column connect right side to right side with a loop of 40 px). Pure and deterministic.
- `GraphView`: an `svg` with `role="img"` and an `aria-label` summarizing node and edge counts, plus a legend; edges colored by type (`supports` safe, `contradicts` danger, `related` muted dashed, `cites` cyan, `part_of`/`version_of` subtle, `superseded_by` warn dashed); nodes are `g` elements with `tabIndex=0`, `role="button"`, `aria-label="<type>: <label>"`, and labels truncated to 34 characters with the full text in a `title` element; Enter or click selects a node and shows a details box listing its full label and its incoming and outgoing edges; a redacted passage node shows "Hidden source" (Review Focus 5). The svg sits in a horizontally scrollable container.
- `GraphPage` (`/analyses/:id/graph`): `getGraph`, loading, empty ("This analysis has no claims"), and error states; a link back to the analysis.

- [ ] **Step 1: Write failing tests**: `graphLayout.test.ts`: `columns follow the type order`; `layout is deterministic` (same input twice -> identical output); `no node rectangles overlap` (random graphs of up to 60 nodes); `barycenter ordering` (two passages each linked to one of two claims are ordered like their claims); `every edge path starts and ends on its nodes' sides`. `HistoryGraph.test.tsx`: `history shows badges and loads more`; `admin can switch to workspace view` (request has `mine=false`); `graph nodes are focusable and selectable by keyboard` (Tab to a node, Enter -> details box shows its label and edges); `redacted passage node is labeled hidden`; `graph empty state`.
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**; delete `DashboardPage.tsx` with `git rm`.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add analysis history and evidence graph view`.

### Task 10: Quick check page and settings

**Files:**
- Create: `frontend/src/pages/QuickCheckPage.tsx`, `frontend/src/pages/SettingsPage.tsx`, `frontend/src/pages/QuickSettings.test.tsx`
- Modify: `frontend/src/components/ClaimCard.tsx` (reads `LegacyAnalyzeResponse` claims from `api/types.ts`; tier labels become `SUPPORTED`, `NOT ENOUGH EVIDENCE` for `uncertain`, and `CONTRADICTED` for `hallucinated`, each still with its icon; the three verifier cells become `NLI` (entailment, neutral, contradiction) and `Calibrated` (`P(supported)`, `P(contradicted)`, `P(insufficient)`) from `verifier_details.calibrated`; feedback calls the legacy endpoint through `endpoints.ts`), `frontend/src/components/DocumentInput.tsx` (unchanged API), `frontend/src/App.tsx`
- Delete: `frontend/src/api/client.ts` (confirm nothing imports it)

**Interfaces:**
- `QuickCheckPage` (`/quick`): the former paste-a-document flow and layout (hero, `LOAD DEMO`, `DocumentInput`, run button, `ScoreGauge`, claim cards, `ChunkViewer`) calling `legacyAnalyze`; copy updated to "Paste a document and a question. EvidenceGraph answers from that document only and checks every claim against it."; a note links to `/collections` for versioned, multi-document work; 504 `analysis_pending` shows "Still running" with a link to the analysis id from the `Location` header.
- `SettingsPage` (`/settings`), three tabs with `role="tablist"`: `API keys` (list with name, prefix shown as `egk_<prefix>_...`, created, last used; create form `Key name`; the created token appears once in a read-only field with a `Copy` button and the warning "This key will not be shown again."; `Revoke` behind a confirm step); `Members` (owners and admins: list with role, add by `Email` and `Role`, remove with confirm; members see the list read-only); `Audit log` (owners and admins: action, user id, target, time, `Load more`).

- [ ] **Step 1: Write failing tests**: `quick check renders legacy result with new labels` (fake `/api/analyze` returns two claims, one `hallucinated` -> `CONTRADICTED` label and calibrated probabilities shown); `quick check pending shows link`; `api key token shown once` (after create, token visible with the warning; after reload of the list, no token anywhere); `revoke requires confirmation`; `members tab hidden actions for members`; `audit tab loads more`; `no module imports the old client` (static check over `src/` that `api/client` is not imported).
- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement**; delete `src/api/client.ts` with `git rm`.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add quick check and workspace settings pages`.

### Task 11: Demo corpus, `halluciscope seed-demo`, and `scripts/dev-up.sh`

**Files:**
- Create: `demo/northwind/terms-of-service.v1.md`, `terms-of-service.v2.md`, `employee-handbook.v1.md`, `employee-handbook.v2.md`, `payments-api.v1.md`, `payments-api.v2.md`, `security-whitepaper.md`, `demo/northwind/manifest.json`, `backend/cli/seed_demo.py`, `scripts/dev-up.sh`, `tests/test_seed_demo.py`
- Modify: `backend/cli/main.py`

**Interfaces:**
- Demo corpus (authored for this repository; Northwind Cloud is a fictional company, stated at the top of each file): each file 600 to 1,500 words of realistic policy prose in Markdown with headings. Version changes by construction: terms of service v1 effective `2026-01-15T00:00:00Z` -> v2 `2026-06-01T00:00:00Z` (refund window 30 -> 14 days; data retention after cancellation 90 -> 30 days; liability cap 12 -> 6 months of fees); handbook v1 `2026-02-01` -> v2 `2026-07-01` (annual paid leave 20 -> 25 days; remote work 2 -> 3 days per week); payments API v1 `2026-03-01` -> v2 `2026-08-15` (rate limit 100 -> 60 requests per second; access tokens expire after 60 -> 15 minutes; `/v1/charges/legacy` removed); the security whitepaper (single version `2026-04-01`) states the 90-day retention, so it conflicts with terms of service v2 for the conflicting-evidence demo.
- `manifest.json`: `{"collection": "Northwind Cloud policies", "documents": [{"key", "title", "versions": [{"file", "effective_from"}]}], "questions": [{"question", "as_of": str | null, "expected": str}]}` with six questions covering each change and the conflict (for example `{"question": "How long is the refund window?", "as_of": "2026-03-01T00:00:00Z", "expected": "30 days"}` and the same question live with `"14 days"`).
- `backend/cli/seed_demo.py`: `async def seed_demo(services: Services, *, email: str, password: str | None, manifest_path: Path) -> SeedResult` (`SeedResult(user_created: bool, generated_password: str | None, collection_id: UUID, versions_created: int, duplicates: int)`): creates the user and personal tenant through `AuthService.register` when missing (password from the argument, else `secrets.token_urlsafe(18)` returned once), creates the collection when missing, and for each document uploads its versions in order with their `effective_from` through `intake_upload(..., enqueue_job=False)` and `process_upload`, so the seed works without a running worker; re-running yields only duplicates. CLI `halluciscope seed-demo [--email demo@northwind.example] [--manifest demo/northwind/manifest.json]` reads the password from `HALLUCISCOPE_DEMO_PASSWORD` when set, prints the generated password only when it generated one, and never logs it.
- `scripts/dev-up.sh` (`set -euo pipefail`, LF): starts Postgres with `scripts/dev-postgres.sh start`; runs `halluciscope migrate`; warns (does not fail) when `backend/calibration/artifacts/default.json` is missing or `curl -fsS http://localhost:11434/api/version` fails; exports `HALLUCISCOPE_DEV_INSECURE_COOKIES=true`; starts `halluciscope worker` (log `.local/worker.log`), `uvicorn backend.api.startup:create_configured_app --factory --host 127.0.0.1 --port 8000` (log `.local/api.log`), and `npm --prefix frontend run dev`; a `trap` stops all three on exit; prints `Open http://localhost:5173`.

- [ ] **Step 1: Write failing tests** in `tests/test_seed_demo.py` (all `pg`; `build_services` patched with the fakes from `tests/fakes.py`): `test_manifest_valid` (every file exists, effective dates increase per document, six questions); `test_seed_creates_versions` (three documents with two versions and one with one, effective dates as in the manifest, `versions_created == 7`, `generated_password` present when none given); `test_seed_idempotent` (second run: `versions_created == 0`, `duplicates == 7`, `user_created` false, no generated password); `test_demo_files_state_fictional_company` (each Markdown file's first 300 characters mention "fictional"); `test_cli_does_not_echo_supplied_password` (CLI with `HALLUCISCOPE_DEMO_PASSWORD` set: output does not contain it).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_seed_demo.py -q`. Expected: FAIL.
- [ ] **Step 3: Write** the corpus and manifest; implement the seed module and CLI command; write `scripts/dev-up.sh` and run `bash -n scripts/dev-up.sh`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add fictional demo corpus and local stack scripts`.

### Task 12: Docker deployment

**Files:**
- Create: `deploy/Dockerfile.api`, `deploy/Dockerfile.frontend`, `deploy/nginx.conf`, `deploy/docker-compose.yml`, `deploy/initdb/01-roles.sh`, `deploy/.env.example`, `.dockerignore`, `tests/test_deploy_files.py`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]` with `"backend.db" = ["migrations/*.sql"]` and `"backend.calibration" = ["artifacts/*.json"]`), `.gitignore` (`deploy/.env`)

**Interfaces:**
- `Dockerfile.api`: `FROM python:3.14-slim`; `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models`; installs `torch==2.14.1` from `https://download.pytorch.org/whl/cpu` first, then the project with `pip install .`; `WORKDIR /app` with `backend/` and `pyproject.toml` copied; a system user `app` (uid 10001) owns `/app` and `/models`; `USER app`; `EXPOSE 8000`; default `CMD ["uvicorn", "backend.api.startup:create_configured_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]` (Plan A final review ruling: the login throttle keys on the client IP, so the API must trust the X-Forwarded-For set by the nginx container; the API port is never published, so only nginx can reach it), and nginx sets `X-Forwarded-For $remote_addr` (overwriting any client value). The same image runs `halluciscope migrate` and `halluciscope worker`.
- `Dockerfile.frontend`: stage `node:24-alpine` runs `npm ci` and `npm run build` in `frontend/`; final stage `nginxinc/nginx-unprivileged:1.29-alpine` with the build in `/usr/share/nginx/html` and `deploy/nginx.conf` as `/etc/nginx/conf.d/default.conf`; listens on 8080.
- `nginx.conf`: SPA fallback `try_files $uri /index.html`; `location /api/` proxies to `http://api:8000` with `X-Request-ID $request_id` when the client sent none, `X-Forwarded-For`, and `client_max_body_size 21m`; static responses carry `X-Content-Type-Options nosniff`, `Referrer-Policy no-referrer`, `X-Frame-Options DENY`, and `Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"` (the font hosts are the ones `frontend/index.html` loads); `/assets/` cached for a year, `index.html` `no-cache`.
- `01-roles.sh` (run by the Postgres image on first init): creates `eg_owner` and `eg_app` (`LOGIN NOSUPERUSER NOBYPASSRLS`) with passwords from `EG_OWNER_PASSWORD` and `EG_APP_PASSWORD`, database `evidencegraph` owned by `eg_owner`, and extensions `vector` and `citext` in it.
- `docker-compose.yml` (project name `evidencegraph`): `postgres` (`pgvector/pgvector:pg17`, volume `pgdata`, the init script mounted read-only, `pg_isready` healthcheck, no published ports); `migrate` (api image, `halluciscope migrate`, `restart: "no"`, depends on healthy postgres); `api` (depends on `migrate` `service_completed_successfully`; `HALLUCISCOPE_DATABASE_URL` and `HALLUCISCOPE_MIGRATION_DATABASE_URL` built from the passwords; `HALLUCISCOPE_CORS_ORIGINS=["http://localhost:8080"]`; `HALLUCISCOPE_DEV_INSECURE_COOKIES=true` with a comment that it must be false behind TLS; `HALLUCISCOPE_OLLAMA_BASE_URL=${EG_OLLAMA_URL:-http://host.docker.internal:11434}`; `extra_hosts: ["host.docker.internal:host-gateway"]`; healthcheck `python -c` fetching `/api/v1/health/live`; model cache volume `models`); `worker` (same environment, `halluciscope worker`, volume `models`, `stop_grace_period: 60s`); `frontend` (ports `"127.0.0.1:8080:8080"`); `ollama` (profile `ollama`, `ollama/ollama:0.40.11`, volume `ollama`; with this profile set `EG_OLLAMA_URL=http://ollama:11434`). Every secret uses `${VAR:?set it in deploy/.env}`.
- `deploy/.env.example`: `EG_OWNER_PASSWORD=change-me`, `EG_APP_PASSWORD=change-me`, `POSTGRES_PASSWORD=change-me`, `HALLUCISCOPE_METRICS_TOKEN=`, `EG_OLLAMA_URL=`.

- [ ] **Step 1: Write failing tests** in `tests/test_deploy_files.py`: `test_wheel_ships_migrations_and_artifact` (`pip wheel . --no-deps -w <tmp>`; the wheel lists every `backend/db/migrations/*.sql` and, when present, `backend/calibration/artifacts/default.json`); `test_api_image_runs_as_non_root` (`USER app` after the last `RUN`; base image pinned to `python:3.14-slim`); `test_frontend_image_unprivileged`; `test_no_public_ports` (every published port mapping in the compose file starts with `127.0.0.1:`); `test_compose_services_present` (`postgres`, `migrate`, `api`, `worker`, `frontend`, `ollama`); `test_env_example_has_no_secrets` (every value is `change-me` or empty); `test_csp_matches_index_html_hosts` (every external host in `frontend/index.html` appears in the nginx CSP); `test_dockerignore_excludes_local_state` (`.venv`, `.local`, `evaluation/data`, `evaluation/results`, `node_modules`, `.git`, `docs/superpowers`, `.superpowers`, `deploy/.env`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_deploy_files.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the files and the packaging change.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Verify the stack for real.** If `docker` is not installed, install Colima and the Docker CLI with Homebrew (`brew install colima docker docker-compose`; mention it in the task report as an environment change) and start `colima start --memory 6 --cpu 4`. Then, with Ollama on the host: `cp deploy/.env.example deploy/.env` with generated passwords, `docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build -d`, wait for health, `curl -fsS http://127.0.0.1:8080/api/v1/health/ready`, register through the API, run `docker compose ... exec api halluciscope seed-demo` (password from the environment), and run one ask through the API to `succeeded`. Record exact commands and outcomes in the report. If the machine cannot run the stack (memory), record the observed failure verbatim, stop the containers, and report it as a known limitation; do not claim the Docker path works. Afterwards `docker compose ... down` (keep volumes) and `colima stop`.
- [ ] **Step 6: Commit** the deploy files, `.dockerignore`, `.gitignore`, `pyproject.toml`, and the test with `build: add Docker deployment for the full stack`.

### Task 13: Continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `tests/test_deploy_files.py`

**Interfaces:**
- Workflow `ci` on `push` to any branch and `pull_request`; `permissions: contents: read`; `concurrency` cancels superseded runs per ref.
- Job `backend` (`ubuntu-24.04`): service `postgres` (`pgvector/pgvector:pg17`, `POSTGRES_HOST_AUTH_METHOD=trust`, port `5432:5432`, `pg_isready` health options); steps: checkout; setup Python `3.14`; `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1`; `pip install -e ".[dev,eval]"`; `ruff check .`; `pytest -q` with `HALLUCISCOPE_TEST_ADMIN_URL=postgresql://postgres@localhost:5432/postgres` (the `models` marker stays excluded; the `citext` extension ships with the image's contrib modules).
- Job `frontend` (`ubuntu-24.04`): checkout; setup Node `24` with npm cache keyed on `frontend/package-lock.json`; `npm ci`, `npm run lint`, `npm test`, `npm run build` in `frontend/`.
- Every `uses:` pins a full 40-character commit SHA with the release tag in a trailing comment, looked up at implementation time with `gh api repos/<owner>/<repo>/releases/latest --jq .tag_name` and `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_deploy_files.py`): `test_ci_actions_pinned_by_sha` (every `uses:` line matches `@[0-9a-f]{40}`); `test_ci_permissions_read_only`; `test_ci_runs_all_gates` (the workflow text contains `ruff check .`, `pytest -q`, `npm run lint`, `npm test`, `npm run build`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_deploy_files.py -q`. Expected: FAIL.
- [ ] **Step 3: Write** the workflow with looked-up SHAs.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS. The workflow itself is verified on the first push in the finishing step (`gh run watch`), not here.
- [ ] **Step 5: Commit** with `ci: run backend and frontend gates on every push`.

### Task 14: Architecture, decision records, security, operations, and evaluation docs

**Files:**
- Create: `docs/architecture.md`, `docs/decisions/0001-postgres-centric-monolith.md`, `0002-bm25-in-sql.md`, `0003-api-worker-split.md`, `0004-calibrated-verdicts-with-risk-control.md`, `0005-immutable-versions-and-provenance.md`, `0006-row-level-security-tenancy.md`, `0007-evaluation-protocol.md`, `docs/security.md`, `docs/operations.md`, `docs/evaluation.md`, `tests/test_docs.py`

**Interfaces (content requirements; the writer reads the spec, the plans, and the code, and describes what the code actually does):**
- `architecture.md`: overview; process model with a Mermaid diagram (browser, nginx or Vite, API, Postgres, worker, Ollama); request flows for upload, ask, verify, quick check, and re-verification as numbered steps naming the modules; data model summary (tables and the integrity guarantees: slice trigger, composite foreign keys, immutability triggers, RLS); retrieval (visible set, BM25 formula and parameters, dense with iterative scan, RRF, rerank); verification (claims, evidence gathering, NLI, the 12 features, the calibrator, the decision rule); provenance and the bitemporal as-of model (R9); observability; known scale limits.
- Each ADR has the headings `Status`, `Context`, `Decision`, `Consequences`, and `Alternatives considered`, with status `Accepted` and date 2026-10-26, drawn from spec sections 5, 10, 14, 15, 17, 20 and rulings R1 to R10.
- `security.md`: assets, actors, trust boundaries; a controls table (`Control | Implementation | Test`) naming the module and test file for each of: password hashing, session tokens, CSRF, login throttling, API keys, roles and collection permissions, 404-not-403 for invisible objects, RLS, BM25 statistics scoped to readable passages, evidence redaction after revocation, upload limits and parse isolation, prompt-injection containment, log hygiene, security headers, CORS allowlist, audit log, pinned dependencies; known gaps (collection names are visible within a workspace through the duplicate-name 409, by design; adding a member by email reveals whether that email has an account and its display name to workspace admins, and adds without the user's consent (no invitation flow); no SSO, no MFA, no malware scanning of uploads, soft delete only, rate limits only on login and analyses, single Postgres, `DEV_INSECURE_COOKIES` must be off behind TLS); how to report a vulnerability (open a private GitHub security advisory).
- `operations.md`: a configuration table listing every `Settings` field as its environment variable (`HALLUCISCOPE_<FIELD>` uppercase), default, and meaning; running locally (`scripts/dev-postgres.sh`, `scripts/dev-up.sh`) and with Docker; migrations; worker operation (leases, retries, dead jobs and how to inspect them with SQL); health and readiness semantics; metrics list with types and labels; backups (`pg_dump -Fc`) and restore; model cache; troubleshooting (`OMP: Error #15`, `calibrator_missing`, `calibrator_mismatch`, `llm_unavailable`, stale worker, `search_timeout`).
- `evaluation.md`: datasets with pinned sources and hashes, splits and leakage controls (R1, source-level, test evaluated once and logged), units and label alignment, systems and baselines, metrics definitions, risk-control procedure and what the guarantee does and does not cover, ablations, synthetic temporal design, exact reproduction commands for every run, links to `docs/results/*.md`, and limitations (spec section 25).
- `tests/test_docs.py`: `test_operations_lists_every_setting` (every `Settings.model_fields` name appears as `HALLUCISCOPE_<NAME>`); `test_adrs_have_required_sections`; `test_relative_links_resolve` (every relative Markdown link in `README.md`, `docs/*.md`, `docs/decisions/*.md` points to an existing file); `test_no_em_dashes_in_docs` (`README.md`, `docs/**/*.md` outside `docs/superpowers`, every `AGENTS.md`, and `demo/**/*.md`); `test_security_table_tests_exist` (every test file named in the security controls table exists).

- [ ] **Step 1: Write the failing tests** in `tests/test_docs.py`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_docs.py -q`. Expected: FAIL.
- [ ] **Step 3: Write** the documents.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `docs: add architecture, decisions, security, operations, and evaluation guides`.

### Task 15: End-to-end walkthrough with real models, screenshots, and fixes

This task needs a browser automation tool (the Playwright MCP browser tools; otherwise the Playwright CLI at the exact version printed by `npm view playwright version`, recorded in the report) and real models. If the implementer has no browser tool, the controller runs this task directly and records the same evidence.

**Files:**
- Create: `docs/images/ask-result.png`, `docs/images/evidence-inspector.png`, `docs/images/source-updated.png`, `docs/images/version-diff.png`, `docs/images/evidence-graph.png`, `docs/images/quick-check.png`
- Modify: any file needed to fix a defect found, each with a failing test first

- [ ] **Step 1: Start** Ollama (`ollama serve`, `llama3.2:3b` pulled), then `bash scripts/dev-up.sh`; `HALLUCISCOPE_DEMO_PASSWORD=<generated> .venv/bin/python -m backend.cli.main seed-demo` with only v1 of the terms of service (temporarily point `--manifest` at a copy listing v1 only, kept under `.local/`).
- [ ] **Step 2: Walk through** at 1440x900: sign in as the demo user; open the collection and a document; ask "How long is the refund window?" live and confirm a supported claim citing terms of service v1 with the span on the refund sentence; open the inspector (screenshot `evidence-inspector.png`, and the result view as `ask-result.png`); upload terms of service v2 through the collection page and wait for `Ready (v2, ...)`; wait for re-verification and confirm the claim shows `Source updated` and is now contradicted (`source-updated.png`); open the diff from the banner (`version-diff.png`); ask "How long is data kept after cancellation?" live and confirm the conflicting-evidence view between terms of service v2 and the security whitepaper; ask the refund question with `As of 2026-03-01` and confirm it answers from v1; open the evidence graph (`evidence-graph.png`); run the quick check demo (`quick-check.png`); create and revoke an API key.
- [ ] **Step 3: Check hygiene:** the browser console shows no errors or React warnings on any page visited; `grep -c "refund window" .local/api.log .local/worker.log` prints 0 for both; every API response in the network log for the visited pages is 2xx or an expected 4xx.
- [ ] **Step 4: Fix defects.** For each defect found, follow superpowers:systematic-debugging (`/Users/omarmorsi/.kiro/skills/sp-systematic-debugging/SKILL.md`): reproduce, write a failing test in the owning module's test file, fix, and commit each fix separately (`fix: ...`). Repeat Step 2 for the affected flow.
- [ ] **Step 5: Verify** all gates: `.venv/bin/ruff check .`, `.venv/bin/python -m pytest -q`, and `npm run lint && npm test && npm run build` in `frontend/`.
- [ ] **Step 6: Commit** the screenshots with `docs: add product walkthrough screenshots`. The task report lists every step's observed outcome and every defect fixed.

### Task 16: README, results summary, license, and agent guidance

**Files:**
- Create: `docs/results/summary.json`, `LICENSE`, `tests/test_readme_summary.py`
- Modify: `evaluation/report.py`, `README.md` (rewritten), `AGENTS.md`, `backend/AGENTS.md`, `evaluation/AGENTS.md`, `tests/AGENTS.md`, `frontend/AGENTS.md`, `frontend/README.md` (replace the Vite template text with a short pointer to the root README and the frontend commands)

**Interfaces:**
- `evaluation/report.py`: `def build_summary(retrieval: Sequence[Mapping], factuality: Mapping, temporal: Mapping, latency: Mapping) -> dict` with exactly: per BEIR dataset `bm25` and `hybrid_rerank` nDCG@10 and Recall@100; RAGTruth test `evidencegraph` and `legacy_as_shipped` detection F1 with intervals, false-supported rate with intervals, accepted risk with its Clopper-Pearson upper bound, coverage, and ECE for `evidencegraph`; temporal correctness, evidence validity rate, and `cross_tenant_links`; latency p50 and p95 for `total`; plus `env` (CPU, RAM, git commit of each run) and `sources` (the results file names). `def render_readme_block(summary: Mapping) -> str` (a Markdown table per area with each figure's dataset and scale, and links to the full reports). CLI `python -m evaluation.report summary --retrieval P... --factuality P --temporal P --latency P --out docs/results/summary.json` and `python -m evaluation.report readme --summary docs/results/summary.json --readme README.md` (replaces the text between `<!-- results:start -->` and `<!-- results:end -->`; missing markers -> exit 2).
- README sections: title and one-paragraph description; screenshot (`docs/images/ask-result.png`); what it does (upload, ask, verify, inspect evidence, contradictions, history, graph); how it works (short, linking `docs/architecture.md`); results (the generated block, with a sentence that every figure comes from `docs/results/` and the scripts that produced it); quickstart for macOS with Homebrew (`postgresql@17`, `pgvector`, Ollama, Python 3.14, Node 24 or newer) and for Docker; demo (`halluciscope seed-demo` and the six manifest questions); reproducing the benchmarks (commands from `docs/evaluation.md`); API overview (main routes, auth model, link to `/docs` OpenAPI); security summary linking `docs/security.md`; limitations (spec section 25 plus any Docker limitation found in Task 12); project layout; license.
- `LICENSE`: MIT, `Copyright (c) 2026 Omar Morsi`.
- AGENTS.md files describe the current layout, commands, invariants (exact offsets, verdicts from the calibrator only, no content in logs or job payloads, no `dangerouslySetInnerHTML`), and the committed benchmark artifacts.

- [ ] **Step 1: Write failing tests** in `tests/test_readme_summary.py`: `test_build_summary_from_fixtures` (fixture results -> exactly the specified keys); `test_readme_block_matches_summary` (the text between the markers in `README.md` equals `render_readme_block(json.load(open("docs/results/summary.json")))`); `test_summary_values_trace_to_reports` (each headline number in `summary.json`, formatted as in the reports, appears in the corresponding `docs/results/*.md` file).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_readme_summary.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the two functions and CLI commands; generate `summary.json` from the real results JSON files under `evaluation/results/`; write the README with the markers and generate the block; write `LICENSE` and the AGENTS.md updates.
- [ ] **Step 4: Run** every gate: `.venv/bin/ruff check .`, `.venv/bin/python -m pytest -q`, `npm run lint && npm test && npm run build` in `frontend/`. Expected: PASS.
- [ ] **Step 5: Commit** with `docs: rewrite README with generated results and setup guide` (README, summary, report changes, test, LICENSE, AGENTS files, frontend README).
