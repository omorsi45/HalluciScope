# Task 16 report: evidence graph, claim feedback, claim check routes

Branch `feature/evidencegraph`, worktree `.worktrees/evidencegraph`. Base commit `4ec6772`.

## RED

Tests written first in `tests/test_graph_feedback.py` (14 test cases: the 6 from the brief plus labels cut at
120 characters, graph of an invisible analysis, feedback validation (5 params), feedback note at the 1000 limit).

Command (inside the Windows kill bound):

```
.venv\Scripts\python.exe .local\run.py 340 .local\red.log -- wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 280 python -m pytest tests/test_graph_feedback.py -q --no-header -p no:cacheprovider --tb=line
```

Result: `14 failed in 6.59s`, exit 1. Every failure is `assert 404 == 200/201/422` with the body
`{"type":"/errors/not_found","code":"not_found"}`: the routes do not exist yet. Seeding (team, collections, ask and
verify analyses through the worker) worked, so the failures are the missing feature, not test setup.


## What was built

- `backend/provenance/graph.py` (new)
  - `build_graph(conn, analysis_id, readable) -> {"nodes", "edges"}`. Node ids `claim:<uuid>`, `sentence:<start>:<end>`,
    `passage:<uuid>`, `version:<uuid>`, `document:<uuid>`; labels cut at 120 characters (claim text, answer sentence,
    passage text), `v<number>`, document title. Edge types `supports|contradicts|related` (claim to passage), `cites`
    (sentence to passage, valid citations only), `part_of`, `version_of`, `superseded_by` (version to the document's
    current version when different). Nodes unique and sorted by id; edges unique and sorted by (source, target, type).
  - `passage_collection_ids(conn, analysis_id)`: collections of every passage the analysis links or cites; the route
    feeds it to `readable_subset`.
  - Redaction is enforced in SQL: the passage query only returns passages of readable collections, so withheld text,
    titles and version data never leave the database. A passage not returned becomes one node with `label None`,
    `data {"redacted": true}`, no `part_of` edge, and no version or document node. Edges into it (claim and `cites`)
    stay.
- `backend/api/schemas_v1.py`: `FeedbackNote` (max 1000, surrogate and NUL checked), `FeedbackIn`, `FeedbackOut`,
  `GraphNodeOut`, `GraphEdgeOut`, `GraphOut`.
- `backend/api/routes/analyses.py`:
  - `GET /api/v1/analyses/{id}/graph` (`require_analysis`, then `readable_subset`, then `build_graph`).
  - `POST /api/v1/claims/{claim_id}/feedback` -> 201 `{id}`, upsert per user via `analyses.upsert_feedback`.
  - `GET /api/v1/claims/{claim_id}/checks` -> `list[ClaimCheckOut]`, newest first (`created_at desc, id desc`).
  - `_require_claim`: loads the claim (RLS hides other tenants) and applies `require_analysis`; any refusal becomes
    404 `claim_not_found`. The visibility rule stays in one place (`policy.require_analysis`).
- `tests/test_graph_feedback.py` (new): 14 test cases, all `pg`.

Files changed (committed): `backend/provenance/graph.py`, `backend/api/routes/analyses.py`,
`backend/api/schemas_v1.py`, `tests/test_graph_feedback.py`.

## GREEN

```
pytest tests/test_graph_feedback.py -q  ->  14 passed in 5.94s
```

Mutation check (to prove the redaction test has teeth): temporarily set `readable = collection_ids` in the graph
route. `test_graph_redacts_unreadable` failed with `assert 'Refunds take 30 days.' is None`. Reverted.

## Full suite and lint

- `pytest -q` (WSL, timeout 900): `1394 passed, 3 deselected, 2 warnings in 226.66s`, exit 0. The 2 warnings are the
  existing `tests/test_generator.py` "coroutine never awaited" warnings (unrelated).
- `ruff check .`: `All checks passed!`
- `ruff format --check` on the four changed files: clean after formatting my two new files.
- Commit hooks (gitleaks) ran: no leaks.

## Commit

`da39823 feat: add evidence graph, claim feedback, and claim check routes` (4 files, staged by name, not pushed).

## Tests (tests/test_graph_feedback.py)

From the brief: `test_graph_shape`, `test_graph_superseded_edge`, `test_graph_redacts_unreadable`,
`test_feedback_upsert_route`, `test_feedback_on_invisible_claim_404`, `test_checks_route_empty_then_listed`.
Added: `test_graph_labels_are_cut_at_120_characters`, `test_graph_of_invisible_analysis_404`,
`test_feedback_validation` (no label, bad label, note 1001 chars, lone surrogate, NUL), `test_feedback_note_limit_is_accepted`.

Notes on the tests: the shape test derives its expectations from `GET /analyses/{id}` plus a DB lineage lookup and
compares full node and edge sets, label contents, sorting, uniqueness, endpoint closure and two-call equality. Its
answer has one sentence citing two passages and one sentence citing only the unknown marker `[P9]`, so it also pins
that an unknown marker gives no sentence node and no `cites` edge. My first answer fixture had a third uncited
sentence, which the pipeline turned into a third fallback claim; I changed the fixture, not an assertion.
The feedback test uses two users (Bob the creator posts twice, Alice the tenant owner once) to show the upsert is per
user: same id for Bob, a different id for Alice, two rows.

## Decisions

1. Sentence nodes exist only for sentences with at least one valid citation. The brief does not say; a sentence with
   only unknown markers would be an isolated node, so it is left out.
2. `FeedbackNote` is `Description` (max 1000, surrogate check) plus the existing NUL check. Postgres text cannot hold
   NUL, so without it a note containing NUL would be a 500 instead of a 422. Slightly stricter than the brief's
   "shared surrogate-checked string type".
3. The checks list query and the claim lookup are SQL inside the routes module, like the existing
   `_with_changed_checks`, `_citation_passages` and `_span_texts`, to stay inside the files the brief lists. No new repo
   function.
4. Node `data`: claim `{ordinal, verdict}`; sentence `{start, end}`; passage `{redacted: false, start_char, end_char,
   page_start, page_end}` (redacted: `{redacted: true}`); version `{version_number, is_current}`; document `{filename}`.
   The brief fixes only the redacted shape.
5. The current version node of a superseded version also gets its `version_of` edge to the document, so the graph has
   no dangling version.
6. The graph is not paginated or size capped (an analysis has at most 50 claims times 4 evidence items plus citations).

## Concerns

- `GET /claims/{id}/checks` returns all checks unpaginated, as the brief specifies. Checks append on each source
  change, so a long lived claim could grow the list.
- `backend/api/schemas_v1.py` is CRLF in the working copy (existing state), LF in the index; the diff shows only my 38
  added lines, so no line-ending churn was committed.
