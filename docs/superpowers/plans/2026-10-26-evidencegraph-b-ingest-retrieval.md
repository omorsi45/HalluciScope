# EvidenceGraph Plan B: Ingestion and Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Versioned, offset-exact document ingestion (PDF, text, Markdown, HTML, CSV) into Postgres, original hybrid retrieval (BM25 in SQL, pgvector dense, RRF, cross-encoder rerank) with access control and bitemporal as-of, and a real BEIR retrieval benchmark.

**Architecture:** Parsing runs in a killable subprocess. Ingestion in the worker writes an immutable version, its passages (slice-checked by a trigger), and term postings (filled by a trigger) in one transaction, reusing embeddings by passage hash. `PostgresRetriever` computes BM25 statistics over the caller's visible passage set only, fuses BM25 and dense rankings with RRF, and optionally reranks. The API loads only the embedder (R7); reranked search runs as a worker job.

**Tech Stack:** Python 3.14.7, PostgreSQL 17.11 + pgvector 0.8.7 (`pgvector==0.5.1` adapter), psycopg 3.3.6, pypdf 6.21.2, sentence-transformers 6.2.0 (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`), stdlib `html.parser`, `csv`, `difflib`.

**Spec:** `docs/superpowers/specs/2026-10-26-evidencegraph-design.md` (sections 8, 9, 10, 15 diff, 16, 20.1, and rulings R2, R4, R5, R7, R9, R10a in section 26, which override earlier sections). Plan A (`docs/superpowers/plans/2026-10-26-evidencegraph-a-foundation.md`) is complete before this plan starts.

## Global Constraints

- Every Plan A Global Constraint applies unchanged (worktree path, `/usr/bin/git`, staging by name, commit style, no em dashes, pinned versions, parameterized SQL, logging hygiene, problem+json errors, `/api/v1`, `pg` marker tests fail loudly, unit tests never download models).
- Job failure hooks (Plan A final review ruling): every job kind whose failure must reach a domain row registers a hook in `build_failure_handlers(services)` (backend/jobs/registry.py) with signature `async def hook(ctx: JobContext, error_code: str) -> None`; the worker calls it when a job ends `failed` or `dead` (including lease expiry). Plan B registers `ingest_upload` (sets the upload `failed` with that `error_code` unless it is already terminal) and `search` (sets the `search_requests` row `failed`). Hooks are idempotent and never raise.
- Collection kinds (Plan A final review ruling): `require_collection(conn, principal, collection_id, needed, *, kinds=("normal",))` filters by kind; `intake_upload` gains `collection_kinds: Sequence[str] = ("normal",)` passed through to it. Benchmark ingestion calls `create_version` directly and never goes through `intake_upload`.
- Error codes from Plan A: unknown API keys are `api_key_not_found`; non-member grants are `grantee_not_member`. Request string fields use the shared length- and surrogate-checked type from `backend/api/schemas_v1.py`.
- Tenant integrity (ruling from Plan A Task 7 review): every foreign key from one tenant-scoped table to another includes `tenant_id`. Each referenced tenant table declares `unique (tenant_id, id)` (and composite uniques such as `unique (tenant_id, id, version_id)` where a composite reference needs them), and references are written `foreign key (tenant_id, <x>_id) references <table> (tenant_id, id)`. References to global tables (`users`, `tenants`) stay single-column. A test per migration asserts that a row pointing at another tenant's parent raises `ForeignKeyViolation`.
- Canonical text offsets are Python `str` indices (Unicode code points). Postgres `substr` on UTF-8 text uses the same units. A passage's `text` always equals `version_text[start_char:end_char]`.
- Ingestion limits: 20,971,520 bytes per file; 2,000,000 canonical characters per version; 5,000 passages per version; 50,000 CSV data rows; 60 s parse timeout.
- Passage defaults: `target_chars = 800`, `max_chars = 1200`, `overlap_sentences = 1`.
- BM25 parameters fixed a priori: `k1 = 0.9`, `b = 0.4`. RRF `k = 60`. Default `candidates = 50`. HNSW `m = 16`, `ef_construction = 64`, query `hnsw.ef_search = 100`, `hnsw.iterative_scan = relaxed_order`.
- Embedder `BAAI/bge-small-en-v1.5`, dimension 384, L2-normalized; queries prefixed with `Represent this sentence for searching relevant passages: `; passages unprefixed. Reranker `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Job payloads and job errors contain only ids, counts, and error codes (R4). Query text lives in the RLS table `search_requests`.
- Ingest error codes (stable strings): `unsupported_type`, `bad_encoding`, `bad_pdf`, `encrypted_pdf`, `no_text`, `too_many_rows`, `bad_csv`, `too_large`, `too_long`, `too_many_passages`, `parse_timeout`, `effective_from_out_of_order`, and the upload-only code `document_deleted`.
- Benchmark data downloads go to `evaluation/data/` (gitignored). Raw results go to `evaluation/results/` (gitignored). Only `docs/results/*.md` and `evaluation/manifests/*.json` are committed. Never write a number into docs that a committed script did not produce.

## Review Focus

1. Uploading identical bytes again, or a re-saved file whose canonical text is unchanged, must not create a version or a re-verification job: upload status `duplicate`, `version_id` = current version. Tests in Tasks 9 and 10.
2. Non-ASCII text (combining accents, emoji, CJK) must keep exact offsets end to end: passage insert passes the slice trigger and `GET .../text?start=&end=` returns the same characters. Test in Task 9.
3. A passage from a collection the caller cannot read, or from a soft-deleted document, never appears in any search mode, and BM25 scores for a readable passage are identical whether or not an unreadable collection contains the query terms. Tests in Tasks 13 and 16.
4. As-of queries: before a document's first version it is invisible; between versions the older version is used; a version uploaded later with a backdated `effective_from` does not change results for an earlier `known_at`. Tests in Tasks 13 and 16.
5. An empty file, a whitespace-only file, or an image-only PDF ends as upload status `failed` with `error_code = no_text`, and the job is `failed` without retries. Test in Task 10.

## File Map

Created
- `backend/ingest/__init__.py`, `types.py`, `normalize.py`, `assemble.py`, `segment.py`, `passages.py`, `errors.py`, `service.py`, `intake.py`, `handlers.py`, `diff.py`, `parse_child.py`
- `backend/ingest/parsers/__init__.py` (dispatch), `text.py`, `markdown.py`, `html.py`, `csv_table.py`, `pdf.py`
- `backend/db/migrations/0003_documents.sql`, `backend/db/repos/documents.py`
- `backend/models/protocols.py`, `backend/services.py`
- `backend/retrieval/__init__.py`, `types.py`, `visible.py`, `bm25.py`, `dense.py`, `fts.py`, `fusion.py`, `rerank.py`, `service.py`, `handlers.py`
- `backend/api/routes/documents.py`, `backend/api/routes/search.py`
- `evaluation/data.py`, `evaluation/beir.py`, `evaluation/metrics/__init__.py`, `evaluation/metrics/retrieval.py`, `evaluation/run_retrieval.py`, `evaluation/report.py`, `evaluation/manifests/beir.json`, `docs/results/retrieval.md`
- Tests: `tests/fakes.py`, `tests/test_normalize.py`, `tests/test_segment.py`, `tests/test_passages.py`, `tests/test_parsers_text_md.py`, `tests/test_parsers_html_csv.py`, `tests/test_parsers_pdf.py`, `tests/test_schema_documents.py`, `tests/test_model_adapters.py`, `tests/test_ingest_versions.py`, `tests/test_ingest_upload.py`, `tests/test_document_routes.py`, `tests/test_diff.py`, `tests/test_bm25.py`, `tests/test_dense_fts.py`, `tests/test_fusion_rerank.py`, `tests/test_retriever.py`, `tests/test_search_route.py`, `tests/test_retrieval_metrics.py`, `tests/test_beir.py`, `tests/test_report.py`

Modified
- `backend/config.py`, `backend/db/pool.py`, `backend/models/loader.py`, `backend/jobs/registry.py`, `backend/cli/main.py`, `backend/api/app.py`, `backend/api/startup.py`, `backend/api/schemas_v1.py`, `backend/api/routes/collections.py`, `tests/pg_fixtures.py`, `tests/test_rls.py` (`GLOBAL_TABLES` unchanged; new tables are tenant tables), `.gitignore` (add `evaluation/data/`)

---

### Task 1: Ingestion types, normalization, and the text assembler

**Files:**
- Create: `backend/ingest/__init__.py`, `backend/ingest/types.py`, `backend/ingest/errors.py`, `backend/ingest/normalize.py`, `backend/ingest/assemble.py`, `tests/test_normalize.py`

**Interfaces:**
- `backend/ingest/types.py` (all frozen dataclasses): `PageSpan(page: int, start: int, end: int)`; `SectionSpan(start: int, end: int, path: str)` (path like `"Refunds > Exceptions"`); `ParsedDocument(text: str, pages: tuple[PageSpan, ...], sections: tuple[SectionSpan, ...], metadata: dict[str, str], parser: str, parser_version: str)`; `Sentence(start: int, end: int)`; `PassageDraft(ordinal: int, start: int, end: int, text: str, page_start: int | None, page_end: int | None, section: str | None, text_sha256: bytes)`. Ruling: `sections` is added to the spec's `ParsedDocument` so passages can respect section boundaries.
- `backend/ingest/errors.py`: `class IngestError(Exception)` with `__init__(self, code: str)` and attribute `code` (one of the Global Constraints codes; unknown codes raise `ValueError`).
- `backend/ingest/normalize.py`: `def normalize(text: str) -> str` applying, in this order: (1) Unicode NFC; (2) `\r\n` and `\r` to `\n`; (3) remove characters of category `Cc` except `\n` and `\t`; (4) `\t` and every category `Zs` character to a space; (5) runs of spaces to one space; (6) strip spaces at the end of each line; (7) runs of 3 or more `\n` to `\n\n`; (8) strip leading and trailing whitespace.
- `backend/ingest/assemble.py`: `class TextAssembler` with `add(self, piece: str, *, page: int | None = None) -> tuple[int, int] | None` (normalizes the piece; empty after normalization returns None; otherwise appends it, separated from the previous piece by `"\n\n"`, extends the open page span when `page` equals the last page, and returns the piece's `(start, end)` in the final text); `start_section(self, path: str | None) -> None` (closes the open section at the current end; opens a new one at the next piece's start when `path` is not None); `build(self, metadata: dict[str, str], parser: str, parser_version: str) -> ParsedDocument` (closes the open section; drops empty sections). Invariant: `normalize(result.text) == result.text`.

- [ ] **Step 1: Write failing tests** in `tests/test_normalize.py`: `test_normalize_rules` (exact cases: `"a\r\nb\rc"` -> `"a\nb\nc"`; `"a\t\u00a0 b"` -> `"a b"`; `"a \x01 b"` -> `"a b"`; `"x  \ny"` -> `"x\ny"`; `"a\n\n\n\nb"` -> `"a\n\nb"`; `"  \n a \n "` -> `"a"`; `"e\u0301"` -> `"\u00e9"`); `test_normalize_idempotent` (for 200 seeded random strings over the alphabet `["a", " ", "\t", "\n", "\r", "\x00", "\u00a0", "e", "\u0301", "😀", "."]` of length 0 to 40, `normalize(normalize(s)) == normalize(s)`); `test_assembler_offsets_and_pages` (pieces `"Page one."` on page 1, `"  "` on page 2 (skipped), `"Page\tthree."` on page 3 -> text `"Page one.\n\nPage three."`, pages `(PageSpan(1, 0, 9), PageSpan(3, 11, 22))`, and `text[s:e]` equals the normalized piece for each returned span); `test_assembler_sections` (section `"A"` with two pieces, section `"A > B"` with one piece, then `start_section(None)` and one piece -> two `SectionSpan`s covering exactly the first three pieces' range split at the third piece start); `test_assembled_text_is_normalized` (random pieces as above; `normalize(doc.text) == doc.text`); `test_unknown_ingest_error_code_rejected`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_normalize.py -q`. Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** the four modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** the files with `feat: add ingestion types, normalization, and text assembler`.

### Task 2: Sentence segmentation with offsets

**Files:**
- Create: `backend/ingest/segment.py`, `tests/test_segment.py`

**Interfaces:**
- `ABBREVIATIONS: frozenset[str] = {"mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "jr", "sr", "st", "vs", "etc", "e.g", "i.e", "u.s", "no", "fig", "eq", "co", "corp", "dept", "approx"}` (compared lowercased, without the final period).
- `def segment_sentences(text: str, start: int = 0, end: int | None = None) -> list[Sentence]` over `text[start:end]` with offsets into `text`. Each sentence excludes surrounding whitespace. Rules, in priority order: (1) `"\n\n"` always ends a sentence; (2) a single `"\n"` ends a sentence when the next line starts with a list marker matching `^(?:[-*+\u2022]|\d{1,3}[.)])\s` or the current line ends with `:`; otherwise it is treated as a space (PDF line wraps); (3) `.`, `!`, `?` (optionally followed by closing quotes or brackets `"'\u201d\u2019)]`) end a sentence when followed by whitespace and then an uppercase letter, digit, opening quote or bracket, or a list marker; (4) never split after an abbreviation in `ABBREVIATIONS`, after a single uppercase letter followed by `.` (initials such as `J. R. R.`), or inside a number (`3.14`, `1.5x`); (5) trailing text without terminal punctuation is a sentence.

- [ ] **Step 1: Write failing tests** with a helper `texts(s) = [s[x.start:x.end] for x in segment_sentences(s)]`: `test_basic_split` (`"One. Two! Three?"` -> 3 sentences); `test_abbreviations_and_initials` (`"Dr. Smith met J. R. Tolkien at 3 p.m. on Main St. today. Next."` -> exactly 2, the first ending with `today.`); `test_decimal_not_split` (`"Pi is 3.14 today. Yes."` -> 2); `test_quotes_close_sentence` (`'He said "Stop." Then left.'` -> `['He said "Stop."', "Then left."]`); `test_paragraph_and_wrap` (`"Line one wraps\nhere. Next para\n\nSecond para"` -> `["Line one wraps\nhere.", "Next para", "Second para"]`); `test_bullets_are_sentences` (`"Rules:\n- first item\n- second item"` -> 3); `test_offsets_with_range_and_unicode` (`s = "Skip. Café 😀 opens. Done."`, `segment_sentences(s, 6)` -> texts `["Café 😀 opens.", "Done."]` and every offset satisfies `s[a.start:a.end] == text`); `test_empty_and_whitespace` (`""` and `"  \n "` -> `[]`); `test_lowercase_after_period_not_split` (`"See fig. 2 and the e.g. case. End."` -> 2).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_segment.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** a single left-to-right scanner (no regex lookbehind chains over the whole text); precompile the list-marker regex.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add offset-preserving sentence segmentation`.

### Task 3: Sentence-aligned passage builder

**Files:**
- Create: `backend/ingest/passages.py`, `tests/test_passages.py`

**Interfaces:**
- `def build_passages(text: str, pages: Sequence[PageSpan] = (), sections: Sequence[SectionSpan] = (), *, target_chars: int = 800, max_chars: int = 1200, overlap_sentences: int = 1) -> list[PassageDraft]`.
- Algorithm: split `text` into regions at section boundaries (each `SectionSpan`, plus the gaps between them as unsectioned regions); segment each region with `segment_sentences(text, region_start, region_end)`; split any sentence longer than `max_chars` into pieces at the last whitespace at or before `max_chars` (hard cut at `max_chars` when there is none), each piece trimmed of whitespace. Within a region, accumulate sentences greedily: a passage closes when adding the next sentence would make `end - start` exceed `target_chars` (a passage always holds at least one sentence). The next passage starts with the last `overlap_sentences` sentences of the previous passage only when the previous passage held more than `overlap_sentences` sentences and the overlap plus the next sentence fits within `max_chars`. `text = text[start:end]`, `text_sha256 = sha256(text.encode("utf-8")).digest()`, `page_start`/`page_end` = first and last pages whose span intersects `[start, end)` (None without pages), `section` = path of the region (None outside sections), ordinals `0..n-1` in document order.

- [ ] **Step 1: Write failing tests**: `test_invariants_on_generated_text` (seeded generator of 30 paragraphs of 1 to 12 sentences of 5 to 40 words, plus one 3,000-character sentence without punctuation; for every passage `p.text == text[p.start:p.end]`, `len(p.text) <= 1200`, `p.text_sha256 == sha256(p.text.encode()).digest()`, ordinals consecutive, starts non-decreasing, every sentence of `segment_sentences(text)` is inside at least one passage); `test_overlap_one_sentence` (ten 100-character sentences, `target_chars=350` -> each passage after the first starts at the previous passage's last sentence start); `test_no_overlap_when_single_sentence_passages` (five 700-character sentences -> five passages, no shared sentence); `test_sections_never_crossed` (two `SectionSpan`s; no passage spans the boundary; `section` values match); `test_pages_mapping` (two pages; a passage spanning the boundary has `page_start=1, page_end=2`); `test_empty_text` (`build_passages("") == []`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_passages.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `build_passages`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add sentence-aligned passage builder`.

### Task 4: Plain text and Markdown parsers

**Files:**
- Create: `backend/ingest/parsers/__init__.py` (empty in this task), `backend/ingest/parsers/text.py`, `backend/ingest/parsers/markdown.py`, `tests/test_parsers_text_md.py`

**Interfaces:**
- `def decode_utf8(data: bytes) -> str` in `text.py`: strict UTF-8 with an optional BOM (`utf-8-sig`); failure raises `IngestError("bad_encoding")`.
- `def parse_text(data: bytes) -> ParsedDocument`: paragraphs (split on blank lines) added to a `TextAssembler`; parser `text`, version `text/1`, no pages, no sections, empty metadata.
- `def parse_markdown(data: bytes) -> ParsedDocument`, line-based, original (no Markdown library). Parser `markdown`, version `markdown/1`. Rules: ATX headings `^(#{1,6})\s+(.*?)\s*#*\s*$` update a heading stack by level, call `start_section(" > ".join(stack))`, and add the heading text as its own piece; the first level-1 heading becomes `metadata["title"]`. Fenced code (```` ``` ```` or `~~~`) lines are kept verbatim as one piece. Outside code: inline code backticks removed (content kept); `![alt](url)` -> `alt`; `[text](url)` -> `text`; reference definitions `^\s*\[[^\]]+\]:\s*\S+` dropped; `**x**`, `__x__`, `*x*`, `_x_` markers removed when they wrap non-space text (intra-word underscores such as `snake_case` kept); blockquote `>` prefixes removed; list markers `*` and `+` normalized to `-`, numbered markers kept; horizontal rules (`---`, `***`, `___` alone) dropped; HTML comments `<!-- ... -->` dropped (including multi-line); pipe tables: separator rows dropped, other rows become cells joined by `"; "`. Consecutive non-blank lines form one paragraph piece (joined with `"\n"`); blank lines separate pieces.

- [ ] **Step 1: Write failing tests**: `test_text_paragraphs_and_bom` (`b"\xef\xbb\xbfFirst para.\r\n\r\nSecond."` -> text `"First para.\n\nSecond."`); `test_text_bad_utf8` (`b"\xff\xfe\x00"` -> `IngestError` code `bad_encoding`); `test_markdown_strips_syntax` (input with `# Policy`, `**Refunds** are [allowed](http://x) within *30 days*.`, `` `code` ``, `![logo](x.png)`, `[ref]: http://y` -> text contains `"Policy"`, `"Refunds are allowed within 30 days."`, `"code"`, `"logo"`; contains none of `"**"`, `"http://"`, `"[ref]"`); `test_markdown_sections` (`# A`, para, `## B`, para, `# C`, para -> sections with paths `"A"`, `"A > B"`, `"C"`, and the second paragraph's offsets lie inside the `"A > B"` span); `test_markdown_title_metadata`; `test_markdown_code_fence_verbatim` (a fence containing `**not bold**` keeps the asterisks); `test_markdown_table_rows` (`| a | b |`, `|---|---|`, `| 1 | 2 |` -> lines `"a; b"` and `"1; 2"`); `test_snake_case_kept`; `test_markdown_output_normalized` (`normalize(doc.text) == doc.text`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_parsers_text_md.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both parsers on top of `TextAssembler`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add text and markdown parsers`.

### Task 5: HTML and CSV parsers

**Files:**
- Create: `backend/ingest/parsers/html.py`, `backend/ingest/parsers/csv_table.py`, `tests/test_parsers_html_csv.py`

**Interfaces:**
- `def parse_html(data: bytes) -> ParsedDocument` with an `html.parser.HTMLParser` subclass (`convert_charrefs=True`). Parser `html`, version `html/1`. Contents of `script`, `style`, `noscript`, `template`, `svg`, `head` are dropped, except `<title>` text becomes `metadata["title"]`. Block tags (`p, div, section, article, header, footer, main, nav, aside, ul, ol, li, table, tr, h1..h6, pre, blockquote, dl, dt, dd, figure, figcaption, hr, form`) flush the current paragraph as a piece; `<br>` inserts `"\n"`; `<li>` content is prefixed with `"- "`; `td`/`th` cells in a row are joined with `"; "`; `<pre>` keeps internal newlines; headings update a level stack and call `start_section` like Markdown and are added as their own piece. Unclosed tags must not raise.
- `def parse_csv(data: bytes) -> ParsedDocument` with stdlib `csv.reader` (dialect `excel`) over `decode_utf8(data)`. Parser `csv`, version `csv/1`. First row is the header; blank header cells become `column_<1-based index>`; each data row becomes one piece `"h1: v1; h2: v2"` with empty values omitted (rows with no values skipped); more than 50,000 data rows raises `IngestError("too_many_rows")`; `csv.Error` raises `IngestError("bad_csv")`; a header-only file raises `IngestError("no_text")`.

- [ ] **Step 1: Write failing tests**: `test_html_drops_scripts_and_styles` (body text kept; `alert(`, `color:` absent; title from `<title>`); `test_html_blocks_lists_tables` (`<p>One</p><p>Two</p><ul><li>a</li><li>b</li></ul><table><tr><th>k</th><th>v</th></tr><tr><td>x</td><td>1</td></tr></table>` -> pieces in order `One`, `Two`, `- a`, `- b`, `k; v`, `x; 1`); `test_html_entities_and_br` (`&amp;` -> `&`, `<br>` -> newline inside the paragraph); `test_html_sections_from_headings`; `test_html_malformed_does_not_raise` (`<p>open <b>bold <div>x`); `test_csv_rows_render` (`b"name,days\nRefund,30\nExchange,\n"` -> text `"name: Refund; days: 30\n\nname: Exchange"`); `test_csv_blank_header`; `test_csv_row_limit` (50,001 data rows -> `too_many_rows`); `test_csv_header_only_no_text`; `test_csv_bad_encoding`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_parsers_html_csv.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both parsers.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add html and csv parsers`.

### Task 6: PDF parser, type dispatch, and the killable parse subprocess

**Files:**
- Create: `backend/ingest/parsers/pdf.py`, `backend/ingest/parse_child.py`, `tests/test_parsers_pdf.py`
- Modify: `backend/ingest/parsers/__init__.py`, `backend/core/document.py` (PDF/TXT reading delegates to the new parsers; signature unchanged; `tests/test_document.py` stays green)

**Interfaces:**
- `def parse_pdf(data: bytes) -> ParsedDocument` with `pypdf.PdfReader(io.BytesIO(data))`. Parser `pdf`, version `pdf/1`. When `reader.is_encrypted`, try `reader.decrypt("")`; a result of `PasswordType.NOT_DECRYPTED` raises `IngestError("encrypted_pdf")`. Each page's `extract_text()` (None treated as `""`) is added to the assembler with `page=<1-based index>`. Metadata: `title`, `author` (from `reader.metadata`, when non-empty strings), `created` (ISO 8601 of `creation_date` when present). No text on any page raises `IngestError("no_text")`. Any `pypdf.errors.PyPdfError`, `ValueError`, `KeyError`, or `TypeError` from pypdf raises `IngestError("bad_pdf")`.
- `backend/ingest/parsers/__init__.py`: `EXTENSIONS = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown", ".html": "text/html", ".htm": "text/html", ".csv": "text/csv"}`; `def mime_for(filename: str) -> str` (unknown extension raises `IngestError("unsupported_type")`); `def parse(data: bytes, filename: str) -> ParsedDocument` (dispatch by lowercased extension; `.pdf` requires `data.startswith(b"%PDF-")`, and a non-PDF extension whose data starts with `%PDF-` raises `unsupported_type`; after parsing, more than 2,000,000 characters raises `IngestError("too_long")` and empty text raises `IngestError("no_text")`).
- `backend/ingest/parse_child.py`: `python -m backend.ingest.parse_child <filename>` reads the file bytes from stdin, calls `parse`, and writes one JSON object to stdout: success `{"ok": true, "doc": {text, pages: [[page, start, end]], sections: [[start, end, path]], metadata, parser, parser_version}}`, ingest error `{"ok": false, "code": "<code>"}`; exit code 0 either way; it imports nothing outside `backend.ingest` and the stdlib plus pypdf. `async def parse_with_timeout(data: bytes, filename: str, timeout_s: float = 60.0) -> ParsedDocument` (in `parsers/__init__.py`) runs it with `asyncio.create_subprocess_exec(sys.executable, "-m", "backend.ingest.parse_child", filename, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)`; on timeout kills the process and raises `IngestError("parse_timeout")`; a non-zero exit or unparsable output raises `IngestError("bad_pdf")` for PDFs and `IngestError("unsupported_type")` otherwise; an `{"ok": false}` result raises `IngestError(code)`.
- Test PDFs are generated in the test with a minimal hand-written PDF builder (`tests/test_parsers_pdf.py::make_pdf(pages: list[str]) -> bytes`, Helvetica text objects, correct xref offsets); no binary fixtures are committed. An image-only PDF is `make_pdf([""])`.

- [ ] **Step 1: Write failing tests**: `test_pdf_pages_and_offsets` (`make_pdf(["Refunds take 30 days.", "Exchanges take 14 days."])` -> two `PageSpan`s, page 2's slice contains `"Exchanges"`); `test_pdf_no_text` (`make_pdf([""])` -> `no_text`); `test_pdf_garbage_bad_pdf` (`b"%PDF-1.4 garbage"` -> `bad_pdf`); `test_dispatch_by_extension` (`.md`, `.HTML`, `.csv`, `.txt` route to their parsers; `.docx` -> `unsupported_type`); `test_pdf_magic_required` (`"x.pdf"` with text bytes -> `unsupported_type`; `"x.txt"` with `%PDF-` bytes -> `unsupported_type`); `test_too_long` (`"a. " * 700_001` as `.txt` -> `too_long`); `test_parse_with_timeout_roundtrip` (text document through the subprocess equals in-process `parse`); `test_parse_with_timeout_error_code` (empty `.txt` -> `IngestError` `no_text`); `test_parse_with_timeout_kills` (`timeout_s=0.001` -> `parse_timeout`); `test_legacy_parse_document_still_works` (`backend.core.document.parse_document(file_path=<tmp .txt>)` returns its text).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_parsers_pdf.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the PDF parser, dispatch, child module, and `parse_with_timeout`; update `backend/core/document.py`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add pdf parser and isolated parse subprocess`.

### Task 7: Documents, versions, passages schema with integrity triggers

**Files:**
- Create: `backend/db/migrations/0003_documents.sql`, `tests/test_schema_documents.py`
- Modify: `backend/db/pool.py` (pool `configure` callback runs `pgvector.psycopg.register_vector_async(conn)`), `tests/pg_fixtures.py` (truncate list unchanged: it truncates all non-migration tables)

**Interfaces (all tenant tables follow the Plan A Task 7 RLS pattern):**
- `create function eg_token_count(tsvector) returns int language sql immutable parallel safe as $$ select coalesce(sum(coalesce(array_length(positions, 1), 1)), 0)::int from unnest($1) $$;` (verified on PG 17.11: usable in a stored generated column).
- `blobs(tenant_id uuid not null references tenants(id), sha256 bytea not null, bytes bytea not null, size int not null, created_at timestamptz not null default now(), primary key (tenant_id, sha256))`.
- `documents(id uuid primary key default gen_random_uuid(), tenant_id, collection_id uuid not null references collections(id), title text not null, filename text not null, mime_type text not null, current_version_id uuid null, created_by uuid references users(id), created_at, deleted_at timestamptz null, unique (id, collection_id))`; index `(tenant_id, collection_id, created_at desc, id desc)`.
- `document_versions(id uuid primary key default gen_random_uuid(), tenant_id, document_id uuid not null references documents(id), version_number int not null, effective_from timestamptz not null, content_sha256 bytea not null, blob_sha256 bytea not null, text text not null, page_map jsonb not null default '[]', sections jsonb not null default '[]', metadata jsonb not null default '{}', parser text not null, parser_version text not null, char_count int not null, created_by uuid references users(id), created_at timestamptz not null default now(), unique (document_id, version_number), unique (id, document_id))` (no content-hash uniqueness, R5); `documents.current_version_id` gets a deferrable FK to `document_versions(id)` added after both tables exist; trigger `document_versions_immutable` raises `'document_versions are immutable'` on UPDATE or DELETE.
- `passages(id uuid primary key default gen_random_uuid(), tenant_id, document_id uuid not null, version_id uuid not null, collection_id uuid not null references collections(id), ordinal int not null, start_char int not null, end_char int not null, page_start int, page_end int, section text, text text not null, text_sha256 bytea not null, tsv tsvector generated always as (to_tsvector('english', text)) stored, token_count int generated always as (eg_token_count(to_tsvector('english', text))) stored, embedding vector(384) not null, embedding_model text not null, is_current boolean not null default false, created_at, unique (version_id, ordinal), unique (id, version_id), foreign key (version_id, document_id) references document_versions(id, document_id), check (0 <= start_char and start_char < end_char))`. Indexes: `passages_embedding_hnsw using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64)`, GIN on `tsv`, btree `(tenant_id, is_current, collection_id)`, `(document_id, version_id)`, `(tenant_id, text_sha256, embedding_model)`.
- Trigger `passages_slice_check` (BEFORE INSERT): the version's `text` must satisfy `substr(v.text, new.start_char + 1, new.end_char - new.start_char) = new.text`, `new.end_char <= v.char_count`, and `new.text_sha256 = sha256(convert_to(new.text, 'UTF8'))`; otherwise `raise exception 'passage text does not match version slice'`. Trigger `passages_immutable` (BEFORE UPDATE) allows changes only to `is_current` and `collection_id`; any other column change raises `'passage content is immutable'`; DELETE raises too.
- `passage_terms(tenant_id, passage_id uuid not null references passages(id), term text not null, tf int not null, primary key (passage_id, term))`, index `(tenant_id, term)`; trigger `passages_fill_terms` (AFTER INSERT on passages) inserts `select new.tenant_id, new.id, u.lexeme, coalesce(array_length(u.positions, 1), 1) from unnest(new.tsv) u`.
- `uploads(id uuid primary key default gen_random_uuid(), tenant_id, document_id uuid not null references documents(id), blob_sha256 bytea not null, filename text not null, declared_mime text not null, effective_from timestamptz not null, status text not null default 'queued' check (status in ('queued','processing','created_version','duplicate','failed')), version_id uuid null references document_versions(id), error_code text null, stats jsonb not null default '{}', created_by uuid references users(id), created_at, updated_at)`, foreign key `(tenant_id, blob_sha256) references blobs(tenant_id, sha256)`.
- `search_requests(id uuid primary key default gen_random_uuid(), tenant_id, user_id uuid not null references users(id), query text not null, params jsonb not null, status text not null default 'queued' check (status in ('queued','done','failed')), hits jsonb null, error_code text null, created_at, completed_at timestamptz null)` (R4; ruling: one table holds both the query and its result so job payloads carry only the id).
- `GLOBAL_TABLES` in `tests/test_rls.py` is not extended: every table above is tenant-scoped.
- The same migration also adds the indexes deferred from the Plan A final review: `memberships (user_id)`, `collection_grants (tenant_id, user_id)`, `sessions (tenant_id, user_id)`, `api_keys (tenant_id, user_id)`.

- [ ] **Step 1: Write failing tests** in `tests/test_schema_documents.py` (all `pg`; a helper `insert_version(conn, text)` creates tenant, collection, document, and version rows directly in SQL): `test_passage_must_match_slice` (`"Refunds take 30 days."`: passage `(0, 7, "Refunds")` inserts; `(0, 7, "Refundz")` raises `psycopg.errors.RaiseException`); `test_unicode_slice_matches` (version text `"Café 😀 naïve 東京 ok"`; passage of `text[5:12]` computed in Python inserts successfully); `test_terms_and_token_count_filled` (passage `"Refunds are refunded within 30 days"` -> `token_count == 5`, terms include `("refund", 2)`); `test_versions_immutable` (UPDATE and DELETE raise); `test_passage_content_immutable_but_flags_mutable` (UPDATE `text` raises; UPDATE `is_current` succeeds); `test_passage_version_document_mismatch_rejected` (passage whose `document_id` differs from its version's document -> `ForeignKeyViolation`); `test_new_tables_have_forced_rls` (re-run the Plan A RLS check; all new tables covered); `test_embedding_roundtrip` (a numpy float32 vector of 384 values reads back equal through the pool's registered adapter).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_schema_documents.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the migration and the pool change.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add versioned document schema with integrity triggers`.

### Task 8: Model protocols, real adapters, service container, test fakes

**Files:**
- Create: `backend/models/protocols.py`, `backend/services.py`, `tests/fakes.py`, `tests/test_model_adapters.py`
- Modify: `backend/models/loader.py`, `backend/config.py`

**Interfaces:**
- `backend/models/protocols.py`: `class Embedder(Protocol)` with attributes `model_id: str`, `dim: int` and methods `embed_passages(self, texts: Sequence[str]) -> np.ndarray` and `embed_queries(self, texts: Sequence[str]) -> np.ndarray` (shape `(n, dim)`, float32, rows L2-normalized; empty input returns shape `(0, dim)`); `class Reranker(Protocol)` with `model_id: str` and `score(self, query: str, passages: Sequence[str]) -> list[float]` (raw logits, input order).
- `Settings` additions: `embedder_model: str = "BAAI/bge-small-en-v1.5"`, `embedding_dim: int = 384`, `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"`, `model_device: str = "auto"`, `max_upload_bytes: int = 20_971_520`, `parse_timeout_s: float = 60.0`, `embed_batch_size: int = 32`. The legacy `embedding_model_name` stays for the legacy pipeline.
- `backend/models/loader.py` additions: `def resolve_device(setting: str) -> str` (`auto` -> `"mps"` when `torch.backends.mps.is_available()`, else `"cuda"` when available, else `"cpu"`; explicit values returned as given); `class SentenceTransformerEmbedder(model_name: str, *, device: str, query_prefix: str = BGE_QUERY_PREFIX, batch_size: int = 32)` with `BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "` and `model_id = model_name`; `class CrossEncoderReranker(model_name: str, *, device: str, batch_size: int = 32)` (sentence-transformers `CrossEncoder`, returns raw scores); `def load_embedder(settings: Settings) -> SentenceTransformerEmbedder`, `def load_reranker(settings: Settings) -> CrossEncoderReranker`. Imports of torch and sentence-transformers stay inside these functions and classes so importing `backend.models.loader` stays cheap. Existing loader functions stay.
- `backend/services.py`: `@dataclass class Services` with `settings: Settings`, `db: Database`, `embedder: Embedder`, `reranker: Reranker | None = None`. Plan C adds fields with defaults.
- `tests/fakes.py`: `class HashEmbedder(dim: int = 384)` (`model_id = "fake-hash-384"`; lowercase `\w+` tokens, each token hashed with `hashlib.blake2b(token.encode(), digest_size=8)` to an index and a sign, counts summed, L2-normalized; empty text gets a fixed unit vector; queries and passages embedded identically so shared words raise cosine); `class KeywordReranker` (`model_id = "fake-keyword"`; score = number of distinct lowercase query tokens present in the passage). Both deterministic across processes (no Python `hash`).
- Model smoke tests are marked `models` (excluded by default).

- [ ] **Step 1: Write failing tests**: `test_hash_embedder_normalized_and_deterministic` (shape `(2, 384)`, norms 1 within 1e-6, two instances agree); `test_hash_embedder_similarity_tracks_overlap` (cos("refund policy days", "refund policy") > cos("refund policy days", "quantum physics")); `test_keyword_reranker_counts`; `test_resolve_device_explicit` (`"cpu"` -> `"cpu"`); `test_settings_model_defaults`; `@pytest.mark.models test_real_embedder_dim_and_prefix` (real bge-small: dim 384; query and passage vectors for the same text differ, proving the prefix is applied); `@pytest.mark.models test_real_reranker_orders` (query `"how long do refunds take"`: the refund passage outscores an unrelated one).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_model_adapters.py -q` and `.venv/bin/python -m pytest tests/test_model_adapters.py -q -m models`. Expected: FAIL.
- [ ] **Step 3: Implement** protocols, adapters, settings, services, fakes.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` (PASS) and `.venv/bin/python -m pytest tests/test_model_adapters.py -q -m models` (PASS; downloads about 220 MB of models once into the Hugging Face cache).
- [ ] **Step 5: Commit** with `feat: add embedder and reranker adapters with test fakes`.

### Task 9: Version creation with dedupe and incremental embeddings

**Files:**
- Create: `backend/db/repos/documents.py`, `backend/ingest/service.py`, `tests/test_ingest_versions.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VersionResult:
    status: Literal["created_version", "duplicate"]
    version_id: UUID; version_number: int
    previous_version_id: UUID | None
    n_passages: int; n_reused: int; n_new: int

async def create_version(conn, embedder: Embedder, *, tenant_id: UUID, document_id: UUID,
                         parsed: ParsedDocument, blob_sha256: bytes, effective_from: datetime,
                         created_by: UUID | None, settings: Settings) -> VersionResult
```

- Runs on a connection from `tenant_tx(tenant_id)`; the caller owns the transaction. Steps: `select ... from documents where id = %s for update` (missing or soft-deleted raises `LookupError`; the handler maps it to `document_deleted`); compute `content_sha256 = sha256(parsed.text.encode()).digest()`; when it equals the current version's hash return `duplicate` with the current version (R5: only the current version is compared); `effective_from` earlier than the current version's raises `IngestError("effective_from_out_of_order")`; `build_passages(parsed.text, parsed.pages, parsed.sections)` with the Global Constraints defaults, more than 5,000 raises `IngestError("too_many_passages")`, zero raises `IngestError("no_text")`; fetch reusable embeddings with `select distinct on (text_sha256) text_sha256, embedding from passages where text_sha256 = any(%s) and embedding_model = %s` (RLS keeps it in the tenant); embed only the remaining texts with `await asyncio.to_thread(embedder.embed_passages, texts)` in batches of `settings.embed_batch_size`; insert the version (`version_number = coalesce(max, 0) + 1`, `char_count = len(text)`, `page_map` as `[[page, start, end], ...]`, `sections` as `[[start, end, path], ...]`), insert passages with `executemany` and `is_current = true`, set `is_current = false` on the previous version's passages, set `documents.current_version_id`; observe `INGEST_PASSAGES{outcome="new"|"reused"}`.
- `backend/db/repos/documents.py` (tenant connections): `insert_blob(conn, tenant_id, sha256, data) -> None` (`on conflict do nothing`), `get_blob(conn, sha256) -> bytes | None`, `insert_document(conn, *, tenant_id, collection_id, title, filename, mime_type, created_by) -> UUID`, `get_document(conn, document_id, *, include_deleted=False) -> dict | None`, `list_documents(conn, collection_ids, page) -> list[dict]` (with `current_version_number` and `version_count`), `list_versions(conn, document_id) -> list[dict]` (without `text`), `get_version(conn, document_id, version_number) -> dict | None`, `get_version_text(conn, version_id, start, end) -> str` (`substr` in SQL), `list_passages(conn, version_id) -> list[dict]` (boundaries only, no text), `soft_delete_document(conn, document_id) -> bool`, `count_live_documents(conn, collection_id) -> int`.

- [ ] **Step 1: Write failing tests** (all `pg`, `HashEmbedder` wrapped in a counting spy, documents created through the repo): `test_first_version_creates_passages` (status `created_version`, `version_number == 1`, passages `is_current`, `n_new == n_passages`); `test_identical_text_is_duplicate` (Review Focus 1: second call with the same parsed text -> `duplicate`, same `version_id`, no new rows, embedder not called); `test_changed_version_reuses_unchanged_passages` (v2 changes one paragraph of five -> `n_reused >= 3`, the spy embedded only `n_new` texts, v1 passages now `is_current = false`, v2 passages true, `current_version_id` is v2); `test_revert_creates_new_version` (v1 text, v2 text, then v1 text again -> `version_number == 3`, `n_new == 0`); `test_effective_from_out_of_order` (v2 earlier than v1 -> `effective_from_out_of_order`, transaction rolled back, still one version); `test_unicode_offsets_end_to_end` (Review Focus 2: text `"Café 😀 opens at 9. 東京 office closes at 5. Naïve test."` -> insert succeeds; `get_version_text(version_id, p.start_char, p.end_char) == p.text` for every passage); `test_too_many_passages` (monkeypatched limit of 2 -> `too_many_passages`); `test_reuse_does_not_cross_tenants` (same text in tenant B embeds anew: spy call count grows).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_ingest_versions.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the repo and `create_version`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add version creation with dedupe and embedding reuse`.

### Task 10: Upload intake, the `ingest_upload` job, and worker services

**Files:**
- Create: `backend/ingest/intake.py`, `backend/ingest/handlers.py`, `tests/test_ingest_upload.py`
- Modify: `backend/jobs/registry.py`, `backend/cli/main.py` (worker builds `Services` with `load_embedder` and `load_reranker`)

**Interfaces:**

```python
@dataclass(frozen=True)
class IntakeResult:
    document_id: UUID; upload_id: UUID; status: Literal["queued", "duplicate"]

async def intake_upload(db: Database, principal: Principal, *, data: bytes, filename: str,
                        collection_id: UUID | None = None, document_id: UUID | None = None,
                        title: str | None = None, effective_from: datetime | None = None,
                        settings: Settings) -> IntakeResult
async def handle_ingest_upload(ctx: JobContext) -> None
```

- `intake_upload` (API side, one `tenant_tx`): exactly one of `collection_id` (new document, `write` permission) or `document_id` (new version, `write` on its collection; deleted or unreadable -> `ApiError(404, "document_not_found", ...)`) is given, else `ValueError`. Filename sanitized to its basename with control characters removed, at most 255 characters; `mime_for(filename)` failures become `ApiError(415, "unsupported_type", ...)`; more than `max_upload_bytes` becomes `ApiError(413, "too_large", ...)`; `effective_from` later than now plus 60 s becomes `ApiError(422, "effective_from_in_future", ...)`, default now. Title defaults to the filename stem, at most 300 characters. When the document's current version has the same `blob_sha256`, the upload row is written with status `duplicate` and `version_id` = current version and no job is enqueued. Otherwise the blob, document (if new), upload row (`queued`), and job `ingest_upload` with payload `{"upload_id": "<uuid>"}`, `tenant_id`, `priority = 0`, `idempotency_key = "upload:<uuid>"`, `request_id` from the context var, are written in the same transaction. Audit `document_uploaded` with ids only.
- `handle_ingest_upload`: payload `upload_id`; in `tenant_tx(job.tenant_id)` load the upload; status other than `queued` or `processing` returns (idempotent); set `processing`; load the blob; parse with `parse_with_timeout(data, filename, settings.parse_timeout_s)` outside the transaction; on `IngestError` set status `failed`, `error_code`, then raise `PermanentJobError(code)`; then in a new `tenant_tx` call `create_version` (an `IngestError` there is handled the same way; a missing or deleted document fails with `document_deleted`); set upload `status`, `version_id`, `stats = {"n_passages", "n_reused", "n_new"}`; when status is `created_version` and `previous_version_id` is not None, enqueue `reverify_document` with payload `{"document_id", "old_version_id", "new_version_id"}`, priority 5, `idempotency_key = "reverify:<new_version_id>"` (Plan C registers its handler; until then the job waits). Embedding failures raise `RetryableJobError("embed_failed")`.
- `build_handlers(services)` registers `"ingest_upload": handle_ingest_upload`. `JobContext.services` is a `Services`.
- Test helper in `tests/pg_fixtures.py`: fixture `services` (`Services(settings, db, HashEmbedder(), KeywordReranker())`) and `async def drain(db, services, kinds=None) -> int` (runs `Worker(db, build_handlers(services), services=services).run_once()` until it returns False; returns the count).

- [ ] **Step 1: Write failing tests** (all `pg`): `test_upload_then_drain_creates_version` (txt upload -> `queued`; `drain` -> upload `created_version`, stats present, document has version 1); `test_upload_is_atomic_with_job` (patch `enqueue` to raise -> no upload, document, or blob rows remain); `test_same_bytes_duplicate_without_job` (Review Focus 1: second upload of identical bytes as a new version -> `IntakeResult.status == "duplicate"`, no job enqueued); `test_resaved_same_text_duplicate` (different bytes, same canonical text: `"a\r\nb"` vs `"a\nb"` -> after `drain` upload status `duplicate`, no `reverify_document` job); `test_new_version_enqueues_reverify` (changed text -> one queued `reverify_document` job with the three ids and no text in its payload); `test_empty_and_image_only_fail_no_text` (Review Focus 5: empty `.txt`, whitespace-only `.txt`, and an image-only PDF from `make_pdf([""])` -> upload `failed`, `error_code == "no_text"`, job status `failed`, `attempts == 1`); `test_unsupported_extension_415`; `test_too_large_413` (monkeypatched limit); `test_effective_from_future_422`; `test_member_without_write_404_or_403` (read-only grant -> 403 `insufficient_permission`; no grant -> 404 `collection_not_found`); `test_filename_sanitized` (`"../../etc/passwd.txt"` stored as `passwd.txt`); `test_handler_idempotent` (running the handler twice on a finished upload changes nothing).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_ingest_upload.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** intake, handler, registry, worker services wiring.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add upload intake and ingestion job`.

### Task 11: Document, version, and upload routes

**Files:**
- Create: `backend/api/routes/documents.py`, `tests/test_document_routes.py`
- Modify: `backend/api/schemas_v1.py`, `backend/api/app.py`, `backend/api/routes/collections.py` (DELETE of a collection with live documents -> 409 `collection_not_empty`)

**Interfaces (all under `/api/v1`, cookie or API key auth, read permission unless stated):**
- `POST /collections/{collection_id}/documents` (write; multipart `file`, optional form fields `title`, `effective_from` ISO 8601) -> 202 `UploadAcceptedOut(document_id: UUID, upload_id: UUID, status: str)`.
- `POST /documents/{document_id}/versions` (write; multipart `file`, optional `effective_from`) -> 202 `UploadAcceptedOut`.
- `GET /uploads/{upload_id}` -> `UploadOut(id, document_id, status, version_id, version_number: int | None, error_code, stats: dict, created_at, updated_at)`.
- `GET /documents?collection_id=<uuid>&limit=&cursor=` -> `Page[DocumentOut]` with `DocumentOut(id, collection_id, title, filename, mime_type, current_version_number: int | None, version_count: int, created_at)`; without `collection_id`, all readable `normal` collections.
- `GET /documents/{document_id}` -> `DocumentDetailOut(DocumentOut + versions: list[VersionSummaryOut])`, `VersionSummaryOut(id, version_number, effective_from, created_at, char_count, n_passages, parser, is_current)`.
- `DELETE /documents/{document_id}` (write) -> 204, soft delete, audit `document_deleted`.
- `GET /documents/{document_id}/versions/{n}` -> `VersionOut(VersionSummaryOut + page_map: list[list[int]], sections: list, metadata: dict, parser_version: str)`.
- `GET /documents/{document_id}/versions/{n}/text?start=0&end=` -> `VersionTextOut(start, end, char_count, text)`; `end` defaults to `min(start + 100_000, char_count)`; `start < 0`, `end > char_count`, `end < start`, or `end - start > 100_000` -> 422 `bad_range`.
- `GET /documents/{document_id}/versions/{n}/passages` -> `list[PassageBoundaryOut(id, ordinal, start_char, end_char, page_start, page_end, section)]`.
- Missing, deleted, unreadable, or cross-tenant documents and uploads -> 404 `document_not_found` / `upload_not_found`; missing version number -> 404 `version_not_found`.

- [ ] **Step 1: Write failing tests** (all `pg`, `client` fixture with `api_app.state.services = services`): `test_upload_drain_and_read_back` (multipart `.md` upload -> 202; `drain`; `GET /uploads/{id}` -> `created_version`, `version_number == 1`; `GET /documents/{id}` lists one version; `GET .../versions/1/text` equals the parsed text); `test_text_range_validation` (`end - start = 100_001` -> 422 `bad_range`; `end > char_count` -> 422); `test_passage_boundaries_cover_text`; `test_new_version_route_and_listing` (v2 upload -> `current_version_number == 2`, `version_count == 2`); `test_cross_tenant_404s` (Carol gets 404 on every route above for Alice's document and upload); `test_read_only_member_cannot_upload` (403 `insufficient_permission`); `test_soft_deleted_document_404` (DELETE then GET -> 404; listing excludes it; uploading a new version -> 404); `test_collection_with_documents_cannot_be_deleted` (409 `collection_not_empty`; succeeds after the document is deleted).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_document_routes.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** routes and schemas.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add document, version, and upload routes`.

### Task 12: Version diff

**Files:**
- Create: `backend/ingest/diff.py`, `tests/test_diff.py`
- Modify: `backend/api/routes/documents.py`, `backend/api/schemas_v1.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class DiffHunk:
    op: Literal["equal", "insert", "delete", "replace"]
    from_start: int; from_end: int; to_start: int; to_end: int
    from_text: str; to_text: str     # both "" for equal hunks

def diff_texts(old: str, new: str) -> list[DiffHunk]
```

- Sentences of each text from `segment_sentences`; `difflib.SequenceMatcher(None, old_sentences, new_sentences, autojunk=False)` over the sentence strings; each opcode becomes a hunk whose offsets run from the first sentence's start to the last sentence's end in each text (empty ranges for pure inserts and deletes sit at the neighboring sentence boundary); adjacent hunks with the same op merge.
- Route `GET /documents/{document_id}/diff?from=<n>&to=<n>` (read) -> `DiffOut(from_version: int, to_version: int, hunks: list[DiffHunkOut], stats: {"equal": int, "insert": int, "delete": int, "replace": int})` (counts of hunks); unknown version numbers -> 404 `version_not_found`; `from == to` -> one `equal` hunk.

- [ ] **Step 1: Write failing tests**: `test_identical_texts_single_equal`; `test_replace_one_sentence` (`"Refunds take 30 days. Shipping is free."` vs `"Refunds take 14 days. Shipping is free."` -> hunks `replace` then `equal`; `old[h.from_start:h.from_end] == "Refunds take 30 days."`, `new[h.to_start:h.to_end] == "Refunds take 14 days."`); `test_insert_and_delete` (an appended sentence is `insert` with `from_start == from_end == len(old)`); `test_offsets_slice_exactly` (seeded random edits of a 40-sentence text; for every non-equal hunk `old[from_start:from_end] == from_text` and `new[to_start:to_end] == to_text`); `@pytest.mark.pg test_diff_route` (two uploaded versions -> 200 with one `replace` hunk; unknown `to` -> 404 `version_not_found`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_diff.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `diff_texts` and the route.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add sentence-level version diff`.

### Task 13: Retrieval types, the visible passage set, and BM25 in SQL

**Files:**
- Create: `backend/retrieval/__init__.py`, `backend/retrieval/types.py`, `backend/retrieval/visible.py`, `backend/retrieval/bm25.py`, `tests/test_bm25.py`

**Interfaces:**
- `backend/retrieval/types.py`: `SearchMode = Literal["hybrid", "bm25", "dense", "fts"]`; frozen `SearchScope(tenant_id: UUID, collection_ids: tuple[UUID, ...], as_of: datetime | None = None, known_at: datetime | None = None)` with property `is_live -> bool` (`as_of is None and known_at is None`); frozen `PassageHit` with the spec fields plus `effective_from: datetime` (`passage_id, document_id, version_id, version_number, effective_from, document_title, text, start_char, end_char, page_start, page_end, section, scores: dict[str, float], rank: int`); `class Retriever(Protocol)` with `async search(self, query: str, scope: SearchScope, *, k: int = 10, mode: SearchMode = "hybrid", rerank: bool = True, candidates: int = 50, exact: bool = False) -> list[PassageHit]`; `class RerankUnavailable(RuntimeError)`.
- `backend/retrieval/visible.py`: `async def visible_version_ids(conn, scope: SearchScope) -> list[UUID] | None` (None for live scopes; otherwise, per R9, with `known = coalesce(known_at, now())` and `valid = coalesce(as_of, known)`: `select distinct on (v.document_id) v.id from document_versions v join documents d on d.id = v.document_id where d.collection_id = any(%s) and d.created_at <= known and (d.deleted_at is null or d.deleted_at > known) and v.created_at <= known and v.effective_from <= valid order by v.document_id, v.version_number desc`); `def passage_filter(scope: SearchScope, version_ids: list[UUID] | None) -> tuple[str, list[Any]]` returning a WHERE fragment over aliases `p` (passages) and `d` (documents): live `p.is_current and d.deleted_at is null and d.collection_id = any(%s)`, temporal `p.version_id = any(%s)`. Every retrieval query joins `documents d on d.id = p.document_id` and uses this fragment.
- `backend/retrieval/bm25.py`:

```python
def bm25_term_score(tf: int, df: int, n_docs: int, doc_len: int, avg_doc_len: float,
                    k1: float = 0.9, b: float = 0.4) -> float
def bm25_reference(query_terms: Sequence[str], term_freqs: Mapping[str, Mapping[str, int]],
                   doc_lens: Mapping[str, int], k1: float = 0.9, b: float = 0.4) -> dict[str, float]
async def query_terms(conn, query: str) -> list[str]
async def bm25_search(conn, query: str, scope: SearchScope, *, limit: int,
                      version_ids: list[UUID] | None, k1: float = 0.9, b: float = 0.4) -> list[tuple[UUID, float]]
```

- `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`; term score `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`; distinct query terms (ruling: repeated query words count once). `query_terms` = distinct lexemes of `to_tsvector('english', query)`, sorted. `bm25_search` is one SQL statement: CTE `vis` (visible passage ids and `token_count`), `stats` (`count(*)::float8`, `avg(token_count)::float8` over `vis`), `df` (per term, count of `passage_terms` rows joined to `vis`), and the scored sum over `passage_terms` joined to `vis`, `df`, and `stats`; ordered by score desc then passage id; `limit`. Empty terms or empty `vis` return `[]`. `bm25_reference` takes per-passage term frequencies and lengths and returns scores with N and avgdl taken over all passages given.
- Test corpus helper (`tests/pg_fixtures.py`): `async def seed_docs(db, services, tenant_id, collection_id, texts: Sequence[str], *, effective_from=None) -> list[UUID]` creating one document per text through `create_version` with `parse_text`.

- [ ] **Step 1: Write failing tests**: `test_term_score_literals` (`bm25_term_score(2, 1, 3, 10, 8)` == `1.246538849981458` within 1e-12; `bm25_term_score(1, 3, 3, 5, 5)` == `0.13353139262452254`); `@pg test_sql_matches_reference` (12 passages over 3 documents, 5 queries; read `passage_terms` and `token_count` back for the visible set, compute `bm25_reference`, compare every returned score within 1e-9 and the order); `@pg test_stats_only_over_visible_set` (Review Focus 3: scores for collection X are identical before and after adding 50 documents containing the query terms to collection Y outside the scope); `@pg test_deleted_documents_excluded`; `@pg test_stopword_only_query_empty` (`"the and of"` -> `[]`); `@pg test_as_of_versions` (Review Focus 4: v1 `effective_from` 2026-01-01, v2 2026-06-01; `as_of` 2025-12-01 -> no versions; 2026-03-01 -> v1; live -> v2 passages; capture `known = now()` after v2 commits, create v3 effective 2026-07-01; `as_of` 2026-08-01 with `known_at = known` -> v2, with `known_at` None -> v3); `@pg test_deleted_after_known_at_still_visible` (document deleted after `known`; scope with `known_at = known` still sees it, live scope does not).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_bm25.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules and `seed_docs`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add bm25 scoring in sql over the visible passage set`.

### Task 14: Dense vector search and the full-text baseline

**Files:**
- Create: `backend/retrieval/dense.py`, `backend/retrieval/fts.py`, `tests/test_dense_fts.py`

**Interfaces:**
- `async def dense_search(conn, query_vec: np.ndarray, scope: SearchScope, *, limit: int, version_ids: list[UUID] | None, exact: bool = False) -> list[tuple[UUID, float]]`: before the query `set local hnsw.ef_search = <max(100, limit)>` (integer formatted from code, not user input) and `set local hnsw.iterative_scan = relaxed_order`; with `exact` instead `set local enable_indexscan = off` and `set local enable_bitmapscan = off`; `order by p.embedding <=> %s limit %s` with the visible filter in WHERE; returns `(passage_id, 1 - distance)` re-sorted by similarity desc then id.
- `async def fts_search(conn, query: str, scope: SearchScope, *, limit: int, version_ids: list[UUID] | None) -> list[tuple[UUID, float]]`: OR query built in SQL from the query lexemes (`array_to_string(array(select quote_literal(lexeme) from unnest(to_tsvector('english', %s))), ' | ')::tsquery`), filter `p.tsv @@ q`, score `ts_rank_cd(p.tsv, q)`, ordered desc then id; no lexemes -> `[]`.

- [ ] **Step 1: Write failing tests** (all `pg`, `HashEmbedder`): `test_dense_nearest_first` (query `"refund window days"` ranks the refund passage first); `test_exact_and_hnsw_agree` (40 passages, 10 queries: top-5 id sets equal); `test_dense_filter_does_not_starve` (3 passages in readable collection X, 300 passages in collection Y closer to the query; `limit=3` returns exactly the 3 X passages); `test_dense_respects_as_of` (temporal scope returns only v1 passages); `test_fts_or_semantics` (a passage containing only one of three query terms is returned); `test_fts_stopwords_empty`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_dense_fts.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add filtered pgvector search and full-text baseline`.

### Task 15: Reciprocal rank fusion and reranking

**Files:**
- Create: `backend/retrieval/fusion.py`, `backend/retrieval/rerank.py`, `tests/test_fusion_rerank.py`

**Interfaces:**
- `def rrf_fuse(rankings: Sequence[Sequence[K]], k: int = 60) -> list[tuple[K, float]]` (K hashable and orderable): score `sum(1 / (k + rank))` with 1-based ranks; ties broken by best individual rank, then by key ascending; a key repeated within one ranking raises `ValueError`.
- `async def rerank_hits(reranker: Reranker, query: str, hits: Sequence[PassageHit], k: int) -> list[PassageHit]`: scores via `await asyncio.to_thread(reranker.score, query, [h.text for h in hits])`, returns the top `k` as new `PassageHit`s (`dataclasses.replace`) with `scores["rerank"]` set, previous scores kept, sorted by rerank score desc then previous rank, `rank` renumbered from 1; empty input returns `[]` without calling the model.

- [ ] **Step 1: Write failing tests**: `test_rrf_literals` (`[["a","b","c"], ["b","c","d"]]` -> keys `["b","c","a","d"]`, scores `0.03252247488101534`, `0.03200204813108039`, `0.01639344262295082`, `0.015873015873015872` within 1e-15); `test_rrf_tie_break` (key order on a full tie: `[["b"], ["a"]]` -> `["a","b"]`; best rank before key on equal scores: `rrf_fuse([["m","a","c","d","e"], ["x","a","y","w","m"]], k=1)` gives `m` and `a` the same score `0.6666666666666666` (ranks 1 and 5 versus 2 and 2, exact float equality verified) and must order `m` before `a`); `test_rrf_duplicate_raises`; `test_rrf_empty`; `test_rerank_orders_and_renumbers` (`KeywordReranker`, three hits -> best keyword match ranked 1, `scores["rerank"]` present, earlier `scores["rrf"]` kept); `test_rerank_empty_no_call` (a reranker whose `score` raises is never called).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_fusion_rerank.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add reciprocal rank fusion and reranking`.

### Task 16: `PostgresRetriever`

**Files:**
- Create: `backend/retrieval/service.py`, `tests/test_retriever.py`

**Interfaces:**

```python
class PostgresRetriever:
    def __init__(self, db: Database, embedder: Embedder, reranker: Reranker | None = None,
                 *, k1: float = 0.9, b: float = 0.4, rrf_k: int = 60): ...
    async def search(self, query: str, scope: SearchScope, *, k: int = 10, mode: SearchMode = "hybrid",
                     rerank: bool = True, candidates: int = 50, exact: bool = False) -> list[PassageHit]
```

- Validates `1 <= k <= 300`, `1 <= candidates <= 1000`, mode in `SearchMode` (`ValueError` otherwise). Empty `scope.collection_ids` returns `[]`. `rerank` with no reranker raises `RerankUnavailable` before any work. Query embedding (`embed_queries`, in a thread) happens before opening the transaction. Inside `db.tenant_tx(scope.tenant_id)`: `version_ids = visible_version_ids(...)`; component limit `n = max(k, candidates)` when mode is `hybrid` or `rerank` is set, else `k`; `hybrid` fuses `bm25_search` and `dense_search` with `rrf_fuse`; scores dict carries every component score present plus `rrf`; hydrate the top `n` ids in one query joining `document_versions` and `documents`; reranking applies to the hydrated top `n`; the result is cut to `k` with ranks `1..k`. `STAGE_SECONDS{stage="retrieve"}` observed.

- [ ] **Step 1: Write failing tests** (all `pg`, fakes): `test_hybrid_combines_lexical_and_dense` (a passage found only by BM25 and one found only by dense both appear in hybrid top `k`); `test_rerank_reorders`; `test_rerank_without_reranker_raises`; `test_scope_enforced_in_every_mode` (Review Focus 3: for `bm25`, `dense`, `fts`, `hybrid`, with and without rerank, no hit comes from the out-of-scope collection or a deleted document); `test_as_of_in_every_mode` (Review Focus 4: every mode returns only v1 text for an as-of scope between v1 and v2); `test_hit_fields_resolve_exactly` (for every hit `text == version_text[start_char:end_char]`, `version_number` and `document_title` correct); `test_cross_tenant_identical_text_isolated` (tenant B holds the same text; tenant A results contain only tenant A passage ids); `test_validation` (`k=0`, `k=301`, `mode="x"` -> `ValueError`); `test_empty_scope_returns_empty`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_retriever.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `PostgresRetriever`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add hybrid Postgres retriever with access-controlled scope`.

### Task 17: Search route and the `search` job

**Files:**
- Create: `backend/retrieval/handlers.py`, `backend/api/routes/search.py`, `tests/test_search_route.py`
- Modify: `backend/jobs/registry.py`, `backend/api/schemas_v1.py`, `backend/api/app.py`, `backend/api/startup.py` (lifespan sets `app.state.embedder = load_embedder(settings)`, R7), `backend/config.py` (`search_wait_s: float = 15.0`)

**Interfaces:**
- `POST /api/v1/search` body `SearchIn(query: str (1..2000 chars after strip), collection_ids: list[UUID] | None = None, as_of: datetime | None = None, k: int = 10 (1..50), mode: SearchMode = "hybrid", rerank: bool = False)` -> `SearchOut(hits: list[PassageHitOut], mode: str, reranked: bool, took_ms: int)`; `PassageHitOut` mirrors `PassageHit`. Scope: `readable_collection_ids(conn, principal, collection_ids)` (kind `normal`); `known_at = now` when `as_of` is given, else live. Without `rerank` the route runs `PostgresRetriever(db, app.state.embedder, None)` in the API process. With `rerank`, one `tenant_tx` inserts a `search_requests` row (`query`, `params` = the JSON of `k`, `mode`, `collection_ids`, `as_of`, `known_at`) and enqueues `search` with payload `{"search_request_id": "<uuid>"}`, priority 10; the route then polls the row every 100 ms for up to `settings.search_wait_s`; `done` -> read `hits`, delete the row, respond; `failed` -> 502 `search_failed`; timeout -> 504 `search_timeout`.
- `async def handle_search(ctx: JobContext) -> None`: loads the row in `tenant_tx`; non-`queued` rows return; runs `PostgresRetriever(services.db, services.embedder, services.reranker).search(..., rerank=True)`; stores `hits` (list of `PassageHitOut` JSON) with status `done` and `completed_at`; deletes this tenant's `search_requests` rows older than 1 hour; any exception sets `failed` with `error_code = "search_failed"` and raises `PermanentJobError("search_failed")`. Registered as `"search"` in `build_handlers`.

- [ ] **Step 1: Write failing tests** (all `pg`; `api_app.state.embedder = HashEmbedder()`): `test_bm25_search_returns_hits_with_text`; `test_hybrid_without_rerank_runs_in_api` (no worker running; 200, `reranked` false); `test_rerank_through_worker` (a background task loops `drain` every 50 ms while the request runs -> 200, `reranked` true, every hit has `scores["rerank"]`, the `search_requests` row is gone afterwards); `test_rerank_timeout_504` (`search_wait_s=0.2`, no worker -> 504 `search_timeout`); `test_job_payload_has_only_id` (R4: the queued job's payload keys are exactly `{"search_request_id"}` and the query string appears nowhere in `jobs`); `test_unreadable_collection_404`; `test_query_validation` (empty and 2,001-character queries -> 422); `test_as_of_search_returns_old_version`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_search_route.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** handler, route, schemas, startup change.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add search endpoint with worker-side reranking`.

### Task 18: Retrieval metrics

**Files:**
- Create: `evaluation/metrics/__init__.py`, `evaluation/metrics/retrieval.py`, `tests/test_retrieval_metrics.py`

**Interfaces (original implementations, no metric libraries):**
- `def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float`: linear gain (gain = relevance grade, as trec_eval `ndcg_cut`), discount `log2(rank + 1)`, ideal DCG from the qrels grades sorted descending; grades `<= 0` are not relevant; no relevant documents raises `ValueError`.
- `def recall_at_k(ranked, qrels, k) -> float` (relevant retrieved in top k over all relevant); `def mrr_at_k(ranked, qrels, k) -> float` (reciprocal rank of the first relevant document in the top k, else 0).
- `def maxp(passage_ranking: Sequence[str], passage_to_doc: Mapping[str, str]) -> list[str]` (documents in order of their first passage; this is MaxP for any monotone score).
- `def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, float]` with keys `ndcg@10`, `recall@10`, `recall@100`, `mrr@10`, `n_queries`; averages over queries with at least one relevant document; a query missing from `run` counts as an empty ranking; a ranking with a repeated document raises `ValueError`.

- [ ] **Step 1: Write failing tests**: `test_ndcg_literal` (`ndcg_at_k(["d1","d2","d3"], {"d1": 1, "d3": 2}, 3)` == `0.7601875334318685` within 1e-12); `test_ndcg_perfect_and_zero` (ideal order -> 1.0; no relevant retrieved -> 0.0); `test_ndcg_ignores_nonpositive_grades`; `test_recall_and_mrr` (`["a","b","c"]`, `{"b": 1, "z": 1}` -> recall@3 `0.5`, mrr@3 `0.5`); `test_maxp_first_occurrence` (`["p1","p2","p3"]` with `p1 -> A, p2 -> B, p3 -> A` -> `["A","B"]`); `test_evaluate_run_missing_query_counts_zero`; `test_queries_without_relevant_excluded` (`n_queries` counts only judged queries); `test_duplicate_docs_raise`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_retrieval_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add retrieval metrics`.

### Task 19: Pinned dataset downloads and BEIR loading and ingestion

**Files:**
- Create: `evaluation/data.py`, `evaluation/beir.py`, `tests/test_beir.py`
- Modify: `.gitignore` (add `evaluation/data/`)

**Interfaces:**
- `evaluation/data.py`: frozen `DatasetFile(name: str, url: str, sha256: str, kind: Literal["zip", "file"])`; `DATA_DIR = Path("evaluation/data")`; `REGISTRY: dict[str, DatasetFile]` with `scifact` (`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`), `nfcorpus` (`.../nfcorpus.zip`), `ragtruth_response` and `ragtruth_source` (`https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/response.jsonl` and `.../source_info.jsonl`, commit pinned on 2026-10-26); `class DatasetIntegrityError(RuntimeError)`; `def fetch(name: str, data_dir: Path = DATA_DIR, *, client: httpx.Client | None = None) -> Path` (streams to a temporary file, verifies sha256 and deletes the file on mismatch, extracts zips into `data_dir/<name>/` rejecting absolute member paths and any `..` component, skips the download when the file exists with the right hash, returns the extracted directory or file path). `python -m evaluation.data fetch <name>...` and `python -m evaluation.data hash <name>` (downloads without verification and prints the sha256, used once to pin the hashes).
- `evaluation/beir.py`: `@dataclass BeirData(name: str, corpus: dict[str, tuple[str, str]], queries: dict[str, str], qrels: dict[str, dict[str, int]])`; `def load_beir(root: Path, name: str, split: str = "test") -> BeirData` (`corpus.jsonl` fields `_id`, `title`, `text`; `queries.jsonl` fields `_id`, `text`; `qrels/<split>.tsv` with a header row `query-id corpus-id score`; queries restricted to those in the qrels); `async def ensure_bench_scope(db: Database, tenant_name: str, collection_name: str) -> tuple[UUID, UUID]` (creates or reuses a tenant and a `benchmark` collection); `async def ingest_beir(services: Services, data: BeirData, *, tenant_id: UUID, collection_id: UUID, doc_ids: Iterable[str] | None = None, filename_prefix: str = "") -> IngestSummary` (`IngestSummary(doc_map: dict[str, UUID], n_ingested: int, n_existing: int, n_skipped_empty: int)`; document text `title + "\n\n" + text` (title omitted when empty) through `parse_text` and `create_version`; filename `<prefix><beir_id>.txt` identifies existing documents so reruns are idempotent; documents with no text after normalization are skipped and counted; 64 documents per transaction).

- [ ] **Step 1: Pin the hashes:** run `.venv/bin/python -m evaluation.data hash scifact` (and `nfcorpus`, `ragtruth_response`, `ragtruth_source`) once the module skeleton exists, and write the printed hex digests into `REGISTRY`. Record them in the commit message body.
- [ ] **Step 2: Write failing tests** (no network: `httpx.MockTransport`): `test_fetch_verifies_hash` (wrong bytes -> `DatasetIntegrityError`, no file left); `test_fetch_rejects_zip_slip` (a zip with member `../evil.txt` -> `DatasetIntegrityError`); `test_fetch_skips_existing` (transport not called when the file exists with the right hash); `test_registry_hashes_pinned` (every `sha256` is 64 hex characters); `test_load_beir_tiny` (tmp folder with 3 docs, 3 queries of which 2 are in `qrels/test.tsv` -> 2 queries, grades parsed as int); `@pg test_ingest_idempotent` (`HashEmbedder`; second call reports `n_existing == 3`, `n_ingested == 0`); `@pg test_empty_doc_skipped`.
- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_beir.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement**.
- [ ] **Step 5: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 6: Commit** `.gitignore evaluation/data.py evaluation/beir.py tests/test_beir.py` with `feat: add pinned dataset downloads and BEIR ingestion`.

### Task 20: Retrieval benchmark runner and report renderer

**Files:**
- Create: `evaluation/run_retrieval.py`, `evaluation/report.py`, `tests/test_report.py`
- Modify: `tests/test_beir.py` (smoke run)

**Interfaces:**
- `python -m evaluation.run_retrieval --dataset {scifact,nfcorpus} [--database-url URL] [--systems fts,bm25,dense,dense_exact,hybrid,hybrid_rerank] [--limit-queries N] [--embedder real|fake] [--out PATH]`. Defaults: database URL from `HALLUCISCOPE_BENCH_DATABASE_URL`, else `postgresql://eg_app@localhost:54329/evidencegraph_bench`; `--out evaluation/results/retrieval_<dataset>.json`. `async def run(args) -> dict` is the testable entry point.
- Systems (R10a, fixed a priori): every system calls `PostgresRetriever.search(query, scope, k=300, candidates=300, rerank=False, mode=..., exact=...)` and collapses passages to documents with `maxp`; `fts` (mode `fts`), `bm25`, `dense`, `dense_exact` (`dense`, `exact=True`), `hybrid`; `hybrid_rerank` takes the hybrid passage list, keeps the top 50 MaxP documents, reranks every retrieved passage of those documents with the reranker, scores each document by its best rerank score, and appends the remaining hybrid documents in their original order. Latency per query is wall time of the search plus reranking.
- Checks recorded in the output and enforced (non-zero exit when violated): `acl_leaks` (a `distractor` collection in the same tenant holds copies of the first 100 corpus documents with prefix `distractor-`; any hit from it in any system is a leak; must be 0); `bm25_parity_max_abs_diff` (for the first 20 queries, SQL BM25 top-10 scores against `bm25_reference` over all visible passages' `passage_terms`; must be below 1e-9).
- Output JSON: `{"dataset", "split", "n_queries", "n_docs", "n_passages", "n_skipped_docs", "systems": {name: {"ndcg@10", "recall@10", "recall@100", "mrr@10", "latency_ms_p50", "latency_ms_p95"}}, "checks": {...}, "params": {"k1": 0.9, "b": 0.4, "rrf_k": 60, "candidates": 300, "rerank_depth": 50, "ef_search": 300, "passage_target_chars": 800}, "models": {"embedder", "reranker"}, "env": {"git_commit", "git_dirty", "cpu", "ram_gb", "python", "torch", "postgres", "pgvector"}, "started_at", "finished_at", "ingest_seconds", "runtime_seconds"}`. Git info via `/usr/bin/git`; CPU from `sysctl -n machdep.cpu.brand_string`; RAM from `sysctl -n hw.memsize`.
- `evaluation/report.py`: `def render_retrieval(results: Sequence[Mapping]) -> str` (one section per dataset: environment and parameter lines, a table `| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |` with metrics to 4 decimals and latency to 1 decimal, the checks, and a fixed methodology note: BM25 parameters fixed before evaluation, MaxP over sentence-aligned passages, no tuning on test qrels); `python -m evaluation.report retrieval <results.json>... --out docs/results/retrieval.md`.

- [ ] **Step 1: Write failing tests**: `test_render_retrieval_golden` (`tests/test_report.py`: a fixture results dict renders a table row `| bm25 | 0.6123 | 0.7000 | 0.9000 | 0.5500 | 12.3 | 45.6 |` and contains the commit and CPU strings); `@pg test_runner_smoke` (`tests/test_beir.py`: the tiny dataset written to a tmp BEIR folder, `--embedder fake`, `--limit-queries 2`, a test database URL -> every system present, `checks.acl_leaks == 0`, `checks.bm25_parity_max_abs_diff < 1e-9`, metrics within `[0, 1]`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_report.py tests/test_beir.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the runner (with a `--data-root` option used by the smoke test) and the renderer.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat: add retrieval benchmark runner and report`.

### Task 21: Run the BEIR benchmark and publish the measured results

This task produces real numbers. Nothing in `docs/results/retrieval.md` may be typed by hand.

**Files:**
- Create: `evaluation/manifests/beir.json`, `docs/results/retrieval.md`

- [ ] **Step 1: Prepare** `bash scripts/dev-postgres.sh start`; create the benchmark database as `postgres`: `createdb -h localhost -p 54329 -U postgres -O eg_owner evidencegraph_bench` then `psql ... -d evidencegraph_bench -c "create extension if not exists vector; create extension if not exists citext"`; migrate with `HALLUCISCOPE_MIGRATION_DATABASE_URL=postgresql://eg_owner@localhost:54329/evidencegraph_bench .venv/bin/python -m backend.cli.main migrate`. Fetch data: `.venv/bin/python -m evaluation.data fetch scifact nfcorpus`.
- [ ] **Step 2: Run** `.venv/bin/python -m evaluation.run_retrieval --dataset scifact` then `--dataset nfcorpus` (real models; expect tens of minutes each on the M1). Expected: exit 0, `acl_leaks == 0`, parity below 1e-9.
- [ ] **Step 3: Sanity check before publishing:** published flat BM25 nDCG@10 is about 0.665 on SciFact and 0.325 on NFCorpus (BEIR paper). Our passage-level MaxP BM25 with the Postgres English stemmer may differ by a few points. If our `bm25` nDCG@10 is below 0.55 on SciFact or below 0.25 on NFCorpus, stop and debug with superpowers:systematic-debugging (`/Users/omarmorsi/.kiro/skills/sp-systematic-debugging/SKILL.md`) before publishing; the threshold is a bug detector, not a target, and the published table shows whatever the corrected code measures.
- [ ] **Step 4: Render** `.venv/bin/python -m evaluation.report retrieval evaluation/results/retrieval_scifact.json evaluation/results/retrieval_nfcorpus.json --out docs/results/retrieval.md`. Write `evaluation/manifests/beir.json` with, per dataset, `url`, `sha256`, `split`, `n_docs`, `n_queries`, `n_passages`, and `fetched_at`, copied from the results JSON and `REGISTRY` by a short script (not by hand).
- [ ] **Step 5: Verify** `git status` shows only the two new files (raw results and data stay ignored), and `.venv/bin/python -m pytest -q` still passes.
- [ ] **Step 6: Commit** `docs/results/retrieval.md evaluation/manifests/beir.json` with `docs: add measured BEIR retrieval results`.
