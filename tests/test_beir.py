"""Pinned dataset downloads (no network: httpx.MockTransport) and BEIR loading and ingestion."""

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import PassageHit, SearchScope
from evaluation import beir, data, run_retrieval
from evaluation.beir import BeirData
from evaluation.data import DatasetFile, DatasetIntegrityError
from tests.fakes import KeywordReranker

pg = pytest.mark.pg

GOOD_ZIP_MEMBERS = {"tiny/corpus.jsonl": b'{"_id": "d1"}\n', "tiny/readme.txt": b"hello"}


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return buffer.getvalue()


def serve(payload: bytes, status: int = 200) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client answering every request with `payload`, plus the list of requests it received."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def register(monkeypatch, name: str, payload: bytes, kind: str, filename: str) -> DatasetFile:
    """Pin `payload` under `name`; the URL basename decides the local file name."""
    entry = DatasetFile(
        name=name,
        url=f"https://example.test/files/{filename}",
        sha256=hashlib.sha256(payload).hexdigest(),
        kind=kind,
    )
    monkeypatch.setitem(data.REGISTRY, name, entry)
    return entry


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_registry_hashes_pinned():
    for entry in data.REGISTRY.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256), entry.name


def test_fetch_downloads_verifies_and_extracts_zip(tmp_path, monkeypatch):
    archive = make_zip(GOOD_ZIP_MEMBERS)
    register(monkeypatch, "tiny", archive, "zip", "tiny.zip")
    client, calls = serve(archive)

    result = data.fetch("tiny", tmp_path / "data", client=client)

    assert result == tmp_path / "data" / "tiny"
    assert (result / "tiny" / "corpus.jsonl").read_bytes() == GOOD_ZIP_MEMBERS["tiny/corpus.jsonl"]
    assert (result / "tiny" / "readme.txt").read_bytes() == b"hello"
    assert len(calls) == 1
    # Neither a partial download nor an extraction scratch directory is left behind.
    assert names_in(tmp_path / "data") == ["tiny", "tiny.zip"]


def test_fetch_file_kind_returns_the_file(tmp_path, monkeypatch):
    payload = b'{"id": "1"}\n{"id": "2"}\n'
    register(monkeypatch, "lines", payload, "file", "lines.jsonl")
    client, _ = serve(payload)

    result = data.fetch("lines", tmp_path / "data", client=client)

    assert result == tmp_path / "data" / "lines.jsonl"
    assert result.read_bytes() == payload
    assert names_in(tmp_path / "data") == ["lines.jsonl"]


def test_fetch_verifies_hash(tmp_path, monkeypatch):
    register(monkeypatch, "tiny", b"the pinned bytes", "file", "tiny.jsonl")
    client, calls = serve(b"tampered bytes")

    with pytest.raises(DatasetIntegrityError):
        data.fetch("tiny", tmp_path / "data", client=client)

    assert len(calls) == 1
    assert names_in(tmp_path / "data") == []


def test_fetch_http_error_leaves_no_file(tmp_path, monkeypatch):
    register(monkeypatch, "tiny", b"the pinned bytes", "file", "tiny.jsonl")
    client, _ = serve(b"not found", status=404)

    with pytest.raises(httpx.HTTPStatusError):
        data.fetch("tiny", tmp_path / "data", client=client)

    assert names_in(tmp_path / "data") == []


@pytest.mark.parametrize(
    "member",
    [
        "../evil.txt",
        "tiny/../../evil.txt",
        "/abs/evil.txt",
        "..\\evil.txt",
        "C:\\evil.txt",
    ],
)
def test_fetch_rejects_zip_slip(tmp_path, monkeypatch, member):
    archive = make_zip({"tiny/ok.txt": b"fine", member: b"evil"})
    register(monkeypatch, "tiny", archive, "zip", "tiny.zip")
    client, _ = serve(archive)

    with pytest.raises(DatasetIntegrityError):
        data.fetch("tiny", tmp_path / "data", client=client)

    assert list(tmp_path.rglob("evil.txt")) == []
    # Nothing was extracted, not even the harmless member listed before the bad one.
    assert not (tmp_path / "data" / "tiny").exists()
    assert set(names_in(tmp_path / "data")) <= {"tiny.zip"}


def test_fetch_skips_existing(tmp_path, monkeypatch):
    archive = make_zip(GOOD_ZIP_MEMBERS)
    register(monkeypatch, "tiny", archive, "zip", "tiny.zip")
    data_dir = tmp_path / "data"
    first_client, _ = serve(archive)
    data.fetch("tiny", data_dir, client=first_client)

    second_client, calls = serve(b"must not be requested")
    result = data.fetch("tiny", data_dir, client=second_client)

    assert calls == []
    assert result == data_dir / "tiny"
    assert (result / "tiny" / "readme.txt").read_bytes() == b"hello"


def test_fetch_extracts_when_archive_exists_without_directory(tmp_path, monkeypatch):
    archive = make_zip(GOOD_ZIP_MEMBERS)
    register(monkeypatch, "tiny", archive, "zip", "tiny.zip")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiny.zip").write_bytes(archive)
    client, calls = serve(b"must not be requested")

    result = data.fetch("tiny", data_dir, client=client)

    assert calls == []
    assert (result / "tiny" / "corpus.jsonl").is_file()


def test_fetch_replaces_existing_file_with_wrong_hash(tmp_path, monkeypatch):
    payload = b"the pinned bytes"
    register(monkeypatch, "tiny", payload, "file", "tiny.jsonl")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiny.jsonl").write_bytes(b"corrupted earlier download")
    client, calls = serve(payload)

    result = data.fetch("tiny", data_dir, client=client)

    assert len(calls) == 1
    assert result.read_bytes() == payload


def test_hash_command_prints_digest_and_keeps_no_file(tmp_path, monkeypatch, capsys):
    payload = b"bytes that are not pinned yet"
    monkeypatch.setitem(
        data.REGISTRY, "tiny", DatasetFile("tiny", "https://example.test/tiny.jsonl", "", "file")
    )
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "data")
    client, calls = serve(payload)

    exit_code = data.main(["hash", "tiny"], client=client)

    assert exit_code == 0
    assert len(calls) == 1
    assert capsys.readouterr().out.strip() == hashlib.sha256(payload).hexdigest()
    assert names_in(tmp_path / "data") == []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_tiny_beir(folder: Path) -> None:
    """Three documents and three queries; q3 is judged only in the train split."""
    (folder / "qrels").mkdir(parents=True)
    write_jsonl(
        folder / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Alpha", "text": "First body.", "metadata": {}},
            {"_id": "d2", "title": "", "text": "No title body.", "metadata": {}},
            {"_id": "d3", "title": "Gamma", "text": "Third body.", "metadata": {}},
        ],
    )
    write_jsonl(
        folder / "queries.jsonl",
        [
            {"_id": "q1", "text": "first question", "metadata": {}},
            {"_id": "q2", "text": "second question", "metadata": {}},
            {"_id": "q3", "text": "third question", "metadata": {}},
        ],
    )
    # CRLF line endings, as in the published SciFact qrels.
    (folder / "qrels" / "test.tsv").write_bytes(
        b"query-id\tcorpus-id\tscore\r\nq1\td1\t1\r\nq1\td3\t0\r\nq2\td2\t2\r\n"
    )
    (folder / "qrels" / "train.tsv").write_bytes(b"query-id\tcorpus-id\tscore\r\nq3\td3\t1\r\n")


@pytest.mark.parametrize("layout", ["nested", "flat"])
def test_load_beir_tiny(tmp_path, layout):
    # nested: the zip's own top-level folder, as fetch() extracts it; flat: the folder itself.
    folder = tmp_path / "tiny" if layout == "nested" else tmp_path
    write_tiny_beir(folder)

    loaded = beir.load_beir(tmp_path, "tiny")

    assert loaded.name == "tiny"
    assert loaded.corpus == {
        "d1": ("Alpha", "First body."),
        "d2": ("", "No title body."),
        "d3": ("Gamma", "Third body."),
    }
    assert loaded.queries == {"q1": "first question", "q2": "second question"}
    assert loaded.qrels == {"q1": {"d1": 1, "d3": 0}, "q2": {"d2": 2}}


def test_load_beir_other_split(tmp_path):
    write_tiny_beir(tmp_path / "tiny")

    loaded = beir.load_beir(tmp_path, "tiny", split="train")

    assert loaded.queries == {"q3": "third question"}
    assert loaded.qrels == {"q3": {"d3": 1}}


def test_load_beir_rejects_qrels_without_header(tmp_path):
    write_tiny_beir(tmp_path / "tiny")
    (tmp_path / "tiny" / "qrels" / "test.tsv").write_bytes(b"q1\td1\t1\r\n")

    with pytest.raises(ValueError, match="header"):
        beir.load_beir(tmp_path, "tiny")


def small_corpus(**extra: tuple[str, str]) -> BeirData:
    corpus = {
        "d1": ("Alpha", "Alpha is the first body."),
        "d2": ("Beta", "Beta is the second body."),
        "d3": ("", "Gamma has no title."),
        **extra,
    }
    return BeirData("tiny", corpus, {}, {})


async def counts(db, tenant_id) -> dict[str, int]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select (select count(*) from documents) as documents, "
            "(select count(*) from document_versions) as versions, "
            "(select count(*) from documents where current_version_id is not null) as current"
        )
        return dict(await cur.fetchone())


@pg
async def test_ensure_bench_scope_reuses_tenant_and_benchmark_collection(db):
    first = await beir.ensure_bench_scope(db, "bench", "scifact")
    again = await beir.ensure_bench_scope(db, "bench", "scifact")
    other = await beir.ensure_bench_scope(db, "bench", "nfcorpus")

    assert again == first
    assert other[0] == first[0]
    assert other[1] != first[1]
    async with db.tenant_tx(first[0]) as conn:
        cur = await conn.execute("select name, kind from collections order by name")
        assert [(row["name"], row["kind"]) for row in await cur.fetchall()] == [
            ("nfcorpus", "benchmark"),
            ("scifact", "benchmark"),
        ]
    async with db.system_tx() as conn:
        cur = await conn.execute("select count(*) as n from tenants")
        assert (await cur.fetchone())["n"] == 1


@pg
async def test_ensure_bench_scope_refuses_a_collection_that_is_not_a_benchmark(db):
    tenant_id, _ = await beir.ensure_bench_scope(db, "bench", "scifact")
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute(
            "insert into collections (tenant_id, name) values (%s, 'people-docs')", (tenant_id,)
        )

    with pytest.raises(ValueError, match="benchmark"):
        await beir.ensure_bench_scope(db, "bench", "people-docs")


@pg
async def test_ingest_idempotent(db, services):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = small_corpus()

    first = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )
    second = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )

    assert (first.n_ingested, first.n_existing, first.n_skipped_empty) == (3, 0, 0)
    assert set(first.doc_map) == {"d1", "d2", "d3"}
    assert (second.n_ingested, second.n_existing, second.n_skipped_empty) == (0, 3, 0)
    assert second.doc_map == first.doc_map
    # The rerun created neither documents nor versions.
    assert await counts(db, tenant_id) == {"documents": 3, "versions": 3, "current": 3}


@pg
async def test_ingest_doc_ids_restricts_and_later_runs_add_the_rest(db, services):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = small_corpus()

    subset = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id, doc_ids=["d2"]
    )
    rest = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )

    assert set(subset.doc_map) == {"d2"}
    assert (subset.n_ingested, subset.n_existing) == (1, 0)
    assert (rest.n_ingested, rest.n_existing) == (2, 1)
    assert rest.doc_map["d2"] == subset.doc_map["d2"]
    assert await counts(db, tenant_id) == {"documents": 3, "versions": 3, "current": 3}


@pg
async def test_ingest_text_title_and_filename(db, services):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = small_corpus()

    summary = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id, filename_prefix="s-"
    )
    # Same BEIR ids under another prefix are different documents.
    other = await beir.ingest_beir(
        services,
        corpus,
        tenant_id=tenant_id,
        collection_id=collection_id,
        doc_ids=["d1"],
        filename_prefix="t-",
    )

    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select d.id, d.filename, d.title, d.mime_type, v.text from documents d "
            "join document_versions v on v.id = d.current_version_id"
        )
        rows = {row["id"]: row for row in await cur.fetchall()}
    assert other.n_ingested == 1
    d1 = rows[summary.doc_map["d1"]]
    assert (d1["filename"], d1["title"], d1["mime_type"]) == ("s-d1.txt", "Alpha", "text/plain")
    assert d1["text"] == "Alpha\n\nAlpha is the first body."
    # No title: the text is the body alone, and the BEIR id stands in as the document title.
    d3 = rows[summary.doc_map["d3"]]
    assert (d3["filename"], d3["title"]) == ("s-d3.txt", "d3")
    assert d3["text"] == "Gamma has no title."
    assert rows[other.doc_map["d1"]]["filename"] == "t-d1.txt"
    assert other.doc_map["d1"] != summary.doc_map["d1"]


@pg
async def test_empty_doc_skipped(db, services):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = BeirData(
        "tiny",
        {"e1": ("", ""), "e2": ("  ", " \n\t "), "ok": ("Title", "A real body.")},
        {},
        {},
    )

    first = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )
    second = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )

    assert (first.n_ingested, first.n_existing, first.n_skipped_empty) == (1, 0, 2)
    assert set(first.doc_map) == {"ok"}
    assert (second.n_ingested, second.n_existing, second.n_skipped_empty) == (0, 1, 2)
    # No document row, not even an empty one, was created for the skipped documents.
    assert await counts(db, tenant_id) == {"documents": 1, "versions": 1, "current": 1}


@pg
async def test_ingest_commits_in_batches_of_64_documents(db, services, monkeypatch):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = BeirData(
        "tiny", {f"d{i}": ("", f"Document number {i} body.") for i in range(130)}, {}, {}
    )
    transactions = []
    real_tenant_tx = db.tenant_tx

    def counting_tenant_tx(tenant):
        transactions.append(tenant)
        return real_tenant_tx(tenant)

    monkeypatch.setattr(db, "tenant_tx", counting_tenant_tx)

    summary = await beir.ingest_beir(
        services, corpus, tenant_id=tenant_id, collection_id=collection_id
    )

    monkeypatch.undo()
    assert summary.n_ingested == 130
    assert len(summary.doc_map) == 130
    assert len(transactions) == 3  # 64 + 64 + 2
    assert await counts(db, tenant_id) == {"documents": 130, "versions": 130, "current": 130}


DOC_A, DOC_B, DOC_C, LEAKED = (UUID(int=n) for n in (1, 2, 3, 99))
NAMES = {DOC_A: "A", DOC_B: "B", DOC_C: "C"}


def make_hit(number: int, document_id: UUID, text: str = "text") -> PassageHit:
    """A complete hit; `number` is both its passage identity and its 1-based hybrid rank."""
    return PassageHit(
        passage_id=UUID(int=1000 + number),
        document_id=document_id,
        version_id=UUID(int=0),
        version_number=1,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        document_title="title",
        text=text,
        start_char=0,
        end_char=len(text),
        page_start=None,
        page_end=None,
        section=None,
        scores={},
        rank=number,
    )


def test_percentile_is_nearest_rank():
    values = list(range(20, 0, -1))  # 20 down to 1: the input need not be sorted

    assert run_retrieval.percentile(values, 50) == 10
    assert run_retrieval.percentile(values, 95) == 19
    assert run_retrieval.percentile([7.5], 95) == 7.5
    with pytest.raises(ValueError, match="empty"):
        run_retrieval.percentile([], 50)


def test_split_leaks_drops_and_counts_distractor_hits():
    hits = [make_hit(1, DOC_A), make_hit(2, LEAKED), make_hit(3, DOC_B), make_hit(4, LEAKED)]

    clean, leaks = run_retrieval.split_leaks(hits, NAMES, {LEAKED})

    assert [hit.rank for hit in clean] == [1, 3]
    assert leaks == 2


def test_split_leaks_rejects_a_hit_from_outside_the_benchmark():
    with pytest.raises(RuntimeError, match="outside"):
        run_retrieval.split_leaks([make_hit(1, UUID(int=555))], NAMES, {LEAKED})


def test_document_ranking_orders_documents_by_their_best_passage():
    hits = [make_hit(1, DOC_A), make_hit(2, DOC_B), make_hit(3, DOC_A), make_hit(4, DOC_C)]

    assert run_retrieval.document_ranking(hits, NAMES) == ["A", "B", "C"]


async def test_rerank_documents_reranks_only_the_top_documents():
    # Depth 2 reranks the passages of A and B. C would score 4 and win if it were included;
    # it must keep its place after them instead.
    hits = [
        make_hit(1, DOC_A, "unrelated words"),
        make_hit(2, DOC_B, "alpha beta gamma"),
        make_hit(3, DOC_A, "alpha"),
        make_hit(4, DOC_C, "alpha beta gamma delta"),
    ]

    ranking = await run_retrieval.rerank_documents(
        KeywordReranker(), "alpha beta gamma delta", hits, NAMES, depth=2
    )

    assert ranking == ["B", "A", "C"]


async def test_rerank_documents_scores_a_document_by_its_best_passage():
    # A's first passage scores 0 and its second scores 4; B's only passage scores 2.
    hits = [
        make_hit(1, DOC_B, "alpha beta"),
        make_hit(2, DOC_A, "unrelated words"),
        make_hit(3, DOC_A, "alpha beta gamma delta"),
    ]

    ranking = await run_retrieval.rerank_documents(
        KeywordReranker(), "alpha beta gamma delta", hits, NAMES, depth=50
    )

    assert ranking == ["A", "B"]


def test_git_info_is_null_outside_a_repository(tmp_path):
    assert run_retrieval.git_info(tmp_path) == (None, None)


@pytest.mark.parametrize(
    ("leaks", "parity", "expected"),
    [(0, 0.0, 0), (0, 9.9e-10, 0), (1, 0.0, 1), (0, 1e-9, 1), (0, 2e-9, 1)],
)
def test_main_exits_nonzero_when_a_check_fails(monkeypatch, leaks, parity, expected):
    async def fake_run(args):
        return {"checks": {"acl_leaks": leaks, "bm25_parity_max_abs_diff": parity}}

    monkeypatch.setattr(run_retrieval, "run", fake_run)

    assert run_retrieval.main(["--dataset", "scifact"]) == expected


@pg
async def test_run_system_counts_distractor_hits_only_when_the_collection_is_in_scope(db, services):
    tenant_id, main_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    _, distractor_id = await beir.ensure_bench_scope(db, "bench", "tiny-distractor")
    corpus = small_corpus()
    main = await beir.ingest_beir(services, corpus, tenant_id=tenant_id, collection_id=main_id)
    distractor = await beir.ingest_beir(
        services,
        corpus,
        tenant_id=tenant_id,
        collection_id=distractor_id,
        filename_prefix="distractor-",
    )
    names = {document_id: beir_id for beir_id, document_id in main.doc_map.items()}
    leaked = set(distractor.doc_map.values())
    retriever = PostgresRetriever(db, services.embedder, services.reranker)

    async def leaks_with(collection_ids):
        scope = SearchScope(tenant_id, collection_ids)
        _, latencies, leaks = await run_retrieval.run_system(
            "bm25", retriever, services.reranker, scope, {"q": "alpha body"}, names, leaked
        )
        assert len(latencies) == 1
        return leaks

    assert await leaks_with((main_id,)) == 0
    assert await leaks_with((main_id, distractor_id)) > 0


@pg
async def test_bm25_parity_flags_a_retriever_with_other_parameters(db, services):
    tenant_id, collection_id = await beir.ensure_bench_scope(db, "bench", "tiny")
    corpus = BeirData(
        "tiny",
        {
            "d1": ("", "alpha alpha alpha beta"),
            "d2": ("", "alpha gamma gamma gamma gamma delta epsilon"),
            "d3": ("", "gamma delta"),
        },
        {},
        {},
    )
    await beir.ingest_beir(services, corpus, tenant_id=tenant_id, collection_id=collection_id)
    scope = SearchScope(tenant_id, (collection_id,))
    same = PostgresRetriever(db, services.embedder, k1=0.9, b=0.4)
    other = PostgresRetriever(db, services.embedder, k1=1.5, b=0.75)

    assert await run_retrieval.bm25_parity(db, same, scope, ["alpha", "gamma delta"]) < 1e-9
    assert await run_retrieval.bm25_parity(db, other, scope, ["alpha"]) > 1e-3


@pg
async def test_runner_smoke(db, pg_urls, tmp_path):
    write_tiny_beir(tmp_path / "data" / "scifact")
    out = tmp_path / "results" / "retrieval_scifact.json"
    args = run_retrieval.parse_args(
        [
            "--dataset",
            "scifact",
            "--database-url",
            pg_urls["app"],
            "--embedder",
            "fake",
            "--limit-queries",
            "2",
            "--data-root",
            str(tmp_path / "data"),
            "--out",
            str(out),
        ]
    )

    result = await run_retrieval.run(args)

    assert set(result["systems"]) == {
        "fts",
        "bm25",
        "dense",
        "dense_exact",
        "hybrid",
        "hybrid_rerank",
    }
    assert (result["dataset"], result["split"], result["n_queries"]) == ("scifact", "test", 2)
    assert (result["n_docs"], result["n_passages"], result["n_skipped_docs"]) == (3, 3, 0)
    assert result["checks"]["acl_leaks"] == 0
    assert result["checks"]["bm25_parity_max_abs_diff"] < 1e-9
    for name, metrics in result["systems"].items():
        for key in ("ndcg@10", "recall@10", "recall@100", "mrr@10"):
            assert 0.0 <= metrics[key] <= 1.0, (name, key)
        assert 0.0 < metrics["latency_ms_p50"] <= metrics["latency_ms_p95"], name
    # Dense search returns every passage of a three-passage corpus, and so do the systems built
    # on it: both relevant documents are in each top 100.
    for name in ("dense", "dense_exact", "hybrid", "hybrid_rerank"):
        assert result["systems"][name]["recall@100"] == 1.0, name
    assert result["models"] == {"embedder": "fake-hash-384", "reranker": "fake-keyword"}
    assert result["params"] == {
        "k1": 0.9,
        "b": 0.4,
        "rrf_k": 60,
        "candidates": 300,
        "rerank_depth": 50,
        "ef_search": 300,
        "passage_target_chars": 800,
    }
    env = result["env"]
    assert set(env) == {
        "git_commit",
        "git_dirty",
        "cpu",
        "ram_gb",
        "python",
        "torch",
        "postgres",
        "pgvector",
    }
    assert env["postgres"]
    assert env["pgvector"]
    assert result["ingest_seconds"] >= 0.0
    assert result["runtime_seconds"] >= result["ingest_seconds"]
    assert json.loads(out.read_text(encoding="utf-8")) == result
    # The distractor collection really was filled with prefixed copies, so a zero leak count
    # means the access control held, not that there was nothing to leak.
    async with db.system_tx() as conn:
        cur = await conn.execute("select id from tenants where name = 'bench'")
        tenant_id = (await cur.fetchone())["id"]
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select c.name, count(*) as n, bool_and(starts_with(d.filename, 'distractor-')) as p "
            "from documents d join collections c on c.id = d.collection_id "
            "group by c.name order by c.name"
        )
        rows = [(row["name"], row["n"], row["p"]) for row in await cur.fetchall()]
    assert rows == [("scifact", 3, False), ("scifact-distractor", 3, True)]
