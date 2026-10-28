"""Pinned dataset downloads (no network: httpx.MockTransport) and BEIR loading and ingestion."""

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import httpx
import pytest

from evaluation import beir, data
from evaluation.beir import BeirData
from evaluation.data import DatasetFile, DatasetIntegrityError

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
