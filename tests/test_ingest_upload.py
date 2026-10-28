"""Upload intake, the ingest_upload job, and its failure hook."""

import asyncio
import io
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from backend.api.errors import ApiError
from backend.auth.service import Principal
from backend.config import Settings
from backend.db.repos import collections, documents, tenants, users
from backend.ingest import handlers, intake
from backend.ingest.handlers import handle_ingest_upload
from backend.ingest.intake import IntakeResult, intake_upload
from backend.jobs import queue
from backend.jobs.queue import Job
from backend.jobs.registry import build_failure_handlers, build_handlers
from backend.jobs.worker import JobContext, Worker
from backend.obs.logging import JsonFormatter, bind, tenant_id_var
from backend.services import Services
from tests.fakes import HashEmbedder, KeywordReranker
from tests.pg_fixtures import drain
from tests.test_parsers_pdf import make_pdf

pytestmark = pytest.mark.pg

SETTINGS = Settings()
V1 = b"Refunds are issued within 30 days. Keep the receipt."
V2 = b"Refunds are issued within 14 days. Keep the receipt and the packaging."
COUNTS_SQL = (
    "select (select count(*) from blobs) as blobs, "
    "(select count(*) from documents) as documents, "
    "(select count(*) from uploads) as uploads, "
    "(select count(*) from document_versions) as versions, "
    "(select count(*) from passages) as passages"
)
NO_ROWS = {"blobs": 0, "documents": 0, "uploads": 0, "versions": 0, "passages": 0}


@dataclass(frozen=True)
class World:
    tenant_id: UUID
    owner: Principal
    collection_id: UUID


class FailingEmbedder(HashEmbedder):
    def embed_passages(self, texts):
        raise RuntimeError("SECRET-TEXT the model fell over")


async def add_user(db, tenant_id: UUID, role: str = "member") -> Principal:
    email = f"user-{uuid4().hex[:10]}@example.com"
    async with db.system_tx() as conn:
        user = await users.insert_user(conn, email, "not-a-real-hash", "Test User")
        await tenants.add_membership(conn, tenant_id, user["id"], role)
    return Principal(user["id"], email, "Test User", tenant_id, role, "session")


@pytest.fixture
async def world(db) -> World:
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, "acme"))["id"]
    owner = await add_user(db, tenant_id, "owner")
    async with db.tenant_tx(tenant_id) as conn:
        collection = await collections.insert_collection(
            conn, tenant_id, "Policies", "", owner.user_id
        )
    return World(tenant_id, owner, collection["id"])


async def grant(db, world: World, user: Principal, permission: str) -> None:
    async with db.tenant_tx(world.tenant_id) as conn:
        await collections.insert_grant(
            conn, world.tenant_id, world.collection_id, user.user_id, permission
        )


async def upload(db, principal: Principal, data: bytes, filename: str = "policy.txt", **kwargs):
    kwargs.setdefault("settings", SETTINGS)
    return await intake_upload(db, principal, data=data, filename=filename, **kwargs)


async def fetch(db, tenant_id: UUID, query: str, params: tuple = ()) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def fetch_one(db, tenant_id: UUID, query: str, params: tuple = ()) -> dict:
    (row,) = await fetch(db, tenant_id, query, params)
    return row


async def upload_row(db, world: World, upload_id: UUID) -> dict:
    return await fetch_one(db, world.tenant_id, "select * from uploads where id = %s", (upload_id,))


async def counts(db, world: World) -> dict:
    return await fetch_one(db, world.tenant_id, COUNTS_SQL)


async def jobs(db, kind: str) -> list[dict]:
    async with db.system_tx() as conn:
        cur = await conn.execute("select * from jobs where kind = %s order by id", (kind,))
        return await cur.fetchall()


async def make_due(db, job_id: int) -> None:
    """Skip the retry backoff so the next drain can claim the job."""
    async with db.system_tx() as conn:
        await conn.execute("update jobs set run_after = now() where id = %s", (job_id,))


def context_for(job_row: dict, db, services) -> JobContext:
    job = Job(
        id=job_row["id"],
        tenant_id=job_row["tenant_id"],
        kind=job_row["kind"],
        payload=job_row["payload"],
        priority=job_row["priority"],
        attempts=job_row["attempts"],
        max_attempts=job_row["max_attempts"],
        request_id=job_row["request_id"],
    )
    return JobContext(job=job, db=db, worker_id="test-worker", services=services)


@contextmanager
def json_logs():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


async def test_upload_then_drain_creates_version(db, services, world):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)

    assert isinstance(result, IntakeResult)
    assert result.status == "queued"
    queued = await upload_row(db, world, result.upload_id)
    assert (queued["status"], queued["document_id"]) == ("queued", result.document_id)

    assert await drain(db, services) == 1

    done = await upload_row(db, world, result.upload_id)
    assert done["status"] == "created_version"
    assert done["error_code"] is None
    assert set(done["stats"]) == {"n_passages", "n_reused", "n_new"}
    assert done["stats"]["n_passages"] >= 1
    assert done["stats"]["n_new"] + done["stats"]["n_reused"] == done["stats"]["n_passages"]
    document = await fetch_one(
        db,
        world.tenant_id,
        "select current_version_id from documents where id = %s",
        (result.document_id,),
    )
    version = await fetch_one(
        db,
        world.tenant_id,
        "select id, version_number from document_versions where document_id = %s",
        (result.document_id,),
    )
    assert version["version_number"] == 1
    assert document["current_version_id"] == version["id"] == done["version_id"]


async def test_new_document_takes_its_fields_from_the_filename(db, world):
    before = datetime.now(UTC)

    result = await upload(
        db, world.owner, V1, "Refund Policy.txt", collection_id=world.collection_id
    )

    document = await fetch_one(
        db, world.tenant_id, "select * from documents where id = %s", (result.document_id,)
    )
    assert (document["title"], document["filename"], document["mime_type"]) == (
        "Refund Policy",
        "Refund Policy.txt",
        "text/plain",
    )
    assert document["collection_id"] == world.collection_id
    assert document["created_by"] == world.owner.user_id
    row = await upload_row(db, world, result.upload_id)
    assert (row["filename"], row["declared_mime"]) == ("Refund Policy.txt", "text/plain")
    assert row["created_by"] == world.owner.user_id
    assert before - timedelta(seconds=1) <= row["effective_from"] <= datetime.now(UTC)


async def test_explicit_title_is_kept_and_capped_at_300_characters(db, world):
    short = await upload(db, world.owner, V1, collection_id=world.collection_id, title="Handbook")
    long = await upload(db, world.owner, V2, collection_id=world.collection_id, title="T" * 400)

    titles = {
        row["id"]: row["title"]
        for row in await fetch(db, world.tenant_id, "select id, title from documents")
    }
    assert titles[short.document_id] == "Handbook"
    assert titles[long.document_id] == "T" * 300


async def test_job_shape_and_request_id(db, world):
    with bind(request_id="req-42"):
        result = await upload(db, world.owner, V1, collection_id=world.collection_id)

    (job,) = await jobs(db, "ingest_upload")
    assert job["payload"] == {"upload_id": str(result.upload_id)}
    assert job["tenant_id"] == world.tenant_id
    assert job["priority"] == 0
    assert job["idempotency_key"] == f"upload:{result.upload_id}"
    assert job["request_id"] == "req-42"
    assert job["status"] == "queued"


async def test_upload_is_audited_with_ids_only(db, world):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)

    async with db.system_tx() as conn:
        cur = await conn.execute("select * from audit_log where action = 'document_uploaded'")
        (entry,) = await cur.fetchall()
    assert entry["tenant_id"] == world.tenant_id
    assert entry["user_id"] == world.owner.user_id
    assert entry["target_id"] == str(result.document_id)
    assert entry["details"] == {"upload_id": str(result.upload_id)}


async def test_upload_is_atomic_with_job(db, world, monkeypatch):
    async def broken_enqueue(*args, **kwargs):
        raise RuntimeError("queue down")

    monkeypatch.setattr(intake, "enqueue", broken_enqueue)

    with pytest.raises(RuntimeError):
        await upload(db, world.owner, V1, collection_id=world.collection_id)

    assert await counts(db, world) == NO_ROWS
    assert await jobs(db, "ingest_upload") == []


async def test_same_bytes_duplicate_without_job(db, services, world):
    first = await upload(db, world.owner, V1, collection_id=world.collection_id)
    await drain(db, services)
    jobs_before = [job["id"] for job in await jobs(db, "ingest_upload")]

    second = await upload(db, world.owner, V1, document_id=first.document_id)

    assert second.status == "duplicate"
    assert second.document_id == first.document_id
    assert [job["id"] for job in await jobs(db, "ingest_upload")] == jobs_before
    document = await fetch_one(
        db,
        world.tenant_id,
        "select current_version_id from documents where id = %s",
        (first.document_id,),
    )
    row = await upload_row(db, world, second.upload_id)
    assert (row["status"], row["version_id"]) == ("duplicate", document["current_version_id"])
    assert await drain(db, services) == 0
    assert (await counts(db, world))["versions"] == 1


async def test_resaved_same_text_duplicate(db, services, world):
    first = await upload(db, world.owner, b"a\r\nb", collection_id=world.collection_id)
    await drain(db, services)

    second = await upload(db, world.owner, b"a\nb", document_id=first.document_id)

    assert second.status == "queued"
    assert await drain(db, services) == 1
    row = await upload_row(db, world, second.upload_id)
    first_row = await upload_row(db, world, first.upload_id)
    assert row["status"] == "duplicate"
    assert row["version_id"] == first_row["version_id"]
    assert (await counts(db, world))["versions"] == 1
    assert await jobs(db, "reverify_document") == []


async def test_new_version_enqueues_reverify(db, services, world):
    first = await upload(db, world.owner, V1, collection_id=world.collection_id)
    await drain(db, services)
    assert await jobs(db, "reverify_document") == []

    second = await upload(db, world.owner, V2, document_id=first.document_id)
    assert await drain(db, services) == 1

    versions = await fetch(
        db,
        world.tenant_id,
        "select id from document_versions where document_id = %s order by version_number",
        (first.document_id,),
    )
    assert len(versions) == 2
    assert (await upload_row(db, world, second.upload_id))["version_id"] == versions[1]["id"]
    (job,) = await jobs(db, "reverify_document")
    assert job["payload"] == {
        "document_id": str(first.document_id),
        "old_version_id": str(versions[0]["id"]),
        "new_version_id": str(versions[1]["id"]),
    }
    assert job["tenant_id"] == world.tenant_id
    assert job["status"] == "queued"
    assert job["priority"] == 5
    assert job["idempotency_key"] == f"reverify:{versions[1]['id']}"
    assert "Refunds" not in json.dumps(job["payload"])


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        pytest.param(b"", "empty.txt", id="empty"),
        pytest.param(b" \n\t \n", "blank.txt", id="whitespace-only"),
        pytest.param(make_pdf([""]), "scan.pdf", id="image-only-pdf"),
    ],
)
async def test_empty_and_image_only_fail_no_text(db, services, world, data, filename):
    result = await upload(db, world.owner, data, filename, collection_id=world.collection_id)

    assert await drain(db, services) == 1

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"], row["version_id"]) == ("failed", "no_text", None)
    (job,) = await jobs(db, "ingest_upload")
    assert (job["status"], job["attempts"], job["last_error"]) == ("failed", 1, "no_text")
    assert (await counts(db, world))["versions"] == 0


@pytest.mark.parametrize(
    ("data", "filename", "code"),
    [
        pytest.param(b"\xff\xfe\x00", "latin.txt", "bad_encoding", id="bad-encoding"),
        pytest.param(b"plain text", "fake.pdf", "unsupported_type", id="not-a-pdf"),
    ],
)
async def test_unparsable_upload_fails_without_retry(db, services, world, data, filename, code):
    result = await upload(db, world.owner, data, filename, collection_id=world.collection_id)

    assert await drain(db, services) == 1

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", code)
    (job,) = await jobs(db, "ingest_upload")
    assert (job["status"], job["attempts"], job["last_error"]) == ("failed", 1, code)


async def test_deleted_document_fails_the_upload(db, services, world):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)
    async with db.tenant_tx(world.tenant_id) as conn:
        assert await documents.soft_delete_document(conn, result.document_id)

    assert await drain(db, services) == 1

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", "document_deleted")
    (job,) = await jobs(db, "ingest_upload")
    assert (job["status"], job["attempts"], job["last_error"]) == ("failed", 1, "document_deleted")


async def test_out_of_order_effective_from_fails_the_upload_and_adds_nothing(db, services, world):
    now = datetime.now(UTC)
    first = await upload(
        db,
        world.owner,
        V1,
        collection_id=world.collection_id,
        effective_from=now - timedelta(hours=1),
    )
    await drain(db, services)

    second = await upload(
        db,
        world.owner,
        V2,
        document_id=first.document_id,
        effective_from=now - timedelta(hours=2),
    )
    assert await drain(db, services) == 1

    row = await upload_row(db, world, second.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", "effective_from_out_of_order")
    assert (await counts(db, world))["versions"] == 1
    assert await jobs(db, "reverify_document") == []


async def test_default_effective_from_not_before_current_version(db, services, world):
    # v1 is dated slightly in the future (allowed by the clock-skew tolerance), so a default of
    # "now" for v2 would precede it. No clock step is needed to reproduce the failure.
    first = await upload(
        db,
        world.owner,
        V1,
        collection_id=world.collection_id,
        effective_from=datetime.now(UTC) + timedelta(seconds=30),
    )
    await drain(db, services)

    second = await upload(db, world.owner, V2, document_id=first.document_id)
    await drain(db, services)

    row = await upload_row(db, world, second.upload_id)
    assert (row["status"], row["error_code"]) == ("created_version", None)
    versions = await fetch(
        db,
        world.tenant_id,
        "select effective_from from document_versions where document_id = %s "
        "order by version_number",
        (first.document_id,),
    )
    assert len(versions) == 2
    assert versions[1]["effective_from"] == versions[0]["effective_from"]


@pytest.mark.parametrize("filename", ["notes.exe", "README", ""])
async def test_unsupported_extension_415(db, world, filename):
    with pytest.raises(ApiError) as info:
        await upload(db, world.owner, V1, filename, collection_id=world.collection_id)

    assert (info.value.status, info.value.code) == (415, "unsupported_type")
    assert await counts(db, world) == NO_ROWS


async def test_too_large_413(db, world):
    small = Settings(max_upload_bytes=10)

    with pytest.raises(ApiError) as info:
        await upload(db, world.owner, b"x" * 11, collection_id=world.collection_id, settings=small)

    assert (info.value.status, info.value.code) == (413, "too_large")
    assert await counts(db, world) == NO_ROWS


async def test_upload_of_exactly_the_limit_is_accepted(db, world):
    small = Settings(max_upload_bytes=10)

    result = await upload(
        db, world.owner, b"x" * 10, collection_id=world.collection_id, settings=small
    )

    assert result.status == "queued"


async def test_effective_from_future_422(db, world):
    with pytest.raises(ApiError) as info:
        await upload(
            db,
            world.owner,
            V1,
            collection_id=world.collection_id,
            effective_from=datetime.now(UTC) + timedelta(minutes=5),
        )

    assert (info.value.status, info.value.code) == (422, "effective_from_in_future")
    assert await counts(db, world) == NO_ROWS


async def test_effective_from_within_clock_skew_is_stored(db, world):
    soon = datetime.now(UTC) + timedelta(seconds=30)

    result = await upload(
        db, world.owner, V1, collection_id=world.collection_id, effective_from=soon
    )

    assert (await upload_row(db, world, result.upload_id))["effective_from"] == soon


@pytest.mark.parametrize("target", ["neither", "both"])
async def test_exactly_one_target_is_required(db, world, target):
    kwargs = {"collection_id": world.collection_id, "document_id": uuid4()}
    if target == "neither":
        kwargs = {}

    with pytest.raises(ValueError):
        await upload(db, world.owner, V1, **kwargs)

    assert await counts(db, world) == NO_ROWS


async def test_member_without_write_404_or_403(db, world):
    reader = await add_user(db, world.tenant_id)
    writer = await add_user(db, world.tenant_id)
    stranger = await add_user(db, world.tenant_id)
    await grant(db, world, reader, "read")
    await grant(db, world, writer, "write")
    existing = await upload(db, world.owner, V1, collection_id=world.collection_id)
    before = await counts(db, world)

    outcomes = {}
    for name, principal, kwargs in [
        ("reader_new", reader, {"collection_id": world.collection_id}),
        ("stranger_new", stranger, {"collection_id": world.collection_id}),
        ("reader_version", reader, {"document_id": existing.document_id}),
        ("stranger_version", stranger, {"document_id": existing.document_id}),
    ]:
        with pytest.raises(ApiError) as info:
            await upload(db, principal, V2, **kwargs)
        outcomes[name] = (info.value.status, info.value.code)

    assert outcomes == {
        "reader_new": (403, "insufficient_permission"),
        "stranger_new": (404, "collection_not_found"),
        "reader_version": (403, "insufficient_permission"),
        "stranger_version": (404, "document_not_found"),
    }
    assert await counts(db, world) == before
    new_document = await upload(db, writer, V2, collection_id=world.collection_id)
    new_version = await upload(db, writer, V2, document_id=existing.document_id)
    assert (new_document.status, new_version.status) == ("queued", "queued")


async def test_missing_or_deleted_document_is_404(db, world):
    existing = await upload(db, world.owner, V1, collection_id=world.collection_id)
    async with db.tenant_tx(world.tenant_id) as conn:
        await documents.soft_delete_document(conn, existing.document_id)

    for document_id in (existing.document_id, uuid4()):
        with pytest.raises(ApiError) as info:
            await upload(db, world.owner, V2, document_id=document_id)
        assert (info.value.status, info.value.code) == (404, "document_not_found")


async def test_collection_kinds_are_passed_to_the_policy(db, world):
    async with db.tenant_tx(world.tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name, kind) values (%s, 'Scratch', 'quick') "
            "returning id",
            (world.tenant_id,),
        )
        quick_id = (await cur.fetchone())["id"]

    with pytest.raises(ApiError) as info:
        await upload(db, world.owner, V1, collection_id=quick_id)
    assert (info.value.status, info.value.code) == (404, "collection_not_found")

    created = await upload(
        db, world.owner, V1, collection_id=quick_id, collection_kinds=("quick",)
    )
    assert created.status == "queued"

    with pytest.raises(ApiError) as info:
        await upload(db, world.owner, V2, document_id=created.document_id)
    assert (info.value.status, info.value.code) == (404, "document_not_found")
    again = await upload(
        db, world.owner, V2, document_id=created.document_id, collection_kinds=("quick",)
    )
    assert again.status == "queued"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("../../etc/passwd.txt", "passwd.txt", id="posix-traversal"),
        pytest.param("..\\..\\windows\\notes.txt", "notes.txt", id="windows-traversal"),
        pytest.param("re\x00po\x07rt\n.txt", "report.txt", id="control-characters"),
        pytest.param("a" * 300 + ".txt", "a" * 251 + ".txt", id="long-name-keeps-extension"),
    ],
)
async def test_filename_sanitized(db, world, raw, expected):
    result = await upload(db, world.owner, V1, raw, collection_id=world.collection_id)

    row = await upload_row(db, world, result.upload_id)
    document = await fetch_one(
        db, world.tenant_id, "select filename from documents where id = %s", (result.document_id,)
    )
    assert row["filename"] == expected
    assert document["filename"] == expected


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        pytest.param(V1, "policy.txt", id="created-version"),
        pytest.param(b"", "empty.txt", id="failed"),
    ],
)
async def test_handler_idempotent(db, services, world, monkeypatch, data, filename):
    await upload(db, world.owner, data, filename, collection_id=world.collection_id)
    await drain(db, services)
    (job,) = await jobs(db, "ingest_upload")
    state_before = (
        await counts(db, world),
        await fetch(db, world.tenant_id, "select * from uploads order by id"),
        await jobs(db, "ingest_upload"),
        await jobs(db, "reverify_document"),
    )

    async def must_not_parse(*args, **kwargs):
        raise AssertionError("a finished upload must not be parsed again")

    monkeypatch.setattr(handlers, "parse_with_timeout", must_not_parse)
    ctx = context_for(job, db, services)
    await handle_ingest_upload(ctx)
    await handle_ingest_upload(ctx)

    state_after = (
        await counts(db, world),
        await fetch(db, world.tenant_id, "select * from uploads order by id"),
        await jobs(db, "ingest_upload"),
        await jobs(db, "reverify_document"),
    )
    assert state_after == state_before


async def test_handler_ignores_an_unknown_upload(db, services, world):
    async with db.system_tx() as conn:
        job_id = await queue.enqueue(
            conn,
            "ingest_upload",
            {"upload_id": str(uuid4())},
            tenant_id=world.tenant_id,
        )

    assert await drain(db, services) == 1

    (job,) = await jobs(db, "ingest_upload")
    assert (job["id"], job["status"]) == (job_id, "succeeded")


async def test_handler_binds_string_context_values(db, services, world, monkeypatch):
    seen = []
    real_parse = handlers.parse_with_timeout

    async def spy(data, filename, timeout_s):
        seen.append(tenant_id_var.get())
        return await real_parse(data, filename, timeout_s)

    monkeypatch.setattr(handlers, "parse_with_timeout", spy)
    await upload(db, world.owner, V1, collection_id=world.collection_id)

    await drain(db, services)

    assert seen == [str(world.tenant_id)]
    assert isinstance(seen[0], str)
    assert tenant_id_var.get() is None


async def test_embedding_failure_retries_then_fails_the_upload(db, world):
    broken = Services(SETTINGS, db, FailingEmbedder(), KeywordReranker())
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)

    for attempt in (1, 2, 3):
        assert await drain(db, broken) == 1
        (job,) = await jobs(db, "ingest_upload")
        assert (job["attempts"], job["last_error"]) == (attempt, "embed_failed")
        row = await upload_row(db, world, result.upload_id)
        if attempt < 3:
            assert (job["status"], row["status"]) == ("queued", "processing")
            await make_due(db, job["id"])

    assert job["status"] == "dead"
    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", "embed_failed")
    assert (await counts(db, world))["versions"] == 0


async def test_lease_expiry_leaves_the_upload_failed(db, services, world):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)
    (job,) = await jobs(db, "ingest_upload")
    async with db.system_tx() as conn:
        claimed = await queue.claim(conn, "dead-worker", ["ingest_upload"])
        await conn.execute(
            "update jobs set max_attempts = 1, heartbeat_at = now() - interval '1 hour' "
            "where id = %s",
            (job["id"],),
        )
    assert claimed is not None
    async with db.tenant_tx(world.tenant_id) as conn:
        await conn.execute(
            "update uploads set status = 'processing' where id = %s", (result.upload_id,)
        )

    worker = Worker(
        db,
        build_handlers(services),
        failure_handlers=build_failure_handlers(services),
        services=services,
        poll_interval_s=0.05,
        stale_after_s=60.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        for _ in range(250):
            row = await upload_row(db, world, result.upload_id)
            if row["status"] == "failed":
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(task, 10)
    finally:
        task.cancel()

    assert (row["status"], row["error_code"]) == ("failed", "lease_expired")
    (job,) = await jobs(db, "ingest_upload")
    assert job["status"] == "dead"


@pytest.mark.parametrize("start", ["queued", "processing"])
async def test_failure_hook_fails_an_open_upload(db, services, world, start):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)
    (job,) = await jobs(db, "ingest_upload")
    async with db.tenant_tx(world.tenant_id) as conn:
        await conn.execute("update uploads set status = %s", (start,))
    hook = build_failure_handlers(services)["ingest_upload"]

    await hook(context_for(job, db, services), "job_timeout")

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", "job_timeout")


@pytest.mark.parametrize(
    ("terminal", "code"),
    [("created_version", None), ("duplicate", None), ("failed", "no_text")],
)
async def test_failure_hook_leaves_a_terminal_upload_alone(db, services, world, terminal, code):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)
    (job,) = await jobs(db, "ingest_upload")
    async with db.tenant_tx(world.tenant_id) as conn:
        await conn.execute("update uploads set status = %s, error_code = %s", (terminal, code))
    hook = build_failure_handlers(services)["ingest_upload"]

    await hook(context_for(job, db, services), "lease_expired")

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == (terminal, code)


async def test_failure_hook_is_idempotent(db, services, world):
    result = await upload(db, world.owner, V1, collection_id=world.collection_id)
    (job,) = await jobs(db, "ingest_upload")
    hook = build_failure_handlers(services)["ingest_upload"]
    ctx = context_for(job, db, services)

    await hook(ctx, "lease_expired")
    await hook(ctx, "job_timeout")

    row = await upload_row(db, world, result.upload_id)
    assert (row["status"], row["error_code"]) == ("failed", "lease_expired")


class BrokenDb:
    def tenant_tx(self, tenant_id):
        raise ConnectionError("SECRET-TEXT database down")


@pytest.mark.parametrize("case", ["no_upload_id", "no_tenant", "database_down", "bad_upload_id"])
async def test_failure_hook_never_raises(db, services, world, case):
    await upload(db, world.owner, V1, collection_id=world.collection_id)
    (job,) = await jobs(db, "ingest_upload")
    job_row = dict(job)
    if case == "no_upload_id":
        job_row["payload"] = {}
    elif case == "no_tenant":
        job_row["tenant_id"] = None
    elif case == "bad_upload_id":
        job_row["payload"] = {"upload_id": "not-a-uuid"}
    ctx = context_for(job_row, BrokenDb() if case == "database_down" else db, services)
    hook = build_failure_handlers(services)["ingest_upload"]

    with json_logs() as stream:
        assert await hook(ctx, "internal_error") is None

    assert "SECRET-TEXT" not in stream.getvalue()


async def test_logs_never_carry_filenames_or_text(db, services, world):
    broken = Services(SETTINGS, db, FailingEmbedder(), KeywordReranker())
    secret = b"SECRET-TEXT is in this file."
    with json_logs() as stream:
        await upload(db, world.owner, secret, "SECRET-NAME.txt", collection_id=world.collection_id)
        await upload(db, world.owner, b"", "SECRET-EMPTY.txt", collection_id=world.collection_id)
        assert await drain(db, services) == 2
        await upload(
            db,
            world.owner,
            b"SECRET-TEXT is different in this file.",
            "SECRET-OTHER.txt",
            collection_id=world.collection_id,
        )
        assert await drain(db, broken, kinds=["ingest_upload"]) == 1

    logs = stream.getvalue()
    assert "SECRET" not in logs
    entries = [json.loads(line) for line in logs.splitlines()]
    (ingested,) = [entry for entry in entries if entry["msg"] == "upload ingested"]
    assert ingested["outcome"] == "created_version"
    assert ingested["n_passages"] >= 1
