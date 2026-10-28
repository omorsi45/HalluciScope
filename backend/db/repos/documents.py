"""Documents, versions, passages, and blobs repository. Connections come from ``tenant_tx``."""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from psycopg import AsyncConnection

if TYPE_CHECKING:
    from backend.api.pagination import PageParams

_DOCUMENT_COLUMNS = (
    "d.id, d.collection_id, d.title, d.filename, d.mime_type, d.current_version_id, "
    "d.created_by, d.created_at, d.deleted_at, "
    "v.version_number as current_version_number, "
    "(select count(*) from document_versions dv where dv.document_id = d.id)::int "
    "as version_count"
)
_DOCUMENT_FROM = "from documents d left join document_versions v on v.id = d.current_version_id "

# Everything except the (large) text. n_passages and is_current are computed in the same
# statement as the row, so they always agree with it.
_VERSION_COLUMNS = (
    "v.id, v.document_id, v.version_number, v.effective_from, v.content_sha256, v.blob_sha256, "
    "v.metadata, v.parser, v.parser_version, v.char_count, v.created_by, v.created_at, "
    "(select count(*) from passages p where p.version_id = v.id)::int as n_passages, "
    "(v.id = d.current_version_id) is true as is_current"
)
_VERSION_FROM = "from document_versions v join documents d on d.id = v.document_id "


async def insert_blob(conn: AsyncConnection, tenant_id: UUID, sha256: bytes, data: bytes) -> None:
    await conn.execute(
        "insert into blobs (tenant_id, sha256, bytes, size) values (%s, %s, %s, %s) "
        "on conflict do nothing",
        (tenant_id, sha256, data, len(data)),
    )


async def get_blob(conn: AsyncConnection, sha256: bytes) -> bytes | None:
    cur = await conn.execute("select bytes from blobs where sha256 = %s", (sha256,))
    row = await cur.fetchone()
    return bytes(row["bytes"]) if row else None


async def insert_document(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    title: str,
    filename: str,
    mime_type: str,
    created_by: UUID | None,
) -> UUID:
    cur = await conn.execute(
        "insert into documents (tenant_id, collection_id, title, filename, mime_type, created_by) "
        "values (%s, %s, %s, %s, %s, %s) returning id",
        (tenant_id, collection_id, title, filename, mime_type, created_by),
    )
    return (await cur.fetchone())["id"]


async def get_document(
    conn: AsyncConnection, document_id: UUID, *, include_deleted: bool = False
) -> dict[str, Any] | None:
    deleted_sql = "" if include_deleted else "and d.deleted_at is null "
    cur = await conn.execute(
        f"select {_DOCUMENT_COLUMNS} {_DOCUMENT_FROM}where d.id = %s {deleted_sql}",
        (document_id,),
    )
    return await cur.fetchone()


async def list_documents(
    conn: AsyncConnection, collection_ids: list[UUID], page: "PageParams"
) -> list[dict[str, Any]]:
    """Live documents of the collections, newest first.

    Returns up to ``page.limit + 1`` rows so the caller can tell whether another page exists.
    The cursor key must already be validated as a UUID (``decode_uuid_cursor``).
    """
    cursor_sql = "and (d.created_at, d.id) < (%s, %s::uuid) " if page.after else ""
    params: list[Any] = [collection_ids]
    if page.after:
        ts, key = page.after
        params.extend((ts, str(key)))
    params.append(page.limit + 1)
    cur = await conn.execute(
        f"select {_DOCUMENT_COLUMNS} {_DOCUMENT_FROM}"
        "where d.collection_id = any(%s) and d.deleted_at is null "
        f"{cursor_sql}"
        "order by d.created_at desc, d.id desc limit %s",
        params,
    )
    return await cur.fetchall()


async def list_versions(conn: AsyncConnection, document_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"select {_VERSION_COLUMNS} {_VERSION_FROM}where v.document_id = %s "
        "order by v.version_number",
        (document_id,),
    )
    return await cur.fetchall()


async def get_version(
    conn: AsyncConnection, document_id: UUID, version_number: int
) -> dict[str, Any] | None:
    """One version including its text, page map, and sections."""
    cur = await conn.execute(
        f"select {_VERSION_COLUMNS}, v.text, v.page_map, v.sections {_VERSION_FROM}"
        "where v.document_id = %s and v.version_number = %s",
        (document_id, version_number),
    )
    return await cur.fetchone()


async def get_version_text(conn: AsyncConnection, version_id: UUID, start: int, end: int) -> str:
    """The characters ``text[start:end]`` of a version, sliced in SQL (code point offsets)."""
    cur = await conn.execute(
        "select substr(text, %s, %s) as piece from document_versions where id = %s",
        (start + 1, end - start, version_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise LookupError("version not found")
    return row["piece"]


async def list_passages(conn: AsyncConnection, version_id: UUID) -> list[dict[str, Any]]:
    """Passage boundaries of a version in order, without the passage text."""
    cur = await conn.execute(
        "select id, ordinal, start_char, end_char, page_start, page_end, section "
        "from passages where version_id = %s order by ordinal",
        (version_id,),
    )
    return await cur.fetchall()


async def get_upload(conn: AsyncConnection, upload_id: UUID) -> dict[str, Any] | None:
    """An upload of a live document, with the document's collection and the version number."""
    cur = await conn.execute(
        "select u.id, u.document_id, u.status, u.version_id, v.version_number, u.error_code, "
        "u.stats, u.created_at, u.updated_at, d.collection_id "
        "from uploads u join documents d on d.id = u.document_id "
        "left join document_versions v on v.id = u.version_id "
        "where u.id = %s and d.deleted_at is null",
        (upload_id,),
    )
    return await cur.fetchone()


async def soft_delete_document(conn: AsyncConnection, document_id: UUID) -> bool:
    """Mark the document deleted. False when it is missing or already deleted."""
    cur = await conn.execute(
        "update documents set deleted_at = now() where id = %s and deleted_at is null returning id",
        (document_id,),
    )
    return await cur.fetchone() is not None


async def count_live_documents(conn: AsyncConnection, collection_id: UUID) -> int:
    cur = await conn.execute(
        "select count(*) as n from documents where collection_id = %s and deleted_at is null",
        (collection_id,),
    )
    return (await cur.fetchone())["n"]
