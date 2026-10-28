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

# Everything except the (large) text.
_VERSION_COLUMNS = (
    "id, document_id, version_number, effective_from, content_sha256, blob_sha256, "
    "metadata, parser, parser_version, char_count, created_by, created_at"
)


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
        f"select {_VERSION_COLUMNS} from document_versions where document_id = %s "
        "order by version_number",
        (document_id,),
    )
    return await cur.fetchall()


async def get_version(
    conn: AsyncConnection, document_id: UUID, version_number: int
) -> dict[str, Any] | None:
    """One version including its text, page map, and sections."""
    cur = await conn.execute(
        f"select {_VERSION_COLUMNS}, text, page_map, sections from document_versions "
        "where document_id = %s and version_number = %s",
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
