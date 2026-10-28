"""Document, version, and upload routes under /api/v1.

Read routes need read permission on the document's collection, uploads and deletes need
write. A document the caller cannot see is reported exactly like one that does not exist.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Request, Response, UploadFile
from psycopg import AsyncConnection
from pydantic import AwareDatetime

from backend.api.errors import ApiError
from backend.api.pagination import (
    Page,
    PageParams,
    decode_uuid_cursor,
    encode_cursor,
    page_params,
)
from backend.api.schemas_v1 import (
    DocumentDetailOut,
    DocumentOut,
    PassageBoundaryOut,
    Title,
    UploadAcceptedOut,
    UploadOut,
    VersionOut,
    VersionSummaryOut,
    VersionTextOut,
)
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import readable_collection_ids, require_collection
from backend.db.pool import Database
from backend.db.repos import audit, documents
from backend.ingest.intake import IntakeResult, intake_upload

router = APIRouter()

MAX_TEXT_CHARS = 100_000


def _document_not_found() -> ApiError:
    return ApiError(404, "document_not_found", "Document not found")


def _upload_not_found() -> ApiError:
    return ApiError(404, "upload_not_found", "Upload not found")


def _version_not_found() -> ApiError:
    return ApiError(404, "version_not_found", "Version not found")


async def _require_collection_or(
    conn: AsyncConnection,
    principal: Principal,
    collection_id: UUID,
    needed: Literal["read", "write"],
    not_found: Callable[[], ApiError],
) -> None:
    """Check the permission, reporting an unreachable collection as ``not_found()``.

    A collection the caller cannot read gives no hint that the object inside it exists. A
    caller who can read but lacks ``needed`` still gets 403.
    """
    try:
        await require_collection(conn, principal, collection_id, needed)
    except ApiError as exc:
        if exc.status == 404:
            raise not_found() from None
        raise


async def _load_document(
    conn: AsyncConnection,
    principal: Principal,
    document_id: UUID,
    needed: Literal["read", "write"],
) -> dict[str, Any]:
    document = await documents.get_document(conn, document_id)
    if document is None:
        raise _document_not_found()
    await _require_collection_or(
        conn, principal, document["collection_id"], needed, _document_not_found
    )
    return document


async def _find_version(
    conn: AsyncConnection, document_id: UUID, version_number: int
) -> dict[str, Any]:
    """The version's summary row, without its text."""
    for version in await documents.list_versions(conn, document_id):
        if version["version_number"] == version_number:
            return version
    raise _version_not_found()


def _accepted(result: IntakeResult) -> UploadAcceptedOut:
    return UploadAcceptedOut(
        document_id=result.document_id, upload_id=result.upload_id, status=result.status
    )


@router.post(
    "/collections/{collection_id}/documents", response_model=UploadAcceptedOut, status_code=202
)
async def upload_document(
    request: Request,
    collection_id: UUID,
    file: UploadFile,
    title: Annotated[Title | None, Form()] = None,
    effective_from: Annotated[AwareDatetime | None, Form()] = None,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> UploadAcceptedOut:
    result = await intake_upload(
        db,
        principal,
        data=await file.read(),
        filename=file.filename or "",
        collection_id=collection_id,
        title=title,
        effective_from=effective_from,
        settings=request.app.state.settings,
    )
    return _accepted(result)


@router.post("/documents/{document_id}/versions", response_model=UploadAcceptedOut, status_code=202)
async def upload_version(
    request: Request,
    document_id: UUID,
    file: UploadFile,
    effective_from: Annotated[AwareDatetime | None, Form()] = None,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> UploadAcceptedOut:
    result = await intake_upload(
        db,
        principal,
        data=await file.read(),
        filename=file.filename or "",
        document_id=document_id,
        effective_from=effective_from,
        settings=request.app.state.settings,
    )
    return _accepted(result)


@router.get("/uploads/{upload_id}", response_model=UploadOut)
async def get_upload(
    upload_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> UploadOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        row = await documents.get_upload(conn, upload_id)
        if row is None:
            raise _upload_not_found()
        await _require_collection_or(
            conn, principal, row["collection_id"], "read", _upload_not_found
        )
    return UploadOut(**row)


@router.get("/documents", response_model=Page[DocumentOut])
async def list_documents(
    collection_id: UUID | None = None,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
    page: PageParams = Depends(page_params),
) -> Page[DocumentOut]:
    if page.after:
        # The repository binds the key as a UUID: reject anything else as a bad cursor first.
        decode_uuid_cursor(page.after)
    async with db.tenant_tx(principal.tenant_id) as conn:
        collection_ids = await readable_collection_ids(
            conn, principal, None if collection_id is None else [collection_id]
        )
        rows = await documents.list_documents(conn, list(collection_ids), page)
    has_more = len(rows) > page.limit
    rows = rows[: page.limit]
    next_cursor = encode_cursor(rows[-1]["created_at"], str(rows[-1]["id"])) if has_more else None
    return Page[DocumentOut](items=[DocumentOut(**row) for row in rows], next_cursor=next_cursor)


@router.get("/documents/{document_id}", response_model=DocumentDetailOut)
async def get_document(
    document_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> DocumentDetailOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        document = await _load_document(conn, principal, document_id, "read")
        versions = await documents.list_versions(conn, document_id)
    return DocumentDetailOut(
        **document, versions=[VersionSummaryOut(**version) for version in versions]
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> Response:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await _load_document(conn, principal, document_id, "write")
        if not await documents.soft_delete_document(conn, document_id):
            raise _document_not_found()
        await audit.record(
            conn,
            action="document_deleted",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            target_type="document",
            target_id=str(document_id),
        )
    return Response(status_code=204)


@router.get("/documents/{document_id}/versions/{version_number}", response_model=VersionOut)
async def get_version(
    document_id: UUID,
    version_number: int,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> VersionOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await _load_document(conn, principal, document_id, "read")
        version = await documents.get_version(conn, document_id, version_number)
    if version is None:
        raise _version_not_found()
    return VersionOut(**version)


@router.get(
    "/documents/{document_id}/versions/{version_number}/text", response_model=VersionTextOut
)
async def get_version_text(
    document_id: UUID,
    version_number: int,
    start: int = 0,
    end: int | None = None,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> VersionTextOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await _load_document(conn, principal, document_id, "read")
        version = await _find_version(conn, document_id, version_number)
        char_count = version["char_count"]
        stop = min(start + MAX_TEXT_CHARS, char_count) if end is None else end
        if start < 0 or stop > char_count or stop < start or stop - start > MAX_TEXT_CHARS:
            raise ApiError(
                422,
                "bad_range",
                "Invalid text range",
                f"Need 0 <= start <= end <= {char_count} and end - start <= {MAX_TEXT_CHARS}",
            )
        text = await documents.get_version_text(conn, version["id"], start, stop)
    return VersionTextOut(start=start, end=stop, char_count=char_count, text=text)


@router.get(
    "/documents/{document_id}/versions/{version_number}/passages",
    response_model=list[PassageBoundaryOut],
)
async def list_version_passages(
    document_id: UUID,
    version_number: int,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[PassageBoundaryOut]:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await _load_document(conn, principal, document_id, "read")
        version = await _find_version(conn, document_id, version_number)
        rows = await documents.list_passages(conn, version["id"])
    return [PassageBoundaryOut(**row) for row in rows]
