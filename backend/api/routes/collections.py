"""Collection and grant routes under /api/v1/collections."""

from uuid import UUID

import psycopg.errors
from fastapi import APIRouter, Depends, Response

from backend.api.errors import ApiError
from backend.api.pagination import (
    Page,
    PageParams,
    decode_uuid_cursor,
    encode_cursor,
    page_params,
)
from backend.api.schemas_v1 import (
    CollectionIn,
    CollectionOut,
    CollectionPatch,
    GrantOut,
    GrantsIn,
)
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import is_tenant_admin, require_collection
from backend.db.pool import Database
from backend.db.repos import audit, collections

router = APIRouter()

_NAME_CONSTRAINT = "collections_tenant_id_name_key"


def _name_taken() -> ApiError:
    return ApiError(409, "collection_name_taken", "A collection with this name already exists")


def _not_empty() -> ApiError:
    return ApiError(
        409,
        "collection_not_empty",
        "Collection still has documents",
        "A collection that holds documents, including deleted ones whose history is kept, "
        "cannot be deleted",
    )


def _is_name_conflict(exc: psycopg.errors.UniqueViolation) -> bool:
    return exc.diag.constraint_name == _NAME_CONSTRAINT


@router.get("", response_model=Page[CollectionOut])
async def list_collections(
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
    page: PageParams = Depends(page_params),
) -> Page[CollectionOut]:
    after = decode_uuid_cursor(page.after) if page.after else None
    async with db.tenant_tx(principal.tenant_id) as conn:
        rows = await collections.list_collections(
            conn,
            principal.user_id,
            is_tenant_admin=is_tenant_admin(principal),
            limit=page.limit,
            after=after,
        )
    has_more = len(rows) > page.limit
    rows = rows[: page.limit]
    next_cursor = encode_cursor(rows[-1]["created_at"], str(rows[-1]["id"])) if has_more else None
    return Page[CollectionOut](
        items=[CollectionOut(**row) for row in rows], next_cursor=next_cursor
    )


@router.post("", response_model=CollectionOut, status_code=201)
async def create_collection(
    body: CollectionIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> CollectionOut:
    try:
        async with db.tenant_tx(principal.tenant_id) as conn:
            row = await collections.insert_collection(
                conn, principal.tenant_id, body.name, body.description, principal.user_id
            )
            await collections.insert_grant(
                conn, principal.tenant_id, row["id"], principal.user_id, "admin"
            )
            await audit.record(
                conn,
                action="collection_created",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                target_type="collection",
                target_id=str(row["id"]),
            )
    except psycopg.errors.UniqueViolation as exc:
        if _is_name_conflict(exc):
            raise _name_taken() from None
        raise
    return CollectionOut(**row, permission="admin")


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(
    collection_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> CollectionOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        row = await require_collection(conn, principal, collection_id, "read")
    return CollectionOut(**row)


@router.patch("/{collection_id}", response_model=CollectionOut)
async def update_collection(
    collection_id: UUID,
    body: CollectionPatch,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> CollectionOut:
    changed = len(body.model_fields_set & {"name", "description"})
    try:
        async with db.tenant_tx(principal.tenant_id) as conn:
            current = await require_collection(conn, principal, collection_id, "admin")
            if changed == 0:
                return CollectionOut(**current)
            row = await collections.update_collection(
                conn, collection_id, body.name, body.description
            )
            await audit.record(
                conn,
                action="collection_updated",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                target_type="collection",
                target_id=str(collection_id),
                details={"fields_changed": changed},
            )
    except psycopg.errors.UniqueViolation as exc:
        if _is_name_conflict(exc):
            raise _name_taken() from None
        raise
    return CollectionOut(**row, permission=current["permission"])


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> Response:
    try:
        async with db.tenant_tx(principal.tenant_id) as conn:
            await require_collection(conn, principal, collection_id, "admin")
            await collections.delete_collection(conn, collection_id)
            await audit.record(
                conn,
                action="collection_deleted",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                target_type="collection",
                target_id=str(collection_id),
            )
    except psycopg.errors.ForeignKeyViolation:
        # Documents reference their collection. Soft deletion keeps the document, its versions
        # and its passages (provenance for past analyses), so a collection that ever held a
        # document can never be removed.
        raise _not_empty() from None
    return Response(status_code=204)


@router.get("/{collection_id}/grants", response_model=list[GrantOut])
async def list_grants(
    collection_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[GrantOut]:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await require_collection(conn, principal, collection_id, "admin")
        rows = await collections.list_grants(conn, collection_id)
    return [GrantOut(**row) for row in rows]


@router.put("/{collection_id}/grants", response_model=list[GrantOut])
async def replace_grants(
    collection_id: UUID,
    body: GrantsIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[GrantOut]:
    user_ids = [grant.user_id for grant in body.grants]
    async with db.tenant_tx(principal.tenant_id) as conn:
        await require_collection(conn, principal, collection_id, "admin")
        if len(set(user_ids)) != len(user_ids):
            raise ApiError(422, "validation_error", "Validation error", "grants: duplicate user_id")
        if await collections.count_members(conn, principal.tenant_id, user_ids) != len(user_ids):
            raise ApiError(
                422, "grantee_not_member", "Every grantee must be a member of this workspace"
            )
        await collections.replace_grants(
            conn,
            principal.tenant_id,
            collection_id,
            [(grant.user_id, grant.permission) for grant in body.grants],
        )
        await audit.record(
            conn,
            action="grants_changed",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            target_type="collection",
            target_id=str(collection_id),
            details={"grant_count": len(body.grants)},
        )
        rows = await collections.list_grants(conn, collection_id)
    return [GrantOut(**row) for row in rows]
