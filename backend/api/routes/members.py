"""Tenant member and audit routes under /api/v1/tenants/current."""

from uuid import UUID

import psycopg.errors
from fastapi import APIRouter, Depends, Response

from backend.api.errors import ApiError
from backend.api.pagination import (
    Page,
    PageParams,
    decode_int_cursor,
    encode_cursor,
    page_params,
)
from backend.api.schemas_v1 import AuditOut, MemberAddIn, MemberOut
from backend.auth.deps import current_principal, get_db, require_role, require_session
from backend.auth.service import Principal
from backend.db.pool import Database
from backend.db.repos import audit, members, tenants, users

router = APIRouter()

_ADMINS = require_role("owner", "admin")


def _insufficient_permission() -> ApiError:
    return ApiError(403, "insufficient_permission", "Insufficient permission for this action")


@router.get("/members", response_model=list[MemberOut])
async def list_members(
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[MemberOut]:
    async with db.system_tx() as conn:
        rows = await members.list_members(conn, principal.tenant_id)
    return [MemberOut(**row) for row in rows]


@router.post("/members", response_model=MemberOut, status_code=201)
async def add_member(
    body: MemberAddIn,
    _session: Principal = Depends(require_session),
    principal: Principal = Depends(_ADMINS),
    db: Database = Depends(get_db),
) -> MemberOut:
    if body.role == "owner" and principal.role != "owner":
        raise _insufficient_permission()
    try:
        async with db.system_tx() as conn:
            user = await users.get_user_by_email(conn, body.email.strip().lower())
            if user is None:
                raise ApiError(404, "user_not_found", "No user with this email")
            await tenants.add_membership(conn, principal.tenant_id, user["id"], body.role)
            row = await members.member_out(conn, principal.tenant_id, user["id"])
            await audit.record(
                conn,
                action="member_added",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                target_type="user",
                target_id=str(user["id"]),
                details={"role": body.role},
            )
    except psycopg.errors.UniqueViolation:
        raise ApiError(409, "already_member", "This user is already a member") from None
    return MemberOut(**row)


@router.delete("/members/{user_id}", status_code=204)
async def remove_member(
    user_id: UUID,
    _session: Principal = Depends(require_session),
    principal: Principal = Depends(_ADMINS),
    db: Database = Depends(get_db),
) -> Response:
    async with db.system_tx() as conn:
        # Locking the owner rows first makes the last-owner check and the delete race-safe.
        owner_ids = await members.lock_owner_ids(conn, principal.tenant_id)
        if principal.role == "owner" and principal.user_id not in owner_ids:
            # Ownership was lost between authentication and this transaction.
            raise _insufficient_permission()
        target = await tenants.get_membership(conn, principal.tenant_id, user_id)
        if target is None:
            raise ApiError(404, "member_not_found", "No such member")
        if target["role"] == "owner":
            if principal.role != "owner":
                raise _insufficient_permission()
            if len(owner_ids) <= 1:
                raise ApiError(409, "last_owner", "A workspace needs at least one owner")
        await members.delete_membership(conn, principal.tenant_id, user_id)
        await members.revoke_user_credentials(conn, principal.tenant_id, user_id)
        await audit.record(
            conn,
            action="member_removed",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            target_type="user",
            target_id=str(user_id),
            details={"role": target["role"]},
        )
    return Response(status_code=204)


@router.get("/audit", response_model=Page[AuditOut])
async def list_audit(
    principal: Principal = Depends(_ADMINS),
    db: Database = Depends(get_db),
    page: PageParams = Depends(page_params),
) -> Page[AuditOut]:
    after = decode_int_cursor(page.after) if page.after else None
    async with db.system_tx() as conn:
        rows = await members.list_audit(conn, principal.tenant_id, limit=page.limit, after=after)
    has_more = len(rows) > page.limit
    rows = rows[: page.limit]
    next_cursor = encode_cursor(rows[-1]["at"], str(rows[-1]["id"])) if has_more else None
    return Page[AuditOut](items=[AuditOut(**row) for row in rows], next_cursor=next_cursor)
