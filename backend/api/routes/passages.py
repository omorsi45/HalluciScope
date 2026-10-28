"""Passage provenance route under /api/v1.

A passage is visible to anyone who can read its collection. A passage the caller cannot see is
reported exactly like one that does not exist, including another tenant's.
"""

from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, Depends

from backend.api.errors import ApiError
from backend.api.schemas_v1 import PassageOut
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import readable_subset
from backend.db.pool import Database
from backend.provenance.resolve import resolve_passage

router = APIRouter()


@router.get("/passages/{passage_id}", response_model=PassageOut)
async def get_passage(
    passage_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> PassageOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        resolved = await resolve_passage(conn, passage_id)
        if resolved is None or resolved.collection_id not in await readable_subset(
            conn, principal, [resolved.collection_id]
        ):
            raise ApiError(404, "passage_not_found", "Passage not found")
    return PassageOut(**asdict(resolved))
