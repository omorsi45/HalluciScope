"""API key management routes under /api/v1/api-keys."""

from uuid import UUID

from fastapi import APIRouter, Depends, Response

from backend.api.schemas_v1 import ApiKeyCreatedOut, ApiKeyCreateIn, ApiKeyOut
from backend.auth.deps import current_principal, get_auth
from backend.auth.service import AuthService, Principal

router = APIRouter()


@router.get("", response_model=list[ApiKeyOut])
async def list_api_keys(
    principal: Principal = Depends(current_principal),
    auth: AuthService = Depends(get_auth),
) -> list[ApiKeyOut]:
    return [ApiKeyOut(**key) for key in await auth.list_api_keys(principal)]


@router.post("", response_model=ApiKeyCreatedOut, status_code=201)
async def create_api_key(
    body: ApiKeyCreateIn,
    principal: Principal = Depends(current_principal),
    auth: AuthService = Depends(get_auth),
) -> ApiKeyCreatedOut:
    info, token = await auth.create_api_key(principal, body.name)
    return ApiKeyCreatedOut(**info, token=token)


@router.delete("/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: UUID,
    principal: Principal = Depends(current_principal),
    auth: AuthService = Depends(get_auth),
) -> Response:
    await auth.revoke_api_key(principal, key_id)
    return Response(status_code=204)
