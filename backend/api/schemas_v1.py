"""Request and response models for /api/v1."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints


class RegisterIn(BaseModel):
    email: str
    password: str
    display_name: str


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: UUID
    email: str
    display_name: str


class TenantOut(BaseModel):
    id: UUID
    name: str
    role: str


class SessionOut(BaseModel):
    user: UserOut
    tenant: TenantOut
    csrf_token: str


class MeOut(BaseModel):
    user: UserOut
    tenant: TenantOut
    tenants: list[TenantOut]


class SwitchTenantIn(BaseModel):
    tenant_id: UUID


class ApiKeyCreateIn(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class ApiKeyOut(BaseModel):
    id: UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None


class ApiKeyCreatedOut(ApiKeyOut):
    token: str


class CollectionIn(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    description: Annotated[str, StringConstraints(max_length=1000)] = ""


class CollectionPatch(BaseModel):
    name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
        | None
    ) = None
    description: Annotated[str, StringConstraints(max_length=1000)] | None = None


class CollectionOut(BaseModel):
    id: UUID
    name: str
    description: str
    kind: str
    created_at: datetime
    permission: str


class GrantIn(BaseModel):
    user_id: UUID
    permission: Literal["read", "write", "admin"]


class GrantsIn(BaseModel):
    grants: list[GrantIn] = Field(max_length=500)


class GrantOut(BaseModel):
    user_id: UUID
    email: str
    display_name: str
    permission: str


class MemberOut(BaseModel):
    user_id: UUID
    email: str
    display_name: str
    role: str
    created_at: datetime


class MemberAddIn(BaseModel):
    email: str
    role: Literal["owner", "admin", "member"]


class AuditOut(BaseModel):
    id: int
    action: str
    user_id: UUID | None
    target_type: str | None
    target_id: str | None
    at: datetime
    details: dict | None
