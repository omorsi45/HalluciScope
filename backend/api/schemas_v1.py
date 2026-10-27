"""Request and response models for /api/v1."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, StringConstraints


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
