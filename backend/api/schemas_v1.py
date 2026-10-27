"""Request and response models for /api/v1."""

from uuid import UUID

from pydantic import BaseModel


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
