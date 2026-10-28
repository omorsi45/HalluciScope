"""Request and response models for /api/v1."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

_SURROGATES = range(0xD800, 0xE000)


def _reject_surrogates(value: str) -> str:
    """Lone surrogates cannot be encoded as UTF-8 and would crash hashing or storage."""
    if any(ord(char) in _SURROGATES for char in value):
        raise ValueError("must not contain lone surrogate characters")
    return value


def _reject_nul(value: str) -> str:
    """Postgres text cannot hold NUL, so it would crash the insert instead of failing validation."""
    if "\x00" in value:
        raise ValueError("must not contain NUL characters")
    return value


# Every request string ends with the surrogate check so a JSON "\ud800" is a 422, not a 500.
# The constraints come first: stacking them on an already validated str skips the strip.
_NO_SURROGATES = AfterValidator(_reject_surrogates)
_NO_NUL = AfterValidator(_reject_nul)
Email = Annotated[str, StringConstraints(max_length=320), _NO_SURROGATES]
Password = Annotated[str, StringConstraints(max_length=256), _NO_SURROGATES]
DisplayName = Annotated[str, StringConstraints(max_length=100), _NO_SURROGATES]
Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    _NO_SURROGATES,
]
Description = Annotated[str, StringConstraints(max_length=1000), _NO_SURROGATES]
Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    _NO_SURROGATES,
    _NO_NUL,
]


class RegisterIn(BaseModel):
    email: Email
    password: Password
    display_name: DisplayName


class LoginIn(BaseModel):
    email: Email
    password: Password


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
    name: Name


class ApiKeyOut(BaseModel):
    id: UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None


class ApiKeyCreatedOut(ApiKeyOut):
    token: str


class CollectionIn(BaseModel):
    name: Name
    description: Description = ""


class CollectionPatch(BaseModel):
    name: Name | None = None
    description: Description | None = None


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
    email: Email
    role: Literal["owner", "admin", "member"]


class AuditOut(BaseModel):
    id: int
    action: str
    user_id: UUID | None
    target_type: str | None
    target_id: str | None
    at: datetime
    details: dict | None


class UploadAcceptedOut(BaseModel):
    document_id: UUID
    upload_id: UUID
    status: str


class UploadOut(BaseModel):
    id: UUID
    document_id: UUID
    status: str
    version_id: UUID | None
    version_number: int | None
    error_code: str | None
    stats: dict
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    filename: str
    mime_type: str
    current_version_number: int | None
    version_count: int
    created_at: datetime


class VersionSummaryOut(BaseModel):
    id: UUID
    version_number: int
    effective_from: datetime
    created_at: datetime
    char_count: int
    n_passages: int
    parser: str
    is_current: bool


class DocumentDetailOut(DocumentOut):
    versions: list[VersionSummaryOut]


class VersionOut(VersionSummaryOut):
    page_map: list[list[int]]
    sections: list
    metadata: dict
    parser_version: str


class VersionTextOut(BaseModel):
    start: int
    end: int
    char_count: int
    text: str


class PassageBoundaryOut(BaseModel):
    id: UUID
    ordinal: int
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    section: str | None


class DiffHunkOut(BaseModel):
    op: Literal["equal", "insert", "delete", "replace"]
    from_start: int
    from_end: int
    to_start: int
    to_end: int
    from_text: str
    to_text: str


class DiffOut(BaseModel):
    from_version: int
    to_version: int
    hunks: list[DiffHunkOut]
    stats: dict[str, int]  # hunk counts per op
