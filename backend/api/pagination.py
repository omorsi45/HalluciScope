"""Cursor pagination shared by list endpoints.

Lists are ordered by ``created_at desc, id desc`` and continue with the condition
``(created_at, id) < (cursor_ts, cursor_key)``. A cursor is opaque to clients.
"""

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from fastapi import Query
from pydantic import BaseModel

from backend.api.errors import ApiError

_BIGINT_MAX = 2**63 - 1
_BIGINT_MAX_DIGITS = len(str(_BIGINT_MAX))


def _bad_cursor() -> ApiError:
    return ApiError(400, "bad_cursor", "Invalid pagination cursor")


def encode_cursor(ts: datetime, key: str) -> str:
    raw = json.dumps([ts.isoformat(), key]).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        ts_text, key = payload
        if not isinstance(ts_text, str) or not isinstance(key, str):
            raise _bad_cursor()
        ts = datetime.fromisoformat(ts_text)
    except (ValueError, TypeError, binascii.Error, UnicodeError):
        raise _bad_cursor() from None
    if ts.tzinfo is None:
        raise _bad_cursor()
    return ts, key


def decode_uuid_cursor(cursor: tuple[datetime, str]) -> tuple[datetime, UUID]:
    """For lists keyed by a UUID: a key that is not a UUID is a bad cursor."""
    ts, key = cursor
    try:
        return ts, UUID(key)
    except ValueError:
        raise _bad_cursor() from None


def decode_int_cursor(cursor: tuple[datetime, str]) -> tuple[datetime, int]:
    """For lists keyed by a bigint id: a key that is not a bigint is a bad cursor."""
    ts, key = cursor
    # The length check comes first: int() on a very long digit string raises ValueError.
    if (
        len(key) > _BIGINT_MAX_DIGITS
        or not (key.isascii() and key.isdecimal())
        or int(key) > _BIGINT_MAX
    ):
        raise _bad_cursor()
    return ts, int(key)


@dataclass(frozen=True)
class PageParams:
    limit: int
    after: tuple[datetime, str] | None


def page_params(limit: int = Query(20, ge=1, le=100), cursor: str | None = None) -> PageParams:
    return PageParams(limit=limit, after=decode_cursor(cursor) if cursor else None)


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
