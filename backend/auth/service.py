"""Authentication service: registration, login with throttling, sessions, API keys.

Every time-dependent decision (session expiry, throttle windows) uses the injected clock
and passes timestamps to SQL as parameters, so tests control time with a fake clock.
Audit details never carry passwords or tokens.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from psycopg.errors import UniqueViolation

from backend.auth.passwords import (
    WeakPassword,
    hash_password,
    validate_password,
    verify_dummy,
    verify_password,
)
from backend.auth.tokens import (
    new_api_key,
    new_csrf_token,
    new_session_token,
    parse_api_key,
    token_hash,
)
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import api_keys, audit, sessions, tenants, throttle, users

Role = Literal["owner", "admin", "member"]

_EMAIL_RE = re.compile(r"[^@\s]{1,64}@[^@\s]{1,255}")
_DISPLAY_NAME_MAX = 100
_EMAIL_MAX = 320
_THROTTLE_WINDOW_MINUTES = 15
_EMAIL_FAILURE_LIMIT = 5
_IP_FAILURE_LIMIT = 20
_KEY_ADMIN_ROLES = frozenset({"owner", "admin"})


def utcnow() -> datetime:
    return datetime.now(UTC)


class AuthError(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    email: str
    display_name: str
    tenant_id: UUID
    role: Role
    auth: Literal["session", "api_key"]
    session_id: UUID | None = None
    csrf_hash: bytes | None = None
    api_key_id: UUID | None = None


@dataclass(frozen=True)
class LoginResult:
    session_token: str
    csrf_token: str
    principal: Principal


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not _EMAIL_RE.fullmatch(normalized):
        raise AuthError("invalid_email", 422)
    return normalized


def _require_session(principal: Principal) -> None:
    """API keys must not mint or revoke API keys, so a leaked key cannot persist itself."""
    if principal.auth == "api_key":
        raise AuthError("session_required", 403)


class AuthService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._db = db
        self._settings = settings
        self._clock = clock

    def _window_start(self, now: datetime) -> datetime:
        minute = now.minute - now.minute % _THROTTLE_WINDOW_MINUTES
        return now.replace(minute=minute, second=0, microsecond=0)

    async def register(self, email: str, password: str, display_name: str) -> tuple[UUID, UUID]:
        if not self._settings.allow_registration:
            raise AuthError("registration_disabled", 403)
        normalized = _normalize_email(email)
        name = display_name.strip()
        if not 1 <= len(name) <= _DISPLAY_NAME_MAX:
            raise AuthError("invalid_display_name", 422)
        try:
            validate_password(password)
        except WeakPassword:
            raise AuthError("weak_password", 422) from None
        password_hash = await asyncio.to_thread(hash_password, password)

        try:
            async with self._db.system_tx() as conn:
                user = await users.insert_user(conn, normalized, password_hash, name)
                tenant = await tenants.insert_tenant(conn, f"{name}'s workspace")
                await tenants.add_membership(conn, tenant["id"], user["id"], "owner")
                await audit.record(
                    conn, action="register", tenant_id=tenant["id"], user_id=user["id"]
                )
        except UniqueViolation:
            raise AuthError("email_taken", 409) from None
        return user["id"], tenant["id"]

    async def login(self, email: str, password: str, *, ip: str) -> LoginResult:
        normalized = email.strip().lower()
        if len(normalized) > _EMAIL_MAX:
            # No account can have this email; skip the throttle table, whose index has a row limit.
            raise AuthError("invalid_credentials", 401)
        now = self._clock()
        window = self._window_start(now)
        email_key = f"email:{normalized}"
        ip_key = f"ip:{ip}"

        # (a) Short transaction: look the user up and reserve the throttle slots atomically.
        async with self._db.system_tx() as conn:
            user = await users.get_login_record(conn, normalized)
            reserved = await self._reserve_attempt(conn, email_key, ip_key, window)
            if not reserved:
                await audit.record(
                    conn,
                    action="login_throttled",
                    tenant_id=None,
                    user_id=user["id"] if user else None,
                    details={"ip": ip},
                )

        # Errors are raised outside the transactions so the reservation and audit rows commit.
        if not reserved:
            raise AuthError("too_many_attempts", 429)

        # (b) No transaction or pooled connection is held while scrypt runs.
        if user is None:
            await asyncio.to_thread(verify_dummy, password)
            valid = False
        else:
            valid = await asyncio.to_thread(verify_password, password, user["password_hash"])

        # (c) Short transaction: write the session and audit row, or the failure audit row.
        async with self._db.system_tx() as conn:
            result = None
            if valid and user["disabled_at"] is None:
                result = await self._open_session(conn, user, ip, now, window, email_key, ip_key)
            if result is None:
                await audit.record(
                    conn,
                    action="login_failed",
                    tenant_id=None,
                    user_id=user["id"] if user else None,
                    details={"ip": ip},
                )
        if result is None:
            raise AuthError("invalid_credentials", 401)
        return result

    async def _reserve_attempt(
        self, conn: Any, email_key: str, ip_key: str, window: datetime
    ) -> bool:
        """Count this attempt against both keys, or against neither when either is full.

        The ip key goes first: attacker-chosen emails then cannot create email rows once an
        ip is at its limit.
        """
        if not await throttle.reserve(conn, ip_key, window, _IP_FAILURE_LIMIT):
            return False
        if not await throttle.reserve(conn, email_key, window, _EMAIL_FAILURE_LIMIT):
            await throttle.release(conn, ip_key, window)
            return False
        return True

    async def _open_session(
        self,
        conn: Any,
        user: dict[str, Any],
        ip: str,
        now: datetime,
        window: datetime,
        email_key: str,
        ip_key: str,
    ) -> LoginResult | None:
        memberships = await tenants.list_memberships(conn, user["id"])
        if not memberships:
            return None

        # The reserved attempt was optimistic: a success clears the email key and
        # gives back the ip slot, so the ip key counts failures only.
        await throttle.clear(conn, email_key, window)
        await throttle.release(conn, ip_key, window)
        membership = memberships[0]
        session_token = new_session_token()
        csrf_token = new_csrf_token()
        session = await sessions.insert_session(
            conn,
            token_hash(session_token),
            token_hash(csrf_token),
            user["id"],
            membership["tenant_id"],
            now + timedelta(days=self._settings.session_ttl_days),
        )
        await audit.record(
            conn,
            action="login",
            tenant_id=membership["tenant_id"],
            user_id=user["id"],
            details={"ip": ip},
        )
        principal = Principal(
            user_id=user["id"],
            email=user["email"],
            display_name=user["display_name"],
            tenant_id=membership["tenant_id"],
            role=membership["role"],
            auth="session",
            session_id=session["id"],
            csrf_hash=token_hash(csrf_token),
        )
        return LoginResult(session_token, csrf_token, principal)

    async def logout(self, session_token: str) -> None:
        async with self._db.system_tx() as conn:
            await sessions.revoke_session(conn, token_hash(session_token))

    async def resolve_session(self, session_token: str) -> Principal | None:
        async with self._db.system_tx() as conn:
            session = await sessions.get_active_session(
                conn, token_hash(session_token), self._clock()
            )
            if session is None:
                return None
            membership = await tenants.get_membership(
                conn, session["tenant_id"], session["user_id"]
            )
        if membership is None:
            return None
        return Principal(
            user_id=session["user_id"],
            email=session["email"],
            display_name=session["display_name"],
            tenant_id=session["tenant_id"],
            role=membership["role"],
            auth="session",
            session_id=session["id"],
            csrf_hash=bytes(session["csrf_hash"]),
        )

    async def resolve_api_key(self, token: str) -> Principal | None:
        prefix = parse_api_key(token)
        if prefix is None:
            return None
        async with self._db.system_tx() as conn:
            key = await api_keys.get_active_by_hash(conn, token_hash(token))
            if key is None or key["prefix"] != prefix:
                return None
            membership = await tenants.get_membership(conn, key["tenant_id"], key["user_id"])
            if membership is None:
                return None
            await api_keys.touch(conn, key["id"])
        return Principal(
            user_id=key["user_id"],
            email=key["email"],
            display_name=key["display_name"],
            tenant_id=key["tenant_id"],
            role=membership["role"],
            auth="api_key",
            api_key_id=key["id"],
        )

    async def switch_tenant(self, principal: Principal, tenant_id: UUID) -> Principal:
        if principal.auth == "api_key" or principal.session_id is None:
            raise AuthError("api_key_tenant_fixed", 403)
        async with self._db.system_tx() as conn:
            membership = await tenants.get_membership(conn, tenant_id, principal.user_id)
            if membership is None:
                raise AuthError("not_a_member", 404)
            await sessions.set_session_tenant(conn, principal.session_id, tenant_id)
        return Principal(
            user_id=principal.user_id,
            email=principal.email,
            display_name=principal.display_name,
            tenant_id=tenant_id,
            role=membership["role"],
            auth="session",
            session_id=principal.session_id,
            csrf_hash=principal.csrf_hash,
        )

    async def create_api_key(self, principal: Principal, name: str) -> tuple[dict, str]:
        _require_session(principal)
        token, prefix = new_api_key()
        async with self._db.system_tx() as conn:
            info = await api_keys.insert_api_key(
                conn, principal.tenant_id, principal.user_id, name, prefix, token_hash(token)
            )
            await audit.record(
                conn,
                action="api_key_created",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                target_type="api_key",
                target_id=str(info["id"]),
            )
        return info, token

    async def list_api_keys(self, principal: Principal) -> list[dict]:
        async with self._db.system_tx() as conn:
            return await api_keys.list_for_user(conn, principal.tenant_id, principal.user_id)

    async def revoke_api_key(self, principal: Principal, key_id: UUID) -> None:
        _require_session(principal)
        async with self._db.system_tx() as conn:
            key = await api_keys.get(conn, key_id, principal.tenant_id)
            may_revoke_any = principal.role in _KEY_ADMIN_ROLES
            if key is None or not (may_revoke_any or key["user_id"] == principal.user_id):
                raise AuthError("api_key_not_found", 404)
            if await api_keys.revoke(conn, key_id, principal.tenant_id):
                await audit.record(
                    conn,
                    action="api_key_revoked",
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    target_type="api_key",
                    target_id=str(key_id),
                )
