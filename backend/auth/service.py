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
_THROTTLE_WINDOW_MINUTES = 15
_EMAIL_FAILURE_LIMIT = 5
_IP_FAILURE_LIMIT = 20


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
        now = self._clock()
        window = self._window_start(now)
        email_key = f"email:{normalized}"
        ip_key = f"ip:{ip}"

        async with self._db.system_tx() as conn:
            user = await users.get_user_by_email(conn, normalized)
            user_id = user["id"] if user else None

            email_failures = await throttle.get_failures(conn, email_key, window)
            ip_failures = await throttle.get_failures(conn, ip_key, window)
            if email_failures >= _EMAIL_FAILURE_LIMIT or ip_failures >= _IP_FAILURE_LIMIT:
                await audit.record(
                    conn,
                    action="login_throttled",
                    tenant_id=None,
                    user_id=user_id,
                    details={"ip": ip},
                )
                outcome: LoginResult | AuthError = AuthError("too_many_attempts", 429)
            else:
                outcome = await self._check_credentials(
                    conn, user, password, ip, now, window, email_key, ip_key
                )

        # Raised after the transaction closes so the throttle and audit rows commit.
        if isinstance(outcome, AuthError):
            raise outcome
        return outcome

    async def _check_credentials(
        self,
        conn: Any,
        user: dict[str, Any] | None,
        password: str,
        ip: str,
        now: datetime,
        window: datetime,
        email_key: str,
        ip_key: str,
    ) -> LoginResult | AuthError:
        if user is None:
            await asyncio.to_thread(verify_dummy, password)
            valid = False
        else:
            valid = await asyncio.to_thread(verify_password, password, user["password_hash"])
        memberships = await tenants.list_memberships(conn, user["id"]) if user and valid else []
        if not valid or user["disabled_at"] is not None or not memberships:
            await throttle.record_failure(conn, email_key, window)
            await throttle.record_failure(conn, ip_key, window)
            await audit.record(
                conn,
                action="login_failed",
                tenant_id=None,
                user_id=user["id"] if user else None,
                details={"ip": ip},
            )
            return AuthError("invalid_credentials", 401)

        await throttle.clear(conn, email_key, window)
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
        token, prefix = new_api_key()
        async with self._db.system_tx() as conn:
            info = await api_keys.insert_api_key(
                conn, principal.tenant_id, principal.user_id, name, prefix, token_hash(token)
            )
        return info, token

    async def list_api_keys(self, principal: Principal) -> list[dict]:
        async with self._db.system_tx() as conn:
            return await api_keys.list_for_user(conn, principal.tenant_id, principal.user_id)

    async def revoke_api_key(self, principal: Principal, key_id: UUID) -> None:
        async with self._db.system_tx() as conn:
            key = await api_keys.get(conn, key_id, principal.tenant_id)
            if key is None or (principal.role == "member" and key["user_id"] != principal.user_id):
                raise AuthError("not_found", 404)
            await api_keys.revoke(conn, key_id, principal.tenant_id)
