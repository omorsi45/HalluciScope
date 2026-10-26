"""Session, CSRF and API key token primitives."""

import hashlib
import re
import secrets

_API_KEY_RE = re.compile(r"egk_([0-9a-f]{8})_[A-Za-z0-9_-]{43}")


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def new_api_key() -> tuple[str, str]:
    prefix = secrets.token_hex(4)
    return f"egk_{prefix}_{secrets.token_urlsafe(32)}", prefix


def parse_api_key(token: str) -> str | None:
    match = _API_KEY_RE.fullmatch(token)
    return match.group(1) if match else None
