"""Password hashing with stdlib scrypt. Stored format: scrypt$n$r$p$salt_b64$hash_b64."""

import base64
import hashlib
import hmac
import re
import secrets
from functools import cache

PASSWORD_MIN = 12
PASSWORD_MAX = 256

_N = 2**14
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_MAXMEM = 64 * 1024 * 1024
_N_LIMIT = 2**20
_WORK_FACTOR = 16
_INT_RE = re.compile(r"[0-9]{1,10}")


class WeakPassword(ValueError):
    """Raised when a password violates the length policy."""


def validate_password(password: str) -> None:
    if len(password) < PASSWORD_MIN:
        raise WeakPassword(f"Password must be at least {PASSWORD_MIN} characters")
    if len(password) > PASSWORD_MAX:
        raise WeakPassword(f"Password must be at most {PASSWORD_MAX} characters")


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode().rstrip("=")


def _decode(text: str) -> bytes:
    # Our format is unpadded only, so any "=" is rejected rather than tolerated.
    if "=" in text:
        raise ValueError("padding is not part of the stored format")
    return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM
    )
    return f"scrypt${_N}${_R}${_P}${_encode(salt)}${_encode(digest)}"


def _parse_stored(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return None
    if not all(_INT_RE.fullmatch(field) for field in parts[1:4]):
        return None
    n, r, p = (int(field) for field in parts[1:4])
    try:
        salt = _decode(parts[4])
        expected = _decode(parts[5])
    except ValueError:
        return None
    if len(salt) != _SALT_BYTES or len(expected) != _DKLEN:
        return None
    if n < 2 or n & (n - 1) or n > _N_LIMIT or r < 1 or p < 1:
        return None
    # Bound total CPU work: memory alone allows hours-long verifications.
    if n * r * p > _N * _R * _P * _WORK_FACTOR:
        return None
    # scrypt working memory is about 128 * r * (n + p + 2) bytes.
    if 128 * r * (n + p + 2) > _MAXMEM:
        return None
    return n, r, p, salt, expected


def verify_password(password: str, stored: str) -> bool:
    parsed = _parse_stored(stored)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        actual = hashlib.scrypt(
            password.encode(),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=_MAXMEM,
        )
    except ValueError:  # includes unencodable passwords (lone surrogates)
        return False
    return hmac.compare_digest(actual, expected)


@cache
def _dummy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


def verify_dummy(password: str) -> None:
    """Spend the cost of a real verification, to equalize timing for unknown users."""
    verify_password(password, _dummy_hash())
