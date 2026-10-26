import base64
import hashlib

import pytest

from backend.auth import passwords
from backend.auth.passwords import (
    WeakPassword,
    hash_password,
    validate_password,
    verify_dummy,
    verify_password,
)

PW = "correct horse battery"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode().rstrip("=")


def _stored(password: str, n: int, r: int, p: int, salt: bytes, dklen: int) -> str:
    """Build a stored hash independently of hash_password, with explicit parameters."""
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=64 * 1024 * 1024
    )
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(digest)}"


@pytest.fixture(scope="module")
def good_hash() -> str:
    return hash_password(PW)


def test_hash_format_and_roundtrip(good_hash):
    assert good_hash.startswith("scrypt$16384$8$1$")
    assert verify_password(PW, good_hash) is True
    assert verify_password(PW + "x", good_hash) is False


def test_hash_fields_are_unpadded_standard_base64(good_hash):
    parts = good_hash.split("$")
    assert len(parts) == 6
    assert "=" not in good_hash
    salt = base64.b64decode(parts[4] + "=" * (-len(parts[4]) % 4), validate=True)
    digest = base64.b64decode(parts[5] + "=" * (-len(parts[5]) % 4), validate=True)
    assert len(salt) == 16
    assert len(digest) == 32


def test_hashes_are_salted():
    assert hash_password(PW) != hash_password(PW)


def test_malformed_hash_is_false():
    assert verify_password("x" * 12, "garbage") is False


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "scrypt$16384$8$1$c2FsdA",  # 5 fields
        "scrypt$16384$8$1$c2FsdA$aGFzaA$extra",  # 7 fields
        "bcrypt$16384$8$1$c2FsdA$aGFzaA",  # wrong scheme
        "scrypt$abc$8$1$c2FsdA$aGFzaA",  # non-integer n
        "scrypt$16384$8$x$c2FsdA$aGFzaA",  # non-integer p
        "scrypt$+16384$8$1$c2FsdA$aGFzaA",  # sign is not a plain integer
        "scrypt$16384$8$1$c2FsdA==$aGFzaA",  # padded base64 is not our encoding
        "scrypt$16384$8$1$c2Fs!A$aGFzaA",  # invalid base64 character
        "scrypt$16384$8$1$c$aGFzaA",  # impossible base64 length
        "scrypt$16384$8$1$$aGFzaA",  # empty salt
        "scrypt$16384$8$1$c2FsdA$",  # empty hash
        "scrypt$16384$8$1$c2Fsd\u00e9$aGFzaA",  # non-ascii
    ],
)
def test_strictly_malformed_stored_values_are_false(stored):
    assert verify_password(PW, stored) is False


def test_verify_uses_parameters_recorded_in_stored_hash():
    salt = b"0123456789abcdef"
    stored = _stored(PW, n=2**10, r=4, p=2, salt=salt, dklen=24)
    assert verify_password(PW, stored) is True
    assert verify_password(PW + "x", stored) is False


@pytest.mark.parametrize(
    "n,r,p",
    [
        (2**14 + 1, 8, 1),  # not a power of two
        (3, 8, 1),  # not a power of two
        (0, 8, 1),
        (2**21, 8, 1),  # above 2**20
        (2**14, 64, 1),  # r*p memory exceeds maxmem
        (2**14, 8, 0),  # p must be positive
        (2**14, 0, 1),  # r must be positive
    ],
)
def test_unsafe_scrypt_parameters_are_false(n, r, p):
    stored = f"scrypt${n}${r}${p}${_b64(b'0123456789abcdef')}${_b64(b'h' * 32)}"
    assert verify_password(PW, stored) is False


def test_password_length_bounds():
    with pytest.raises(WeakPassword):
        validate_password("x" * 11)
    validate_password("x" * 12)
    validate_password("x" * 256)
    with pytest.raises(WeakPassword):
        validate_password("x" * 257)


def test_weak_password_is_value_error_without_password_in_message():
    secret = "short-pw1"
    with pytest.raises(WeakPassword) as exc:
        validate_password(secret)
    assert isinstance(exc.value, ValueError)
    assert secret not in str(exc.value)


def test_verify_dummy_runs_scrypt_and_returns_none(monkeypatch):
    verify_dummy(PW)  # warm up the lazily computed dummy hash
    calls = []
    real = hashlib.scrypt

    def counting(*args, **kwargs):
        calls.append(kwargs.get("n"))
        return real(*args, **kwargs)

    monkeypatch.setattr(passwords.hashlib, "scrypt", counting)
    assert verify_dummy(PW) is None
    assert calls == [2**14]
