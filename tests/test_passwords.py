import base64
import hashlib
import types

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
MAXMEM = 64 * 1024 * 1024
# Production defaults, restated so the tests do not import private names.
_N, _R, _P = 2**14, 8, 1
# Salt whose standard base64 contains both '+' and '/', so url-safe swaps are detectable.
SALT = b"\xfb\xff\xff" + b"0123456789abc"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode().rstrip("=")


def _stored(password: str, n: int, r: int, p: int, salt: bytes, dklen: int) -> str:
    """Build a stored hash independently of hash_password, with explicit parameters."""
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=MAXMEM
    )
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(digest)}"


@pytest.fixture(scope="module")
def good_hash() -> str:
    return hash_password(PW)


@pytest.fixture(scope="module")
def fast_hash() -> str:
    """A cheap hash that verifies, used as the base for every parser mutation."""
    return _stored(PW, n=2**10, r=8, p=1, salt=SALT, dklen=32)


@pytest.fixture
def scrypt_calls(monkeypatch):
    """Spy on the scrypt call made by passwords.py only (test helpers keep the real one).

    Real scrypt runs only when n*r*p is within the production bound. Anything larger
    returns zeros immediately, so a missing guard fails fast instead of burning CPU.
    """
    calls = []
    real = hashlib.scrypt
    cheap_enough = _N * _R * _P * 16

    def spy(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["n"] * kwargs["r"] * kwargs["p"] > cheap_enough:
            return bytes(32)
        return real(*args, **kwargs)

    monkeypatch.setattr(passwords, "hashlib", types.SimpleNamespace(scrypt=spy))
    return calls


def _with_fields(stored: str, **fields: str) -> str:
    """Replace named fields of a stored hash (scheme, n, r, p, salt, digest)."""
    names = ["scheme", "n", "r", "p", "salt", "digest"]
    parts = stored.split("$")
    for name, value in fields.items():
        parts[names.index(name)] = value
    return "$".join(parts)


def _pad(text: str) -> str:
    return text + "=" * (-len(text) % 4)


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


def test_fast_hash_fixture_is_valid_and_exercises_url_safe_alphabet(fast_hash):
    assert "+" in fast_hash.split("$")[4]
    assert "/" in fast_hash.split("$")[4]
    assert len(SALT) == 16


def test_default_parameters_call_scrypt_once(good_hash, scrypt_calls):
    assert verify_password(PW, good_hash) is True
    assert len(scrypt_calls) == 1
    assert (scrypt_calls[0]["n"], scrypt_calls[0]["r"], scrypt_calls[0]["p"]) == (2**14, 8, 1)


def test_worst_case_cpu_hash_never_reaches_scrypt(fast_hash, scrypt_calls):
    """n*r*p here is 262,140 times the default; memory alone would have allowed it."""
    salt, digest = fast_hash.split("$")[4:]
    hostile = f"scrypt$131072$2$131070${salt}${digest}"
    assert verify_password(PW, fast_hash) is True
    assert len(scrypt_calls) == 1
    scrypt_calls.clear()
    assert verify_password(PW, hostile) is False
    assert scrypt_calls == []


def test_work_bound_is_inclusive(fast_hash, monkeypatch):
    """n*r*p == 16x the default is accepted; one more unit of p is not."""
    calls = []

    def fake_scrypt(*args, **kwargs):
        calls.append(kwargs)
        return base64.b64decode(_pad(fast_hash.split("$")[5]))

    monkeypatch.setattr(passwords.hashlib, "scrypt", fake_scrypt)
    at_limit = _with_fields(fast_hash, n=str(2**14), r="8", p="16")
    assert verify_password(PW, at_limit) is True
    assert len(calls) == 1
    calls.clear()
    over = _with_fields(fast_hash, n=str(2**14), r="8", p="17")
    assert verify_password(PW, over) is False
    assert calls == []


def _mutations(fast_hash: str) -> dict[str, str]:
    salt, digest = fast_hash.split("$")[4:]
    salt15 = _stored(PW, 2**10, 8, 1, SALT[:15], 32)
    salt17 = _stored(PW, 2**10, 8, 1, SALT + b"!", 32)
    digest24 = _stored(PW, 2**10, 8, 1, SALT, 24)
    digest33 = _stored(PW, 2**10, 8, 1, SALT, 33)
    return {
        "padded salt": _with_fields(fast_hash, salt=_pad(salt)),
        "padded digest": _with_fields(fast_hash, digest=_pad(digest)),
        "appended = on salt": _with_fields(fast_hash, salt=_pad(salt) + "="),
        "appended = on digest": _with_fields(fast_hash, digest=_pad(digest) + "="),
        "single = appended to digest": _with_fields(fast_hash, digest=digest + "="),
        "url-safe salt": _with_fields(fast_hash, salt=salt.replace("+", "-").replace("/", "_")),
        "url-safe digest": _with_fields(
            fast_hash, digest=digest.replace("+", "-").replace("/", "_")
        ),
        "leading space on whole string": " " + fast_hash,
        "trailing newline on whole string": fast_hash + "\n",
        "space inside salt": _with_fields(fast_hash, salt=salt[:4] + " " + salt[4:]),
        "newline inside digest": _with_fields(fast_hash, digest=digest[:4] + "\n" + digest[4:]),
        "space around n": _with_fields(fast_hash, n=" 1024"),
        "four fields": "$".join(fast_hash.split("$")[:5]),
        "seven fields": fast_hash + "$extra",
        "empty string": "",
        "wrong literal": _with_fields(fast_hash, scheme="bcrypt"),
        "wrong literal case": _with_fields(fast_hash, scheme="SCRYPT"),
        "signed n": _with_fields(fast_hash, n="+1024"),
        "signed r": _with_fields(fast_hash, r="+8"),
        "signed p": _with_fields(fast_hash, p="+1"),
        "negative p": _with_fields(fast_hash, p="-1"),
        "non-ascii digits in n": _with_fields(fast_hash, n="\uff11\uff10\uff12\uff14"),
        "non-ascii digits in r": _with_fields(fast_hash, r="\u0668"),
        "non-ascii digits in p": _with_fields(fast_hash, p="\u0661"),
        "non-digit p": _with_fields(fast_hash, p="x"),
        "salt 15 bytes": salt15,
        "salt 17 bytes": salt17,
        "empty salt": _with_fields(fast_hash, salt=""),
        "digest 24 bytes": digest24,
        "digest 33 bytes": digest33,
        "empty digest": _with_fields(fast_hash, digest=""),
        "non-ascii in salt": _with_fields(fast_hash, salt=salt[:-1] + "\u00e9"),
        "n not a power of two": _with_fields(fast_hash, n="1000"),
        "n one above a power of two": _with_fields(fast_hash, n="1025"),
        "n is 3": _with_fields(fast_hash, n="3"),
        "n is 0": _with_fields(fast_hash, n="0"),
        "n is 1": _with_fields(fast_hash, n="1"),
        "n above 2**20": _with_fields(fast_hash, n=str(2**21), r="1"),
        "r is 0": _with_fields(fast_hash, r="0"),
        "p is 0": _with_fields(fast_hash, p="0"),
        "memory above maxmem": _with_fields(fast_hash, n=str(2**14), r="64"),
        "work just above the bound": _with_fields(fast_hash, n=str(2**14), r="8", p="17"),
        "work far above the bound": _with_fields(fast_hash, n="131072", r="2", p="131070"),
    }


_MUTATION_NAMES = list(_mutations("scrypt$1024$8$1$AAAAAAAAAAAAAAAAAAAAAA$" + "A" * 43))


@pytest.mark.parametrize("name", _MUTATION_NAMES)
def test_parser_rejects_mutation_before_scrypt(name, fast_hash, scrypt_calls):
    # The unmutated hash verifies, so the only difference is the mutation.
    assert verify_password(PW, fast_hash) is True
    assert len(scrypt_calls) == 1
    scrypt_calls.clear()

    mutated = _mutations(fast_hash)[name]

    assert mutated != fast_hash
    assert verify_password(PW, mutated) is False
    assert scrypt_calls == []  # rejected by the parser, not by a digest mismatch


def test_mutation_table_is_stable(fast_hash):
    assert list(_mutations(fast_hash)) == _MUTATION_NAMES


def test_verify_uses_parameters_recorded_in_stored_hash(scrypt_calls):
    salt = b"0123456789abcdef"
    stored = _stored(PW, n=2**10, r=4, p=2, salt=salt, dklen=32)
    scrypt_calls.clear()
    assert verify_password(PW, stored) is True
    assert verify_password(PW + "x", stored) is False
    assert [(c["n"], c["r"], c["p"]) for c in scrypt_calls] == [(2**10, 4, 2)] * 2


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
