import hashlib
import re

from backend.auth.tokens import (
    new_api_key,
    new_csrf_token,
    new_session_token,
    parse_api_key,
    token_hash,
)

API_KEY_RE = re.compile(r"^egk_[0-9a-f]{8}_[A-Za-z0-9_-]{43}$")


def test_token_hash_is_sha256():
    assert token_hash("abc") == hashlib.sha256(b"abc").digest()
    assert len(token_hash("abc")) == 32


def test_api_key_shape_and_parse():
    token, prefix = new_api_key()
    assert API_KEY_RE.match(token)
    assert re.fullmatch(r"[0-9a-f]{8}", prefix)
    assert token.startswith(f"egk_{prefix}_")
    assert parse_api_key(token) == prefix


def test_parse_api_key_rejects_malformed():
    assert parse_api_key("egk_zz_short") is None
    assert parse_api_key("Bearer x") is None
    assert parse_api_key("") is None


def test_parse_api_key_rejects_trailing_newline_and_extra_chars():
    token, _ = new_api_key()
    assert parse_api_key(token + "\n") is None
    assert parse_api_key(token + "x") is None
    assert parse_api_key(" " + token) is None


def test_session_tokens_unique():
    assert len({new_session_token() for _ in range(100)}) == 100


def test_session_and_csrf_tokens_are_43_urlsafe_chars():
    for token in (new_session_token(), new_csrf_token()):
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
