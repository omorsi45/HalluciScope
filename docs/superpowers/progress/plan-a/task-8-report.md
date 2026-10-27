# Task 8 report: password hashing and token primitives

Status: DONE. Commit: f979bdd `feat: add password hashing and token primitives`.

## Built
- `backend/auth/passwords.py`: `PASSWORD_MIN/MAX`, `WeakPassword(ValueError)`, `validate_password` (message never contains the password), `hash_password` (scrypt n=2**14 r=8 p=1 dklen=32, 16-byte salt, `scrypt$n$r$p$salt$hash`, unpadded standard base64, maxmem 64 MiB), `verify_password`, `verify_dummy` (lazily cached module-level hash via `functools.cache`).
- `backend/auth/tokens.py`: `new_session_token`, `new_csrf_token`, `token_hash` (sha256 digest), `new_api_key`, `parse_api_key` (`fullmatch`, so trailing newline is rejected).
- `backend/auth/__init__.py` (empty).
- Tests: `tests/test_passwords.py`, `tests/test_tokens.py`.

## verify_password hardening
- Strict parse: exactly 6 `$` fields, literal `scrypt`, n/r/p must be 1 to 10 ASCII digits (no sign, whitespace, unicode digits), base64 re-padded then decoded with `validate=True` (padded input, bad chars, impossible lengths rejected), non-empty salt, hash length 1..64.
- Uses n, r, p and dklen (= decoded hash length) from the stored string. Rejects n < 2, non power of two, n > 2**20, r < 1, p < 1, and 128*r*(n+p+2) > maxmem (r*p / n memory estimate).
- Any parse error, `ValueError` from scrypt, or unencodable password returns False. Comparison via `hmac.compare_digest`.

## RED
`.venv/bin/python -m pytest tests/test_passwords.py tests/test_tokens.py -q`:
`ModuleNotFoundError: No module named 'backend.auth'`, 2 collection errors (feature missing, not a typo).

## GREEN
Same command: `34 passed in 0.37s` (slowest test 0.11s).
Full suite: `107 passed, 2 warnings` (the 2 warnings are the pre-existing un-awaited AsyncMock in `tests/test_generator.py`, unrelated).
Lint: `.venv/bin/ruff check .` -> All checks passed. Commit hook gitleaks: no leaks.

## Decisions
- Added tests beyond the brief for the stated security details: 13 strict-parse cases, honoring stored params (built independently with hashlib at n=2**10, r=4, p=2, dklen=24), unsafe-parameter rejection, WeakPassword message/type, verify_dummy performing exactly one scrypt call (spy wraps the real function), 43-char token shape, trailing-newline rejection.
- hash_password does not call validate_password (brief keeps them separate).
- Cap of 64 on stored hash length and 10-digit cap on integer fields are my choices to bound work/ints; not in the brief.

## Concerns
- Constant-time comparison is not unit-testable without mocking; the `hmac.compare_digest` use is verified by reading only.
- The n > 2**20 case is also caught by the memory estimate; both guards kept as specified.

## Fix round 1

Correction to the original report: the claim that "padded input was rejected" was wrong. `_decode` re-padded and decoded, so `c2FsdA==` and a single appended `=` were accepted (a padded variant of a valid hash verified True). The old strict-parse cases passed only because the digest mismatched, and the unsafe-parameter cases would also have returned False via hashlib's own ValueError with the guards deleted.

### Changes (backend/auth/passwords.py)
- `_decode` rejects any `=` (format is unpadded only).
- Salt must decode to exactly 16 bytes and digest to exactly 32 bytes (`_SALT_BYTES`, `_DKLEN`); removed the `_DKLEN_LIMIT` 1..64 allowance.
- New work bound before any scrypt call: `n * r * p <= _N * _R * _P * 16` (`_WORK_FACTOR = 16`). Memory guard, n power-of-two and n limit kept.

### Changes (tests/test_passwords.py)
- Kept the earlier uncommitted edits, fixed two defects: the spy delegated to real scrypt for every call (the cause of the one-hour stall), and it patched the global `hashlib.scrypt`, so the `_stored` helper polluted the call record. The spy now replaces only `passwords.hashlib` with a namespace; it runs real scrypt only when `n*r*p <= 16x default`, otherwise returns 32 zero bytes immediately.
- `fast_hash` fixture (n=2**10, salt containing `+` and `/`) is the base for every case.
- `test_parser_rejects_mutation_before_scrypt`: 43 parametrized cases. Each asserts the unmutated hash verifies (True, one spy call), then the mutation returns False with zero scrypt calls. Cases: padded salt/digest, appended `=` (salt, digest, single), url-safe salt/digest, whitespace (outer, inside salt/digest, around n), 4 and 7 fields, empty, wrong literal (2), signed/negative/non-ASCII digits (n, r, p), non-digit p, salt 15/17 bytes/empty, digest 24/33 bytes/empty, non-ASCII in salt, n not power of two (1000, 1025, 3, 0, 1), n above 2**20, r=0, p=0, memory above maxmem, work just above and far above the bound.
- `test_worst_case_cpu_hash_never_reaches_scrypt`: `scrypt$131072$2$131070$<valid salt>$<valid digest>` returns False with zero calls.
- `test_work_bound_is_inclusive`: n*r*p == 16x default accepted, p=17 rejected (fake scrypt).
- `test_default_parameters_call_scrypt_once`: one real call with (2**14, 8, 1).
- `test_verify_uses_parameters_recorded_in_stored_hash` now uses dklen=32 and the spy.

### RED
`perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_passwords.py -q -p no:cacheprovider` before the code change:
`11 failed, 45 passed in 1.02s`. Failures: test_worst_case_cpu_hash_never_reaches_scrypt, test_work_bound_is_inclusive, mutations [padded salt], [padded digest], [single = appended to digest], [salt 15 bytes], [salt 17 bytes], [digest 24 bytes], [digest 33 bytes], [work just above the bound], [work far above the bound]. For the far-above case the spy recorded `n=131072, r=2, p=131070` and returned instantly (no real scrypt).

### GREEN
Same command after the change: `56 passed in 0.97s` (slowest 0.11s).
Full suite `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q`: `135 passed, 2 warnings in 9.22s` (warnings pre-existing, unrelated).
Lint `.venv/bin/ruff check .`: All checks passed.

### Decisions
- Format has no dklen field, so digest is fixed at 32 bytes and salt at 16, per controller ruling. Hashes from other sources with different lengths are intentionally rejected.
