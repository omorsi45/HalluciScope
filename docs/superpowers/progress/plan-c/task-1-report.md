# Plan C Task 1 report: Ollama chat client with typed failures + ScriptedLLM

Status: DONE_WITH_CONCERNS (concerns are minor decisions, listed below)

Commit: `11d0db5 feat: add Ollama chat client with typed failures` on `feature/evidencegraph` (not pushed).
Staged by name: `backend/llm/__init__.py`, `backend/llm/client.py`, `tests/test_llm_client.py`, `tests/fakes.py`, `backend/config.py`. Working tree clean after commit. The gitleaks pre-commit hook ran and passed (no leaks).

## What was built

- `backend/llm/client.py`
  - `LLMFailure(Exception)` with class attribute `code`; subclasses `LLMUnavailable` (`llm_unavailable`), `LLMTimeout` (`llm_timeout`), `LLMError` (`llm_error`). Base default is `llm_error`.
  - `LLMClient` Protocol: `model_id: str` and `async chat(messages, *, temperature=0.0, json_schema=None, timeout_s=None) -> str`.
  - `OllamaClient(http, base_url, model, *, timeout_s=120.0)`. `chat` posts `{"model", "messages", "stream": false, "options": {"temperature": t}}` to `{base_url}/api/chat`, adds `"format": json_schema` only when given, passes `timeout=` per request (per-call `timeout_s` overrides the client default), returns `body["message"]["content"]`.
  - Mapping: `httpx.ConnectError` -> `LLMUnavailable`; `httpx.TimeoutException` -> `LLMTimeout`; any other `httpx.HTTPError` -> `LLMError` (see decisions); non-2xx -> `LLMError("Ollama returned HTTP <status> for model '<model>'.")` with no body; unparseable JSON, missing `message.content`, or non-str content -> `LLMError("... malformed response ...")` with no body.
  - Kept the actionable legacy guidance: "Make sure Ollama is running (`ollama serve`)." and "The model may still be loading, try again in a moment." The legacy em dash was replaced by a comma.
  - No logging in the client. No prompt, message content, or response body in any exception message.
- `backend/llm/__init__.py`: one-line module docstring.
- `backend/config.py`: `llm_model: str = "llama3.2:3b"`, `llm_timeout_s: float = 120.0`. Legacy `ollama_model` untouched.
- `tests/fakes.py`: `ScriptedLLM(model_id="fake-llm")` with `responses: list[str | Exception]` (consumed in order, exceptions are raised), `default` callable `(messages) -> str` used when `responses` is empty (initial value returns `""`), and `calls: list[dict]` recording `messages` (copied), `temperature`, `json_schema` for every call including ones that raise. Module docstring updated to mention the LLM fake.
- `tests/test_llm_client.py`: 19 test cases (13 test functions, some parametrized), all with `httpx.MockTransport`, no network.

## RED evidence

Tests written first, before any production code existed:

```
wsl ... wsl.sh timeout 300 python -m pytest tests/test_llm_client.py -q
tests/test_llm_client.py:6: in <module>
    from backend.llm.client import LLMError, LLMFailure, LLMTimeout, LLMUnavailable, OllamaClient
E   ModuleNotFoundError: No module named 'backend.llm'
ERROR tests/test_llm_client.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.75s          (exit=2)
```

Failure reason is the missing feature (the module under test does not exist), not a typo. No network or slow path was reachable in the RED run.

## GREEN evidence

```
wsl ... wsl.sh timeout 300 python -m pytest tests/test_llm_client.py -v
collected 19 items
test_chat_payload_and_content PASSED
test_default_temperature_is_zero PASSED
test_json_schema_sent_as_format PASSED
test_connect_error_unavailable PASSED
test_timeout[ReadTimeout] PASSED
test_timeout[ConnectTimeout] PASSED
test_timeout_uses_client_default_and_per_call_override PASSED
test_http_error_hides_body[404] PASSED
test_http_error_hides_body[500] PASSED
test_http_error_hides_body[503] PASSED
test_malformed_body_llm_error[not-json] PASSED
test_malformed_body_llm_error[json-list] PASSED
test_malformed_body_llm_error[no-message] PASSED
test_malformed_body_llm_error[no-content] PASSED
test_malformed_body_llm_error[message-not-object] PASSED
test_malformed_body_llm_error[content-not-str] PASSED
test_malformed_body_llm_error[null] PASSED
test_other_transport_failure_is_llm_error PASSED
test_scripted_llm_records_calls PASSED
19 passed in 0.33s   (exit=0)
```

Brief test names covered: `test_chat_payload_and_content`, `test_json_schema_sent_as_format`, `test_connect_error_unavailable`, `test_timeout`, `test_http_error_hides_body`, `test_malformed_body_llm_error`, `test_scripted_llm_records_calls`. Added: `test_default_temperature_is_zero`, `test_timeout_uses_client_default_and_per_call_override`, `test_other_transport_failure_is_llm_error`.

Test-quality notes: payload test uses exact dict equality, so a stray `format` key when no schema is given fails it. The timeout override test reads `request.extensions["timeout"]` from the MockTransport request, so it proves the configured value reaches httpx (45.0 default vs 7.5 override), not just that the parameter exists. Hide-the-body tests embed `SECRET-PROMPT` in the raw body of every error and malformed case and assert it is absent from `str()` and `repr()` of the exception.

## Full suite and lint

- Full suite: `wsl.sh timeout 900 python -m pytest -q` -> `1035 passed, 2 deselected, 2 warnings in 144.35s`, exit 0. The 2 warnings are pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` from legacy `tests/test_generator.py` (not touched). The 2 deselected are tests marked `models` (real ML models), excluded by the existing `addopts = "-m 'not models'"` in `pyproject.toml` (not changed by this task). No failures. Postgres container `eg-pg` was up for the DB-backed tests.
- Lint: `wsl.sh timeout 120 ruff check .` -> `All checks passed!`, exit 0.
- Em dash check (U+2014) on all 5 touched files: none.

## Decisions

1. Other `httpx.HTTPError` subclasses (dropped connection, `RemoteProtocolError`, `ReadError`) map to `LLMError` with the exception class name only. The brief lists three mappings; without this, a raw httpx exception would escape the typed-failure contract and later tasks would record `internal_error` instead of `llm_error`. Covered by `test_other_transport_failure_is_llm_error`. Easy to drop if the reviewer considers it out of scope.
2. `ScriptedLLM.default` starts as a callable returning `""`, since the brief only says it is the fallback when `responses` is empty. `responses` and `default` are plain attributes that tests set after construction; the constructor signature is exactly `ScriptedLLM(model_id="fake-llm")`.
3. `calls` records exactly the three keys named in the brief. `timeout_s` is accepted by `ScriptedLLM.chat` (to satisfy the protocol) but is not recorded.
4. `ScriptedLLM.chat` copies each message dict into `calls`, so later mutation of the caller's list does not rewrite the record.
5. Non-2xx detection uses `response.is_success` rather than `raise_for_status()`, so no `HTTPStatusError` (whose text includes the URL) is created; 1xx and 3xx also become `LLMError`.
6. No test added for the two new `Settings` fields: asserting default values would be a change detector, and the first consumers (Tasks 2, 3, 18) will exercise them. `tests/test_config.py` was left untouched.
7. `tests/fakes.py` was `w/lf` before my edit; the edit tool rewrote it as CRLF, so I converted it back to LF (`git ls-files --eol` shows `i/lf w/lf`). `backend/config.py` stayed CRLF as it was. The index content is LF for all files either way.

## Concerns

- Timeout semantics: httpx applies `timeout_s` to connect, read, write, and pool phases separately, not as one wall-clock deadline. For a non-streaming Ollama call the read phase is the whole generation wait, so this behaves as the 120 s limit in practice, but a pathological server trickling bytes could exceed it. Not addressed since the brief specifies a plain timeout.
- `base_url` is used verbatim as `{base_url}/api/chat`; a trailing slash in config would produce `//api/chat`. The default setting has no trailing slash and the brief specifies the verbatim form, so I did not normalize it.
