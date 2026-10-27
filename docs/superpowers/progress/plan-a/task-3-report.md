# Task 3 report: decomposition uses its own system prompt

Status: DONE_WITH_CONCERNS (minor, see Concerns)

## Built
- `backend/core/generator.py`: new `chat_completion(messages, base_url, model, temperature=0.0, client=None) -> str` holding the payload build, shared-client vs temporary-client paths, and the ConnectError / TimeoutException / HTTPStatusError to `OllamaError` mapping (messages unchanged byte-for-byte). `generate_answer` keeps its signature, builds the same two RAG messages, and returns `await chat_completion(messages, base_url, model, temperature, client)`.
- `backend/core/decomposer.py`: imports `chat_completion`; `decompose_claims` sends `[system=DECOMPOSE_SYSTEM, user=DECOMPOSE_USER.format(answer=answer)]` at temperature 0.0.
- `tests/test_decomposer.py`: `test_decompose_claims` now patches `backend.core.decomposer.chat_completion`; new `test_decompose_uses_decompose_system_prompt`.
- `consistency.py` untouched; it still imports `generate_answer` and `decompose_claims` from the same modules.

## RED
`.venv/bin/python -m pytest tests/test_decomposer.py -q`
- 2 failed, 4 passed. Both failures: `AttributeError: module 'backend.core.decomposer' does not have the attribute 'chat_completion'` (feature missing, as expected).

## GREEN
- `.venv/bin/python -m pytest -q`: 53 passed, 2 warnings (52 before + 1 new test). Includes tests/test_generator.py and tests/test_consistency.py.
- Warnings: RuntimeWarning "coroutine AsyncMockMixin._execute_mock_call was never awaited" at `response.raise_for_status()` in tests/test_generator.py (test_generate_answer, test_generate_answer_with_temperature). Same test mocks and same line of logic as before; not introduced by this task.

## Lint
- `ruff check` on the three changed files: 1 error, I001 import ordering in tests/test_decomposer.py. The import block layout (pytest, then unittest.mock, then backend) is pre-existing; I only extended the names on the backend import line. Not fixed (out of scope). generator.py and the pre-change generator pass clean.
- Whole repo `ruff check .`: 46 errors, pre-existing, not from this task.

## Commit
- `fix: use the decomposition system prompt for claim extraction` (generator.py, decomposer.py, test_decomposer.py staged by name).

## Decisions / concerns
- The timeout OllamaError message contains an em dash ("loading — try again"). It is pre-existing and the dispatch required messages byte-for-byte, so it was kept. The rules file bans em dashes; if the project wants it removed, tests/test_generator.py assertions and the message need to change together.
- No direct unit tests were added for `chat_completion` itself; its behavior is covered through `generate_answer` in tests/test_generator.py, per the brief.
