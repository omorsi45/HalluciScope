"""Chat client for the local LLM (Ollama) with typed failures.

Callers depend on the `LLMClient` protocol so tests can swap in a scripted fake. Failure
messages never include response bodies or prompts: an HTTP error body may echo the prompt.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx


class LLMFailure(Exception):
    """Base class for LLM failures. `code` is the analysis error code recorded for the failure."""

    code: str = "llm_error"


class LLMUnavailable(LLMFailure):
    """The LLM server could not be reached."""

    code = "llm_unavailable"


class LLMTimeout(LLMFailure):
    """The LLM server did not answer within the time limit."""

    code = "llm_timeout"


class LLMError(LLMFailure):
    """The LLM server answered with an error or an unusable response."""

    code = "llm_error"


class LLMClient(Protocol):
    model_id: str

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        json_schema: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Return the assistant reply for `messages`.

        `json_schema` constrains the reply to that JSON schema. `timeout_s` overrides the
        client default for this call. Raises an `LLMFailure` subclass on any failure.
        """
        ...


class OllamaClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 120.0,
    ) -> None:
        self._http = http
        self._base_url = base_url
        self._timeout_s = timeout_s
        self.model_id = model

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        json_schema: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        timeout = self._timeout_s if timeout_s is None else timeout_s

        try:
            response = await self._http.post(
                f"{self._base_url}/api/chat", json=payload, timeout=timeout
            )
        except httpx.ConnectError as exc:
            raise LLMUnavailable(
                f"Could not connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeout(
                f"Request to Ollama timed out (model={self.model_id!r}). "
                "The model may still be loading, try again in a moment."
            ) from exc
        except httpx.HTTPError as exc:
            # Any other transport failure (dropped connection, protocol error). The message
            # carries the exception class only, never its text.
            raise LLMError(
                f"Request to Ollama failed ({type(exc).__name__}) for model {self.model_id!r}."
            ) from exc

        if not response.is_success:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code} for model {self.model_id!r}."
            )

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(
                f"Ollama returned a malformed response for model {self.model_id!r}."
            ) from exc
        if not isinstance(content, str):
            raise LLMError(
                f"Ollama returned a malformed response for model {self.model_id!r}."
            )
        return content
