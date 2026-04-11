import httpx

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
If the context doesn't contain enough information to answer, say so. Do not make up information."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above."""


class OllamaError(RuntimeError):
    """Raised when the Ollama API is unreachable, times out, or returns an error."""


async def generate_answer(
    question: str,
    context_chunks: list[str],
    base_url: str,
    model: str,
    temperature: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Generate an answer using Ollama given context chunks and a question.

    If a shared client is provided it is used directly. Otherwise a
    temporary client is created for this call (preserves CLI compatibility).

    Raises:
        OllamaError: If Ollama is unreachable, times out, or returns an error
            status. The message includes actionable guidance for each case.
    """
    context = "\n\n".join(context_chunks)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        if client is not None:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

        async with httpx.AsyncClient(timeout=120.0) as _client:
            response = await _client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    except httpx.ConnectError:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        )
    except httpx.TimeoutException:
        raise OllamaError(
            f"Request to Ollama timed out (model={model!r}). "
            "The model may still be loading — try again in a moment."
        )
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.response.status_code} for model {model!r}. "
            f"Response: {exc.response.text[:200]}"
        ) from exc
