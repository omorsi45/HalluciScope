import httpx

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
If the context doesn't contain enough information to answer, say so. Do not make up information."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above."""


async def generate_answer(
    question: str,
    context_chunks: list[str],
    base_url: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Generate an answer using Ollama given context chunks and a question."""
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
