"""Deterministic prompt builder for prefix cache optimization.

The prompt is structured as:
    SYSTEM (fixed) + CONTEXT (sorted chunks) + USER QUERY

Because the system prompt is always identical and context chunks are
sorted by (doc_id, chunk_idx), two requests that retrieve the same
documents will produce an identical prefix. The llm-d EPP will hash
this prefix and route both requests to the vLLM pod that already has
the KV cache populated, skipping prefill computation entirely.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions based only on the "
    "provided context. If the context does not contain enough information "
    "to answer the question, say so clearly. Do not make up information."
)


def build_prompt(context_chunks: list[dict], query: str) -> str:
    """Build a deterministic prompt from retrieved chunks and user query.

    Args:
        context_chunks: Sorted list of {"text": str, "doc_id": str, ...}
        query: The user's question.

    Returns:
        Formatted prompt string with stable prefix for KV cache reuse.
    """
    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)

    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Context:\n{context_block}\n\n"
        f"### User:\n{query}\n\n"
        f"### Assistant:\n"
    )


def build_chat_messages(context_chunks: list[dict], query: str) -> list[dict]:
    """Build OpenAI chat-format messages for /v1/chat/completions.

    Same deterministic ordering as build_prompt.
    """
    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)

    return [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Context:\n{context_block}"
            ),
        },
        {"role": "user", "content": query},
    ]
