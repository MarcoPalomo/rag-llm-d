"""Milvus retrieval for the RAG gateway.

Embeds the user query with bge-m3 and searches Milvus for the top-k
most relevant document chunks.
"""

from rag.ingestion.embedder import embed_query
from rag.ingestion.milvus_store import search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant chunks for a query.

    Returns:
        List of {"text": str, "doc_id": str, "chunk_idx": int, "score": float}
        sorted by (doc_id, chunk_idx) for deterministic prompt building.
    """
    query_embedding = embed_query(query)
    results = search(query_embedding, top_k=top_k)

    # Sort deterministically by doc_id then chunk_idx.
    # This ensures the same set of retrieved chunks always produces
    # the same prompt prefix, maximizing llm-d KV cache hits.
    results.sort(key=lambda r: (r["doc_id"], r["chunk_idx"]))
    return results
