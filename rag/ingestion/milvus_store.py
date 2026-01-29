"""Milvus collection management and document storage.

Collection schema:
  - id        : INT64 (auto-generated primary key)
  - doc_id    : VARCHAR — stable document identifier
  - chunk_idx : INT32  — position within the document
  - text      : VARCHAR — chunk text content
  - embedding : FLOAT_VECTOR(1024) — bge-m3 dense vector
"""

import os

from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 1024

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

_client = None


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        _client = MilvusClient(uri=uri)
    return _client


def ensure_collection():
    """Create the collection and index if they don't exist."""
    client = _get_client()

    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_idx", DataType.INT32)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}' with IVF_FLAT index.")


def insert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """Insert chunked documents with their embeddings into Milvus.

    Args:
        chunks: List of {"doc_id": str, "chunk_idx": int, "text": str}
        embeddings: Corresponding embedding vectors.
    """
    client = _get_client()

    data = [
        {
            "doc_id": chunk["doc_id"],
            "chunk_idx": chunk["chunk_idx"],
            "text": chunk["text"],
            "embedding": emb,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Inserted {result['insert_count']} chunks into '{COLLECTION_NAME}'.")


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for similar chunks. Returns list of {text, doc_id, chunk_idx, score}."""
    client = _get_client()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "doc_id", "chunk_idx"],
    )

    return [
        {
            "text": hit["entity"]["text"],
            "doc_id": hit["entity"]["doc_id"],
            "chunk_idx": hit["entity"]["chunk_idx"],
            "score": hit["distance"],
        }
        for hit in results[0]
    ]
