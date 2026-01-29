"""Embedding wrapper using BAAI/bge-m3 (1024 dimensions).

Runs on CPU to leave the GPU free for vLLM inference.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return normalized vectors."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()
