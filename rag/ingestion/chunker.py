"""Document chunker aligned to 1024-token KV cache slots.

Uses the OpenLLaMA tokenizer for accurate token counting so that chunk
boundaries match the block size vLLM uses for prefix caching.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

CHUNK_SIZE = 1024  # tokens — aligned with KV cache slot boundaries
CHUNK_OVERLAP = 128  # ~12.5% overlap for context continuity
TOKENIZER_NAME = "openlm-research/open_llama_7b"

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def token_length(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def create_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, doc_id: str) -> list[dict]:
    """Split text into chunks and return dicts with metadata.

    Returns:
        List of {"doc_id": str, "chunk_idx": int, "text": str}
    """
    splitter = create_splitter()
    chunks = splitter.split_text(text)
    return [
        {"doc_id": doc_id, "chunk_idx": i, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
