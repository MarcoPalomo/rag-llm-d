"""CLI entrypoint for document ingestion.

Reads all .txt and .pdf files from an input directory, chunks them
at 1024-token boundaries, embeds with bge-m3, and stores in Milvus.

Usage:
    python -m rag.ingestion.ingest --input-dir /data/documents
"""

import argparse
import os
import sys
from pathlib import Path

from rag.ingestion.chunker import chunk_text
from rag.ingestion.embedder import embed_texts
from rag.ingestion.milvus_store import ensure_collection, insert_chunks

SUPPORTED_EXTENSIONS = {".txt", ".md"}
BATCH_SIZE = 32


def read_document(path: Path) -> str:
    """Read a text document. Extend this for PDF support."""
    return path.read_text(encoding="utf-8")


def ingest_directory(input_dir: str):
    """Ingest all supported documents from a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    files = [
        f
        for f in input_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]

    if not files:
        print(f"No supported files found in {input_dir}.")
        return

    print(f"Found {len(files)} document(s) to ingest.")
    ensure_collection()

    all_chunks = []
    for filepath in files:
        doc_id = str(filepath.relative_to(input_path))
        print(f"  Processing: {doc_id}")
        text = read_document(filepath)
        chunks = chunk_text(text, doc_id=doc_id)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunk(s)")

    print(f"\nEmbedding {len(all_chunks)} total chunks...")

    # Process in batches to manage memory
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        insert_chunks(batch, embeddings)
        print(f"  Batch {i // BATCH_SIZE + 1} done ({len(batch)} chunks)")

    print(f"\nIngestion complete. {len(all_chunks)} chunks stored in Milvus.")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing documents to ingest.",
    )
    args = parser.parse_args()
    ingest_directory(args.input_dir)


if __name__ == "__main__":
    main()
