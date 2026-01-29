"""RAG Gateway — FastAPI application.

Orchestrates: query embedding -> Milvus retrieval -> prompt building -> llm-d inference.

Endpoints:
    POST /query   — RAG query (retrieve + generate)
    GET  /health  — liveness/readiness probe
"""

import json
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.gateway.llmd_client import generate, generate_stream
from rag.gateway.prompt_builder import build_prompt
from rag.gateway.retriever import retrieve
from rag.ingestion.milvus_store import ensure_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Gateway", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=False)


class Source(BaseModel):
    doc_id: str
    chunk_idx: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
async def startup():
    logger.info("Ensuring Milvus collection exists...")
    ensure_collection()
    logger.info("RAG Gateway ready.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    logger.info(f"Query: {req.query!r} (top_k={req.top_k}, stream={req.stream})")

    # 1. Retrieve relevant chunks from Milvus
    chunks = retrieve(req.query, top_k=req.top_k)
    logger.info(f"Retrieved {len(chunks)} chunks")

    # 2. Build deterministic prompt for prefix cache optimization
    prompt = build_prompt(chunks, req.query)

    # 3. Stream or generate
    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, chunks),
            media_type="text/event-stream",
        )

    # 4. Non-streaming: call llm-d and return full response
    result = await generate(prompt)
    answer = result["choices"][0]["text"].strip()

    sources = [
        Source(doc_id=c["doc_id"], chunk_idx=c["chunk_idx"], score=c["score"])
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)


async def _stream_response(prompt: str, chunks: list[dict]):
    """SSE wrapper for streaming responses."""
    sources = [
        {"doc_id": c["doc_id"], "chunk_idx": c["chunk_idx"], "score": c["score"]}
        for c in chunks
    ]

    # Send sources first
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # Stream tokens
    async for token_data in generate_stream(prompt):
        yield f"data: {token_data}\n\n"

    yield "data: [DONE]\n\n"
