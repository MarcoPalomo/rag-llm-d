"""HTTP client for the llm-d inference gateway.

Sends requests to the OpenAI-compatible /v1/completions endpoint
exposed by the Envoy Gateway -> EPP -> vLLM chain.
"""

import os

import httpx

LLMD_GATEWAY_URL = os.getenv("LLMD_GATEWAY_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "open-llama-7b")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


async def generate(prompt: str) -> dict:
    """Send a completion request to llm-d and return the response.

    Uses the /v1/completions endpoint (raw prompt, not chat format)
    to ensure the full prompt prefix is visible to the EPP for
    prefix hash computation.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{LLMD_GATEWAY_URL}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()


async def generate_stream(prompt: str):
    """Stream tokens from llm-d via SSE.

    Yields text chunks as they arrive.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{LLMD_GATEWAY_URL}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield line[6:]  # Strip "data: " prefix
