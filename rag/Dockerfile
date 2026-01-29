FROM python:3.11-slim

WORKDIR /app

# System dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image
# This avoids download latency on first pod start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Pre-download the tokenizer for chunking
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openlm-research/open_llama_7b')"

# Copy application code
COPY . /app/rag/
ENV PYTHONPATH=/app

# Default: run the gateway. Ingestion jobs override the command.
CMD ["uvicorn", "rag.gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
