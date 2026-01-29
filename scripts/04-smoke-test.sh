#!/usr/bin/env bash
set -euo pipefail

GATEWAY_URL="${1:-http://$(minikube ip):30080}"

echo "==> Smoke test against $GATEWAY_URL"

echo ""
echo "--- Health check ---"
curl -sf "$GATEWAY_URL/health" | python3 -m json.tool
echo ""

echo "--- RAG query ---"
curl -sf -X POST "$GATEWAY_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?", "top_k": 3}' \
  | python3 -m json.tool

echo ""
echo "==> Smoke test complete."
