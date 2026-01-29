#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Deploying Milvus standalone..."
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/ 2>/dev/null || true
helm repo update
helm upgrade --install milvus zilliztech/milvus \
  -n milvus \
  -f "$PROJECT_DIR/helm-values/milvus-standalone.yaml" \
  --wait --timeout 5m

echo "==> Deploying llm-d infrastructure..."
helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/ 2>/dev/null || true
helm repo update
helm upgrade --install llm-d-infra llm-d-infra/llm-d-infra \
  -n llm-d \
  -f "$PROJECT_DIR/helm-values/llm-d-infra.yaml" \
  --wait --timeout 5m

echo "==> Infrastructure deployed."
