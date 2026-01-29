#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure HF_TOKEN is set
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable is required." >&2
  echo "Export it: export HF_TOKEN=hf_your_token_here" >&2
  exit 1
fi

echo "==> Creating HuggingFace token secret..."
kubectl create secret generic llm-d-hf-token \
  -n llm-d \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Deploying llm-d model service (vLLM + GPTQ)..."
helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/ 2>/dev/null || true
helm repo update
helm upgrade --install llm-d-model llm-d-modelservice/llm-d-modelservice \
  -n llm-d \
  -f "$PROJECT_DIR/helm-values/llm-d-modelservice.yaml" \
  --wait --timeout 10m

echo "==> Applying InferencePool and InferenceModel CRDs..."
kubectl apply -f "$PROJECT_DIR/k8s/inference-pool.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/inference-model.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/httproute.yaml"

echo "==> Model deployment complete."
echo "    Waiting for vLLM pod to be ready (this may take several minutes for model download)..."
kubectl wait --timeout=600s -n llm-d pod -l app=open-llama-7b --for=condition=Ready 2>/dev/null || \
  echo "    Note: Pod may still be downloading the model. Check with: kubectl get pods -n llm-d"
