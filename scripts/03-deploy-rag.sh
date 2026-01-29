#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building RAG image in minikube Docker context..."
eval "$(minikube docker-env)"
docker build -t rag-llm-d:latest -f "$PROJECT_DIR/rag/Dockerfile" "$PROJECT_DIR/rag/"

echo "==> Deploying RAG gateway..."
kubectl apply -f "$PROJECT_DIR/k8s/rag-gateway-deployment.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/rag-gateway-service.yaml"

echo "==> Waiting for RAG gateway to be ready..."
kubectl wait --timeout=300s -n llm-d deployment/rag-gateway --for=condition=Available

GATEWAY_URL=$(minikube service rag-gateway -n llm-d --url 2>/dev/null || echo "http://$(minikube ip):30080")
echo "==> RAG gateway deployed at: $GATEWAY_URL"
echo "    Health check: curl $GATEWAY_URL/health"
