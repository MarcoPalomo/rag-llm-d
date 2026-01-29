#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting minikube with GPU support..."
minikube start \
  --driver=docker \
  --gpus=all \
  --memory=16384 \
  --cpus=6 \
  --disk-size=80g \
  --kubernetes-version=v1.30.0

echo "==> Enabling addons..."
minikube addons enable metrics-server

echo "==> Verifying GPU availability..."
kubectl wait --for=condition=Ready node --all --timeout=120s
if kubectl describe nodes | grep -q "nvidia.com/gpu"; then
  echo "GPU detected on node."
else
  echo "WARNING: No NVIDIA GPU detected. vLLM pods will not schedule."
  echo "Ensure nvidia-container-toolkit is installed on the host."
fi

echo "==> Creating namespaces..."
kubectl create namespace llm-d --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace milvus --dry-run=client -o yaml | kubectl apply -f -

echo "==> Minikube setup complete."
