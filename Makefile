.PHONY: setup infra model rag-build rag-ingest rag-deploy test teardown all help

NAMESPACE_LLM  := llm-d
NAMESPACE_MV   := milvus

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Start minikube with GPU and create namespaces
	bash scripts/00-minikube-setup.sh

infra: ## Deploy Milvus + llm-d infrastructure (Envoy Gateway, CRDs)
	bash scripts/01-deploy-infra.sh

model: ## Deploy vLLM model service with GPTQ (requires HF_TOKEN)
	bash scripts/02-deploy-model.sh

rag-build: ## Build the RAG Docker image in minikube context
	eval $$(minikube docker-env) && \
	docker build -t rag-llm-d:latest -f rag/Dockerfile rag/

rag-ingest: ## Run the document ingestion job
	kubectl apply -f k8s/rag-ingestion-job.yaml
	@echo "Monitor with: kubectl logs -f job/rag-ingestion -n $(NAMESPACE_LLM)"

rag-deploy: ## Deploy the RAG gateway service
	kubectl apply -f k8s/rag-gateway-deployment.yaml
	kubectl apply -f k8s/rag-gateway-service.yaml
	@echo "Access at: http://$$(minikube ip):30080"

test: ## Run smoke tests against the RAG gateway
	bash scripts/04-smoke-test.sh

status: ## Show status of all pods
	@echo "=== Milvus ==="
	@kubectl get pods -n $(NAMESPACE_MV) 2>/dev/null || echo "  Namespace not found"
	@echo ""
	@echo "=== llm-d ==="
	@kubectl get pods -n $(NAMESPACE_LLM) 2>/dev/null || echo "  Namespace not found"

logs-vllm: ## Tail vLLM pod logs
	kubectl logs -f -n $(NAMESPACE_LLM) -l app=open-llama-7b --tail=100

logs-gateway: ## Tail RAG gateway logs
	kubectl logs -f -n $(NAMESPACE_LLM) -l app=rag-gateway --tail=100

teardown: ## Delete everything and stop minikube
	-helm uninstall llm-d-model -n $(NAMESPACE_LLM) 2>/dev/null
	-helm uninstall llm-d-infra -n $(NAMESPACE_LLM) 2>/dev/null
	-helm uninstall milvus -n $(NAMESPACE_MV) 2>/dev/null
	-kubectl delete -f k8s/rag-gateway-deployment.yaml 2>/dev/null
	-kubectl delete -f k8s/rag-gateway-service.yaml 2>/dev/null
	-kubectl delete -f k8s/rag-ingestion-job.yaml 2>/dev/null
	-kubectl delete -f k8s/inference-model.yaml 2>/dev/null
	-kubectl delete -f k8s/inference-pool.yaml 2>/dev/null
	-kubectl delete -f k8s/httproute.yaml 2>/dev/null
	-kubectl delete namespace $(NAMESPACE_LLM) 2>/dev/null
	-kubectl delete namespace $(NAMESPACE_MV) 2>/dev/null
	minikube stop

all: setup infra model rag-build rag-deploy ## Deploy everything end-to-end
