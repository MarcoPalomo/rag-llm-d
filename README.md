# RAG + llm-d Pipeline on Kubernetes

Pipeline RAG (Retrieval-Augmented Generation) avec routing intelligent via [llm-d](https://github.com/llm-d/llm-d), deploye sur Kubernetes local (minikube).

## Architecture

```
User Query -> [RAG Gateway :8080]
  -> embed query (bge-m3, CPU)
  -> search Milvus (top_k=5)
  -> build_prompt(system + chunks tries + query)
  -> [llm-d Envoy Gateway :80]
  -> [EPP: hash prefix -> lookup KV index -> route]
  -> [vLLM Pod (GPTQ, prefix cache)]
  -> response streamed back
```

### Composants

| Composant | Role |
|---|---|
| **Milvus** | Vector store (standalone, embedded etcd) |
| **llm-d infra** | Envoy Gateway + Gateway API Inference Extension |
| **llm-d EPP** | Endpoint Picker — routing prefix-cache-aware |
| **vLLM** | Inference engine avec `--enable-prefix-caching` et `--quantization gptq` |
| **RAG Ingestion** | Chunking 1024 tokens + embedding bge-m3 + stockage Milvus |
| **RAG Gateway** | FastAPI orchestrant retrieval, prompt building et appel llm-d |

### Pourquoi llm-d

llm-d ajoute un scheduling intelligent au-dessus de vLLM. Le prompt builder formate les prompts de maniere deterministe (system fixe + contexte trie par `doc_id`/`chunk_idx`). Deux requetes qui recuperent les memes documents produisent un prefixe identique. L'EPP hashe ce prefixe et route vers le pod vLLM qui a deja le KV cache en memoire GPU, evitant le recalcul du prefill (jusqu'a 88% de reduction du TTFT selon les benchmarks llm-d).

## Prerequis

- [minikube](https://minikube.sigs.k8s.io/) avec driver Docker
- GPU NVIDIA + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) sur l'hote
- [Helm](https://helm.sh/) >= 3.12
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Un token [HuggingFace](https://huggingface.co/settings/tokens)

## Quickstart

```bash
# 1. Configurer le token HuggingFace
export HF_TOKEN=hf_your_token

# 2. Deployer tout d'un coup
make all
```

Ou etape par etape :

```bash
make setup       # minikube + GPU + namespaces
make infra       # Milvus + Envoy Gateway + llm-d CRDs
make model       # vLLM GPTQ + InferencePool + InferenceModel
make rag-build   # Build de l'image Docker RAG
make rag-ingest  # Job d'ingestion des documents
make rag-deploy  # Deploiement du gateway FastAPI
make test        # Smoke test end-to-end
```

## Utilisation

```bash
# Health check
curl http://$(minikube ip):30080/health

# Requete RAG
curl -X POST http://$(minikube ip):30080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

Reponse :

```json
{
  "answer": "...",
  "sources": [
    {"doc_id": "document.txt", "chunk_idx": 0, "score": 0.87},
    {"doc_id": "document.txt", "chunk_idx": 1, "score": 0.82}
  ]
}
```

## Ingestion de documents

Placer les fichiers `.txt` ou `.md` dans un PVC monte sur `/data/documents`, puis :

```bash
make rag-ingest
kubectl logs -f job/rag-ingestion -n llm-d
```

Le chunker decoupe a 1024 tokens (aligne sur les slots KV cache de vLLM) avec 128 tokens d'overlap. L'embedding est fait par bge-m3 (1024 dimensions, COSINE).

## Commandes utiles

```bash
make status       # Etat des pods (Milvus + llm-d)
make logs-vllm    # Logs du pod vLLM
make logs-gateway # Logs du gateway RAG
make teardown     # Suppression complete + arret minikube
make help         # Liste des targets
```

## Structure du projet

```
rag-llm-d/
├── Makefile                          # Orchestration
├── scripts/
│   ├── 00-minikube-setup.sh          # Cluster + GPU + namespaces
│   ├── 01-deploy-infra.sh            # Milvus + llm-d infra
│   ├── 02-deploy-model.sh            # vLLM + CRDs
│   ├── 03-deploy-rag.sh              # Build + deploy gateway
│   └── 04-smoke-test.sh              # Test end-to-end
├── helm-values/
│   ├── milvus-standalone.yaml        # Milvus minimal (embedded etcd)
│   ├── llm-d-infra.yaml              # Gateway API + Envoy
│   └── llm-d-modelservice.yaml       # vLLM GPTQ config
├── k8s/
│   ├── namespace.yaml
│   ├── secrets.yaml.example          # Template HF_TOKEN
│   ├── inference-pool.yaml           # InferencePool CRD
│   ├── inference-model.yaml          # InferenceModel CRD
│   ├── httproute.yaml                # Route /v1/* -> InferencePool
│   ├── rag-ingestion-job.yaml        # Job d'ingestion batch
│   ├── rag-gateway-deployment.yaml
│   └── rag-gateway-service.yaml      # NodePort :30080
└── rag/
    ├── Dockerfile
    ├── requirements.txt
    ├── ingestion/
    │   ├── chunker.py                # 1024 tokens, tokenizer OpenLLaMA
    │   ├── embedder.py               # bge-m3 (CPU)
    │   ├── milvus_store.py           # Collection + IVF_FLAT index
    │   └── ingest.py                 # CLI entrypoint
    └── gateway/
        ├── retriever.py              # Milvus search + tri deterministe
        ├── prompt_builder.py         # Prompt stable pour prefix cache
        ├── llmd_client.py            # Client async httpx + SSE
        └── app.py                    # FastAPI (POST /query, GET /health)
```

## Configuration

Variables d'environnement du gateway RAG :

| Variable | Default | Description |
|---|---|---|
| `MILVUS_HOST` | `localhost` | Hote Milvus |
| `MILVUS_PORT` | `19530` | Port Milvus |
| `LLMD_GATEWAY_URL` | `http://localhost:8000` | URL du gateway Envoy/llm-d |
| `MODEL_NAME` | `open-llama-7b` | Nom du modele (doit matcher l'InferenceModel) |
| `MAX_TOKENS` | `512` | Tokens max en generation |
| `TEMPERATURE` | `0.7` | Temperature de sampling |

## Notes

- Les Helm values `llm-d-*.yaml` peuvent necessiter un ajustement selon la version des charts llm-d (`helm show values <chart>` pour verifier les cles disponibles).
- Sans GPU, les pods vLLM ne scheduleront pas. Le reste du stack (Milvus, gateway RAG) fonctionne sur CPU.
- Le modele `nkpz/open_llama_7b_qlora_uncensored-gptq` est quantize en GPTQ 4-bit, ce qui reduit la VRAM necessaire a ~6 Go.
