# nvyra-x Cache Generation Pipeline

Multi-stage data cache builder for the nvyra-x disinformation detection platform. Processes raw claims through an 8-stage pipeline on H200 GPUs, producing verified factcheck analyses, hybrid embeddings, and knowledge graphs stored to Backblaze B2 and Turso.

## Architecture

```
Arrow File -> DatasetLoader (CPU) -> CacheRefinery (H200 GPU)
                                          |
                                    8-Stage Pipeline
                                          |
                            S3 (B2) + Turso (LibSQL)
```

### Pipeline Stages

1. **Lite Reranking** - Qwen3-Reranker-0.6B filters irrelevant claim-evidence pairs
2. **Corrective RAG** - Evidence quality validation with web search fallback
3. **Heavy Reranking** - Qwen3-Reranker-8B deep semantic analysis for mid-confidence items
4. **GraphRAG** - GLiNER NER extraction + Nemotron-30B relation extraction into NetworkX graphs
5. **Agent Evidence Selection** - RL-trained GAT agent navigates the knowledge graph to select optimal evidence paths
6. **Factcheck + CoVe** - Nemotron-30B-FP8 verdict generation with Chain-of-Verification reflexion loop
7. **Listwise Reranking** - LLM-based evidence reordering
8. **Embeddings + Storage** - Parallel KaLM dense + SPLADE sparse embeddings, multi-backend save

### Storage Layout

```
ai-text-cache/
  v30/{claim_id}/{uuid}.zst                    # full analysis payload
  search/embeddings/{claim_id}/{uuid}.json.zst  # dense + sparse vectors
  search/graph/{claim_id}/{uuid}.json.zst       # entities, relations, graph features
```

Metadata is stored in Turso (LibSQL) for resume/dedup.

### GraphRAG-R1 Agent

A lightweight Graph Attention Network (~1M params) trained via supervised pre-training on FEVER and RL fine-tuning with PPO. The agent navigates extracted knowledge graphs to find multi-hop evidence chains, improving factcheck accuracy over static BM25/reranker selection.

## Project Structure

```
app.py                  # Modal deployment and orchestration
config.py               # All configuration constants
loader.py               # Data ingestion with BM25 ranking and MinHash dedup
models/
  auxiliary.py          # Rerankers, embedders, NER (GLiNER)
  calibrator.py         # OOM-safe dynamic batch size tuner
  engine.py             # SGLang runtime for Nemotron-30B-FP8
pipeline/
  graph_agent.py        # GAT-based graph navigation agent
  graph_rag.py          # Entity/relation extraction and agent integration
  refinery.py           # 8-stage pipeline orchestrator
  reranking.py          # Lite, heavy, and listwise reranking
  verification.py       # Corrective RAG and CoVe reflexion loop
storage/
  backends.py           # S3 (B2) and Turso persistence
  metrics.py            # Pipeline throughput tracking
training/
  prepare_fever.py      # FEVER/LIAR dataset -> graph training data
  reward_model.py       # Nemotron-30B reward wrapper for PPO
  train_agent.py        # Supervised + RL training loop
  evaluate.py           # Agent evaluation on held-out claims
```

## Usage

```bash
# Deploy to Modal
modal deploy app.py

# Run on an Arrow file
modal run app.py --input-file /data/claims.arrow
```

### Training the Agent

```bash
# Prepare FEVER graphs
python training/prepare_fever.py --output_dir ./data --max_graphs 10000

# Phase 1: Supervised pre-training (~30 min on T4)
python training/train_agent.py --data_dir ./data --phase 1 --epochs 5

# Phase 2: RL fine-tuning with PPO (~2 hrs on T4)
python training/train_agent.py --data_dir ./data --phase 2 --epochs 3

# Evaluate
python training/evaluate.py --weights ./checkpoints/policy.pt
```

## GPU Memory Budget (H200 141GB)

| Model | Size | VRAM |
|-------|------|------|
| Nemotron-30B-FP8 (SGLang) | 30B | ~56GB |
| KaLM-Embedding-Gemma3-12B | 12B | ~24GB |
| Qwen3-Reranker-8B-AWQ | 8B | ~4GB |
| Qwen3-Reranker-0.6B | 0.6B | ~1.2GB |
| SPLADE-v3 | 110M | ~0.3GB |
| GLiNER | ~200M | ~0.5GB |
| GraphRAG-R1 Agent | ~1M | ~4MB |

## License

Proprietary. See [LICENSE.md](LICENSE.md).
