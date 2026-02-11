# Confidential - nvyra-x (c) 2025-2026

APP_NAME = "data-cache-generation"
PIPELINE_BATCH_SIZE = 128
MAX_GPUS = 1
DEDUP_LIMIT = 50_000
GPU_MEMORY_GB = 141  # H200 SXM5 HBM3e

# Model paths
FACTCHECK_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
LITE_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
HEAVY_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
HEAVY_RERANKER_AWQ_MODEL = "Feargal/Qwen3-Reranker-8B-AWQ"
DENSE_EMBED_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"
SPARSE_EMBED_MODEL = "naver/splade-v3"
NER_MODEL = "urchade/gliner_large-v2.1"

ALL_MODELS = [
    LITE_RERANKER_MODEL,
    HEAVY_RERANKER_MODEL,
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    FACTCHECK_MODEL,
]

# SGLang
SGLANG_MEM_FRACTION = 0.40

# Reranker thresholds
LITE_THRESHOLD = 0.02
HEAVY_THRESHOLD = 0.75
YES_TOKEN_ID = 7866
NO_TOKEN_ID = 1489

# CoVe (Chain-of-Verification)
MAX_REFLEXION_ITERATIONS = 3
COVE_EARLY_EXIT_CONFIDENCE = 0.9
COVE_EARLY_EXIT_FALSITY_SCORES = {0, 9}

# Corrective RAG
CRAG_GOOD_THRESHOLD = 0.6
CRAG_AMBIGUOUS_THRESHOLD = 0.3

# Storage
S3_BUCKET = "ai-text-cache"
S3_KEY_PREFIX = "v30"
S3_EMBEDDINGS_PREFIX = "search/embeddings"
S3_GRAPH_PREFIX = "search/graph"
ZST_COMPRESSION_LEVEL = 3
S3_MAX_WORKERS = 64

# Batch calibration
CALIBRATION_MAX_LIMIT = 256
CALIBRATION_SAFETY_MARGIN = 0.95
CALIBRATION_SEQ_LEN = 1024

# Data processing
BM25_CONTEXT_LIMIT = 15_000
MINHASH_JACCARD_THRESHOLD = 0.85
MINHASH_NUM_PERM = 128
LITE_RERANKER_MAX_LEN = 1024
HEAVY_RERANKER_MAX_LEN = 4096

# GraphRAG
GRAPH_ENTITY_LABELS = [
    "person", "organization", "location", "date",
    "statistic", "claim", "source", "event",
]
GRAPH_RELATION_MAX_TOKENS = 512

# GraphRAG-R1 Agent
AGENT_MODEL = "Feargal/graphrag-r1-agent-v1"
AGENT_CLAIM_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
AGENT_HIDDEN_DIM = 128
AGENT_MAX_HOPS = 3
AGENT_BEAM_WIDTH = 5

# Environment variables (set at module level for local execution)
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
