# Confidential - nvyra-x (c) 2025-2026

import sys
import os
import asyncio
import modal
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from config import (
    APP_NAME,
    PIPELINE_BATCH_SIZE,
    MAX_GPUS,
    ALL_MODELS,
    NER_MODEL,
    HEAVY_RERANKER_AWQ_MODEL,
    AGENT_MODEL,
    AGENT_CLAIM_ENCODER,
)

data_vol = modal.Volume.from_name("rag-harvest-storage-prod", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

my_secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("turso-api-new"),
    modal.Secret.from_name("b2-credentials"),
]


def download_artifacts():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    models = ALL_MODELS + [NER_MODEL, AGENT_CLAIM_ENCODER]

    # Also attempt AWQ model
    try:
        models.append(HEAVY_RERANKER_AWQ_MODEL)
    except Exception:
        pass

    def fetch(m):
        try:
            print(f"Downloading {m}...")
            snapshot_download(m, ignore_patterns=["*.md", "*.txt"])
            AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            print(f"Ready: {m}")
        except Exception as e:
            print(f"Warning for {m}: {e}")

    with ThreadPoolExecutor(8) as ex:
        ex.map(fetch, models)

    # Download GLiNER separately
    try:
        from gliner import GLiNER
        GLiNER.from_pretrained(NER_MODEL)
        print(f"Ready: {NER_MODEL} (GLiNER)")
    except Exception as e:
        print(f"Warning for GLiNER: {e}")

    print("All models downloaded.")


gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "wget", "libzstd-dev", "build-essential", "ninja-build", "ccache")
    .pip_install("uv")
    .run_commands(
        "uv venv .venv",
        "uv pip install --system --upgrade setuptools pip",
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130",
        "uv pip install --system 'sglang[all]>=0.4.6' --no-build-isolation",
        "uv pip install --system 'transformers>=4.57.0' accelerate>=1.2.0 huggingface_hub hf_transfer",
        "uv pip install --system libsql-experimental boto3 zstandard",
        "uv pip install --system polars pyarrow rank-bm25 datasketch networkx gliner autoawq",
        "uv pip install --system uvloop numpy sentence-transformers torch_geometric",
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch291 --extra-index-url https://download.pytorch.org/whl/cu130",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
        "CUDA_LAUNCH_BLOCKING": "0",
    })
    .run_function(
        download_artifacts,
        secrets=my_secrets,
        volumes={"/root/.cache/huggingface": hf_cache_vol},
    )
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "libsql-experimental", "boto3", "polars", "pyarrow",
        "rank-bm25", "datasketch", "transformers", "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cache_gen_mount = modal.Mount.from_local_dir(
    os.path.dirname(os.path.abspath(__file__)),
    remote_path="/root/cache_generation",
)

app = modal.App(APP_NAME, secrets=my_secrets)


@app.cls(image=cpu_image, volumes={"/data": data_vol}, timeout=1200)
class DatasetLoader:
    @modal.method()
    def process_and_rank(self, input_file: str) -> List[Dict]:
        # Use inline import since this runs on CPU image without the package
        import polars as pl
        import pyarrow as pa
        from rank_bm25 import BM25Okapi
        from transformers import AutoTokenizer
        from datasketch import MinHash

        tok_lite = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True
        )
        tok_heavy = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Reranker-8B", trust_remote_code=True
        )

        print(f"Streaming from {input_file}...")
        try:
            with open(input_file, "rb") as f:
                reader = pa.ipc.open_stream(f)
                pa_table = reader.read_all()
            df = pl.from_arrow(pa_table)
        except Exception as e:
            print(f"Failed to read Arrow Stream: {e}")
            return []

        print("Aggregating and pruning context...")
        df = df.group_by("claim_id").agg([
            pl.col("claim_text").first(),
            pl.col("raw_doc_text").alias("docs"),
            pl.col("url").alias("urls"),
            pl.col("title").alias("titles"),
            pl.col("source").alias("sources"),
        ])

        data = df.to_dicts()
        processed_items = []

        for row in data:
            claim = row["claim_text"]
            docs = row["docs"]

            unique_docs = []
            hashes = []
            for doc in docs:
                m = MinHash(num_perm=128)
                for word in doc.split():
                    m.update(word.encode("utf8"))
                is_duplicate = False
                for h in hashes:
                    if m.jaccard(h) > 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_docs.append(doc)
                    hashes.append(m)

            if len(unique_docs) > 0:
                tokenized_claim = claim.lower().split()
                tokenized_docs = [d.lower().split() for d in unique_docs]
                bm25 = BM25Okapi(tokenized_docs)
                scores = bm25.get_scores(tokenized_claim)
                scored_docs = sorted(
                    zip(unique_docs, scores), key=lambda x: x[1], reverse=True
                )

                selected_docs = []
                current_len = 0
                for d, s in scored_docs:
                    if current_len + len(d) < 15000:
                        selected_docs.append(d)
                        current_len += len(d)
                    else:
                        break
                combined_docs = "\n\n".join(selected_docs)
            else:
                combined_docs = ""

            lite_p = (
                f"Instruct: Retrieve relevant context\n"
                f"Query: {claim}\nDoc: {combined_docs[:512]}"
            )
            heavy_p = (
                f"<Instruct>: Identify contradictions and supporting evidence\n"
                f"<Query>: {claim}\n<Document>: {combined_docs}"
            )

            lite_ids = tok_lite.encode(lite_p, truncation=True, max_length=1024)
            heavy_ids = tok_heavy.encode(heavy_p, truncation=True, max_length=4096)
            prefix = heavy_ids[:64]
            lcp_hash = hash(tuple(prefix))

            processed_items.append({
                "claim_id": row["claim_id"],
                "claim_text": claim,
                "doc_text": combined_docs,
                "doc_metadata": {
                    "urls": row["urls"],
                    "titles": row["titles"],
                    "sources": row["sources"],
                },
                "lite_ids": lite_ids,
                "heavy_ids": heavy_ids,
                "lcp_hash": lcp_hash,
            })

        processed_items.sort(key=lambda x: x["lcp_hash"])
        return processed_items


@app.cls(image=cpu_image, secrets=my_secrets)
class DBScanner:
    @modal.method()
    def get_existing_ids(self) -> List[str]:
        import libsql_experimental as libsql

        print("Connecting to Turso for resume check...")
        try:
            turso_url = os.environ["turso_url"]
            turso_token = os.environ["turso_api"]
            db = libsql.connect(database=turso_url, auth_token=turso_token)
            res = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='claim_metadata'"
            )
            if not res.fetchone():
                return []
            rows = db.execute("SELECT claim_id FROM claim_metadata").fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"Failed to fetch existing IDs: {e}")
            return []


@app.cls(
    image=gpu_image,
    gpu="H200",
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache_vol},
    mounts=[cache_gen_mount],
    max_containers=MAX_GPUS,
    timeout=5400,
)
class CacheRefinery:

    @modal.enter()
    def setup(self):
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        sys.path.insert(0, "/root/cache_generation")

        from models.auxiliary import AuxiliaryModels
        from models.engine import FactcheckEngine
        from storage.backends import StorageManager
        from storage.metrics import PipelineMetrics
        from pipeline.refinery import CacheRefineryCore
        from pipeline.graph_agent import GraphNavigationAgent

        print("Initializing auxiliary models...")
        self.auxiliary = AuxiliaryModels(device="cuda")

        print("Initializing SGLang factcheck engine...")
        self.engine = FactcheckEngine()

        print("Initializing storage backends...")
        self.storage = StorageManager()

        print("Initializing graph navigation agent...")
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(AGENT_MODEL, filename="policy.pt")
            self.agent = GraphNavigationAgent(device="cuda", weights_path=weights_path)
            print("Agent loaded from pretrained weights.")
        except Exception as e:
            print(f"Agent weights not found ({e}), using random initialization.")
            self.agent = GraphNavigationAgent(device="cuda")

        self.metrics = PipelineMetrics()
        self.core = CacheRefineryCore(
            self.auxiliary, self.engine, self.storage, self.metrics, agent=self.agent
        )
        print("Cache refinery ready.")

    @modal.method()
    async def process_batch(self, batch: List[Dict]):
        await self.core.process_batch(batch)


@app.local_entrypoint()
def main(input_file: str):
    if not input_file:
        print("Usage: modal run app.py --input-file data.arrow")
        return

    print("Triggering remote dataset loader...")
    processed_items = DatasetLoader().process_and_rank.remote(input_file)

    if not processed_items:
        print("No items returned.")
        return

    print("Checking for existing progress in DB...")
    existing_ids = set(DBScanner().get_existing_ids.remote())
    print(f"  Found {len(existing_ids)} previously processed claims.")

    initial_count = len(processed_items)
    processed_items = [x for x in processed_items if x["claim_id"] not in existing_ids]
    skipped_count = initial_count - len(processed_items)

    if skipped_count > 0:
        print(f"Skipping {skipped_count} items (already in DB).")
    print(f"Dispatching {len(processed_items)} items to H200...")

    if not processed_items:
        print("Job complete (nothing new to process).")
        return

    refinery = CacheRefinery()

    async def driver():
        futs = []
        for i in range(0, len(processed_items), PIPELINE_BATCH_SIZE):
            batch = processed_items[i : i + PIPELINE_BATCH_SIZE]
            futs.append(asyncio.create_task(refinery.process_batch.remote.aio(batch)))

            if len(futs) > 50:
                done, _ = await asyncio.wait(futs, return_when=asyncio.FIRST_COMPLETED)
                futs = [f for f in futs if not f.done()]
        await asyncio.gather(*futs)

    print("Starting GPU processing...")
    asyncio.run(driver())
    print("Job fully complete.")
