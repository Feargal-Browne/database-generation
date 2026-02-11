# Confidential - nvyra-x (c) 2025-2026

import asyncio
from typing import List, Tuple, Optional

import torch
import numpy as np

from config import (
    LITE_RERANKER_MODEL,
    HEAVY_RERANKER_MODEL,
    HEAVY_RERANKER_AWQ_MODEL,
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    NER_MODEL,
    YES_TOKEN_ID,
    NO_TOKEN_ID,
    GRAPH_ENTITY_LABELS,
)
from models.calibrator import calibrate_batch_size


class AuxiliaryModels:
    """Manages all non-LLM models: rerankers, embedders, NER."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM

        self.device = device
        self.aux_stream = torch.cuda.Stream()

        def load_bf16_fa3(model_name, cls, **kwargs):
            return (
                cls.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_3",
                    **kwargs,
                )
                .to(device)
                .eval()
            )

        def load_robust_splade(model_name, cls, **kwargs):
            for attn in ("flash_attention_3", "flash_attention_2", "sdpa", "eager"):
                try:
                    print(f"  SPLADE: trying {attn}...")
                    return (
                        cls.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            attn_implementation=attn,
                            **kwargs,
                        )
                        .to(device)
                        .eval()
                    )
                except Exception:
                    continue
            raise RuntimeError("SPLADE failed to load with any attention backend")

        print("Loading lite reranker...")
        self.model_lite = load_bf16_fa3(
            LITE_RERANKER_MODEL, AutoModelForCausalLM, trust_remote_code=True
        )

        print("Loading heavy reranker (AWQ with BF16 fallback)...")
        try:
            from awq import AutoAWQForCausalLM

            self.model_heavy = AutoAWQForCausalLM.from_quantized(
                HEAVY_RERANKER_AWQ_MODEL,
                fuse_layers=True,
                trust_remote_code=True,
            ).to(device).eval()
            print(f"  Loaded AWQ model: {HEAVY_RERANKER_AWQ_MODEL}")
        except Exception as e:
            print(f"  AWQ load failed ({e}), falling back to BF16...")
            self.model_heavy = load_bf16_fa3(
                HEAVY_RERANKER_MODEL, AutoModelForCausalLM, trust_remote_code=True
            )

        print("Loading dense embedding model...")
        self.model_dense = load_bf16_fa3(
            DENSE_EMBED_MODEL, AutoModel, trust_remote_code=True
        )

        print("Loading sparse embedding model...")
        self.model_sparse = load_robust_splade(
            SPARSE_EMBED_MODEL, AutoModelForMaskedLM, trust_remote_code=True
        )

        print("Loading NER model (GLiNER)...")
        from gliner import GLiNER

        self.model_ner = GLiNER.from_pretrained(NER_MODEL).to(device).eval()

        # Tokenizers
        self.tok_sparse = AutoTokenizer.from_pretrained(
            SPARSE_EMBED_MODEL, trust_remote_code=True
        )
        self.tok_dense = AutoTokenizer.from_pretrained(
            DENSE_EMBED_MODEL, trust_remote_code=True
        )

        # Calibrate batch size using the heavy reranker
        self.optimal_batch_size = calibrate_batch_size(
            self.model_heavy, device, self.aux_stream
        )

    def run_rerank(
        self, model, input_ids: List[List[int]], micro_batch_size: int = None
    ) -> np.ndarray:
        """Run reranker inference, returns numpy array of relevance scores."""
        if micro_batch_size is None:
            micro_batch_size = self.optimal_batch_size

        results = []
        with torch.cuda.stream(self.aux_stream):
            with torch.inference_mode():
                for i in range(0, len(input_ids), micro_batch_size):
                    batch_slice = [
                        x[:1024] for x in input_ids[i : i + micro_batch_size]
                    ]
                    max_len = max(len(x) for x in batch_slice)
                    padded = [x + [0] * (max_len - len(x)) for x in batch_slice]
                    input_tensor = torch.tensor(padded, device=self.device)
                    attn_mask = (input_tensor != 0).long()

                    logits = model(
                        input_ids=input_tensor, attention_mask=attn_mask
                    ).logits[:, -1, :]
                    scores = (
                        torch.softmax(logits[:, [NO_TOKEN_ID, YES_TOKEN_ID]], dim=1)[
                            :, 1
                        ]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    results.extend(scores)
                    del input_tensor, attn_mask
                    torch.cuda.empty_cache()

        return np.array(results)

    def run_dense_embed(
        self, input_ids: List[List[int]], micro_batch_size: int = None
    ) -> List[List[int]]:
        """Compute binarized 1024-dim dense embeddings."""
        if micro_batch_size is None:
            micro_batch_size = self.optimal_batch_size

        results = []
        with torch.cuda.stream(self.aux_stream):
            with torch.inference_mode():
                for i in range(0, len(input_ids), micro_batch_size):
                    batch_slice = [
                        x[:1024] for x in input_ids[i : i + micro_batch_size]
                    ]
                    max_len = max(len(x) for x in batch_slice)
                    padded = [x + [0] * (max_len - len(x)) for x in batch_slice]
                    input_tensor = torch.tensor(padded, device=self.device)
                    attn_mask = (input_tensor != 0).long()

                    out = self.model_dense(
                        input_ids=input_tensor, attention_mask=attn_mask
                    )
                    last_idx = attn_mask.sum(1) - 1
                    vecs = out[0][torch.arange(len(batch_slice)), last_idx][:, :1024]
                    binarized = (
                        (torch.nn.functional.normalize(vecs, p=2, dim=1) > 0)
                        .int()
                        .cpu()
                        .tolist()
                    )
                    results.extend(binarized)
                    del input_tensor, attn_mask
                    torch.cuda.empty_cache()

        return results

    def run_sparse_embed(self, texts: List[str]) -> Tuple[List[list], List[list]]:
        """Compute SPLADE sparse embeddings with chunking."""
        CH_SIZE, STRIDE = 512, 256
        batch_indices, batch_values = [], []

        with torch.cuda.stream(self.aux_stream):
            with torch.inference_mode():
                for text in texts:
                    tokens = self.tok_sparse(
                        text,
                        return_tensors="pt",
                        add_special_tokens=True,
                        truncation=True,
                        max_length=8192,
                    ).to(self.device)
                    total_len = tokens["input_ids"].shape[1]

                    if total_len <= CH_SIZE:
                        chunks = [tokens]
                    else:
                        chunks = [
                            {
                                k: v[:, i : min(i + CH_SIZE, total_len)]
                                for k, v in tokens.items()
                            }
                            for i in range(0, total_len, STRIDE)
                        ]

                    chunk_vecs = []
                    for c in chunks:
                        out = self.model_sparse(**c)
                        val = torch.max(
                            torch.log(1 + torch.relu(out.logits))
                            * c["attention_mask"].unsqueeze(-1),
                            dim=1,
                        ).values.squeeze()
                        chunk_vecs.append(val)

                    final_vec = (
                        torch.stack(chunk_vecs).max(dim=0).values
                        if len(chunk_vecs) > 1
                        else chunk_vecs[0]
                    )
                    indices = final_vec.nonzero().squeeze().cpu().tolist()
                    values = final_vec[indices].cpu().tolist()
                    batch_indices.append(
                        [indices] if isinstance(indices, int) else indices
                    )
                    batch_values.append(
                        [values] if isinstance(values, float) else values
                    )

        return batch_indices, batch_values

    def run_ner(self, texts: List[str]) -> List[List[dict]]:
        """Extract named entities using GLiNER."""
        results = []
        for text in texts:
            try:
                entities = self.model_ner.predict_entities(
                    text[:4096], GRAPH_ENTITY_LABELS, threshold=0.4
                )
                results.append([
                    {
                        "text": e["text"],
                        "label": e["label"],
                        "score": round(e["score"], 3),
                    }
                    for e in entities
                ])
            except Exception:
                results.append([])
        return results

    async def compute_embeddings_parallel(
        self, texts: List[str], input_ids: List[List[int]]
    ) -> Tuple[List[List[int]], Tuple[List[list], List[list]]]:
        """Compute dense and sparse embeddings in parallel via separate threads."""
        dense_task = asyncio.to_thread(self.run_dense_embed, input_ids)
        sparse_task = asyncio.to_thread(self.run_sparse_embed, texts)
        dense_vecs, (sparse_idx, sparse_val) = await asyncio.gather(
            dense_task, sparse_task
        )
        return dense_vecs, (sparse_idx, sparse_val)
