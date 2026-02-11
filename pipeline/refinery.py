# Confidential - nvyra-x (c) 2025-2026

import asyncio
import uuid
import hashlib
from typing import List, Dict, Optional
from collections import OrderedDict

from config import DEDUP_LIMIT


class DedupCache:
    """LRU deduplication cache for claim analyses."""

    def __init__(self, limit: int = DEDUP_LIMIT):
        self.cache = OrderedDict()
        self.limit = limit

    def check(self, claim_text: str) -> Optional[Dict]:
        h = hashlib.md5(claim_text.encode()).hexdigest()
        if h in self.cache:
            self.cache.move_to_end(h)
            return self.cache[h]
        return None

    def update(self, claim_text: str, result: Dict):
        h = hashlib.md5(claim_text.encode()).hexdigest()
        self.cache[h] = result
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)


class CacheRefineryCore:
    """Main pipeline orchestrator (plain Python, no Modal decorators)."""

    def __init__(self, auxiliary, engine, storage, metrics, agent=None):
        from pipeline.reranking import RerankingPipeline
        from pipeline.graph_rag import GraphRAGPipeline
        from pipeline.verification import VerificationPipeline

        self.auxiliary = auxiliary
        self.engine = engine
        self.storage = storage
        self.metrics = metrics
        self.dedup = DedupCache()

        self.reranking = RerankingPipeline(auxiliary, engine)
        self.graph_rag = GraphRAGPipeline(auxiliary, engine, agent=agent)
        self.verification = VerificationPipeline(auxiliary, engine)

    async def process_batch(self, batch: List[Dict]):
        """Multi-stage cache generation pipeline."""

        # Stage 1: Lite reranking (filter junk)
        survivors = await self.reranking.lite_rerank(batch)
        if not survivors:
            return

        # Stage 2: Corrective RAG (evidence quality check)
        survivors = await self.verification.corrective_rag_filter(
            survivors, self.metrics
        )

        # Stage 3: Heavy reranking (deep analysis for mid-confidence)
        survivors = await self.reranking.heavy_rerank(survivors, self.dedup)
        if not survivors:
            return

        # Stage 4: GraphRAG entity/relation extraction
        survivors = await self.graph_rag.extract_and_enrich(survivors)
        for item in survivors:
            if item.get("graph", {}).get("num_entities", 0) > 0:
                self.metrics.items_graphrag += 1

        # Stage 4.5: Agent evidence path selection
        survivors = await self.graph_rag.agent_navigate(survivors)

        # Stage 5: Factcheck + CoVe reflexion loop
        survivors = await self.verification.factcheck_with_cove(
            survivors, self.dedup, self.metrics
        )

        # Stage 6: Listwise reranking (evidence ordering)
        survivors = await self.reranking.listwise_rerank(survivors)

        # Build final results with embeddings
        final_results, embed_texts = [], []
        for item in survivors:
            if "analysis" not in item:
                continue
            analysis = item["analysis"]
            graph = item.get("graph", {})
            graph_str = ", ".join(
                f'{r["subject"]} {r["predicate"]} {r["object"]}'
                for r in graph.get("relations", [])[:10]
            )
            rich = (
                f"Claim: {item['claim_text']}\n"
                f"Verdict: {analysis.get('verdict', '')}\n"
                f"Score: {analysis.get('falsity_score', '')}\n"
                f"Justification: {analysis.get('falsity_explanation', '')}\n"
                f"Synthesis: {analysis.get('synthesis', '')}\n"
                f"Critique: {analysis.get('critique', '')}\n"
                f"Graph: {graph_str}\n"
                f"Reasoning: {analysis.get('reasoning_trace', '')}"
            )
            embed_texts.append(rich)
            final_results.append(item)

        if not final_results:
            return

        # Stage 7: Parallel dense + sparse embeddings
        encoded_embed = self.auxiliary.tok_dense(
            embed_texts, padding=True, truncation=True, max_length=1024
        )["input_ids"]

        dense_vecs, (sparse_idx, sparse_val) = (
            await self.auxiliary.compute_embeddings_parallel(embed_texts, encoded_embed)
        )

        for i, item in enumerate(final_results):
            item["dense_vec"] = dense_vecs[i]
            item["sparse_idx"] = sparse_idx[i]
            item["sparse_val"] = sparse_val[i]
            item["record_uuid"] = str(uuid.uuid4())

        # Stage 8: Multi-backend save (S3 + Turso)
        await asyncio.to_thread(
            self.storage.save_batch, final_results, self.metrics
        )
