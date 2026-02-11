# Confidential - nvyra-x (c) 2025-2026

import asyncio
import json
import re
from typing import List, Dict, Optional

from config import LITE_THRESHOLD, HEAVY_THRESHOLD


class RerankingPipeline:
    """Multi-stage reranking: lite, heavy, and listwise."""

    def __init__(self, auxiliary, engine):
        self.auxiliary = auxiliary
        self.engine = engine

    async def lite_rerank(self, batch: List[Dict]) -> List[Dict]:
        """Stage 1: Filter with lightweight 0.6B reranker."""
        lite_ids = [x["lite_ids"] for x in batch]
        scores = await asyncio.to_thread(
            self.auxiliary.run_rerank, self.auxiliary.model_lite, lite_ids
        )

        survivors = []
        for i, s in enumerate(scores):
            if s < LITE_THRESHOLD:
                continue
            batch[i]["lite_score"] = float(s)
            survivors.append(batch[i])

        return survivors

    async def heavy_rerank(self, survivors: List[Dict], dedup_cache) -> List[Dict]:
        """Stage 3: Deep reranking with 8B model for mid-confidence items."""
        heavy_ids = []
        passed = []

        for item in survivors:
            cached = dedup_cache.check(item["claim_text"])
            if cached:
                item["analysis"] = cached
                item["cache_hit"] = True
                passed.append(item)
                continue
            if item["lite_score"] > 0.98:
                item["heavy_score"] = 1.0
                passed.append(item)
            else:
                heavy_ids.append(item)

        if heavy_ids:
            ids_list = [x["heavy_ids"] for x in heavy_ids]
            scores = await asyncio.to_thread(
                self.auxiliary.run_rerank, self.auxiliary.model_heavy, ids_list
            )
            for i, item in enumerate(heavy_ids):
                s = float(scores[i])
                if s > HEAVY_THRESHOLD:
                    item["heavy_score"] = s
                    passed.append(item)

        return passed

    async def listwise_rerank(self, items: List[Dict]) -> List[Dict]:
        """Stage 6: LLM-based listwise reranking for evidence ordering."""
        for item in items:
            if item.get("cache_hit"):
                continue

            analysis = item.get("analysis", {})
            citations = analysis.get("citations", [])
            if len(citations) < 3:
                continue

            citation_text = "\n".join(
                f"[{i}] {c}" for i, c in enumerate(citations[:8])
            )
            prompt = (
                f"Given this claim and evidence passages, rank them by relevance "
                f"to the claim. Output ONLY a JSON array of indices, most relevant first.\n\n"
                f"Claim: {item['claim_text']}\n\nPassages:\n{citation_text}\n\n"
                f"Output: [indices in order of relevance]"
            )

            raw = await self.engine.generate(
                prompt,
                system_prompt="You rank evidence passages. Output only a JSON array of integers.",
                max_tokens=64,
                temperature=0.1,
            )

            if raw:
                try:
                    match = re.search(r"\[[\d,\s]+\]", raw)
                    if match:
                        order = json.loads(match.group())
                        reordered = []
                        for idx in order:
                            if isinstance(idx, int) and idx < len(citations):
                                reordered.append(citations[idx])
                        if reordered:
                            item["analysis"]["citations"] = reordered
                            item["analysis"]["listwise_reranked"] = True
                except Exception:
                    pass

        return items
