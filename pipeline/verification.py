# Confidential - nvyra-x (c) 2025-2026

import asyncio
import json
import re
import uuid
from typing import List, Dict, Optional

from config import (
    MAX_REFLEXION_ITERATIONS,
    COVE_EARLY_EXIT_CONFIDENCE,
    COVE_EARLY_EXIT_FALSITY_SCORES,
    CRAG_GOOD_THRESHOLD,
    CRAG_AMBIGUOUS_THRESHOLD,
)


class VerificationPipeline:
    """Chain-of-Verification (CoVe) reflexion loop and Corrective RAG."""

    def __init__(self, auxiliary, engine):
        self.auxiliary = auxiliary
        self.engine = engine

    async def corrective_rag_filter(
        self, items: List[Dict], metrics=None
    ) -> List[Dict]:
        """Stage 2: Corrective RAG - evaluate evidence quality."""
        survivors = []
        for item in items:
            if item.get("cache_hit"):
                survivors.append(item)
                continue

            # Use the heavy reranker score as a proxy for evidence quality
            lite_score = item.get("lite_score", 0.0)

            if lite_score >= CRAG_GOOD_THRESHOLD:
                item["evidence_quality"] = "good"
                survivors.append(item)
            elif lite_score >= CRAG_AMBIGUOUS_THRESHOLD:
                item["evidence_quality"] = "weak"
                survivors.append(item)
            else:
                item["evidence_quality"] = "insufficient"
                item["analysis"] = {
                    "verdict": "unverifiable",
                    "falsity_score": 0.0,
                    "synthesis": "",
                    "critique": "Insufficient evidence quality for reliable verification.",
                    "reasoning_trace": "",
                    "entities": [],
                    "graph_relations": [],
                    "citations": [],
                    "falsity_explanation": "Evidence quality below threshold.",
                    "search_query": "",
                }
                survivors.append(item)
                if metrics:
                    metrics.items_corrective_filtered += 1

        return survivors

    async def factcheck_with_cove(
        self, items: List[Dict], dedup_cache=None, metrics=None
    ) -> List[Dict]:
        """Stage 5: Factcheck generation with Chain-of-Verification reflexion."""
        gen_tasks = []
        gen_indices = []

        for i, item in enumerate(items):
            if item.get("cache_hit") or item.get("evidence_quality") == "insufficient":
                continue
            gen_tasks.append(self._cove_loop(item, metrics))
            gen_indices.append(i)

        if gen_tasks:
            results = await asyncio.gather(*gen_tasks)
            for idx, analysis in zip(gen_indices, results):
                if analysis:
                    items[idx]["analysis"] = analysis
                    if dedup_cache:
                        dedup_cache.update(items[idx]["claim_text"], analysis)

        return items

    async def _cove_loop(self, item: Dict, metrics=None) -> Optional[Dict]:
        """Run up to MAX_REFLEXION_ITERATIONS of Chain-of-Verification."""
        claim = item["claim_text"]
        evidence = item["doc_text"][:15000]
        graph_context = ""
        graph = item.get("graph", {})
        if graph.get("key_entities"):
            entities_str = ", ".join(
                e["entity"] for e in graph["key_entities"]
            )
            graph_context = f"\nKey entities: {entities_str}"
            relations = graph.get("relations", [])
            if relations:
                rel_str = "; ".join(
                    f'{r["subject"]} {r["predicate"]} {r["object"]}'
                    for r in relations[:10]
                )
                graph_context += f"\nRelationships: {rel_str}"

        # Iteration 1: Draft analysis
        system_prompt = (
            "You are an expert fact-checker. Analyze the claim against the evidence. "
            "First, think step-by-step in a <think>...</think> block. "
            "Then output a valid JSON object with keys: synthesis, critique, "
            "reasoning_trace, entities (list), graph_relations (list), "
            "citations (list of ints), verdict, falsity_score (0-9), "
            "falsity_explanation, search_query."
        )
        draft_prompt = f"Claim: {claim}\nEvidence: {evidence}{graph_context}"

        raw_draft = await self.engine.generate(
            draft_prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.8
        )

        draft = self._parse_analysis(raw_draft)
        if not draft:
            return None

        if metrics:
            metrics.items_cove_iterations += 1

        # Early exit for high-confidence clear-cut results
        confidence = draft.get("falsity_score", 5)
        if isinstance(confidence, (int, float)):
            raw_score = int(confidence * 9) if confidence <= 1 else int(confidence)
            if raw_score in COVE_EARLY_EXIT_FALSITY_SCORES:
                draft["cove_iterations"] = 1
                draft["falsity_score"] = float(draft.get("falsity_score", 0)) / 9.0
                return draft

        # Iteration 2: Verification questions
        verify_prompt = (
            f"You previously analyzed this claim and concluded: "
            f"verdict={draft.get('verdict')}, "
            f"reasoning={draft.get('reasoning_trace', '')[:500]}\n\n"
            f"Claim: {claim}\nEvidence: {evidence}{graph_context}\n\n"
            f"Generate 3-5 verification questions to stress-test your conclusion. "
            f"Then answer each question using ONLY the provided evidence. "
            f"If any answer contradicts your verdict, revise it.\n\n"
            f"Output a JSON object with keys: verification_questions (list of "
            f"{{question, answer, supports_verdict}}), revised_verdict, "
            f"revised_falsity_score (0-9), confidence (0.0-1.0), "
            f"revised_reasoning."
        )

        raw_verify = await self.engine.generate(
            verify_prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.5
        )

        if metrics:
            metrics.items_cove_iterations += 1

        verify_result = self._parse_json(raw_verify)
        if verify_result and "revised_verdict" in verify_result:
            draft["verdict"] = verify_result["revised_verdict"]
            draft["verification_trace"] = verify_result.get(
                "verification_questions", []
            )
            new_confidence = verify_result.get("confidence", 0.5)

            # Iteration 3: Final synthesis (only if verdict changed significantly)
            old_score = draft.get("falsity_score", 0.5)
            new_score = float(verify_result.get("revised_falsity_score", 5)) / 9.0
            delta = abs(new_score - (old_score if old_score <= 1 else old_score / 9.0))

            if delta > 0.15 and metrics:
                final_prompt = (
                    f"Your initial verdict was {draft.get('verdict')}. "
                    f"After verification, it became {verify_result['revised_verdict']}.\n\n"
                    f"Claim: {claim}\nEvidence: {evidence}{graph_context}\n\n"
                    f"Provide your final, definitive analysis. Output a JSON object "
                    f"with keys: verdict, falsity_score (0-9), falsity_explanation, "
                    f"reasoning_trace, confidence (0.0-1.0)."
                )
                raw_final = await self.engine.generate(
                    final_prompt,
                    system_prompt=system_prompt,
                    max_tokens=1024,
                    temperature=0.3,
                )
                metrics.items_cove_iterations += 1

                final_result = self._parse_json(raw_final)
                if final_result:
                    draft["verdict"] = final_result.get(
                        "verdict", draft["verdict"]
                    )
                    draft["falsity_explanation"] = final_result.get(
                        "falsity_explanation", draft.get("falsity_explanation", "")
                    )
                    draft["reasoning_trace"] = final_result.get(
                        "reasoning_trace", draft.get("reasoning_trace", "")
                    )
                    new_score = float(
                        final_result.get("falsity_score", 5)
                    ) / 9.0

            draft["falsity_score"] = new_score
            draft["cove_iterations"] = (
                3 if delta > 0.15 else 2
            )
        else:
            draft["falsity_score"] = (
                float(draft.get("falsity_score", 0)) / 9.0
                if draft.get("falsity_score", 0) > 1
                else float(draft.get("falsity_score", 0))
            )
            draft["cove_iterations"] = 1

        return draft

    def _parse_analysis(self, raw: Optional[str]) -> Optional[Dict]:
        """Parse factcheck analysis output, handling <think> blocks."""
        if not raw:
            return None
        try:
            json_match = re.search(r"</think>\s*(\{.*\})", raw, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})$", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return {
                    "synthesis": data.get("synthesis", ""),
                    "critique": data.get("critique", ""),
                    "reasoning_trace": data.get("reasoning_trace", ""),
                    "entities": data.get("entities", []),
                    "graph_relations": data.get("graph_relations", []),
                    "citations": data.get("citations", []),
                    "verdict": data.get("verdict", "Unknown"),
                    "falsity_score": data.get("falsity_score", 0),
                    "falsity_explanation": data.get("falsity_explanation", ""),
                    "search_query": data.get("search_query", ""),
                }
        except Exception:
            pass
        return None

    def _parse_json(self, raw: Optional[str]) -> Optional[Dict]:
        """Parse a JSON object from raw text."""
        if not raw:
            return None
        try:
            json_match = re.search(r"</think>\s*(\{.*\})", raw, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except Exception:
            pass
        return None
