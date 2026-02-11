# Confidential - nvyra-x (c) 2025-2026

import asyncio
from typing import List, Dict

from config import GRAPH_RELATION_MAX_TOKENS, AGENT_MAX_HOPS


class GraphRAGPipeline:
    """Entity/relation extraction, knowledge graph construction, and agent navigation."""

    def __init__(self, auxiliary, engine, agent=None):
        self.auxiliary = auxiliary
        self.engine = engine
        self.agent = agent

    async def extract_and_enrich(self, items: List[Dict]) -> List[Dict]:
        """Stage 4: Extract entities with GLiNER, relations with Nemotron-30B."""
        texts = []
        indices = []
        for i, item in enumerate(items):
            if item.get("cache_hit"):
                continue
            combined = f"{item['claim_text']}\n{item['doc_text'][:3000]}"
            texts.append(combined)
            indices.append(i)

        if not texts:
            return items

        ner_results = await asyncio.to_thread(self.auxiliary.run_ner, texts)

        relation_prompts = []
        relation_indices = []
        for j, (idx, entities) in enumerate(zip(indices, ner_results)):
            if not entities:
                items[idx]["graph"] = {"entities": [], "relations": []}
                continue

            entity_list = ", ".join(
                f'{e["text"]} ({e["label"]})' for e in entities[:20]
            )
            claim = items[idx]["claim_text"]
            evidence = items[idx]["doc_text"][:2000]

            prompt = (
                f"Given these entities extracted from the text:\n{entity_list}\n\n"
                f"Claim: {claim}\nEvidence: {evidence}\n\n"
                f"Extract relationships as a JSON array of objects with keys: "
                f"subject, predicate, object, weight (0.0-1.0).\n"
                f"Focus on relationships relevant to verifying the claim.\n"
                f"Output ONLY the JSON array."
            )
            relation_prompts.append(prompt)
            relation_indices.append((idx, entities))

        if relation_prompts:
            relation_results = await self.engine.generate_batch(
                relation_prompts,
                system_prompt="Extract entity relationships as JSON arrays.",
                max_tokens=GRAPH_RELATION_MAX_TOKENS,
                temperature=0.3,
            )

            import json
            import re

            for raw, (idx, entities) in zip(relation_results, relation_indices):
                relations = []
                if raw:
                    try:
                        match = re.search(r"\[.*\]", raw, re.DOTALL)
                        if match:
                            parsed = json.loads(match.group())
                            for r in parsed:
                                if all(k in r for k in ("subject", "predicate", "object")):
                                    relations.append({
                                        "subject": str(r["subject"]),
                                        "predicate": str(r["predicate"]),
                                        "object": str(r["object"]),
                                        "weight": float(r.get("weight", 0.5)),
                                    })
                    except Exception:
                        pass

                graph = self._build_graph(entities, relations)
                items[idx]["graph"] = graph

        return items

    async def agent_navigate(self, items: List[Dict]) -> List[Dict]:
        """Stage 4.5: RL agent selects optimal evidence paths through the graph."""
        if not self.agent:
            return items

        for item in items:
            if item.get("cache_hit"):
                continue
            graph = item.get("graph", {})
            if graph.get("num_entities", 0) < 2 or not graph.get("relations"):
                continue

            try:
                selected_path = await asyncio.to_thread(
                    self.agent.navigate,
                    item["claim_text"],
                    graph,
                    AGENT_MAX_HOPS,
                )
                item["agent_evidence_path"] = selected_path
                item["doc_text"] = self._assemble_path_evidence(
                    selected_path, item["doc_text"], graph
                )
            except Exception as e:
                print(f"Agent navigation failed: {e}")

        return items

    def _assemble_path_evidence(
        self,
        path: List[str],
        doc_text: str,
        graph: dict,
    ) -> str:
        """Reorder evidence text to prioritize content related to the agent's path."""
        if not path:
            return doc_text

        paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
        if not paragraphs:
            return doc_text

        path_lower = {entity.lower() for entity in path}

        def relevance_score(paragraph: str) -> int:
            text_lower = paragraph.lower()
            return sum(1 for entity in path_lower if entity in text_lower)

        scored = [(p, relevance_score(p)) for p in paragraphs]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Put path-relevant paragraphs first, then the rest
        relevant = [p for p, s in scored if s > 0]
        rest = [p for p, s in scored if s == 0]

        return "\n\n".join(relevant + rest)

    def _build_graph(self, entities: List[dict], relations: List[dict]) -> dict:
        """Build graph features from entities and relations."""
        try:
            import networkx as nx

            G = nx.DiGraph()
            for e in entities:
                G.add_node(e["text"], label=e["label"], score=e["score"])
            for r in relations:
                G.add_edge(
                    r["subject"], r["object"],
                    predicate=r["predicate"],
                    weight=r["weight"],
                )

            centrality = nx.degree_centrality(G) if len(G) > 0 else {}
            key_entities = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return {
                "entities": [
                    {"text": e["text"], "label": e["label"], "score": e["score"]}
                    for e in entities
                ],
                "relations": relations,
                "num_entities": len(entities),
                "num_relations": len(relations),
                "graph_density": nx.density(G) if len(G) > 1 else 0.0,
                "key_entities": [
                    {"entity": name, "centrality": round(score, 3)}
                    for name, score in key_entities
                ],
            }
        except Exception:
            return {
                "entities": [
                    {"text": e["text"], "label": e["label"], "score": e["score"]}
                    for e in entities
                ],
                "relations": relations,
                "num_entities": len(entities),
                "num_relations": len(relations),
                "graph_density": 0.0,
                "key_entities": [],
            }
