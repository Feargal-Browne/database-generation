# Confidential - nvyra-x (c) 2025-2026

"""
FEVER + LIAR dataset preparation for GraphRAG-R1 agent training.
Builds entity graphs from evidence documents for supervised pre-training.

Usage (Colab T4):
    python prepare_fever.py --output_dir ./data --max_samples 50000
"""

import argparse
import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_fever(split: str = "train", max_samples: int = 50_000) -> List[Dict]:
    """Load FEVER dataset from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("fever/fever", "v1.0", split=split, trust_remote_code=True)

    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break
        label = row.get("label")
        if label is None:
            continue

        label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
        verdict = label_map.get(label, "NOT ENOUGH INFO")

        evidence_sents = row.get("evidence_sentence", [])
        evidence_pages = row.get("evidence_wiki_url", [])

        if not evidence_sents and verdict != "NOT ENOUGH INFO":
            continue

        samples.append({
            "claim": row["claim"],
            "verdict": verdict,
            "evidence_sentences": evidence_sents if isinstance(evidence_sents, list) else [evidence_sents],
            "evidence_pages": evidence_pages if isinstance(evidence_pages, list) else [evidence_pages],
        })

    return samples


def load_liar(split: str = "test", max_samples: int = 5000) -> List[Dict]:
    """Load LIAR dataset for validation."""
    from datasets import load_dataset

    ds = load_dataset("liar", split=split, trust_remote_code=True)

    label_map = {
        0: "pants-fire", 1: "false", 2: "barely-true",
        3: "half-true", 4: "mostly-true", 5: "true",
    }

    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break
        label_idx = row.get("label", 0)
        samples.append({
            "claim": row["statement"],
            "verdict": label_map.get(label_idx, "unknown"),
            "speaker": row.get("speaker", ""),
            "context": row.get("context", ""),
            "justification": row.get("justification", ""),
        })

    return samples


def extract_entities_simple(text: str) -> List[Dict]:
    """Extract entities using spaCy NER (lightweight, for training data prep)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text[:5000])
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.text.lower() not in seen:
            entities.append({
                "text": ent.text,
                "label": ent.label_.lower(),
                "score": 1.0,
            })
            seen.add(ent.text.lower())

    return entities


def build_entity_graph(
    claim: str,
    evidence_texts: List[str],
    claim_encoder,
) -> Dict:
    """Build an entity graph from claim + evidence for training."""
    combined = f"{claim}\n" + "\n".join(evidence_texts)
    entities = extract_entities_simple(combined)

    if len(entities) < 2:
        return None

    relations = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            for text in [claim] + evidence_texts:
                if e1["text"].lower() in text.lower() and e2["text"].lower() in text.lower():
                    relations.append({
                        "subject": e1["text"],
                        "predicate": "co_occurs_with",
                        "object": e2["text"],
                        "weight": 0.5,
                    })
                    break

    if not relations:
        return None

    node_texts = [f"{e['text']} ({e['label']})" for e in entities]
    node_embeddings = claim_encoder.encode(node_texts, convert_to_numpy=True)
    claim_embedding = claim_encoder.encode(claim, convert_to_numpy=True)

    return {
        "claim": claim,
        "claim_embedding": claim_embedding.tolist(),
        "entities": entities,
        "relations": relations,
        "node_embeddings": node_embeddings.tolist(),
        "num_entities": len(entities),
        "num_relations": len(relations),
    }


def build_fever_graphs(
    samples: List[Dict],
    claim_encoder,
    max_graphs: int = 10_000,
) -> Tuple[List[Dict], List[Dict]]:
    """Build training graphs from FEVER samples.

    Returns (graphs_with_evidence, graphs_without_evidence) for
    supervised pre-training (knows correct evidence) and RL fine-tuning.
    """
    supervised_graphs = []
    for sample in samples:
        if len(supervised_graphs) >= max_graphs:
            break

        evidence = sample.get("evidence_sentences", [])
        if not evidence:
            continue

        graph = build_entity_graph(
            sample["claim"],
            evidence,
            claim_encoder,
        )
        if graph is None:
            continue

        # Mark which entities appear in the evidence (ground truth for supervised)
        evidence_text_lower = " ".join(e.lower() for e in evidence)
        for entity in graph["entities"]:
            entity["in_evidence"] = entity["text"].lower() in evidence_text_lower

        graph["verdict"] = sample["verdict"]
        graph["evidence_texts"] = evidence
        supervised_graphs.append(graph)

    return supervised_graphs


def main():
    parser = argparse.ArgumentParser(description="Prepare FEVER/LIAR data for agent training")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--max_fever", type=int, default=50_000)
    parser.add_argument("--max_liar", type=int, default=5_000)
    parser.add_argument("--max_graphs", type=int, default=10_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    claim_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading FEVER dataset...")
    fever_samples = load_fever(split="train", max_samples=args.max_fever)
    print(f"  Loaded {len(fever_samples)} FEVER samples.")

    print("Building entity graphs from FEVER...")
    graphs = build_fever_graphs(fever_samples, claim_encoder, max_graphs=args.max_graphs)
    print(f"  Built {len(graphs)} training graphs.")

    fever_path = output_dir / "fever_graphs.jsonl"
    with open(fever_path, "w") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")
    print(f"  Saved to {fever_path}")

    print("Loading LIAR dataset...")
    liar_samples = load_liar(split="test", max_samples=args.max_liar)
    print(f"  Loaded {len(liar_samples)} LIAR samples.")

    liar_path = output_dir / "liar_validation.jsonl"
    with open(liar_path, "w") as f:
        for s in liar_samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved to {liar_path}")

    print(f"Data preparation complete.")
    print(f"  Training graphs: {len(graphs)}")
    print(f"  Validation samples: {len(liar_samples)}")


if __name__ == "__main__":
    main()
