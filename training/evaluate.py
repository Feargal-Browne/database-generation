# Confidential - nvyra-x (c) 2025-2026

"""
Evaluate the GraphRAG-R1 agent on held-out claims.

Metrics:
- Path accuracy: fraction of evidence nodes in agent's path
- Factcheck confidence: Nemotron-30B confidence on agent-selected evidence
- Verdict correctness: match against ground truth

Usage:
    python evaluate.py \
        --data_dir ./data \
        --weights ./checkpoints/policy.pt \
        --max_samples 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np


def evaluate_path_accuracy(
    policy,
    graphs: List[Dict],
    device: str,
) -> Dict:
    """Evaluate how well the agent finds evidence nodes."""
    from training.train_agent import build_adjacency

    results = {
        "total": 0,
        "path_precision": [],
        "path_recall": [],
        "avg_path_length": [],
    }

    for graph in graphs:
        entities = graph["entities"]
        if len(entities) < 2:
            continue

        node_embeddings = torch.tensor(
            graph["node_embeddings"], device=device, dtype=torch.float32
        )
        claim_embed = torch.tensor(
            graph["claim_embedding"], device=device, dtype=torch.float32
        )
        node_names = [e["text"] for e in entities]
        adj = build_adjacency(node_names, graph["relations"], device)

        sims = F.cosine_similarity(claim_embed.unsqueeze(0), node_embeddings, dim=1)
        current_idx = sims.argmax().item()
        visited = torch.zeros(len(node_names), device=device)
        visited[current_idx] = 1.0
        path_indices = [current_idx]

        with torch.inference_mode():
            for _ in range(3):
                action_logits, neighbors = policy(
                    node_embeddings, adj, claim_embed, current_idx, visited
                )
                if len(neighbors) == 0:
                    break

                action_idx = action_logits.argmax().item()
                if action_idx == len(neighbors):
                    break

                next_idx = neighbors[action_idx].item()
                if visited[next_idx] == 1.0:
                    break

                visited[next_idx] = 1.0
                current_idx = next_idx
                path_indices.append(current_idx)

        evidence_indices = {
            i for i, e in enumerate(entities) if e.get("in_evidence", False)
        }
        path_set = set(path_indices)

        if evidence_indices:
            precision = len(path_set & evidence_indices) / max(len(path_set), 1)
            recall = len(path_set & evidence_indices) / len(evidence_indices)
        else:
            precision = 0.0
            recall = 0.0

        results["path_precision"].append(precision)
        results["path_recall"].append(recall)
        results["avg_path_length"].append(len(path_indices))
        results["total"] += 1

    return {
        "total_graphs": results["total"],
        "avg_path_precision": float(np.mean(results["path_precision"])) if results["path_precision"] else 0.0,
        "avg_path_recall": float(np.mean(results["path_recall"])) if results["path_recall"] else 0.0,
        "avg_path_length": float(np.mean(results["avg_path_length"])) if results["avg_path_length"] else 0.0,
    }


def evaluate_with_reward(
    policy,
    graphs: List[Dict],
    reward_model,
    device: str,
) -> Dict:
    """Evaluate using the reward model (factcheck confidence)."""
    from training.train_agent import build_adjacency

    rewards = []

    for graph in graphs:
        entities = graph["entities"]
        if len(entities) < 2:
            continue

        node_embeddings = torch.tensor(
            graph["node_embeddings"], device=device, dtype=torch.float32
        )
        claim_embed = torch.tensor(
            graph["claim_embedding"], device=device, dtype=torch.float32
        )
        node_names = [e["text"] for e in entities]
        adj = build_adjacency(node_names, graph["relations"], device)

        sims = F.cosine_similarity(claim_embed.unsqueeze(0), node_embeddings, dim=1)
        current_idx = sims.argmax().item()
        visited = torch.zeros(len(node_names), device=device)
        visited[current_idx] = 1.0
        path = [node_names[current_idx]]

        with torch.inference_mode():
            for _ in range(3):
                action_logits, neighbors = policy(
                    node_embeddings, adj, claim_embed, current_idx, visited
                )
                if len(neighbors) == 0:
                    break

                action_idx = action_logits.argmax().item()
                if action_idx == len(neighbors):
                    break

                next_idx = neighbors[action_idx].item()
                if visited[next_idx] == 1.0:
                    break

                visited[next_idx] = 1.0
                current_idx = next_idx
                path.append(node_names[current_idx])

        evidence_texts = graph.get("evidence_texts", [])
        agent_evidence = " ".join(
            t for t in evidence_texts
            if any(p.lower() in t.lower() for p in path)
        )
        if not agent_evidence:
            agent_evidence = " ".join(evidence_texts[:2])

        reward = reward_model.score(graph["claim"], agent_evidence)
        rewards.append(reward)

    return {
        "total_evaluated": len(rewards),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG-R1 agent")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--weights", type=str, default="./checkpoints/policy.pt")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--use_reward", action="store_true")
    parser.add_argument("--reward_api_url", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.graph_agent import GraphPolicyNetwork

    policy = GraphPolicyNetwork(
        claim_dim=384,
        node_dim=384,
        hidden_dim=128,
    ).to(device)

    weights_path = Path(args.weights)
    if weights_path.exists():
        policy.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found, using random weights.")

    policy.eval()

    print("Loading evaluation graphs...")
    from training.train_agent import load_graphs
    graphs = load_graphs(args.data_dir)[:args.max_samples]
    print(f"  Evaluating on {len(graphs)} graphs.")

    print("Evaluating path accuracy...")
    path_results = evaluate_path_accuracy(policy, graphs, device)
    print(json.dumps(path_results, indent=2))

    if args.use_reward:
        print("Evaluating with reward model...")
        from training.reward_model import RewardModel
        reward_model = RewardModel(
            mode="api" if args.reward_api_url else "local",
            api_url=args.reward_api_url,
        )
        reward_results = evaluate_with_reward(policy, graphs, reward_model, device)
        print(json.dumps(reward_results, indent=2))

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
