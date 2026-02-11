# Confidential - nvyra-x (c) 2025-2026

"""
PPO training loop for the GraphRAG-R1 navigation agent.
Runs on Colab T4 (~1M trainable params, <2 hours total).

Phase 1: Supervised pre-training on FEVER (cross-entropy on correct edges)
Phase 2: RL fine-tuning with PPO (reward from Nemotron-30B)

Usage:
    python train_agent.py \
        --data_dir ./data \
        --output_dir ./checkpoints \
        --phase 1 \
        --epochs 5
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_graphs(data_dir: str) -> List[Dict]:
    """Load pre-built graphs from JSONL."""
    path = Path(data_dir) / "fever_graphs.jsonl"
    graphs = []
    with open(path) as f:
        for line in f:
            graphs.append(json.loads(line))
    return graphs


def build_adjacency(node_names: List[str], relations: List[Dict], device: str) -> torch.Tensor:
    """Build adjacency matrix from relations."""
    N = len(node_names)
    adj = torch.zeros(N, N, device=device)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    for r in relations:
        src = name_to_idx.get(r["subject"])
        dst = name_to_idx.get(r["object"])
        if src is not None and dst is not None:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0

    for i in range(N):
        adj[i, i] = 1.0

    return adj


def supervised_step(
    policy,
    graph: Dict,
    device: str,
) -> torch.Tensor:
    """One supervised training step: cross-entropy on edges leading to evidence nodes."""
    entities = graph["entities"]
    relations = graph["relations"]
    node_embeddings = torch.tensor(graph["node_embeddings"], device=device, dtype=torch.float32)
    claim_embed = torch.tensor(graph["claim_embedding"], device=device, dtype=torch.float32)

    node_names = [e["text"] for e in entities]
    adj = build_adjacency(node_names, relations, device)

    evidence_mask = torch.tensor(
        [1.0 if e.get("in_evidence", False) else 0.0 for e in entities],
        device=device,
    )

    if evidence_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    sims = F.cosine_similarity(claim_embed.unsqueeze(0), node_embeddings, dim=1)
    current_idx = sims.argmax().item()

    visited_mask = torch.zeros(len(node_names), device=device)
    visited_mask[current_idx] = 1.0

    total_loss = torch.tensor(0.0, device=device)
    steps = 0

    for _ in range(3):  # max hops
        action_logits, neighbors = policy(
            node_embeddings, adj, claim_embed, current_idx, visited_mask
        )

        if len(neighbors) == 0:
            break

        # Prefer neighbors in evidence set
        neighbor_evidence = evidence_mask[neighbors]
        if neighbor_evidence.sum() == 0:
            target = torch.tensor(len(neighbors), device=device, dtype=torch.long)
        else:
            # Pick the evidence neighbor most similar to the claim
            evidence_sims = sims[neighbors] * neighbor_evidence
            target = evidence_sims.argmax()

        loss = F.cross_entropy(action_logits.unsqueeze(0), target.unsqueeze(0))
        total_loss = total_loss + loss
        steps += 1

        if target.item() < len(neighbors):
            next_idx = neighbors[target.item()].item()
            visited_mask[next_idx] = 1.0
            current_idx = next_idx
        else:
            break

    return total_loss / max(steps, 1)


class PPOBuffer:
    """Simple buffer for PPO rollout data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

    def add(self, state, action, log_prob, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns


def ppo_step(
    policy,
    optimizer,
    buffer: PPOBuffer,
    clip_epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
) -> float:
    """One PPO update step."""
    if not buffer.log_probs:
        return 0.0

    returns = buffer.compute_returns()
    old_log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values)

    advantages = returns - values.detach()

    ratio = torch.exp(old_log_probs - old_log_probs.detach())
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    value_loss = F.mse_loss(values, returns)

    loss = policy_loss + value_coeff * value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train_phase1(
    policy,
    graphs: List[Dict],
    device: str,
    epochs: int = 5,
    lr: float = 1e-3,
    output_dir: str = "./checkpoints",
):
    """Phase 1: Supervised pre-training on FEVER evidence paths."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(graphs)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        random.shuffle(graphs)
        total_loss = 0.0
        count = 0

        for i, graph in enumerate(graphs):
            if len(graph["entities"]) < 2:
                continue

            loss = supervised_step(policy, graph, device)
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                count += 1

            if (i + 1) % 500 == 0:
                avg = total_loss / max(count, 1)
                print(f"  Epoch {epoch+1}, step {i+1}/{len(graphs)}, loss={avg:.4f}")

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}/{epochs} complete, avg_loss={avg_loss:.4f}")

        ckpt_path = output_path / f"phase1_epoch{epoch+1}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")

    final_path = output_path / "policy.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Phase 1 complete. Final weights: {final_path}")


def train_phase2(
    policy,
    graphs: List[Dict],
    reward_model,
    device: str,
    epochs: int = 3,
    lr: float = 3e-4,
    output_dir: str = "./checkpoints",
):
    """Phase 2: RL fine-tuning with PPO using Nemotron-30B rewards."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    buffer = PPOBuffer()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        random.shuffle(graphs)
        total_reward = 0.0
        count = 0

        for i, graph in enumerate(graphs):
            if len(graph["entities"]) < 2:
                continue

            entities = graph["entities"]
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
            visited_mask = torch.zeros(len(node_names), device=device)
            visited_mask[current_idx] = 1.0
            path = [node_names[current_idx]]

            for _ in range(3):
                action_logits, neighbors = policy(
                    node_embeddings, adj, claim_embed, current_idx, visited_mask
                )

                if len(neighbors) == 0:
                    break

                probs = F.softmax(action_logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Value estimate from mean logits
                value = action_logits.mean()

                if action.item() == len(neighbors):
                    buffer.add(None, action, log_prob, 0.0, value)
                    break

                next_idx = neighbors[action.item()].item()
                visited_mask[next_idx] = 1.0
                current_idx = next_idx
                path.append(node_names[current_idx])

                buffer.add(None, action, log_prob, 0.0, value)

            evidence_texts = graph.get("evidence_texts", [])
            agent_evidence = " ".join(
                t for t in evidence_texts
                if any(p.lower() in t.lower() for p in path)
            )
            if not agent_evidence:
                agent_evidence = " ".join(evidence_texts[:2])

            reward = reward_model.score(graph["claim"], agent_evidence)
            total_reward += reward
            count += 1

            # Assign terminal reward
            if buffer.rewards:
                buffer.rewards[-1] = reward

            # PPO update every 32 rollouts
            if count % 32 == 0 and buffer.log_probs:
                loss = ppo_step(policy, optimizer, buffer)
                buffer.clear()
                avg_r = total_reward / count
                print(f"  Epoch {epoch+1}, step {count}, "
                      f"avg_reward={avg_r:.3f}, ppo_loss={loss:.4f}")

        # Final PPO update for remaining buffer
        if buffer.log_probs:
            ppo_step(policy, optimizer, buffer)
            buffer.clear()

        avg_reward = total_reward / max(count, 1)
        print(f"Epoch {epoch+1}/{epochs} complete, avg_reward={avg_reward:.3f}")

        ckpt_path = output_path / f"phase2_epoch{epoch+1}.pt"
        torch.save(policy.state_dict(), ckpt_path)

    final_path = output_path / "policy.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Phase 2 complete. Final weights: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GraphRAG-R1 agent")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    print("Loading training graphs...")
    graphs = load_graphs(args.data_dir)
    print(f"  Loaded {len(graphs)} graphs.")

    if args.phase == 1:
        print("Starting Phase 1: Supervised pre-training on FEVER...")
        policy.train()
        train_phase1(
            policy, graphs, device,
            epochs=args.epochs, lr=args.lr,
            output_dir=args.output_dir,
        )

    elif args.phase == 2:
        phase1_path = Path(args.output_dir) / "policy.pt"
        if phase1_path.exists():
            policy.load_state_dict(
                torch.load(phase1_path, map_location=device, weights_only=True)
            )
            print(f"Loaded Phase 1 weights from {phase1_path}")

        from training.reward_model import RewardModel
        reward_model = RewardModel(
            mode="api" if args.reward_api_url else "local",
            api_url=args.reward_api_url,
        )

        print("Starting Phase 2: RL fine-tuning with PPO...")
        policy.train()
        train_phase2(
            policy, graphs, reward_model, device,
            epochs=args.epochs,
            lr=args.lr if args.lr != 1e-3 else 3e-4,
            output_dir=args.output_dir,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
