# Confidential - nvyra-x (c) 2025-2026

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGENT_HIDDEN_DIM, AGENT_MAX_HOPS, AGENT_CLAIM_ENCODER


class GraphAttentionLayer(nn.Module):
    """Single GAT attention head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (num_nodes, in_dim)
        adj: (num_nodes, num_nodes) binary adjacency
        Returns: (num_nodes, out_dim)
        """
        Wh = self.W(h)  # (N, out_dim)
        N = Wh.size(0)

        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))

        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        return alpha @ Wh  # (N, out_dim)


class GraphPolicyNetwork(nn.Module):
    """2-layer GAT with action head for graph navigation."""

    def __init__(
        self,
        claim_dim: int = 384,
        node_dim: int = 384,
        hidden_dim: int = AGENT_HIDDEN_DIM,
        num_heads: int = 4,
    ):
        super().__init__()

        self.gat_heads_1 = nn.ModuleList([
            GraphAttentionLayer(node_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        self.gat_heads_2 = nn.ModuleList([
            GraphAttentionLayer(hidden_dim * num_heads, hidden_dim)
            for _ in range(num_heads)
        ])
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)

        self.claim_proj = nn.Linear(claim_dim, hidden_dim * num_heads)

        # claim + current_node + visited_aggregation
        state_dim = hidden_dim * num_heads * 3
        self.action_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.stop_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        claim_embed: torch.Tensor,
        current_node_idx: int,
        visited_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns action logits over (neighbor_edges + STOP).

        node_features: (N, node_dim)
        adj: (N, N) binary adjacency
        claim_embed: (claim_dim,)
        current_node_idx: int
        visited_mask: (N,) binary mask of visited nodes
        """
        h = torch.cat([head(node_features, adj) for head in self.gat_heads_1], dim=-1)
        h = self.norm1(h)
        h = F.elu(h)

        h = torch.cat([head(h, adj) for head in self.gat_heads_2], dim=-1)
        h = self.norm2(h)
        h = F.elu(h)

        # Build state vector
        claim_proj = self.claim_proj(claim_embed)
        current_node = h[current_node_idx]
        visited_agg = (h * visited_mask.unsqueeze(-1)).sum(dim=0)
        visited_count = visited_mask.sum().clamp(min=1)
        visited_agg = visited_agg / visited_count

        state = torch.cat([claim_proj, current_node, visited_agg])

        neighbors = adj[current_node_idx].nonzero(as_tuple=True)[0]
        if len(neighbors) == 0:
            # No neighbors, only STOP action
            stop_score = self.stop_head(state)
            return stop_score, torch.tensor([], device=node_features.device, dtype=torch.long)

        neighbor_states = []
        for n_idx in neighbors:
            n_state = torch.cat([claim_proj, h[n_idx], visited_agg])
            neighbor_states.append(n_state)

        neighbor_states = torch.stack(neighbor_states)
        edge_scores = self.action_head(neighbor_states).squeeze(-1)
        stop_score = self.stop_head(state).squeeze(-1)

        # [edge_0, edge_1, ..., STOP]
        action_logits = torch.cat([edge_scores, stop_score.unsqueeze(0)])

        return action_logits, neighbors


class GraphNavigationAgent:
    """RL-trained agent that navigates knowledge graphs to find evidence paths."""

    def __init__(self, device: str = "cuda", weights_path: Optional[str] = None):
        self.device = device

        from sentence_transformers import SentenceTransformer
        self.claim_encoder = SentenceTransformer(
            AGENT_CLAIM_ENCODER, device=device
        )
        self.claim_encoder.eval()
        for p in self.claim_encoder.parameters():
            p.requires_grad = False

        claim_dim = self.claim_encoder.get_sentence_embedding_dimension()

        self.policy = GraphPolicyNetwork(
            claim_dim=claim_dim,
            node_dim=claim_dim,
            hidden_dim=AGENT_HIDDEN_DIM,
        ).to(device)

        if weights_path:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
        self.policy.eval()

    def _encode_nodes(self, graph_dict: dict) -> Tuple[torch.Tensor, List[str]]:
        """Encode graph nodes using the claim encoder."""
        entities = graph_dict.get("entities", [])
        relations = graph_dict.get("relations", [])

        node_texts = []
        node_names = []
        for e in entities:
            node_texts.append(f"{e['text']} ({e['label']})")
            node_names.append(e["text"])

        if not node_texts:
            return torch.zeros(1, 384, device=self.device), ["empty"]

        embeddings = self.claim_encoder.encode(
            node_texts, convert_to_tensor=True, device=self.device
        )
        return embeddings, node_names

    def _build_adjacency(
        self, node_names: List[str], relations: List[dict]
    ) -> torch.Tensor:
        """Build adjacency matrix from relations."""
        N = len(node_names)
        adj = torch.zeros(N, N, device=self.device)
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        for r in relations:
            src = name_to_idx.get(r["subject"])
            dst = name_to_idx.get(r["object"])
            if src is not None and dst is not None:
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0  # undirected for navigation

        for i in range(N):
            adj[i, i] = 1.0

        return adj

    @torch.inference_mode()
    def navigate(
        self,
        claim_text: str,
        graph_dict: dict,
        max_hops: int = AGENT_MAX_HOPS,
    ) -> List[str]:
        """Navigate the graph to select an evidence path.

        Returns a list of entity names forming the selected path.
        """
        entities = graph_dict.get("entities", [])
        relations = graph_dict.get("relations", [])
        if len(entities) < 2 or not relations:
            return [e["text"] for e in entities[:5]]

        claim_embed = self.claim_encoder.encode(
            claim_text, convert_to_tensor=True, device=self.device
        )

        node_features, node_names = self._encode_nodes(graph_dict)
        adj = self._build_adjacency(node_names, relations)

        N = len(node_names)
        visited_mask = torch.zeros(N, device=self.device)
        path = []

        # Start at node most similar to the claim
        sims = F.cosine_similarity(
            claim_embed.unsqueeze(0), node_features, dim=1
        )
        current_idx = sims.argmax().item()
        visited_mask[current_idx] = 1.0
        path.append(node_names[current_idx])

        for _ in range(max_hops):
            action_logits, neighbors = self.policy(
                node_features, adj, claim_embed, current_idx, visited_mask
            )

            if len(neighbors) == 0:
                break

            # Greedy action selection
            action_idx = action_logits.argmax().item()

            if action_idx == len(neighbors):
                break

            next_node = neighbors[action_idx].item()
            if visited_mask[next_node] == 1.0:
                # Already visited, try next best
                sorted_actions = action_logits[:-1].argsort(descending=True)
                moved = False
                for alt in sorted_actions:
                    alt_node = neighbors[alt.item()].item()
                    if visited_mask[alt_node] == 0.0:
                        next_node = alt_node
                        moved = True
                        break
                if not moved:
                    break

            current_idx = next_node
            visited_mask[current_idx] = 1.0
            path.append(node_names[current_idx])

        return path

    def save_weights(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.policy.eval()
