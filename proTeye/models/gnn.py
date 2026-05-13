"""
Geometric Graph Neural Network for protein structure encoding.

Implements a message-passing network that aggregates neighbourhood
information in 3D space.  All operations are invariant to global rigid
transformations because edge features are expressed as distances and
normalised direction vectors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    """Single message-passing layer.

    Each node aggregates messages from its neighbours, where a message is
    a function of the sender's features and the shared edge features.

    Parameters
    ----------
    node_dim :
        Dimensionality of node feature vectors.
    edge_dim :
        Dimensionality of edge feature vectors.
    hidden_dim :
        Hidden dimension for the message and update MLPs.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        # Message MLP: (h_i ⊕ h_j ⊕ e_ij) → message
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Update MLP: (h_i ⊕ agg_i) → h_i'
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (N, node_dim)
        edge_index : (2, E)
        edge_attr : (E, edge_dim)

        Returns
        -------
        h' : (N, node_dim)
        """
        src, dst = edge_index[0], edge_index[1]  # both (E,)
        n = h.shape[0]

        if edge_index.shape[1] == 0:
            # No edges: skip aggregation, apply identity update
            return self.norm(h + self.update_mlp(
                torch.cat([h, torch.zeros(n, self.message_mlp[0].out_features,
                                          device=h.device)], dim=-1)
            ))

        # --- Message computation -------------------------------------------
        h_src = h[src]   # (E, node_dim)
        h_dst = h[dst]   # (E, node_dim)
        msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # (E, ...)
        messages = self.message_mlp(msg_input)                     # (E, hidden)

        # --- Aggregation (mean) -------------------------------------------
        agg = torch.zeros(n, messages.shape[-1], device=h.device, dtype=h.dtype)
        count = torch.zeros(n, 1, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(1),
                           torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        count = count.clamp(min=1)
        agg = agg / count

        # --- Update -------------------------------------------------------
        update_input = torch.cat([h, agg], dim=-1)
        h_new = h + self.update_mlp(update_input)  # residual connection
        return self.norm(h_new)


class ProteinGNNEncoder(nn.Module):
    """Multi-layer GNN encoder for protein graphs.

    Produces a latent embedding vector for each residue.

    Parameters
    ----------
    node_input_dim :
        Dimension of raw node features from the graph builder.
    edge_input_dim :
        Dimension of raw edge features from the graph builder.
    hidden_dim :
        Internal representation dimension.
    num_layers :
        Number of message-passing layers.
    output_dim :
        Dimension of the per-residue embedding.
    dropout :
        Dropout probability applied between layers.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Input projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)

        # Message-passing layers
        self.layers = nn.ModuleList(
            [
                GNNLayer(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the protein graph into per-residue embeddings.

        Parameters
        ----------
        node_features : (N, node_input_dim)
        edge_index : (2, E)
        edge_features : (E, edge_input_dim)

        Returns
        -------
        embeddings : (N, output_dim)
        """
        h = F.silu(self.node_proj(node_features))      # (N, hidden)
        e = F.silu(self.edge_proj(edge_features))       # (E, hidden)

        for layer in self.layers:
            h = self.dropout(layer(h, edge_index, e))

        return self.out_proj(h)  # (N, output_dim)
