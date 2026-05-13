"""
Protein graph builder.

Converts a :class:`~proTeye.data.pdb_loader.ProteinStructure` into a
graph representation suitable for message-passing neural networks.

Nodes  – one per residue.  Features: amino-acid one-hot + local geometry.
Edges  – k-nearest neighbours in Cα space.  Features: distance + unit vector.
"""

from __future__ import annotations

from typing import Dict, NamedTuple

import numpy as np
import torch

from proTeye.data.pdb_loader import NUM_AA_TYPES, ProteinStructure


class ProteinGraph(NamedTuple):
    """Tensor representation of a protein's graph.

    All tensors are on CPU and use float32 / int64 dtype.

    Attributes
    ----------
    node_features : Tensor, shape (N, node_dim)
        Per-residue feature vectors.
    ca_coords : Tensor, shape (N, 3)
        Cα coordinates (Å).
    edge_index : Tensor, shape (2, E)
        Column *i* holds the source and destination residue indices of
        edge *i*.  Edges are directed; both (i→j) and (j→i) are present.
    edge_features : Tensor, shape (E, edge_dim)
        Per-edge feature vectors.
    aa_indices : Tensor, shape (N,), dtype int64
        Integer amino-acid class per residue.
    """

    node_features: torch.Tensor
    ca_coords: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    aa_indices: torch.Tensor


class ProteinGraphBuilder:
    """Build kNN protein graphs from :class:`ProteinStructure` objects.

    Parameters
    ----------
    k :
        Number of nearest neighbours per residue.
    max_distance :
        Edges longer than *max_distance* (Å) are discarded even if the
        neighbour is within the top-k list.
    """

    def __init__(self, k: int = 10, max_distance: float = 15.0) -> None:
        self.k = k
        self.max_distance = max_distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, protein: ProteinStructure) -> ProteinGraph:
        """Convert *protein* to a :class:`ProteinGraph`.

        Parameters
        ----------
        protein :
            Parsed protein structure.

        Returns
        -------
        ProteinGraph
        """
        ca = protein.ca_coords  # (N, 3)
        n = ca.shape[0]

        # Replace any NaN Cα with the chain centroid so kNN still works.
        ca = _fill_nan_ca(ca)

        node_feats = self._node_features(protein, ca)
        edge_index, edge_feats = self._build_edges(ca)

        return ProteinGraph(
            node_features=torch.from_numpy(node_feats),
            ca_coords=torch.from_numpy(ca.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index),
            edge_features=torch.from_numpy(edge_feats),
            aa_indices=torch.from_numpy(protein.aa_indices),
        )

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _node_features(
        self, protein: ProteinStructure, ca: np.ndarray
    ) -> np.ndarray:
        """Concatenate amino-acid one-hot and local-geometry features.

        Returns array of shape (N, NUM_AA_TYPES + geometry_dim).
        """
        n = protein.num_residues

        # (1) One-hot amino-acid encoding
        aa_onehot = np.zeros((n, NUM_AA_TYPES), dtype=np.float32)
        aa_onehot[np.arange(n), protein.aa_indices] = 1.0

        # (2) Sequence position encoding (normalised to [0, 1])
        pos_enc = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).reshape(-1, 1)

        # (3) Backbone virtual bond lengths (Cα–Cα distances to neighbours)
        #     Using a simple ±1 window for local geometry.
        bond_feats = _local_ca_geometry(ca)  # (N, 2)

        return np.concatenate([aa_onehot, pos_enc, bond_feats], axis=-1)

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _build_edges(
        self, ca: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build directed kNN edges and their feature vectors.

        Returns
        -------
        edge_index : int64 array, shape (2, E)
        edge_features : float32 array, shape (E, edge_dim)
        """
        n = ca.shape[0]

        # Pairwise squared distances
        diff = ca[:, None, :] - ca[None, :, :]  # (N, N, 3)
        sq_dist = (diff ** 2).sum(-1)           # (N, N)

        # Self-loops get infinity so they are never chosen
        np.fill_diagonal(sq_dist, np.inf)

        k_eff = min(self.k, n - 1)
        src_list, dst_list, feat_list = [], [], []

        max_sq = self.max_distance ** 2
        for i in range(n):
            row = sq_dist[i]
            nn_idx = np.argpartition(row, k_eff)[:k_eff]
            nn_idx = nn_idx[row[nn_idx] <= max_sq]
            for j in nn_idx:
                d = float(np.sqrt(row[j]))
                unit = diff[i, j] / (d + 1e-8)         # (3,)
                src_list.append(i)
                dst_list.append(j)
                # edge features: [distance, Δsequence_position, unit_vector x3]
                seq_sep = float(abs(int(i) - int(j))) / max(n - 1, 1)
                feat_list.append(
                    np.array([d / self.max_distance, seq_sep, *unit],
                             dtype=np.float32)
                )

        if not src_list:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 5), dtype=np.float32)
        else:
            edge_index = np.array([src_list, dst_list], dtype=np.int64)  # (2, E)
            edge_features = np.stack(feat_list, axis=0)                  # (E, 5)

        return edge_index, edge_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_nan_ca(ca: np.ndarray) -> np.ndarray:
    """Replace NaN Cα positions with the chain centroid."""
    ca = ca.copy()
    nan_mask = np.isnan(ca).any(axis=-1)
    if nan_mask.all():
        ca[:] = 0.0
        return ca
    centroid = ca[~nan_mask].mean(axis=0)
    ca[nan_mask] = centroid
    return ca


def _local_ca_geometry(ca: np.ndarray) -> np.ndarray:
    """Compute virtual bond lengths to the previous and next Cα.

    Returns array of shape (N, 2).
    The values are normalised by a typical Cα–Cα virtual bond length (3.8 Å).
    """
    n = ca.shape[0]
    feats = np.zeros((n, 2), dtype=np.float32)
    if n < 2:
        return feats

    # Distance to previous residue
    diffs_prev = ca[1:] - ca[:-1]                        # (N-1, 3)
    dist_prev = np.linalg.norm(diffs_prev, axis=-1)       # (N-1,)
    feats[1:, 0] = dist_prev / 3.8

    # Distance to next residue (same values, shifted)
    feats[:-1, 1] = dist_prev / 3.8

    return feats
