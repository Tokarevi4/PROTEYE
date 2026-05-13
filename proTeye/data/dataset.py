"""
PyTorch Dataset and collation utilities for protein structures.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from proTeye.data.graph_builder import ProteinGraph, ProteinGraphBuilder
from proTeye.data.pdb_loader import PDBLoader, ProteinStructure


class ProteinDataset(Dataset):
    """Dataset of protein structures loaded from PDB files.

    Parameters
    ----------
    pdb_paths :
        List of paths to ``.pdb`` files.
    graph_builder :
        :class:`ProteinGraphBuilder` used to convert structures to graphs.
        If *None*, a default builder (k=10, max_distance=15 Å) is used.
    chain_id :
        If set, only the specified chain is loaded from each file.
    transform :
        Optional callable applied to each :class:`ProteinGraph` after
        construction.  Useful for data augmentation.
    """

    def __init__(
        self,
        pdb_paths: List[str],
        graph_builder: Optional[ProteinGraphBuilder] = None,
        chain_id: Optional[str] = None,
        transform: Optional[Callable[[ProteinGraph], ProteinGraph]] = None,
    ) -> None:
        self.pdb_paths = pdb_paths
        self.graph_builder = graph_builder or ProteinGraphBuilder()
        self.chain_id = chain_id
        self.transform = transform
        self._loader = PDBLoader()

        # Build index: list of (pdb_path, chain_id) tuples
        self._index: List[Tuple[str, str]] = self._build_index()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> ProteinGraph:
        pdb_path, chain = self._index[idx]
        structures = self._loader.load(pdb_path, chain_id=chain)
        if not structures:
            raise ValueError(
                f"No valid chain '{chain}' found in {pdb_path}"
            )
        protein = structures[0]
        graph = self.graph_builder.build(protein)
        if self.transform is not None:
            graph = self.transform(graph)
        return graph

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Tuple[str, str]]:
        """Discover available (file, chain) pairs."""
        loader = PDBLoader()
        index = []
        for path in self.pdb_paths:
            try:
                structures = loader.load(path, chain_id=self.chain_id)
                for protein in structures:
                    index.append((path, protein.chain_id))
            except Exception:  # noqa: BLE001
                pass  # skip unreadable files
        return index


def collate_fn(batch: List[ProteinGraph]) -> dict:
    """Collate a list of :class:`ProteinGraph` objects into a batched dict.

    Because protein graphs have variable sizes they cannot be naively
    stacked.  Instead, node tensors are concatenated and an additional
    ``batch_idx`` tensor records which graph each node belongs to.
    Edge indices are offset so that they refer to the correct positions
    in the concatenated node array.

    Returns
    -------
    dict with keys:
        * ``node_features``  – (total_N, node_dim)
        * ``ca_coords``      – (total_N, 3)
        * ``aa_indices``     – (total_N,)
        * ``edge_index``     – (2, total_E)
        * ``edge_features``  – (total_E, edge_dim)
        * ``batch_idx``      – (total_N,) – graph index per node
        * ``graph_sizes``    – (B,) – number of residues per graph
    """
    node_feats_list = []
    ca_coords_list = []
    aa_idx_list = []
    edge_index_list = []
    edge_feats_list = []
    batch_idx_list = []
    sizes = []

    offset = 0
    for graph_idx, graph in enumerate(batch):
        n = graph.node_features.shape[0]
        node_feats_list.append(graph.node_features)
        ca_coords_list.append(graph.ca_coords)
        aa_idx_list.append(graph.aa_indices)
        edge_index_list.append(graph.edge_index + offset)
        edge_feats_list.append(graph.edge_features)
        batch_idx_list.append(torch.full((n,), graph_idx, dtype=torch.long))
        sizes.append(n)
        offset += n

    return {
        "node_features": torch.cat(node_feats_list, dim=0),
        "ca_coords": torch.cat(ca_coords_list, dim=0),
        "aa_indices": torch.cat(aa_idx_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1) if edge_index_list and edge_index_list[0].shape[1] > 0
                      else torch.zeros((2, 0), dtype=torch.long),
        "edge_features": torch.cat(edge_feats_list, dim=0),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "graph_sizes": torch.tensor(sizes, dtype=torch.long),
    }
