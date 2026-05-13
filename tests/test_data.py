"""
Tests for data loading and graph building.

PDB-related tests use synthetic coordinate data so that BioPython is not
required to have actual PDB files on disk.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from proTeye.data.graph_builder import ProteinGraphBuilder, _fill_nan_ca, _local_ca_geometry
from proTeye.data.pdb_loader import AA_CODES, NUM_AA_TYPES, PDBLoader, ProteinStructure


# ---------------------------------------------------------------------------
# Helpers – synthetic ProteinStructure
# ---------------------------------------------------------------------------

def make_synthetic_protein(n: int = 15, seed: int = 0) -> ProteinStructure:
    """Return a ProteinStructure with random backbone coordinates."""
    rng = np.random.default_rng(seed)
    coords = np.zeros((n, 4, 3), dtype=np.float32)
    # Build a chain along the x-axis with small perturbations
    for i in range(n):
        base = np.array([i * 3.8, 0.0, 0.0], dtype=np.float32)
        coords[i, 0] = base + rng.normal(scale=0.1, size=3)  # N
        coords[i, 1] = base + np.array([1.5, 0.0, 0.0]) + rng.normal(scale=0.1, size=3)  # CA
        coords[i, 2] = base + np.array([3.0, 0.5, 0.0]) + rng.normal(scale=0.1, size=3)  # C
        coords[i, 3] = base + np.array([3.0, 0.5, 1.2]) + rng.normal(scale=0.1, size=3)  # O

    sequence = list(AA_CODES.keys())[:n % len(AA_CODES)] + ["GLY"] * (n - n % len(AA_CODES))
    sequence = sequence[:n]
    aa_indices = np.array([AA_CODES.get(r, 20) for r in sequence], dtype=np.int64)

    return ProteinStructure(
        name="test_protein",
        sequence=sequence,
        aa_indices=aa_indices,
        coords=coords,
        chain_id="A",
    )


# ---------------------------------------------------------------------------
# PDBLoader – from_coords
# ---------------------------------------------------------------------------

class TestProteinStructure:
    def test_num_residues(self):
        p = make_synthetic_protein(10)
        assert p.num_residues == 10

    def test_ca_coords_shape(self):
        p = make_synthetic_protein(8)
        assert p.ca_coords.shape == (8, 3)

    def test_has_missing_atoms_false(self):
        p = make_synthetic_protein(5)
        assert not p.has_missing_atoms()

    def test_has_missing_atoms_true(self):
        p = make_synthetic_protein(5)
        p.coords[2, 0] = np.nan  # introduce a NaN
        assert p.has_missing_atoms()


class TestPDBLoaderFromCoords:
    def test_ca_only_input(self):
        ca = np.random.randn(12, 3).astype(np.float32)
        ps = PDBLoader.from_coords(ca, name="test")
        assert ps.num_residues == 12
        assert ps.coords.shape == (12, 4, 3)
        # Only CA slot should be non-NaN
        assert not np.isnan(ps.coords[:, 1, :]).any()
        # N, C, O slots should be NaN
        assert np.isnan(ps.coords[:, 0, :]).all()

    def test_full_backbone_input(self):
        coords = np.random.randn(5, 4, 3).astype(np.float32)
        ps = PDBLoader.from_coords(coords, sequence=["ALA"] * 5)
        assert ps.coords.shape == (5, 4, 3)

    def test_default_gly_sequence(self):
        ca = np.zeros((7, 3), dtype=np.float32)
        ps = PDBLoader.from_coords(ca)
        assert all(r == "GLY" for r in ps.sequence)


# ---------------------------------------------------------------------------
# ProteinGraphBuilder
# ---------------------------------------------------------------------------

class TestProteinGraphBuilder:
    def test_graph_shapes(self):
        protein = make_synthetic_protein(15)
        builder = ProteinGraphBuilder(k=5)
        graph = builder.build(protein)

        n = protein.num_residues
        assert graph.node_features.shape[0] == n
        assert graph.ca_coords.shape == (n, 3)
        assert graph.aa_indices.shape == (n,)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_features.shape[1] == 5  # dist, seq_sep, unit_x3

    def test_no_self_loops(self):
        protein = make_synthetic_protein(10)
        builder = ProteinGraphBuilder(k=4)
        graph = builder.build(protein)
        src, dst = graph.edge_index
        assert (src == dst).sum().item() == 0, "Self-loops detected"

    def test_node_feature_dim(self):
        protein = make_synthetic_protein(10)
        builder = ProteinGraphBuilder()
        graph = builder.build(protein)
        # NUM_AA_TYPES (21) + 1 (pos enc) + 2 (local geometry)
        expected_dim = NUM_AA_TYPES + 1 + 2
        assert graph.node_features.shape[1] == expected_dim

    def test_single_residue_no_crash(self):
        protein = make_synthetic_protein(1)
        builder = ProteinGraphBuilder(k=5)
        graph = builder.build(protein)
        assert graph.node_features.shape[0] == 1

    def test_all_nan_ca_no_crash(self):
        """Graph building should not raise even when all Cα are NaN."""
        protein = make_synthetic_protein(5)
        protein.coords[:] = np.nan
        builder = ProteinGraphBuilder(k=3)
        graph = builder.build(protein)
        assert graph.ca_coords.shape == (5, 3)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_fill_nan_ca_replaces_nan(self):
        ca = np.array([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
        filled = _fill_nan_ca(ca)
        assert not np.isnan(filled).any()

    def test_local_ca_geometry_shape(self):
        ca = np.random.randn(8, 3).astype(np.float32)
        feats = _local_ca_geometry(ca)
        assert feats.shape == (8, 2)

    def test_local_ca_geometry_single_residue(self):
        ca = np.random.randn(1, 3).astype(np.float32)
        feats = _local_ca_geometry(ca)
        assert feats.shape == (1, 2)
        assert (feats == 0.0).all()
