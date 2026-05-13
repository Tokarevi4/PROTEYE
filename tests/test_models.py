"""
Tests for GNN, diffusion, and conformer model components.
"""

from __future__ import annotations

import pytest
import torch

from proTeye.data.graph_builder import ProteinGraphBuilder
from proTeye.models.conformer import ConformerGenerator
from proTeye.models.diffusion import (
    DenoisingNetwork,
    DiffusionSchedule,
    SinusoidalTimeEmbedding,
)
from proTeye.models.gnn import GNNLayer, ProteinGNNEncoder
from proTeye.training.losses import coordinate_rmsd, diffusion_loss

# Re-use the synthetic protein builder from test_data
from tests.test_data import make_synthetic_protein

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NODE_DIM = 24   # NUM_AA_TYPES (21) + 1 + 2 = 24
EDGE_DIM = 5
HIDDEN  = 32


def make_graph(n: int = 12, k: int = 5, seed: int = 0):
    protein = make_synthetic_protein(n, seed=seed)
    builder = ProteinGraphBuilder(k=k)
    return builder.build(protein)


# ---------------------------------------------------------------------------
# GNNLayer
# ---------------------------------------------------------------------------

class TestGNNLayer:
    def test_output_shape(self):
        layer = GNNLayer(node_dim=HIDDEN, edge_dim=HIDDEN, hidden_dim=HIDDEN)
        graph = make_graph()
        n = graph.node_features.shape[0]
        h = torch.randn(n, HIDDEN)
        # Project edges to same hidden dim
        e = torch.randn(graph.edge_features.shape[0], HIDDEN)
        h_out = layer(h, graph.edge_index, e)
        assert h_out.shape == (n, HIDDEN)

    def test_no_edges(self):
        """Layer should handle an empty edge set without raising."""
        layer = GNNLayer(node_dim=HIDDEN, edge_dim=HIDDEN, hidden_dim=HIDDEN)
        n = 5
        h = torch.randn(n, HIDDEN)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, HIDDEN))
        h_out = layer(h, edge_index, edge_attr)
        assert h_out.shape == (n, HIDDEN)


# ---------------------------------------------------------------------------
# ProteinGNNEncoder
# ---------------------------------------------------------------------------

class TestProteinGNNEncoder:
    def test_output_shape(self):
        encoder = ProteinGNNEncoder(
            node_input_dim=NODE_DIM,
            edge_input_dim=EDGE_DIM,
            hidden_dim=HIDDEN,
            num_layers=2,
            output_dim=HIDDEN,
        )
        graph = make_graph()
        n = graph.node_features.shape[0]
        out = encoder(graph.node_features, graph.edge_index, graph.edge_features)
        assert out.shape == (n, HIDDEN)

    def test_gradient_flows(self):
        encoder = ProteinGNNEncoder(
            node_input_dim=NODE_DIM,
            edge_input_dim=EDGE_DIM,
            hidden_dim=HIDDEN,
            num_layers=2,
            output_dim=HIDDEN,
        )
        graph = make_graph()
        out = encoder(graph.node_features, graph.edge_index, graph.edge_features)
        loss = out.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# DiffusionSchedule
# ---------------------------------------------------------------------------

class TestDiffusionSchedule:
    def test_q_sample_shape(self):
        schedule = DiffusionSchedule(num_steps=50)
        x0 = torch.randn(10, 3)
        t = torch.tensor([25])
        x_t, eps = schedule.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert eps.shape == x0.shape

    def test_q_sample_t0_close_to_x0(self):
        """At t=0 (very small noise), x_t should be close to x0."""
        schedule = DiffusionSchedule(num_steps=50, beta_start=1e-8, beta_end=1e-7)
        x0 = torch.randn(20, 3)
        t = torch.tensor([0])
        x_t, _ = schedule.q_sample(x0, t)
        assert torch.allclose(x_t, x0, atol=1e-3)

    def test_q_sample_large_t_mostly_noise(self):
        """At t=T-1 (max noise), x_t should be mostly Gaussian noise."""
        schedule = DiffusionSchedule(num_steps=100)
        x0 = torch.zeros(50, 3)  # all zeros
        t = torch.tensor([99])
        x_t, eps = schedule.q_sample(x0, t)
        # x_t should be approximately eps (since alpha_bar ≈ 0)
        alpha_bar = schedule.alpha_bars[99]
        expected = schedule.sqrt_one_minus_alpha_bars[99] * eps
        assert torch.allclose(x_t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(embed_dim=64)
        t = torch.tensor([0, 10, 50, 99])
        out = emb(t)
        assert out.shape == (4, 64)

    def test_different_steps_different_embeds(self):
        emb = SinusoidalTimeEmbedding(embed_dim=32)
        t1 = torch.tensor([0])
        t2 = torch.tensor([50])
        assert not torch.allclose(emb(t1), emb(t2))


# ---------------------------------------------------------------------------
# DenoisingNetwork
# ---------------------------------------------------------------------------

class TestDenoisingNetwork:
    def test_output_shape(self):
        net = DenoisingNetwork(
            coord_dim=3,
            cond_dim=HIDDEN,
            edge_dim=EDGE_DIM,
            hidden_dim=HIDDEN,
            num_layers=2,
        )
        graph = make_graph(10)
        n = graph.ca_coords.shape[0]
        x_t = torch.randn(n, 3)
        cond = torch.randn(n, HIDDEN)
        t = torch.tensor([5])
        eps_pred = net(x_t, t, cond, graph.edge_index, graph.edge_features)
        assert eps_pred.shape == (n, 3)


# ---------------------------------------------------------------------------
# ConformerGenerator (end-to-end)
# ---------------------------------------------------------------------------

class TestConformerGenerator:
    def _make_model(self):
        return ConformerGenerator(
            node_input_dim=NODE_DIM,
            edge_input_dim=EDGE_DIM,
            hidden_dim=HIDDEN,
            encoder_layers=2,
            denoiser_layers=2,
            num_diffusion_steps=10,
        )

    def test_forward_shapes(self):
        model = self._make_model()
        graph = make_graph(12)
        eps_pred, eps_true = model(graph)
        assert eps_pred.shape == eps_true.shape
        assert eps_pred.shape == (12, 3)

    def test_forward_loss_finite(self):
        model = self._make_model()
        graph = make_graph(12)
        eps_pred, eps_true = model(graph)
        loss = diffusion_loss(eps_pred, eps_true)
        assert torch.isfinite(loss)

    def test_generate_shapes(self):
        model = self._make_model()
        graph = make_graph(8)
        conformations = model.generate(graph, num_samples=3)
        assert len(conformations) == 3
        for conf in conformations:
            assert conf.shape == (8, 3)

    def test_generate_structures(self):
        from tests.test_data import make_synthetic_protein
        model = self._make_model()
        protein = make_synthetic_protein(8)
        builder = ProteinGraphBuilder(k=4)
        graph = builder.build(protein)
        structs = model.generate_structures(graph, protein, num_samples=2)
        assert len(structs) == 2
        for s in structs:
            assert s.num_residues == 8


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLosses:
    def test_diffusion_loss_zero(self):
        x = torch.randn(10, 3)
        loss = diffusion_loss(x, x)
        assert loss.item() < 1e-10

    def test_diffusion_loss_positive(self):
        pred = torch.randn(10, 3)
        true = torch.randn(10, 3)
        loss = diffusion_loss(pred, true)
        assert loss.item() >= 0.0

    def test_coordinate_rmsd_zero(self):
        coords = torch.randn(10, 3)
        rmsd = coordinate_rmsd(coords, coords)
        assert rmsd.item() < 1e-5

    def test_coordinate_rmsd_with_align(self):
        coords = torch.randn(15, 3)
        # Translate by 5 Å; after alignment RMSD should still be ~0
        shifted = coords + 5.0
        rmsd = coordinate_rmsd(shifted, coords, align=True)
        assert rmsd.item() < 1e-3
