"""
Denoising Diffusion Probabilistic Model (DDPM) for protein coordinates.

Implements the DDPM framework (Ho et al., 2020) applied to Cα coordinate
trajectories, conditioned on per-residue GNN embeddings.

Forward process
    q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)

Reverse process
    p_θ(x_{t-1} | x_t) learned by predicting the noise ε at each step.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """Linear noise schedule for DDPM.

    Parameters
    ----------
    num_steps :
        Total number of diffusion steps *T*.
    beta_start :
        Noise level β at the first step.
    beta_end :
        Noise level β at the last step.
    device :
        Torch device.
    """

    def __init__(
        self,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.T = num_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

        # Quantities needed for the reverse step
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
        self.posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t given x_0 and time step t (closed-form).

        Parameters
        ----------
        x0 : (N, 3) or (B, N, 3)
            Clean coordinates.
        t : (B,) or scalar
            Time indices in [0, T-1].
        noise :
            Pre-sampled Gaussian noise; drawn fresh if *None*.

        Returns
        -------
        x_t : same shape as x0
        noise : same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t]

        # Broadcast over coordinate dimensions
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1m_ab = sqrt_1m_ab.unsqueeze(-1)

        x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse diffusion step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model: "DenoisingNetwork",
        x_t: torch.Tensor,
        t: int,
        condition: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse-diffusion step: x_t → x_{t-1}.

        Parameters
        ----------
        model :
            Trained denoising network.
        x_t : (N, 3)
            Noisy coordinates at step *t*.
        t : int
            Current time step index.
        condition : (N, embed_dim)
            Per-residue conditioning embeddings from the GNN encoder.
        edge_index : (2, E)
        edge_features : (E, edge_dim)

        Returns
        -------
        x_{t-1} : (N, 3)
        """
        t_tensor = torch.tensor([t], device=x_t.device)
        eps_pred = model(x_t, t_tensor, condition, edge_index, edge_features)

        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t]

        # Predicted x0
        x0_pred = (x_t - sqrt_1m_ab * eps_pred) / alpha_bar.sqrt()

        # Posterior mean
        coeff1 = beta * self.alpha_bars[t - 1].sqrt() / (1.0 - alpha_bar) if t > 0 else 0.0
        coeff2 = (1.0 - self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x_t.device)) * alpha.sqrt() / (1.0 - alpha_bar)
        mu = coeff1 * x0_pred + coeff2 * x_t if t > 0 else x0_pred

        if t == 0:
            return mu

        noise = torch.randn_like(x_t)
        std = self.posterior_variance[t].sqrt()
        return mu + std * noise

    @torch.no_grad()
    def sample(
        self,
        model: "DenoisingNetwork",
        n_residues: int,
        condition: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full reverse diffusion chain: Gaussian noise → coordinates.

        Parameters
        ----------
        model : DenoisingNetwork
        n_residues : int
        condition : (N, embed_dim)
        edge_index : (2, E)
        edge_features : (E, edge_dim)
        device : target device

        Returns
        -------
        x0 : (N, 3) – generated Cα coordinates
        """
        device = device or condition.device
        x = torch.randn(n_residues, 3, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample_step(
                model, x, t, condition, edge_index, edge_features
            )
        return x


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion time steps."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) integer time indices

        Returns
        -------
        embedding : (B, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class DenoisingNetwork(nn.Module):
    """GNN-based denoising network for diffusion on Cα coordinates.

    Predicts the noise ε from (x_t, t, conditioning embeddings).

    Architecture
    ------------
    1. Project noisy coordinates + per-residue conditioning + time embedding
       to a common hidden dimension.
    2. Several message-passing layers to model inter-residue dependencies.
    3. Linear head that predicts the noise vector for each residue.

    Parameters
    ----------
    coord_dim :
        Dimensionality of coordinates (3 for 3D Cα).
    cond_dim :
        Dimensionality of the conditioning embedding from the GNN encoder.
    edge_dim :
        Dimensionality of edge features.
    hidden_dim :
        Hidden dimension for all internal layers.
    num_layers :
        Number of message-passing layers.
    time_embed_dim :
        Dimension of sinusoidal time embedding.
    """

    def __init__(
        self,
        coord_dim: int = 3,
        cond_dim: int = 128,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()

        from proTeye.models.gnn import GNNLayer

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
        )

        # Input: [x_t (3), condition (cond_dim), time (hidden_dim)]
        self.input_proj = nn.Linear(coord_dim + cond_dim + hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Predict noise vector (same dim as coordinates)
        self.out = nn.Linear(hidden_dim, coord_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict added noise ε.

        Parameters
        ----------
        x_t : (N, 3)
        t : (1,) or (B,) – time step index (same step for all nodes here)
        condition : (N, cond_dim)
        edge_index : (2, E)
        edge_features : (E, edge_dim)

        Returns
        -------
        eps_pred : (N, 3)
        """
        n = x_t.shape[0]

        # Time embedding broadcast to all nodes
        t_emb = self.time_proj(self.time_embed(t))    # (1_or_B, hidden)
        t_emb = t_emb.expand(n, -1)                   # (N, hidden)

        h = self.input_proj(
            torch.cat([x_t, condition, t_emb], dim=-1)
        )  # (N, hidden)

        e = F.silu(self.edge_proj(edge_features))     # (E, hidden)

        for layer in self.layers:
            h = layer(h, edge_index, e)

        return self.out(h)  # (N, 3)
