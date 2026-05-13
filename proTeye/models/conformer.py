"""
ConformerGenerator – full generative pipeline.

Combines the GNN encoder and the DDPM diffusion model to:
  1. Encode an input protein structure into per-residue embeddings.
  2. Sample alternative Cα coordinate sets via the reverse diffusion chain.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from proTeye.data.graph_builder import ProteinGraph
from proTeye.data.pdb_loader import PDBLoader, ProteinStructure
from proTeye.models.diffusion import DenoisingNetwork, DiffusionSchedule
from proTeye.models.gnn import ProteinGNNEncoder


class ConformerGenerator(nn.Module):
    """Generative model for alternative protein conformations.

    Parameters
    ----------
    node_input_dim :
        Dimension of raw node features produced by the graph builder.
    edge_input_dim :
        Dimension of raw edge features produced by the graph builder.
    hidden_dim :
        Shared hidden dimension for the encoder and denoising network.
    encoder_layers :
        Number of message-passing layers in the GNN encoder.
    denoiser_layers :
        Number of message-passing layers in the denoising network.
    embed_dim :
        Dimensionality of per-residue embeddings (encoder output).
    num_diffusion_steps :
        Number of steps *T* in the DDPM schedule.
    dropout :
        Dropout rate in the encoder.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        encoder_layers: int = 4,
        denoiser_layers: int = 4,
        embed_dim: int = 128,
        num_diffusion_steps: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = ProteinGNNEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.denoiser = DenoisingNetwork(
            coord_dim=3,
            cond_dim=embed_dim,
            edge_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=denoiser_layers,
        )

        self.schedule = DiffusionSchedule(num_steps=num_diffusion_steps)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        graph: ProteinGraph,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute diffusion training loss inputs.

        Randomly samples a diffusion step *t*, adds noise to the ground-
        truth Cα coordinates, and returns the predicted and actual noise.

        Parameters
        ----------
        graph :
            Single protein graph (not batched).
        t :
            Optional pre-sampled time step tensor, shape (1,).
            Drawn uniformly from [0, T-1] if *None*.

        Returns
        -------
        eps_pred : (N, 3)
            Noise predicted by the denoising network.
        eps : (N, 3)
            Actual noise that was added.
        """
        device = graph.ca_coords.device

        # Move schedule to same device if needed
        if self.schedule.alpha_bars.device != device:
            self.schedule = DiffusionSchedule(
                num_steps=self.schedule.T, device=device
            )

        if t is None:
            t = torch.randint(0, self.schedule.T, (1,), device=device)

        # Encode structure
        condition = self.encoder(
            graph.node_features,
            graph.edge_index,
            graph.edge_features,
        )  # (N, embed_dim)

        # Forward diffusion
        x_t, eps = self.schedule.q_sample(graph.ca_coords, t)

        # Predict noise
        eps_pred = self.denoiser(
            x_t, t, condition, graph.edge_index, graph.edge_features
        )

        return eps_pred, eps

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        graph: ProteinGraph,
        num_samples: int = 1,
    ) -> List[torch.Tensor]:
        """Generate alternative Cα conformations for the given structure.

        Parameters
        ----------
        graph :
            Input protein graph (encodes sequence/structural context).
        num_samples :
            Number of independent conformations to generate.

        Returns
        -------
        List of (N, 3) tensors, one per generated conformation.
        """
        device = graph.ca_coords.device
        if self.schedule.alpha_bars.device != device:
            self.schedule = DiffusionSchedule(
                num_steps=self.schedule.T, device=device
            )

        # Encode once; reuse for all samples
        condition = self.encoder(
            graph.node_features,
            graph.edge_index,
            graph.edge_features,
        )  # (N, embed_dim)

        n = graph.ca_coords.shape[0]
        conformations = []
        for _ in range(num_samples):
            coords = self.schedule.sample(
                model=self.denoiser,
                n_residues=n,
                condition=condition,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
                device=device,
            )
            conformations.append(coords)

        return conformations

    @torch.no_grad()
    def generate_structures(
        self,
        graph: ProteinGraph,
        protein: ProteinStructure,
        num_samples: int = 1,
    ) -> List[ProteinStructure]:
        """Generate conformations wrapped as :class:`ProteinStructure` objects.

        Parameters
        ----------
        graph :
            Graph built from *protein*.
        protein :
            Source protein (sequence / chain info retained).
        num_samples :
            Number of conformations.

        Returns
        -------
        List[ProteinStructure] with generated Cα coordinates.
        """
        conformations = self.generate(graph, num_samples)
        results = []
        for idx, coords in enumerate(conformations):
            ps = PDBLoader.from_coords(
                coords.cpu().numpy(),
                sequence=protein.sequence,
                name=f"{protein.name}_conf{idx}",
            )
            results.append(ps)
        return results
