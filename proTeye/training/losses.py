"""
Loss functions for training the ConformerGenerator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_loss(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between predicted and actual noise vectors.

    This is the simplified DDPM objective (Ho et al., 2020).

    Parameters
    ----------
    eps_pred : (N, 3)
        Noise predicted by the denoising network.
    eps_true : (N, 3)
        Ground-truth Gaussian noise that was added in the forward process.

    Returns
    -------
    loss : scalar tensor
    """
    return F.mse_loss(eps_pred, eps_true)


def coordinate_rmsd(
    coords_pred: torch.Tensor,
    coords_true: torch.Tensor,
    align: bool = False,
) -> torch.Tensor:
    """Root mean square deviation between two coordinate sets.

    Parameters
    ----------
    coords_pred : (N, 3)
    coords_true : (N, 3)
    align :
        If *True*, optimally superpose *coords_pred* onto *coords_true*
        before computing RMSD (Kabsch algorithm).

    Returns
    -------
    rmsd : scalar tensor (Å)
    """
    if align:
        from proTeye.utils.geometry import kabsch_align
        coords_pred = kabsch_align(coords_pred, coords_true)

    diff = coords_pred - coords_true
    return torch.sqrt((diff ** 2).sum(-1).mean())
