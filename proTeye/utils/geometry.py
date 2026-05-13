"""
Geometric utilities for protein structure analysis.

All functions accept and return NumPy arrays or PyTorch tensors as
indicated by their signatures.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dihedral angles
# ---------------------------------------------------------------------------

def compute_dihedral(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Compute the dihedral angle (in radians) defined by four points.

    Uses the standard IUPAC definition: the angle between the planes
    (p0, p1, p2) and (p1, p2, p3).

    Parameters
    ----------
    p0, p1, p2, p3 : (3,) arrays

    Returns
    -------
    angle : float in [-π, π]
    """
    b0 = np.asarray(p0, dtype=float) - np.asarray(p1, dtype=float)
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)

    b1_norm = b1 / (np.linalg.norm(b1) + 1e-10)

    # Gram-Schmidt: project b0 and b2 onto the plane normal to b1
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    return math.atan2(y, x)


def compute_backbone_dihedrals(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute backbone dihedral angles (φ, ψ, ω) for all residues.

    Parameters
    ----------
    coords : (N, 4, 3)
        Backbone coordinates. Atom order: N, CA, C, O.

    Returns
    -------
    phi : (N,) – φ angles in radians. NaN for the first residue.
    psi : (N,) – ψ angles in radians. NaN for the last residue.
    omega : (N,) – ω angles in radians. NaN for the first residue.
    """
    n = coords.shape[0]
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    omega = np.full(n, np.nan)

    # Atom indices: 0=N, 1=CA, 2=C, 3=O
    for i in range(n):
        N_i, CA_i, C_i = coords[i, 0], coords[i, 1], coords[i, 2]
        if np.isnan([N_i, CA_i, C_i]).any():
            continue

        # φ (phi): C_{i-1} – N_i – CA_i – C_i
        if i > 0:
            C_prev = coords[i - 1, 2]
            if not np.isnan(C_prev).any():
                phi[i] = compute_dihedral(C_prev, N_i, CA_i, C_i)

        # ψ (psi): N_i – CA_i – C_i – N_{i+1}
        if i < n - 1:
            N_next = coords[i + 1, 0]
            if not np.isnan(N_next).any():
                psi[i] = compute_dihedral(N_i, CA_i, C_i, N_next)

        # ω (omega): CA_{i-1} – C_{i-1} – N_i – CA_i
        if i > 0:
            CA_prev = coords[i - 1, 1]
            C_prev = coords[i - 1, 2]
            if not np.isnan([CA_prev, C_prev]).any():
                omega[i] = compute_dihedral(CA_prev, C_prev, N_i, CA_i)

    return phi, psi, omega


# ---------------------------------------------------------------------------
# Local reference frames
# ---------------------------------------------------------------------------

def compute_local_frames(coords: np.ndarray) -> np.ndarray:
    """Compute an orthonormal local frame for each residue.

    The frame is defined using the N–CA–C triangle:

    * **u** – unit vector from N to CA  
    * **v** – component of (CA→C) orthogonal to **u** (then normalised)  
    * **w** – cross product **u** × **v**

    Parameters
    ----------
    coords : (N, 4, 3)
        Backbone coordinates. Atom order: N, CA, C, O.

    Returns
    -------
    frames : (N, 3, 3)
        Each ``frames[i]`` is the 3×3 rotation matrix [u | v | w] as columns.
        Residues with NaN coordinates get an identity frame.
    """
    n = coords.shape[0]
    frames = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))

    for i in range(n):
        N_pos, CA_pos, C_pos = coords[i, 0], coords[i, 1], coords[i, 2]
        if np.isnan([N_pos, CA_pos, C_pos]).any():
            continue

        u = CA_pos - N_pos
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-8:
            continue
        u = u / norm_u

        v = C_pos - CA_pos
        v = v - np.dot(v, u) * u
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            continue
        v = v / norm_v

        w = np.cross(u, v)
        frames[i] = np.stack([u, v, w], axis=1)  # columns

    return frames


# ---------------------------------------------------------------------------
# RMSD and Kabsch alignment
# ---------------------------------------------------------------------------

def compute_rmsd(
    coords_pred: torch.Tensor,
    coords_true: torch.Tensor,
) -> torch.Tensor:
    """Root mean square deviation (no alignment).

    Parameters
    ----------
    coords_pred, coords_true : (N, 3) tensors

    Returns
    -------
    rmsd : scalar tensor (Å)
    """
    diff = coords_pred - coords_true
    return torch.sqrt((diff ** 2).sum(-1).mean())


def kabsch_align(
    mobile: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Rotate *mobile* to minimise RMSD against *target* (Kabsch algorithm).

    Both coordinate sets are centred before computing the optimal rotation
    and returned in the original (centred) frame.

    Parameters
    ----------
    mobile : (N, 3)
    target : (N, 3)

    Returns
    -------
    mobile_aligned : (N, 3) – *mobile* after optimal rotation (centred).
    """
    mobile_mean = mobile.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    mobile_c = mobile - mobile_mean
    target_c = target - target_mean

    # Covariance matrix
    H = mobile_c.T @ target_c  # (3, 3)

    U, S, Vt = torch.linalg.svd(H)

    # Ensure a proper rotation (det = +1)
    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=mobile.device, dtype=mobile.dtype))

    R = Vt.T @ D @ U.T
    # Rotate mobile (centred) and translate to target centroid
    return mobile_c @ R.T + target_mean
