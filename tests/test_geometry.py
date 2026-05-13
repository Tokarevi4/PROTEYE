"""
Tests for geometric utility functions.
"""

import math

import numpy as np
import pytest
import torch

from proTeye.utils.geometry import (
    compute_backbone_dihedrals,
    compute_dihedral,
    compute_local_frames,
    compute_rmsd,
    kabsch_align,
)


class TestComputeDihedral:
    def test_flat_dihedral_zero(self):
        """Four co-planar points in the xy-plane → dihedral = 0."""
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        angle = compute_dihedral(p0, p1, p2, p3)
        assert abs(angle) < 1e-6

    def test_perpendicular_dihedral(self):
        """Classic 90° dihedral."""
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        angle = compute_dihedral(p0, p1, p2, p3)
        assert abs(abs(angle) - math.pi / 2) < 1e-5

    def test_return_type(self):
        p0 = np.array([1.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 1.0])
        angle = compute_dihedral(p0, p1, p2, p3)
        assert isinstance(angle, float)


class TestBackboneDihedrals:
    def _make_helix_coords(self, n: int = 10) -> np.ndarray:
        """Generate simple helical coordinates for N residues."""
        coords = np.zeros((n, 4, 3), dtype=np.float32)
        for i in range(n):
            t = i * 1.0
            coords[i, 0] = [t, 0.0, 0.0]          # N
            coords[i, 1] = [t + 1.0, 1.0, 0.0]    # CA
            coords[i, 2] = [t + 2.0, 0.0, 0.0]    # C
            coords[i, 3] = [t + 2.0, -0.5, 0.0]   # O
        return coords

    def test_output_shapes(self):
        coords = self._make_helix_coords(8)
        phi, psi, omega = compute_backbone_dihedrals(coords)
        assert phi.shape == (8,)
        assert psi.shape == (8,)
        assert omega.shape == (8,)

    def test_first_phi_is_nan(self):
        coords = self._make_helix_coords(5)
        phi, _, _ = compute_backbone_dihedrals(coords)
        assert np.isnan(phi[0])

    def test_last_psi_is_nan(self):
        coords = self._make_helix_coords(5)
        _, psi, _ = compute_backbone_dihedrals(coords)
        assert np.isnan(psi[-1])


class TestLocalFrames:
    def test_output_shape(self):
        n = 6
        coords = np.random.randn(n, 4, 3).astype(np.float32)
        frames = compute_local_frames(coords)
        assert frames.shape == (n, 3, 3)

    def test_orthonormality(self):
        n = 4
        coords = np.zeros((n, 4, 3), dtype=np.float32)
        for i in range(n):
            coords[i, 0] = [float(i), 0.0, 0.0]
            coords[i, 1] = [float(i) + 1.0, 1.0, 0.0]
            coords[i, 2] = [float(i) + 2.0, 0.0, 0.0]
        frames = compute_local_frames(coords)
        for i in range(n):
            R = frames[i]
            # R^T R should be close to identity
            product = R.T @ R
            np.testing.assert_allclose(product, np.eye(3), atol=1e-5)

    def test_nan_coords_give_identity(self):
        coords = np.full((3, 4, 3), np.nan, dtype=np.float32)
        frames = compute_local_frames(coords)
        np.testing.assert_allclose(frames[0], np.eye(3), atol=1e-6)


class TestRMSD:
    def test_zero_rmsd_identical(self):
        coords = torch.randn(20, 3)
        rmsd = compute_rmsd(coords, coords)
        assert rmsd.item() < 1e-6

    def test_rmsd_positive(self):
        a = torch.zeros(10, 3)
        b = torch.ones(10, 3)
        rmsd = compute_rmsd(a, b)
        assert rmsd.item() > 0.0

    def test_rmsd_known_value(self):
        a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        rmsd = compute_rmsd(a, b)
        # All distances are 1.0, so RMSD = sqrt(mean(1^2)) = 1.0
        assert abs(rmsd.item() - 1.0) < 1e-5


class TestKabschAlign:
    def test_align_identity(self):
        coords = torch.randn(15, 3)
        aligned = kabsch_align(coords, coords)
        # After aligning to itself, RMSD should be essentially zero
        rmsd = compute_rmsd(aligned, coords)
        assert rmsd.item() < 1e-5

    def test_align_rotation(self):
        """Rotating a structure and then aligning should recover low RMSD."""
        coords = torch.randn(20, 3)

        # Apply a 90° rotation around z-axis
        R = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rotated = (coords - coords.mean(0)) @ R.T + coords.mean(0)

        aligned = kabsch_align(rotated, coords)
        rmsd = compute_rmsd(aligned, coords)
        assert rmsd.item() < 1e-4
