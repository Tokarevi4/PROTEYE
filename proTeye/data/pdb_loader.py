"""
PDB structure loader.

Parses PDB files using BioPython and extracts per-residue backbone
coordinates (N, Cα, C, O) together with residue type information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from Bio import PDB as _PDB
    from Bio.PDB import PDBParser as _PDBParser
    _HAS_BIOPYTHON = True
except ImportError:  # pragma: no cover
    _HAS_BIOPYTHON = False

# Standard 20 amino-acid codes; unknown residues map to index 20.
AA_CODES: Dict[str, int] = {
    "ALA": 0,  "ARG": 1,  "ASN": 2,  "ASP": 3,  "CYS": 4,
    "GLN": 5,  "GLU": 6,  "GLY": 7,  "HIS": 8,  "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
NUM_AA_TYPES: int = 21  # 20 standard + 1 unknown

# Backbone heavy-atom names in canonical order.
BACKBONE_ATOMS: List[str] = ["N", "CA", "C", "O"]


@dataclass
class ProteinStructure:
    """Parsed representation of a single protein chain."""

    name: str
    """Identifier (e.g. PDB ID + chain)."""

    sequence: List[str]
    """Three-letter amino-acid codes, one entry per residue."""

    aa_indices: np.ndarray
    """Integer amino-acid indices, shape (N,)."""

    coords: np.ndarray
    """Backbone coordinates, shape (N, 4, 3) – axes: residues × atoms × xyz.
    Atom order: N, CA, C, O.  Missing atoms are filled with NaN."""

    chain_id: str = "A"
    """Source chain identifier."""

    pdb_path: Optional[str] = None
    """Path to the source PDB file, if available."""

    extra: Dict = field(default_factory=dict)
    """Arbitrary metadata."""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_residues(self) -> int:
        return len(self.sequence)

    @property
    def ca_coords(self) -> np.ndarray:
        """Cα coordinates, shape (N, 3)."""
        return self.coords[:, 1, :]  # index 1 → CA

    def has_missing_atoms(self) -> bool:
        """Return True if any backbone atom coordinate is NaN."""
        return bool(np.isnan(self.coords).any())


class PDBLoader:
    """Load protein structures from PDB files.

    Parameters
    ----------
    quiet:
        Suppress BioPython PDBIO warnings when *True*.
    """

    def __init__(self, quiet: bool = True) -> None:
        if not _HAS_BIOPYTHON:
            raise ImportError(
                "BioPython is required: pip install biopython"
            )
        self._parser = _PDBParser(QUIET=quiet)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        pdb_path: str,
        chain_id: Optional[str] = None,
    ) -> List[ProteinStructure]:
        """Parse *pdb_path* and return one :class:`ProteinStructure` per chain.

        Parameters
        ----------
        pdb_path:
            Path to a ``.pdb`` file.
        chain_id:
            If given, only parse that chain; otherwise parse all chains.

        Returns
        -------
        List[ProteinStructure]
            One entry per (selected) chain with at least one residue.
        """
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        structure = self._parser.get_structure(pdb_id, pdb_path)

        results: List[ProteinStructure] = []
        for model in structure:
            for chain in model:
                if chain_id is not None and chain.id != chain_id:
                    continue
                protein = self._parse_chain(
                    chain, name=f"{pdb_id}_{chain.id}", pdb_path=pdb_path
                )
                if protein is not None:
                    results.append(protein)
            break  # use first MODEL only

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chain(
        chain: "_PDB.Chain.Chain",
        name: str,
        pdb_path: Optional[str],
    ) -> Optional[ProteinStructure]:
        """Extract residue data from a single BioPython chain object."""
        sequence: List[str] = []
        coords_list: List[np.ndarray] = []

        for residue in chain:
            # Skip HETATM records (water, ligands …)
            if residue.id[0] != " ":
                continue

            resname = residue.resname.strip()
            sequence.append(resname)

            atom_coords = np.full((4, 3), np.nan, dtype=np.float32)
            for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
                if residue.has_id(atom_name):
                    atom_coords[atom_idx] = residue[atom_name].get_vector().get_array()

            coords_list.append(atom_coords)

        if not sequence:
            return None

        coords = np.stack(coords_list, axis=0)  # (N, 4, 3)
        aa_indices = np.array(
            [AA_CODES.get(r, 20) for r in sequence], dtype=np.int64
        )

        return ProteinStructure(
            name=name,
            sequence=sequence,
            aa_indices=aa_indices,
            coords=coords,
            chain_id=chain.id,
            pdb_path=pdb_path,
        )

    @staticmethod
    def from_coords(
        coords: np.ndarray,
        sequence: Optional[List[str]] = None,
        name: str = "generated",
    ) -> "ProteinStructure":
        """Construct a :class:`ProteinStructure` directly from backbone coordinates.

        Useful for wrapping model-generated conformations.

        Parameters
        ----------
        coords:
            Shape ``(N, 4, 3)`` or ``(N, 3)`` (Cα-only).
        sequence:
            Optional list of three-letter residue codes.
        name:
            Name/identifier for the structure.
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 2:
            # Cα-only: broadcast to full backbone with NaN for other atoms
            n = coords.shape[0]
            full = np.full((n, 4, 3), np.nan, dtype=np.float32)
            full[:, 1, :] = coords  # CA slot
            coords = full

        n = coords.shape[0]
        if sequence is None:
            sequence = ["GLY"] * n

        aa_indices = np.array(
            [AA_CODES.get(r, 20) for r in sequence], dtype=np.int64
        )
        return ProteinStructure(
            name=name,
            sequence=sequence,
            aa_indices=aa_indices,
            coords=coords,
        )
