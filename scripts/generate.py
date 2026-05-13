"""
generate.py – Generate alternative protein conformations with a trained model.

Usage
-----
    python scripts/generate.py --pdb ./input.pdb \\
                               --checkpoint ./checkpoints/model_final.pt \\
                               --num_samples 5 \\
                               --output_dir ./generated
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proTeye.data.graph_builder import ProteinGraphBuilder
from proTeye.data.pdb_loader import PDBLoader, ProteinStructure
from proTeye.models.conformer import ConformerGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate alternative protein conformations with ProtEye"
    )
    parser.add_argument("--pdb", required=True, help="Input .pdb file")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained model (.pt)"
    )
    parser.add_argument(
        "--output_dir", default="./generated",
        help="Directory where generated PDB-like coordinate files are written"
    )
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of conformations to generate")
    parser.add_argument("--chain", default=None, help="Specific chain to use")
    parser.add_argument(
        "--hidden_dim", type=int, default=128,
        help="Must match the value used during training"
    )
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--denoiser_layers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--knn", type=int, default=10)
    return parser.parse_args()


def save_ca_pdb(
    coords: np.ndarray,
    sequence: List[str],
    path: str,
    chain_id: str = "A",
) -> None:
    """Write Cα-only PDB file for a generated conformation.

    Parameters
    ----------
    coords : (N, 3) Cα coordinates
    sequence : list of three-letter residue codes
    path : output file path
    chain_id : chain identifier character
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for i, (resname, xyz) in enumerate(zip(sequence, coords), start=1):
            x, y, z = xyz
            fh.write(
                f"ATOM  {i:5d}  CA  {resname:3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        fh.write("END\n")


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ data
    loader = PDBLoader()
    structures = loader.load(args.pdb, chain_id=args.chain)
    if not structures:
        sys.exit(f"No protein chains found in {args.pdb}")

    protein = structures[0]
    print(
        f"Loaded chain {protein.chain_id} from {args.pdb}  "
        f"({protein.num_residues} residues)"
    )

    graph_builder = ProteinGraphBuilder(k=args.knn)
    graph = graph_builder.build(protein)

    # --------------------------------------------------------------- model
    node_dim = graph.node_features.shape[-1]
    edge_dim = graph.edge_features.shape[-1] if graph.edge_features.shape[0] > 0 else 5

    model = ConformerGenerator(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        denoiser_layers=args.denoiser_layers,
        num_diffusion_steps=args.diffusion_steps,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # ------------------------------------------------------------ generation
    os.makedirs(args.output_dir, exist_ok=True)

    pdb_name = os.path.splitext(os.path.basename(args.pdb))[0]
    conformations = model.generate(graph, num_samples=args.num_samples)

    for idx, coords_tensor in enumerate(conformations):
        coords = coords_tensor.cpu().numpy()
        out_path = os.path.join(args.output_dir, f"{pdb_name}_conf{idx + 1:03d}.pdb")
        save_ca_pdb(coords, protein.sequence, out_path, chain_id=protein.chain_id)
        print(f"  Saved conformation {idx + 1} → {out_path}")

    print(f"\nGenerated {len(conformations)} conformation(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
