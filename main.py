import argparse
import os
from pathlib import Path

import torch

from config.default_config import CONFIG

from training.train_egnn_denoiser import train
from visualization.save_results import save_all_results


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdb",
        type=str,
        default=str(CONFIG["pdb_path"]),
        help="Path to PDB file"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    pdb_path = Path(args.pdb)

    CONFIG["pdb_path"] = pdb_path

    protein_name = pdb_path.stem

    CONFIG["protein_name"] = protein_name

    os.makedirs(
        CONFIG["output_dir"],
        exist_ok=True
    )

    os.makedirs(
        CONFIG["weights_dir"],
        exist_ok=True
    )

    print(f"\n=== ProtEye Pipeline ===")
    print(f"Protein: {protein_name}")

    (
        model,
        noisy_coords,
        clean_coords,
        edge_index,
        edge_features,
        node_features,
        loss_history
    ) = train(CONFIG)

    save_all_results(
        model=model,
        noisy_coords=noisy_coords,
        clean_coords=clean_coords,
        edge_index=edge_index,
        edge_features=edge_features,
        node_features=node_features,
        loss_history=loss_history,
        config=CONFIG
    )

    weights_path = (
        CONFIG["weights_dir"]
        / f"proteye_{protein_name}.pt"
    )

    torch.save(
        model.state_dict(),
        weights_path
    )

    print(f"\nWeights saved: {weights_path}")
    print("\nProtEye pipeline completed.")


if __name__ == "__main__":
    main()