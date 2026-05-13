"""
train.py – Train the ProtEye ConformerGenerator on a set of PDB files.

Usage
-----
    python scripts/train.py --pdb_dir ./data/pdb_files \\
                            --epochs 100 \\
                            --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proTeye.data.dataset import ProteinDataset, collate_fn
from proTeye.data.graph_builder import ProteinGraphBuilder
from proTeye.models.conformer import ConformerGenerator
from proTeye.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ProtEye conformer generator"
    )
    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="Directory containing .pdb files for training",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Graphs are variable-size; keep 1 for simplicity")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--denoiser_layers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--knn", type=int, default=10,
                        help="k for kNN graph construction")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--chain", default=None,
                        help="Restrict to a specific chain ID")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ data
    pdb_paths = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_paths:
        sys.exit(f"No .pdb files found in {args.pdb_dir}")
    print(f"Found {len(pdb_paths)} PDB files.")

    graph_builder = ProteinGraphBuilder(k=args.knn)
    dataset = ProteinDataset(
        pdb_paths=pdb_paths,
        graph_builder=graph_builder,
        chain_id=args.chain,
    )
    print(f"Dataset size (chains): {len(dataset)}")

    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: b[0])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=lambda b: b[0])

    # ----------------------------------------------------------------- model
    # Infer feature dimensions from one sample
    sample_graph = dataset[0]
    node_dim = sample_graph.node_features.shape[-1]
    edge_dim = sample_graph.edge_features.shape[-1] if sample_graph.edge_features.shape[0] > 0 else 5

    model = ConformerGenerator(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        denoiser_layers=args.denoiser_layers,
        num_diffusion_steps=args.diffusion_steps,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --------------------------------------------------------------- training
    trainer = Trainer(
        model=model,
        lr=args.lr,
        checkpoint_dir=args.output_dir,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
    )

    final_path = os.path.join(args.output_dir, "model_final.pt")
    trainer.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
