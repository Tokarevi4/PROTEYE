from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.metrics import (
    compute_rmsd,
    compute_per_residue_error
)


def save_structure_plot(
    coords,
    title,
    save_path,
    node_color="blue"
):

    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.plot(
        x, y, z,
        marker="o",
        markerfacecolor=node_color,
        color="gray"
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_loss_plot(
    loss_history,
    save_path
):

    plt.figure(figsize=(8, 5))

    plt.plot(loss_history)

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_per_residue_plot(
    errors,
    save_path
):

    plt.figure(figsize=(10, 5))

    plt.plot(errors)

    plt.xlabel("Residue Index")

    plt.ylabel("Error")

    plt.title("Per-Residue Reconstruction Error")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_metrics(
    rmsd,
    mean_error,
    max_error,
    save_path
):

    with open(save_path, "w") as f:

        f.write("ProtEye Metrics\n")
        f.write("====================\n\n")

        f.write(f"RMSD: {rmsd:.4f}\n")
        f.write(f"Mean Residue Error: {mean_error:.4f}\n")
        f.write(f"Max Residue Error: {max_error:.4f}\n")


def save_all_results(
    model,
    noisy_coords,
    clean_coords,
    edge_index,
    edge_features,
    node_features,
    loss_history,
    config
):

    protein_name = config["protein_name"]

    output_dir = Path(
        config["output_dir"]
    )

    model.eval()

    with torch.no_grad():

        reconstructed_coords = model(
            node_features,
            noisy_coords,
            edge_index,
            edge_features
        )

    clean_np = clean_coords.cpu().numpy()

    noisy_np = noisy_coords.cpu().numpy()

    reconstructed_np = (
        reconstructed_coords
        .cpu()
        .numpy()
    )

    rmsd = compute_rmsd(
        clean_np,
        reconstructed_np
    )

    residue_errors = (
        compute_per_residue_error(
            clean_np,
            reconstructed_np
        )
    )

    mean_error = np.mean(
        residue_errors
    )

    max_error = np.max(
        residue_errors
    )

    save_structure_plot(
        clean_np,
        "Original Structure",
        output_dir / f"{protein_name}_original.png",
        node_color = "blue"
    )

    save_structure_plot(
        noisy_np,
        "Noisy Structure",
        output_dir / f"{protein_name}_noisy.png",
        node_color = "red"
    )

    save_structure_plot(
        reconstructed_np,
        "Reconstructed Structure",
        output_dir / f"{protein_name}_reconstructed.png",
        node_color = "green"
    )

    save_loss_plot(
        loss_history,
        output_dir / f"{protein_name}_loss.png"
    )

    save_per_residue_plot(
        residue_errors,
        output_dir / f"{protein_name}_residue_error.png"
    )

    save_metrics(
        rmsd,
        mean_error,
        max_error,
        output_dir / f"{protein_name}_metrics.txt"
    )

    print(f"\nRMSD: {rmsd:.4f}")
    print(f"Mean Residue Error: {mean_error:.4f}")
    print(f"Max Residue Error: {max_error:.4f}")

    print("\nAll results saved.")