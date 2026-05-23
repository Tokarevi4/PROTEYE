from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from prot_eye.graph_builder import build_protein_graph
from prot_eye.noise import add_coordinate_noise


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1AAR.pdb"


def plot_noisy_backbone(
    pdb_path,
    noise_std=0.5
):

    graph = build_protein_graph(pdb_path)

    coords = graph["coords"]

    noisy_coords = add_coordinate_noise(
        coords,
        noise_std=noise_std
    )

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    # Original structure
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        label="Original"
    )

    # Noisy structure
    ax.plot(
        noisy_coords[:, 0],
        noisy_coords[:, 1],
        noisy_coords[:, 2],
        label="Noisy"
    )

    ax.legend()

    ax.set_title(
        f"Noisy Protein Backbone (std={noise_std})"
    )

    plt.show()


if __name__ == "__main__":

    plot_noisy_backbone(
        PDB_PATH,
        noise_std=1.0
    )