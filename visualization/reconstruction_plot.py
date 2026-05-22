from pathlib import Path

import torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from prot_eye.tensor_builder import (
    build_graph_tensors
)

from prot_eye.noise import (
    add_coordinate_noise
)

from models.simple_denoiser import (
    SimpleDenoiser
)


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def visualize_reconstruction():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    tensors = build_graph_tensors(
        PDB_PATH
    )

    clean_coords = tensors["coords"].to(device)

    node_features = tensors[
        "node_features"
    ].to(device)

    noisy_coords = torch.tensor(
        add_coordinate_noise(
            clean_coords.cpu().numpy(),
            noise_std=1.0
        ),
        dtype=torch.float32
    ).to(device)

    model = SimpleDenoiser().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = torch.nn.MSELoss()

    # quick training
    for epoch in range(500):

        optimizer.zero_grad()

        predicted_coords = model(
            node_features,
            noisy_coords
        )

        loss = criterion(
            predicted_coords,
            clean_coords
        )

        loss.backward()

        optimizer.step()

    reconstructed = predicted_coords.detach().cpu().numpy()

    clean = clean_coords.cpu().numpy()
    noisy = noisy_coords.cpu().numpy()

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    # Original
    ax.plot(
        clean[:, 0],
        clean[:, 1],
        clean[:, 2],
        label="Original"
    )

    # Noisy
    ax.plot(
        noisy[:, 0],
        noisy[:, 1],
        noisy[:, 2],
        label="Noisy"
    )

    # Reconstructed
    ax.plot(
        reconstructed[:, 0],
        reconstructed[:, 1],
        reconstructed[:, 2],
        label="Reconstructed"
    )

    ax.legend()

    ax.set_title(
        "Protein Reconstruction"
    )

    plt.show()


if __name__ == "__main__":

    visualize_reconstruction()