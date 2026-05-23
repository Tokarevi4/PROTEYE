from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from prot_eye.spatial_tensor_builder import (
    build_spatial_graph_tensors
)

from prot_eye.edge_features import (
    build_edge_features
)

from prot_eye.noise import (
    add_coordinate_noise
)

from models.egnn_denoiser import (
    EGNNDenoiser
)

BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    tensors = build_spatial_graph_tensors(
        PDB_PATH,
        radius=8.0
    )

    clean_coords = tensors[
        "coords"
    ].to(device)

    node_features = tensors[
        "node_features"
    ].to(device)

    edge_index = tensors[
        "edge_index"
    ].to(device)

    noisy_coords = torch.tensor(
        add_coordinate_noise(
            clean_coords.cpu().numpy(),
            noise_std=1.0
        ),
        dtype=torch.float32
    ).to(device)

    edge_features = build_edge_features(
        noisy_coords,
        edge_index
    ).to(device)

    model = EGNNDenoiser().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()
    loss_history = []

    epochs = 500

    for epoch in range(epochs):

        optimizer.zero_grad()

        predicted_coords = model(
            node_features,
            noisy_coords,
            edge_index,
            edge_features
        )

        loss = criterion(
            predicted_coords,
            clean_coords
        )

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f}"
            )
    return (
        model,
        noisy_coords,
        clean_coords,
        edge_index,
        edge_features,
        node_features,
        loss_history
    )

if __name__ == "__main__":

    train()