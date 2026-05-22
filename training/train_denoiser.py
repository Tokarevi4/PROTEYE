from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

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


def train():

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

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    epochs = 500

    for epoch in range(epochs):

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

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f}"
            )


if __name__ == "__main__":

    train()