import numpy as np
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

    # 1. Загрузка исходных тензоров белка
    tensors = build_spatial_graph_tensors(
        PDB_PATH,
        radius=8.0
    )

    clean_coords = tensors["coords"].to(device)
    node_features = tensors["node_features"].to(device)
    edge_index = tensors["edge_index"].to(device)

    # 2. Генерация зашумленных координат для задачи денойзинга
    noisy_coords = torch.tensor(
        add_coordinate_noise(
            clean_coords.cpu().numpy(),
            noise_std=1.0
        ),
        dtype=torch.float32
    ).to(device)

    # 3. Инициализация модели и оптимизатора
    model = EGNNDenoiser().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()
    loss_history = []
    epochs = 500

    print("Старт обучения с физическими ограничениями...")

    for epoch in range(epochs):
        optimizer.zero_grad()


        edge_features = build_edge_features(
            noisy_coords, 
            edge_index
        ).to(device)

        # Прямой проход модели
        predicted_coords = model(
            node_features,
            noisy_coords,
            edge_index,
            edge_features
        )


        coord_loss = criterion(predicted_coords, clean_coords)

        # Вычисляем расстояния между соседними аминокислотами в цепи (i и i+1)
        pred_bonds = torch.linalg.norm(predicted_coords[1:] - predicted_coords[:-1], dim=1)
        clean_bonds = torch.linalg.norm(clean_coords[1:] - clean_coords[:-1], dim=1)
        bond_loss = criterion(pred_bonds, clean_bonds)

        # Итоговый лосс с весовым коэффициентом для стабилизации геометрии
        loss = coord_loss + 0.5 * bond_loss

        # Обратное распространение и шаг оптимизатора
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Coord MSE: {coord_loss.item():.4f} | "
                f"Bond MSE: {bond_loss.item():.4f}"
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
