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


def train(config):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # 1. Загрузка структуры белка
    tensors = build_spatial_graph_tensors(
        config["pdb_path"],
        radius=config["radius"]
    )

    clean_coords = tensors["coords"].to(device)

    node_features = (
        tensors["node_features"]
        .to(device)
    )

    edge_index = (
        tensors["edge_index"]
        .to(device)
    )

    # 2. Генерация шума
    noisy_coords = torch.tensor(
        add_coordinate_noise(
            clean_coords.cpu().numpy(),
            noise_std=config["noise_std"]
        ),
        dtype=torch.float32
    ).to(device)

    # 3. Модель
    model = EGNNDenoiser().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    epochs = config["epochs"]

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5
        )
    )

    # 4. Loss Function
    criterion = nn.HuberLoss(
        delta=1.0
    )

    loss_history = []

    print(
        f"\nStarting EGNN training "
        f"(epochs={epochs})..."
    )

    # 5. Training Loop
    for epoch in range(epochs):

        optimizer.zero_grad()

        # Edge features
        edge_features = build_edge_features(
            noisy_coords,
            edge_index
        ).to(device)

        #Forward pass
        predicted_coords = model(
            node_features,
            noisy_coords,
            edge_index,
            edge_features
        )

        #Coordinate loss
        coord_loss = criterion(
            predicted_coords,
            clean_coords
        )

        #Bond length loss
        pred_bonds = torch.linalg.norm(
            predicted_coords[1:]
            - predicted_coords[:-1],
            dim=1
        )

        clean_bonds = torch.linalg.norm(
            clean_coords[1:]
            - clean_coords[:-1],
            dim=1
        )

        bond_loss = criterion(
            pred_bonds,
            clean_bonds
        )

        #Angle loss
        v1 = (
            predicted_coords[1:-1]
            - predicted_coords[:-2]
        )

        v2 = (
            predicted_coords[2:]
            - predicted_coords[1:-1]
        )

        v1_norm = v1 / (
            torch.linalg.norm(
                v1,
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        v2_norm = v2 / (
            torch.linalg.norm(
                v2,
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        pred_angles = torch.sum(
            v1_norm * v2_norm,
            dim=1
        )

        v1_clean = (
            clean_coords[1:-1]
            - clean_coords[:-2]
        )

        v2_clean = (
            clean_coords[2:]
            - clean_coords[1:-1]
        )

        v1_clean_norm = v1_clean / (
            torch.linalg.norm(
                v1_clean,
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        v2_clean_norm = v2_clean / (
            torch.linalg.norm(
                v2_clean,
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        clean_angles = torch.sum(
            v1_clean_norm * v2_clean_norm,
            dim=1
        )

        angle_loss = criterion(
            pred_angles,
            clean_angles
        )

        # Final Loss
        loss = (
            coord_loss
            + 0.5 * bond_loss
            + 0.3 * angle_loss
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        scheduler.step()

        loss_history.append(
            loss.item()
        )

        # Logs
        if (
            epoch % 100 == 0
            or epoch == epochs - 1
        ):

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch {epoch:04d} | "
                f"Loss: {loss.item():.4f} | "
                f"Coord: {coord_loss.item():.4f} | "
                f"Bond: {bond_loss.item():.4f} | "
                f"Angle: {angle_loss.item():.4f} | "
                f"LR: {current_lr:.6f}"
            )

    print("\nTraining completed.")

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

    from config.default_config import CONFIG

    train(CONFIG)