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
PDB_PATH = BASE_DIR / "data" / "sample" / "1AAR.pdb"

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

    # 2. Генерация зашумленных координат
    noisy_coords = torch.tensor(
        add_coordinate_noise(
            clean_coords.cpu().numpy(),
            noise_std=1.0
        ),
        dtype=torch.float32
    ).to(device)

    # 3. Инициализация модели, оптимизатора и шедулера
    model = EGNNDenoiser().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    
    # Увеличиваем количество эпох для плавной физической оптимизации
    epochs = 1500
    
    # Косинусный шедулер для тонкой финальной настройки весов на поздних эпохах
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=1e-5
    )

    # ПОЛИРОВКА 1: Переход на HuberLoss (Smooth L1) для устойчивости к выбросам на петлях
    criterion = nn.HuberLoss(delta=1.0)
    loss_history = []

    print(f"Старт финального обучения (Эпох: {epochs}) с полным физическим профилем...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Динамическое обновление признаков рёбер
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

        # ПОЛИРОВКА 2: Комплексный Physics-Informed Loss
        # А) Координатный лосс (теперь HuberLoss)
        coord_loss = criterion(predicted_coords, clean_coords)

        # Б) Лосс длин связей (C_alpha - C_alpha)
        pred_bonds = torch.linalg.norm(predicted_coords[1:] - predicted_coords[:-1], dim=1)
        clean_bonds = torch.linalg.norm(clean_coords[1:] - clean_coords[:-1], dim=1)
        bond_loss = criterion(pred_bonds, clean_bonds)

        # В) ПОЛИРОВКА 3: Лосс валентных углов (борьба со сломанными зигзагами в петлях)
        v1 = predicted_coords[1:-1] - predicted_coords[:-2]
        v2 = predicted_coords[2:] - predicted_coords[1:-1]
        v1_norm = v1 / (torch.linalg.norm(v1, dim=1, keepdim=True) + 1e-6)
        v2_norm = v2 / (torch.linalg.norm(v2, dim=1, keepdim=True) + 1e-6)
        pred_angles = torch.sum(v1_norm * v2_norm, dim=1)

        v1_c = clean_coords[1:-1] - clean_coords[:-2]
        v2_c = clean_coords[2:] - clean_coords[1:-1]
        v1_c_norm = v1_c / (torch.linalg.norm(v1_c, dim=1, keepdim=True) + 1e-6)
        v2_c_norm = v2_c / (torch.linalg.norm(v2_c, dim=1, keepdim=True) + 1e-6)
        clean_angles = torch.sum(v1_c_norm * v2_c_norm, dim=1)
        
        angle_loss = criterion(pred_angles, clean_angles)

        # Итоговый сбалансированный лосс
        loss = coord_loss + 0.5 * bond_loss + 0.3 * angle_loss

        # 1. Считаем градиенты
        loss.backward()

        # 2. КРИТИЧЕСКИЙ ШАГ: Делаем реальный шаг оптимизатора (обновляем веса)
        optimizer.step()
        
        # 3. Шаг расписания скорости обучения (СТРОГО ПОСЛЕ optimizer.step())
        scheduler.step()
        
        loss_history.append(loss.item())

        # Выводим логи каждые 100 эпох
        if epoch % 100 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:04d} | "
                f"Loss: {loss.item():.4f} | "
                f"Coord: {coord_loss.item():.4f} | "
                f"Bond: {bond_loss.item():.4f} | "
                f"Angle: {angle_loss.item():.4f} | "
                f"LR: {current_lr:.6f}"
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
