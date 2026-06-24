import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split 
from torch_geometric.loader import DataLoader

from data.dataset import ProteinDataset
from models.egnn_autoencoder import ProtEyeEGNN

def add_coordinate_noise_tensor(coords, noise_std=0.1):
    return coords + torch.randn_like(coords) * noise_std


def compute_graph_geometry_loss(pred_coords, clean_coords, edge_index, criterion):
    """
    ЧЕСТНЫЙ И НАУЧНО КОРРЕКТНЫЙ ГЕОМЕТРИЧЕСКИЙ ЛОСС:
    Мы полностью отказываемся от фейковых валентных углов и ковалентных связей.
    Вместо этого мы штрафуем модель за искажение парных расстояний МЕЖДУ РЕАЛЬНЫМИ 
    СОСЕДЯМИ в графе смежности (kNN/Radius graph), зафиксированном в edge_index.
    """
    row, col = edge_index
    
    # Расстояния между связанными по графу узлами в предсказанной структуре
    pred_dist = torch.sqrt(torch.sum((pred_coords[row] - pred_coords[col]) ** 2, dim=-1) + 1e-6)
    
    # Расстояния между теми же узлами в оригинальной чистой структуре
    true_dist = torch.sqrt(torch.sum((clean_coords[row] - clean_coords[col]) ** 2, dim=-1) + 1e-6)
    
    # Графовый лосс расстояний (Graph Distance Matching Loss)
    graph_dist_loss = criterion(pred_dist, true_dist)
    
    return graph_dist_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    CONFIG = {
        "start_noise_std": 0.5,   # Начальный шум для обучения денойзера
        "end_noise_std": 0.01,    # Финальный микро-шум к 10-й эпохе
        "learning_rate": 5e-4,
        "epochs": 10
    }

    full_dataset = ProteinDataset("data/train")
    total_size = len(full_dataset)
    
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"Dataset Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = ProtEyeEGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    # HuberLoss идеально подходит для регрессии координат и расстояний
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(CONFIG["epochs"]):
        # Планировщик шума (Linear Noise Schedule)
        progress = epoch / CONFIG["epochs"]
        current_noise_std = CONFIG["start_noise_std"] + progress * (CONFIG["end_noise_std"] - CONFIG["start_noise_std"])

        # ==================== ОБУЧЕНИЕ (Geometric Denoising) ====================
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            clean_coords = batch.y 
            
            # Изолированный шум без мутации Data-объекта
            noisy_pos = add_coordinate_noise_tensor(batch.pos, noise_std=current_noise_std)
            
            # Прямой проход модели
            pred_coords = model(x=batch.x, pos=noisy_pos, edge_index=batch.edge_index)
            
            # 1. Основной координатный лосс
            coord_loss = criterion(pred_coords, clean_coords)
            
            # 2. Честный лосс сохранения геометрии графа соседей
            graph_dist_loss = compute_graph_geometry_loss(pred_coords, clean_coords, batch.edge_index, criterion)
            
            # Итоговый сбалансированный лосс без фейковой физики
            loss = coord_loss + 1.0 * graph_dist_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()

        # ==================== ВАЛИДАЦИЯ (True Reconstruction Quality) ====================
        model.eval()
        total_val_loss = 0.0
        all_mae = []
        all_rmse = []
        all_edge_errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                clean_coords = batch.y
                
                # Валидация идет на чистых нативных координатах (проверка реконструкции многообразия)
                pred_coords = model(x=batch.x, pos=batch.pos, edge_index=batch.edge_index)
                
                coord_loss = criterion(pred_coords, clean_coords)
                graph_dist_loss = compute_graph_geometry_loss(pred_coords, clean_coords, batch.edge_index, criterion)
                
                val_loss = coord_loss + 1.0 * graph_dist_loss
                total_val_loss += val_loss.item()
                
                # Метрики качества реконструкции облака точек (в Ангстремах)
                mae = torch.mean(torch.abs(pred_coords - clean_coords))
                all_mae.append(mae.item())
                
                rmse = torch.sqrt(nn.MSELoss()(pred_coords, clean_coords))
                all_rmse.append(rmse.item())
                
                # Реальная средняя ошибка восстановления расстояний между соседями по графу
                row, col = batch.edge_index
                pred_dist = torch.sqrt(torch.sum((pred_coords[row] - pred_coords[col]) ** 2, dim=-1) + 1e-6)
                true_dist = torch.sqrt(torch.sum((clean_coords[row] - clean_coords[col]) ** 2, dim=-1) + 1e-6)
                edge_err = torch.mean(torch.abs(pred_dist - true_dist))
                all_edge_errors.append(edge_err.item())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_mae = sum(all_mae) / len(all_mae)
        avg_rmse = sum(all_rmse) / len(all_rmse)
        avg_edge_err = sum(all_edge_errors) / len(all_edge_errors)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{CONFIG['epochs']} |\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Reconstruction Loss: {avg_val_loss:.4f}\n"
            f"  Val Reconstruction MAE: {avg_mae:.4f} | Val Reconstruction RMSE: {avg_rmse:.4f} Å\n"
            f"  Val Graph Neighbor Distance Error: {avg_edge_err:.4f} Å | Noise STD: {current_noise_std:.3f} | LR: {current_lr:.6f}"
        )
        print("-" * 85)


if __name__ == "__main__":
    train()
