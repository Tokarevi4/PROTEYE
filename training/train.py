import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import random_split 
from torch_geometric.loader import DataLoader


from data.dataset import ProteinDataset
from models.egnn_autoencoder import ProtEyeEGNN

def add_coordinate_noise_tensor(coords, noise_std=0.1):
    return coords + torch.randn_like(coords) * noise_std


def compute_graph_geometry_loss(pred_coords, clean_coords, edge_index, criterion):
    row, col = edge_index
    
    # Расстояния между связанными по графу узлами в предсказанной структуру
    pred_dist = torch.sqrt(torch.sum((pred_coords[row] - pred_coords[col]) ** 2, dim=-1) + 1e-6)
    
    # Расстояния между теми же узлами в оригинальной чистой структуре
    true_dist = torch.sqrt(torch.sum((clean_coords[row] - clean_coords[col]) ** 2, dim=-1) + 1e-6)
    
    # Графовый лосс расстояний (Graph Distance Matching Loss)
    graph_dist_loss = criterion(pred_dist, true_dist)
    
    return graph_dist_loss

def kabsch_superimpose(pred, true):
    """
    Выравнивает предсказанные координаты (pred) по истинным (true) с помощью SVD.
    Исправлено: изменены in-place операции, вызывавшие падение PyTorch Autograd.
    """
    centroid_pred = pred.mean(dim=0, keepdim=True)
    centroid_true = true.mean(dim=0, keepdim=True)
    
    pred_c = pred - centroid_pred
    true_c = true - centroid_true
    
    H = torch.matmul(pred_c.t(), true_c)
    H = H + torch.eye(3, device=H.device) * 1e-6 
    U, _, Vt = torch.linalg.svd(H)
    
    # Вычисляем матрицу поворота R
    R = torch.matmul(Vt.t(), U.t())
    
    # Безопасная out-of-place коррекция зеркального отражения
    if torch.det(R) < 0:
        # Создаем маску знаков вместо изменения Vt на месте
        diag_modifier = torch.ones(3, device=Vt.device)
        diag_modifier[-1] = -1.0
        # Модифицируем Vt через обычное умножение, создавая новый тензор
        Vt_corrected = Vt * diag_modifier.unsqueeze(1)
        R = torch.matmul(Vt_corrected.t(), U.t())
        
    pred_aligned = torch.matmul(pred_c, R.t()) + centroid_true
    return pred_aligned


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    CONFIG = {
        "min_noise_std": 0.1,
        "max_noise_std": 1.5,
        "learning_rate": 1e-4,
        "epochs": 100
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

    model = ProtEyeEGNN(input_feats_dim=24, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    criterion = nn.HuberLoss(delta=1.0)
    best_val_rmse = float('inf')
    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_edge_err": []}

    for epoch in range(CONFIG["epochs"]):
        # ==================== ОБУЧЕНИЕ ====================
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 1. Генерируем чистый случайный Гауссов шум относительно чистых координат batch.y
            noise_vectors = torch.randn_like(batch.y)
            
            # 2. Сэмплируем случайную интенсивность шума для батча (с 10% шансом на чистый белок)
            if np.random.rand() < 0.1:
                batch_noise_std = 0.0
            else:
                batch_noise_std = np.random.uniform(CONFIG["min_noise_std"], CONFIG["max_noise_std"])
            
            # 3. Зашумляем строго чистые координаты из таргета
            noisy_pos = batch.y + noise_vectors * batch_noise_std
            
            # 4. Пропускаем через эквивариантную модель ProtEyeEGNN
            pred_noise = model(
                x=batch.x, 
                pos=noisy_pos, 
                edge_index=batch.edge_index, 
                batch_idx=batch.batch
            )
            
            # 5. СТРАТЕГИЯ 1: Расчет базового лосса денойзинга (Huber Loss)
            loss_noise = criterion(pred_noise, noise_vectors)
            
            # Динамически увеличиваем вес лосса для чистых белков и малых шумов
            if batch_noise_std == 0.0:
                loss_weight = 10.0
            else:
                loss_weight = 1.0 / (batch_noise_std + 0.1)
                
            weighted_loss = loss_noise * loss_weight

            # 6. СТРАТЕГИЯ 2: Внедрение физического инварианта белковой цепи (≈ 3.8 Å)
            # Рассчитываем позиции атомов, какими их видит модель после очистки
            pred_coords = noisy_pos - pred_noise * batch_noise_std
            
            # Извлекаем индексы связанных соседей в графе молекулы
            row, col = batch.edge_index
            
            # Считаем межатомные расстояния в восстановленной структуре
            pred_dist = torch.sqrt(torch.sum((pred_coords[row] - pred_coords[col]) ** 2, dim=-1) + 1e-6)
            
            # ИСПРАВЛЕНО: Считаем MAE вместо MSE (модуль разницы, а не квадрат), чтобы лосс не взлетал
            loss_geometry = torch.mean(torch.abs(pred_dist - 3.8))
            
            # ИСПРАВЛЕНО: Уменьшаем коэффициент до 0.01 для идеального баланса градиентов
            total_loss = weighted_loss + 0.01 * loss_geometry
            
            # 7. Обратный проход и шаг оптимизатора
            total_loss.backward()
            
            # Стабильный клиппинг градиентов для защиты от NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += total_loss.item() * batch.num_graphs

        scheduler.step()
        average_train_loss = total_train_loss / len(train_dataset)

        # ==================== ВАЛИДАЦИЯ ====================
        model.eval()
        total_val_loss = 0.0
        all_mae = []
        all_rmse = []
        all_baseline_rmse = []  # <-- Список для хранения исходного шума до модели
        all_edge_errors = []
        
        # 1. Повышаем шум до 0.5, чтобы модель могла наглядно показать денойзинг
        val_noise_std = 0.5 

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                clean_coords = batch.y  # Чистые исходные координаты
                
                # ИСПРАВЛЕНО: Генерируем тестовый шум относительно ЧИСТЫХ координат clean_coords
                noise_vectors = torch.randn_like(clean_coords)
                noisy_pos = clean_coords + noise_vectors * val_noise_std
                
                # ИСПРАВЛЕНО: Передаем в модель One-Hot аминокислоты x, зашумленные позиции и указатель батча
                pred_noise = model(
                    x=batch.x,
                    pos=noisy_pos,
                    edge_index=batch.edge_index,
                    batch_idx=batch.batch
                )
                
                # Восстанавливаем координаты (модель предсказывает чистый вектор шума)
                pred_coords = noisy_pos - pred_noise * val_noise_std
                
                # Объективный лосс валидации
                val_loss = criterion(pred_noise, noise_vectors)
                total_val_loss += val_loss.item()
                
                # Попробелковое выравнивание Кабша для МЕТРИК
                aligned_pred = torch.zeros_like(pred_coords)
                aligned_noisy = torch.zeros_like(noisy_pos)  
                
                for batch_idx in torch.unique(batch.batch):
                    mask = (batch.batch == batch_idx)
                    # Выравниваем результат работы модели
                    aligned_pred[mask] = kabsch_superimpose(pred_coords[mask], clean_coords[mask])
                    # Выравниваем исходную зашумленную структуру (до модели)
                    aligned_noisy[mask] = kabsch_superimpose(noisy_pos[mask], clean_coords[mask])
                
                # Метрики после работы модели
                mae = torch.mean(torch.abs(aligned_pred - clean_coords))
                all_mae.append(mae.item())
                
                rmse = torch.sqrt(nn.MSELoss()(aligned_pred, clean_coords))
                all_rmse.append(rmse.item())
                
                # 2. Считаем базовый RMSE ДО денойзинга
                b_rmse = torch.sqrt(nn.MSELoss()(aligned_noisy, clean_coords))
                all_baseline_rmse.append(b_rmse.item())
                
                # Ошибка длин связей по ребрам
                row, col = batch.edge_index
                pred_dist = torch.sqrt(torch.sum((pred_coords[row] - pred_coords[col]) ** 2, dim=-1) + 1e-6)
                true_dist = torch.sqrt(torch.sum((clean_coords[row] - clean_coords[col]) ** 2, dim=-1) + 1e-6)
                edge_err = torch.mean(torch.abs(pred_dist - true_dist))
                all_edge_errors.append(edge_err.item())


        # Агрегация результатов валидации
        avg_val_loss = total_val_loss / len(val_loader)
        avg_mae = sum(all_mae) / len(all_mae)
        avg_rmse = sum(all_rmse) / len(all_rmse)
        avg_baseline_rmse = sum(all_baseline_rmse) / len(all_baseline_rmse)  # <-- Средний базовый шум
        avg_edge_err = sum(all_edge_errors) / len(all_edge_errors)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{CONFIG['epochs']} |\n"
            f"  Train Loss: {average_train_loss:.4f} | Val Noise Loss: {avg_val_loss:.4f}\n"
            f"  Val BEFORE Denoising RMSE: {avg_baseline_rmse:.4f} Å\n"  # <-- Каким шум был на входе
            f"  Val AFTER  Denoising RMSE: {avg_rmse:.4f} Å\n"           # <-- Как его исправила модель
            f"  Val Reconstruction MAE: {avg_mae:.4f} |\n"
            f"  Val Graph Neighbor Distance Error: {avg_edge_err:.4f} Å | Validation Noise STD: {val_noise_std:.3f} | LR: {current_lr:.6f}"
        )

        # Запись истории обучения
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["val_rmse"].append(avg_rmse)
        history["val_edge_err"].append(avg_edge_err)

        # Логика сохранения лучшей модели
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            torch.save(model.state_dict(), "weights/test_model_100_2.pt")
            print(f"--> [SAVE] Веса лучшей модели сохранены в weights\\test_model_100_2.pt с Val RMSE: {best_val_rmse:.4f} Å")
        print("-" * 85)


    # ГЕНЕРАЦИЯ ГРАФИКА ДЛЯ ПРЕЗЕНТАЦИИ (сохраняем в папку outputs/)
    print("\nТренировка окончена. Построение графиков...")
    try:
        import matplotlib.pyplot as plt
        os.makedirs("outputs", exist_ok=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Кривая Train Loss
        color = 'tab:red'
        ax1.set_xlabel('Эпохи обучения')
        ax1.set_ylabel('Train Loss (Huber)', color=color)
        ax1.plot(history["epoch"], history["train_loss"], color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        # Кривая Val RMSE
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Val Reconstruction RMSE (Å)', color=color)
        ax2.plot(history["epoch"], history["val_rmse"], color=color, marker='s', label='Val RMSE')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Динамика обучения ProtEyeEGNN (Samsung Innovation Campus)')
        fig.tight_layout()
        
        graph_save_path = os.path.join("outputs", "training_metrics.png")
        plt.savefig(graph_save_path, dpi=300)
        print(f"--> [GRAPH] График обучения успешно сгенерирован и сохранен в: {graph_save_path}")
    except ImportError:
        print("Библиотека matplotlib не обнаружена в виртуальном окружении venv, график не построен.")


if __name__ == "__main__":
    train()
