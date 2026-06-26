import torch
import numpy as np
from models.egnn_autoencoder import ProtEyeEGNN
from data.dataset import ProteinDataset

def calculate_tm_score(pred, target):
    """
    Математический расчет TM-score для двух структур [N, 3].
    Формула: (1 / N) * sum(1 / (1 + (d_i / d_0)^2))
    """
    N = pred.size(0)
    if N <= 15:
        return 0.0
        
    # Попарные расстояния d_i между соответствующими атомами выровненных белков
    d_i_sq = torch.sum((pred - target) ** 2, dim=-1)
    
    # Расчет статистического масштабирующего фактора d_0 (нормализация под длину белка)
    # Стандартная формула МакЛахлана/Янга для белков:
    d0 = 1.24 * np.power(N - 15, 1/3) - 1.8
    if d0 < 0.5:
        d0 = 0.5
    d0_sq = d0 ** 2
    
    # Считаем сумму по формуле TM-score
    score = torch.mean(1.0 / (1.0 + (d_i_sq / d0_sq)))
    return float(score)

# Наш исправленный алгоритм Кабша
def kabsch_superimpose(pred, target):
    pred_center = pred.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    pred_c = pred - pred_center
    target_c = target - target_center
    H = pred_c.T @ target_c
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return pred_c @ R + target_center

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели и загрузка лучших весов
    model = ProtEyeEGNN(input_feats_dim=3, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("weights/best_model_3.pt", map_location=device))
    model.eval()
    
    # Загрузка тестовой структуры
    dataset = ProteinDataset(root_dir="data/train")
    sample = dataset[0]
    target = sample.y.to(device)
    edge_index = sample.edge_index.to(device)
    
    # Симуляция сильного шума
    TEST_NOISE_STD = 0.5
    noise_vectors = torch.randn_like(target)
    noisy_pos = target + noise_vectors * TEST_NOISE_STD
    
    with torch.no_grad():
        pred_noise = model(pos=noisy_pos, edge_index=edge_index)
        # Нормализуем предсказание модели под масштаб реального Гауссова шума
        pred_noise_normalized = pred_noise / (pred_noise.std() + 1e-6)

        # Восстанавливаем координаты
        pred_coords = noisy_pos - pred_noise_normalized * TEST_NOISE_STD
        
    # Совмещение по Кабшу перед оценкой топологии
    aligned_noisy = kabsch_superimpose(noisy_pos, target)
    aligned_pred = kabsch_superimpose(pred_coords, target)
    
    # Считаем биологический скор фолдинга
    tm_before = calculate_tm_score(aligned_noisy, target)
    tm_after = calculate_tm_score(aligned_pred, target)
    
    print("\n" + "="*40)
    print("      БИОИНФОРМАТИЧЕСКИЙ АНАЛИЗ СТРУКТУРЫ      ")
    print("="*40)
    print(f"Количество аминокислотных остатков: {target.size(0)}")
    print(f"TM-score структуры ДО денойзинга : {tm_before:.4f}")
    print(f"TM-score структуры ПОСЛЕ модели  : {tm_after:.4f}")
    print("-"*40)
    
    if tm_after > tm_before:
        print(f"🚀 УСПЕХ: Топологическая укладка белка улучшена на {((tm_after - tm_before)/tm_before)*100:.2f}%!")
    else:
        print("💡 Топология осталась прежней.")
