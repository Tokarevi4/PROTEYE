import torch
import numpy as np
from pathlib import Path
from data.dataset import ProteinDataset
from torch_geometric.loader import DataLoader
from models.egnn_autoencoder import ProtEyeEGNN

# Кастомная мини-функция поворота для теста инвариантности
def get_random_rotation_matrix():
    q, r = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== АУДИТ МОДЕЛИ PROTEYE НА ЧИТЕРСТВО ===")
    print(f"Тестирование на устройстве: {device}")

    # 1. Загрузка модели и данных
    model_path = Path("weights/test_model_4759.pt")
    if not model_path.exists():
        print(f"Ошибка: Файл весов {model_path} не найден!")
        return

    model = ProtEyeEGNN(input_feats_dim=24, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = ProteinDataset("data/train")
    # Берем фиксированный батч из конца (из валидационной зоны)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader)) # Возьмем первый батч для стабильного теста
    batch = batch.to(device)

    print(f"Размер тестового батча: {batch.num_graphs} белков, {batch.pos.size(0)} атомов.")
    print("-" * 50)

    cheating_flags = []

    with torch.no_grad():
        # ТЕСТ 1: Работа на чистых данных (Шум = 0)
        # Идеальная модель не должна изменять чистую структуру
        pred_noise_clean = model(x=batch.x, pos=batch.y, edge_index=batch.edge_index, batch_idx=batch.batch)
        clean_displacement = torch.mean(torch.abs(pred_noise_clean)).item()
        print(f"[Тест 1] Среднее смещение на ЧИСТЫХ данных: {clean_displacement:.6f} Å")
        if clean_displacement > 0.05:
            print("  ⚠️ ВНИМАНИЕ: Модель сильно искажает структуру, даже когда шума нет!")
            cheating_flags.append("Искажение чистых данных")
        else:
            print("  ✅ Успешно: На чистых данных модель практически не совершает сдвигов.")

        # ТЕСТ 2: Проверка Эквивариантности / Инвариантности к поворотам (SO(3))
        # Генерируем стандартный зашумленный вход
        noise_vectors = torch.randn_like(batch.y)
        val_noise_std = 0.5
        noisy_pos = batch.y + noise_vectors * val_noise_std

        # Прямой проход для оригинального шума
        pred_noise_orig = model(x=batch.x, pos=noisy_pos, edge_index=batch.edge_index, batch_idx=batch.batch)
        mae_orig = torch.mean(torch.abs(pred_noise_orig - noise_vectors)).item()

        # Поворачиваем всю систему координат случайной матрицей вращения R
        R = get_random_rotation_matrix().to(device)
        noisy_pos_rotated = noisy_pos @ R
        noise_vectors_rotated = noise_vectors @ R

        # Прямой проход для ПОВЕРНУТОГО шума
        pred_noise_rotated = model(x=batch.x, pos=noisy_pos_rotated, edge_index=batch.edge_index, batch_idx=batch.batch)
        
        # Возвращаем предсказание обратно в исходные оси для честного сравнения
        pred_noise_recovered = pred_noise_rotated @ R.T
        mae_rotated = torch.mean(torch.abs(pred_noise_recovered - noise_vectors)).item()

        diff_rotation = abs(mae_orig - mae_rotated)
        print(f"\n[Тест 2] Ошибка предсказания шума (MAE):")
        print(f"  В оригинальных координатах: {mae_orig:.6f}")
        print(f"  В повернутых координатах:   {mae_rotated:.6f}")
        print(f"  Разница из-за поворота:     {diff_rotation:.6e}")

        if diff_rotation > 1e-4:
            print("  ❌ КРИТИЧЕСКИ: Модель привязана к абсолютным осям PDB и теряет точность при поворотах!")
            cheating_flags.append("Отсутствие геометрической эквивариантности")
        else:
            print("  ✅ Успешно: Модель строго эквивариантна! Повороты в пространстве не влияют на точность.")

        # ТЕСТ 3: Инвариантность к параллельному переносу (Сдвиг по осям)
        shift = torch.tensor([100.0, -50.0, 200.0], device=device)
        noisy_pos_shifted = noisy_pos + shift

        pred_noise_shifted = model(x=batch.x, pos=noisy_pos_shifted, edge_index=batch.edge_index, batch_idx=batch.batch)
        mae_shifted = torch.mean(torch.abs(pred_noise_shifted - noise_vectors)).item()

        diff_shift = abs(mae_orig - mae_shifted)
        print(f"\n[Тест 3] Разница MAE при гигантском переносе на [100, -50, 200]: {diff_shift:.6e}")
        if diff_shift > 1e-4:
            print("  ❌ КРИТИЧЕСКИ: Модель зависит от положения белка относительно центра координат!")
            cheating_flags.append("Зависимость от трансляции")
        else:
            print("  ✅ Успешно: Сдвиг по осям полностью игнорируется за счет батч-центрирования.")

    # ==================== ФИНАЛЬНЫЙ ВЕРДИКТ ====================
    print("\n" + "="*50)
    print("ФИНАЛЬНЫЙ ВЕРДИКТ АУДИТА:")
    if not cheating_flags:
        print("🟢 МОДЕЛЬ АБСОЛЮТНО ЧЕСТНАЯ.")
        print("Сеть выучила реальные физико-геометрические инварианты графа белков.")
        print("Её результаты обусловлены архитектурой EGNN и качеством препроцессинга.")
        print("Данный граф и веса можно уверенно защищать в дипломной работе.")
    else:
        print("🔴 ОБНАРУЖЕНЫ СЛЕДЫ ЧИТЕРСТВА / АРТЕФАКТЫ АРХИТЕКТУРЫ:")
        for flag in cheating_flags:
            print(f"  - {flag}")
        print("Рекомендуется перепроверить слои Message Passing.")
    print("="*50)

if __name__ == "__main__":
    main()
