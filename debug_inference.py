import torch
import torch.nn as nn
import numpy as np

from data.dataset import ProteinDataset
from models.egnn_autoencoder import ProtEyeEGNN


# ----------------------------
# Математически точный алгоритм Кабша
# ----------------------------
def kabsch_superimpose(pred, target):
    # Центрируем обе структуры
    pred_center = pred.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)

    pred_c = pred - pred_center
    target_c = target - target_center

    # Вычисляем матрицу взаимной ковариации 3x3
    H = pred_c.T @ target_c

    # Сингулярное разложение (SVD)
    U, S, Vt = torch.linalg.svd(H)

    # Вычисляем ортогональную матрицу вращения R
    R = U @ Vt

    # Защита от зеркального отражения (если детерминант отрицательный)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Поворачиваем предсказание и смещаем в систему координат таргета
    pred_aligned = pred_c @ R + target_center
    return pred_aligned


def rmsd(a, b):
    # Среднеквадратичное отклонение между двумя облаками точек
    return torch.sqrt(
        torch.mean(
            torch.sum((a - b) ** 2, dim=1)
        )
    )


# ----------------------------
# ИНИЦИАЛИЗАЦИЯ УСТРОЙСТВА И МОДЕЛИ
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Входная размерность input_feats_dim=2 (степень узла + среднее расстояние)
model = ProtEyeEGNN(input_feats_dim=3, hidden_dim=64).to(device)

model.load_state_dict(
    torch.load(
        "weights/best_model_3.pt",
        map_location=device
    )
)
model.eval()

print("Loaded model successfully.")


# ----------------------------
# ЗАГРУЗКА ТЕСТОВОГО СЭМПЛА
# ----------------------------
dataset = ProteinDataset(root_dir="data/train")
sample = dataset[0]

# Модели больше не нужен статический x, берем только геометрию
target = sample.y.to(device)         # Чистый целевой белок
edge_index = sample.edge_index.to(device)

print(f"Nodes in protein structure: {target.shape[0]}")


# ----------------------------
# ГЕНЕРАЦИЯ КОНТРОЛИРУЕМОГО ШУМА (КАК НА ВАЛИДАЦИИ)
# ----------------------------
# Тестируем модель на сильном шуме (0.5), на котором она обучалась
TEST_NOISE_STD = 0.5

# Генерируем чистый случайный Гауссов шум
noise_vectors = torch.randn_like(target)

# Создаем зашумленный вход: чистый таргет белка + наложенный шум
noisy_input_pos = target + noise_vectors * TEST_NOISE_STD


# ----------------------------
# ОЦЕНКА БАЗОВОЙ СТРУКТУРЫ (ДО МОДЕЛИ)
# ----------------------------
raw_baseline_rmsd = rmsd(noisy_input_pos, target)
# Выравниваем исходный шум по Кабшу, чтобы узнать чистую ошибку формы белка
noisy_aligned = kabsch_superimpose(noisy_input_pos, target)
baseline_kabsch_rmsd = rmsd(noisy_aligned, target)

print()
print("===== BASELINE (BEFORE MODEL) =====")
print(f"Raw Input -> Target RMSD   : {float(raw_baseline_rmsd):.4f} Å")
print(f"Kabsch Alignment Baseline  : {float(baseline_kabsch_rmsd):.4f} Å (Чистая ошибка формы)")


# ----------------------------
# РАБОТА ГЕОМЕТРИЧЕСКОГО ДЕНОЙЗЕРА
# ----------------------------
with torch.no_grad():
    # Модель вычисляет динамические признаки h_dyn и предсказывает вектор шума
    pred_noise = model(
        pos=noisy_input_pos,
        edge_index=edge_index
    )
    
    # Математически "очищаем" зашумленные координаты по формуле денойзера
    # Нормализуем предсказание модели под масштаб реального Гауссова шума (STD=1.0)
    pred_noise_normalized = pred_noise / (pred_noise.std() + 1e-6)

    # Восстанавливаем координаты, умножая на целевой уровень шума (0.5)
    pred = noisy_input_pos - pred_noise_normalized * TEST_NOISE_STD


# ----------------------------
# ОЦЕНКА РЕЗУЛЬТАТА (ПОСЛЕ МОДЕЛИ)
# ----------------------------
raw_model_rmsd = rmsd(pred, target)
pred_aligned = kabsch_superimpose(pred, target)
model_kabsch_rmsd = rmsd(pred_aligned, target)

print()
print("===== PROTEYE EGNN OUTPUT (AFTER MODEL) =====")
print(f"Raw Model -> Target RMSD   : {float(raw_model_rmsd):.4f} Å")
print(f"Kabsch Alignment Model RMSD: {float(model_kabsch_rmsd):.4f} Å (Итоговое качество)")


# ----------------------------
# АНАЛИЗ АМПЛИТУДЫ ДВИЖЕНИЯ АТОМОВ
# ----------------------------
# Считаем, на какое расстояние в пространстве модель физически сдвинула атомы
movement = torch.norm(pred - noisy_input_pos, dim=1)

print()
print("===== ANALYSIS OF ATOM MOVEMENT =====")
print(f"Mean Atom Shift            : {float(movement.mean()):.4f} Å")
print(f"Max Atom Shift             : {float(movement.max()):.4f} Å")
print(f"Predicted Structure STD    : {pred.std(dim=0).tolist()}")

print()
if model_kabsch_rmsd < baseline_kabsch_rmsd:
    improvement = baseline_kabsch_rmsd - model_kabsch_rmsd
    print(f"Денойзер успешно исправил геометрию белка. Улучшение структуры на {float(improvement):.4f} Å.")
else:
    print("Модель пока осторожничает. Требуется больше эпох обучения или усложнение признаков.")
