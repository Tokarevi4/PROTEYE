import os
import json
import numpy as np
from pathlib import Path

# Пути к вашим данным
PROCESSED_DIR = Path("data/processed")
SAMPLE_INPUT = Path("data/train/sample_002655/input.npy")
SAMPLE_TARGET = Path("data/train/sample_002655/target.npy")

# Загружаем тестовый сэмпл
x_input = np.load(SAMPLE_INPUT)
y_target = np.load(SAMPLE_TARGET)

print(f"=== Анализ тестового сэмпла ===")
print(f"Размер input: {x_input.shape}, размер target: {y_target.shape}")

# 1. Проверяем связь между input и target (анализ шума)
noise = x_input - y_target
max_noise = np.max(np.abs(noise))
mean_noise = np.mean(noise)
std_noise = np.std(noise)

print(f"\n[Анализ шума в input относительно target]:")
print(f"  Максимальная амплитуда шума: {max_noise:.6f}")
print(f"  Среднее значение шума (mean): {mean_noise:.6f}")
print(f"  Стандартное отклонение (std): {std_noise:.6f}")

if max_noise < 1e-5:
    print("  (!) Внимание: input и target идентичны. Возможно, шум накладывается динамически в DataLoader!")
else:
    print("  (i) Шум зафиксирован в самом датасете.")

# 2. Ищем, из какого исходного белка был сделан этот сэмпл
print(f"\n[Поиск исходного белка в {PROCESSED_DIR}...]")
found_protein = None
start_idx = -1

# Берем первую "чистую" координату из target, чтобы найти её в coords.npy
first_atom_target = y_target[0]

for p_dir in PROCESSED_DIR.iterdir():
    if not p_dir.is_dir():
        continue
        
    coords_path = p_dir / "coords.npy"
    if not coords_path.exists():
        continue
        
    coords = np.load(coords_path)
    
    # Ищем совпадение первой точки target с координатами белка
    # Используем close, так как при сохранении float32 могла слегка поплыть точность
    matches = np.where(np.all(np.isclose(coords, first_atom_target, atol=1e-4), axis=1))[0]
    
    for idx in matches:
        # Проверяем, совпадает ли весь кусок по длине
        if idx + len(y_target) <= len(coords):
            sub_coords = coords[idx : idx + len(y_target)]
            if np.all(np.isclose(sub_coords, y_target, atol=1e-4)):
                found_protein = p_dir.name
                start_idx = idx
                break
                
    if found_protein:
        break

if found_protein:
    print(f"✅ Сэмпл успешно сопоставлен!")
    print(f"  Исходный белок: {found_protein}")
    print(f"  Индекс начала среза (slice start): {start_idx}")
    print(f"  Индекс конца среза (slice end): {start_idx + len(y_target)}")
    
    # Дополнительно проверяем residues, если они обрезались так же
    residues_path = PROCESSED_DIR / found_protein / "residues.npy"
    if residues_path.exists():
        res = np.load(residues_path, allow_pickle=True)
        print(f"  Типы остатков во фрагменте: {res[start_idx : start_idx + 5]}... (длина {len(res)})")
else:
    print("❌ Не удалось найти исходный белок. Проверьте, все ли папки белков скачаны в data/processed.")
