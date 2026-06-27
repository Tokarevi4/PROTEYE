import os
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = Path("data/train")

MAX_WINDOW_SIZE = 457
NOISE_STD = 1.5
NOISE_MEAN = 0.0

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP",
    "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
]

def decode_numeric_residues(numeric_array):
    string_residues = []
    for val in numeric_array:
        idx = int(val)
        if 0 <= idx < len(AMINO_ACIDS):
            string_residues.append(AMINO_ACIDS[idx])
        else:
            string_residues.append("UNK")
    return np.array(string_residues, dtype='U')

def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Инкрементальный фильтр: собираем то, что уже есть (сейчас там пусто)
    existing_samples = {d.name for d in TRAIN_DIR.iterdir() if d.is_dir()}
    protein_dirs = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    
    created_count = 0
    skipped_existing = 0

    print(f"Сборка нового датасета PROTEYE. Найдено белков в processed: {len(protein_dirs)}")

    for p_dir in protein_dirs:
        pdb_id = p_dir.name
        sample_name = f"sample_{pdb_id}"

        # Защита от дубликатов при будущих запусках
        if sample_name in existing_samples:
            skipped_existing += 1
            continue

        coords_path = p_dir / "coords.npy"
        residues_path = p_dir / "residues.npy"
        
        if not coords_path.exists() or not residues_path.exists():
            continue

        coords = np.load(coords_path)
        residues = np.load(residues_path, allow_pickle=True)

        # Длинные белки аккуратно обрезаем до 457, короткие берем как есть
        current_len = min(len(coords), MAX_WINDOW_SIZE)

        target_coords = coords[0:current_len].astype(np.float32)
        numeric_residues = residues[0:current_len]

        # Конвертируем числа в строки ("ALA", "ARG"...), чтобы PyTorch не ругался на pickle
        target_residues = decode_numeric_residues(numeric_residues)

        # Генерируем Гауссовский шум для вашей задачи денойзинга
        noise = np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size=target_coords.shape).astype(np.float32)
        input_coords = target_coords + noise

        # Создаем именную папку (например, data/train/sample_13ZM)
        sample_path = TRAIN_DIR / sample_name
        sample_path.mkdir(exist_ok=True)

        # Сохраняем полный комплект для ProteinDataset
        np.save(sample_path / "input.npy", input_coords)
        np.save(sample_path / "target.npy", target_coords)
        np.save(sample_path / "residues.npy", target_residues)
        
        created_count += 1

    print("\n=== Сборка датасета завершена ===")
    print(f"Успешно создано именных сэмплов: {created_count}")
    print(f"Пропущено (уже были собраны ранее): {skipped_existing}")

if __name__ == "__main__":
    main()
