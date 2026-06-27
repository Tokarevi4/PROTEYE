import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = Path("data/train")

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
    # Принудительно конвертируем в тип 'U' (строки Юникод фиксированной длины)
    # Это исключает создание object array и решает проблему с allow_pickle
    return np.array(string_residues, dtype='U')

def main():
    if not TRAIN_DIR.exists() or not PROCESSED_DIR.exists():
        print("Проверьте пути к папкам data/train и data/processed.")
        return

    print("[1/3] Загрузка оригинальных белков из processed...")
    proteins = []
    for p_dir in PROCESSED_DIR.iterdir():
        if not p_dir.is_dir():
            continue
        coords_path = p_dir / "coords.npy"
        residues_path = p_dir / "residues.npy"
        
        if coords_path.exists() and residues_path.exists():
            proteins.append({
                "name": p_dir.name,
                "coords": np.load(coords_path).astype(np.float32),
                "path": p_dir
            })

    print(f"  Загружено белков для сопоставления: {len(proteins)}")

    # В этот раз мы берем ВСЕ папки сэмплов из train, чтобы ПЕРЕЗАПИСАТЬ некорректный тип object
    print("\n[2/3] Поиск сэмплов для обновления формата...")
    samples_to_repair = []
    for sample_dir in TRAIN_DIR.iterdir():
        if not sample_dir.is_dir():
            continue
        target_path = sample_dir / "target.npy"
        if target_path.exists():
            samples_to_repair.append(sample_dir)

    print(f"  Всего сэмплов для исправления формата: {len(samples_to_repair)}")

    print("\n[3/3] Перезапись остатков в безопасном строковом формате...")
    repaired_count = 0
    failed_count = 0

    for sample_dir in samples_to_repair:
        target_coords = np.load(sample_dir / "target.npy").astype(np.float32)
        sample_len = len(target_coords)
        
        best_match_dir = None
        min_mse = float("inf")
        
        for p in proteins:
            if len(p["coords"]) < sample_len:
                continue
            orig_chunk = p["coords"][0:sample_len]
            mse = np.mean((target_coords - orig_chunk) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_match_dir = p["path"]
                
        if min_mse < 1e-4 and best_match_dir is not None:
            orig_residues = np.load(best_match_dir / "residues.npy", allow_pickle=True)
            chunk_numeric = orig_residues[0:sample_len]
            
            # Теперь функция вернет чистый строковый массив без pickle
            chunk_strings = decode_numeric_residues(chunk_numeric)
            
            output_file = sample_dir / "residues.npy"
            np.save(output_file, chunk_strings)
            repaired_count += 1
        else:
            failed_count += 1

    print(f"\n=== Итоги исправления типов ===")
    print(f"Успешно пересохранено сэмплов: {repaired_count}")
    print(f"Не удалось сопоставить: {failed_count}")

if __name__ == "__main__":
    main()
