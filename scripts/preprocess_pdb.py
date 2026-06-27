import os
import json
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw_pdb")
PROCESSED_DIR = Path("data/processed")

# Фильтры длины (можно настроить под себя)
MIN_RESIDUES = 30
MAX_RESIDUES = 1000

# Маппинг для конвертации названий в числа (как в вашем исходном residues.npy)
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]
AA_TO_NUM = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def parse_pdb_ca_atoms(pdb_path):
    """Быстрый парсер C-alpha атомов из PDB файла."""
    coords = []
    residue_types = []
    
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Нас интересуют только строки с координатами атомов
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                
                # Фильтруем строго по Альфа-Углеродам (основа цепочки белка)
                if atom_name == "CA":
                    res_name = line[17:20].strip().upper()
                    
                    # Извлекаем координаты X, Y, Z по фиксированным позициям в PDB-формате
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError:
                        continue
                        
                    # Переводим аминокислоту в число. Если она нестандартная, ставим -1 (UNK)
                    aa_num = AA_TO_NUM.get(res_name, -1)
                    
                    coords.append([x, y, z])
                    residue_types.append(aa_num)
                    
    return np.array(coords, dtype=np.float32), np.array(residue_types, dtype=np.int64)

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print(f"Ошибка: Директория {RAW_DIR} не найдена!")
        return

    pdb_files = list(RAW_DIR.glob("*.pdb"))
    print(f"Найдено PDB файлов в raw_pdb: {len(pdb_files)}")

    processed_count = 0
    skipped_existing = 0
    skipped_length = 0
    failed_count = 0

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()  # Приводим к верхнему регистру для единообразия (например, 1A0C)
        protein_dir = PROCESSED_DIR / pdb_id

        # Инкрементальность: если белок уже обрабатывался — пропускаем его
        if protein_dir.exists():
            skipped_existing += 1
            continue

        try:
            coords, residues = parse_pdb_ca_atoms(pdb_path)
            n_res = len(coords)
            
            # Если в файле не нашлось C-alpha атомов, пропускаем
            if n_res == 0:
                failed_count += 1
                continue
                
            # Фильтрация по длине из рекомендаций коллег
            if not (MIN_RESIDUES <= n_res <= MAX_RESIDUES):
                skipped_length += 1
                continue
                
            # Сохраняем структуру в том же формате, что был у вас на первом скриншоте
            protein_dir.mkdir(parents=True, exist_ok=True)
            np.save(protein_dir / "coords.npy", coords)
            np.save(protein_dir / "residues.npy", residues)
            
            # Сохраняем метаданные с длиной
            meta_data = {"length": n_res}
            with open(protein_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta_data, f)
                
            processed_count += 1
            
        except Exception as e:
            # Ловим любые битые PDB-файлы, чтобы скрипт не падал на середине многотысячной выборки
            failed_count += 1
            continue

    print("\n=== Результаты препроцессинга PDB ===")
    print(f"Успешно обработано и сохранено: {processed_count}")
    print(f"Пропущено (уже были в processed):  {skipped_existing}")
    print(f"Пропущено по лимиту длины:       {skipped_length}")
    print(f"Ошибки парсинга / пустые файлы:   {failed_count}")

if __name__ == "__main__":
    main()
