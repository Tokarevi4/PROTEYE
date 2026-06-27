import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
OUTPUT_IMAGE = Path("reports/protein_lengths_distribution.png")

def main():
    if not PROCESSED_DIR.exists():
        print(f"Ошибка: Директория {PROCESSED_DIR} не найдена!")
        return

    lengths = []
    
    # Сбор длин из метафайлов каждого обработанного белка
    for p_dir in PROCESSED_DIR.iterdir():
        if not p_dir.is_dir():
            continue
        
        meta_path = p_dir / "meta.json"
        coords_path = p_dir / "coords.npy"
        
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                lengths.append(meta["length"])
        elif coords_path.exists():
            coords = np.load(coords_path)
            lengths.append(len(coords))

    if not lengths:
        print("Не найдено данных для анализа.")
        return

    lengths = np.array(lengths)

    total_count = len(lengths)
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    std_len = np.std(lengths)

    print(f"=== СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАТАСЕТА PROTEYE ===")
    print(f"Всего уникальных белков:     {total_count}")
    print(f"Минимальная длина (а.к.):    {min_len}")
    print(f"Максимальная длина (а.к.):   {max_len}")
    print(f"Средняя длина белка:         {mean_len:.1f}")
    print(f"Медианная длина белка:       {median_len:.1f}")
    print(f"Стандартное отклонение (std): {std_len:.1f}")
    print(f"=============================================")

    # Построение гистограммы распределения длин
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300) 
    
    # ИСПРАВЛЕНО: Заменили n, bins, patches на подчёркивания _, чтобы убрать предупреждения линтера
    _ = plt.hist(lengths, bins=30, range=(0, 1500), 
                 color="#0e7490", edgecolor="white", alpha=0.9)
    
    # Добавляем вертикальную линию среднего значения (новое среднее = 373.9)
    plt.axvline(mean_len, color="#be123c", linestyle="--", linewidth=1.5, 
                label=f"Среднее = {mean_len:.1f}")
    
    # Оформление осей и заголовка
    plt.title("Распределение длин белков в датасете ProtEye", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Количество остатков (длина последовательности)", fontsize=11, labelpad=10)
    plt.ylabel("Количество белков в выборке", fontsize=11, labelpad=10)
    
    # Фиксируем границы осей, чтобы убрать пустой хвост справа
    plt.xlim(-50, 1550)
    
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    
    # Сохраняем картинку
    plt.savefig(OUTPUT_IMAGE)
    print(f"\n📊 Сцентрированный график успешно сохранен по пути: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
