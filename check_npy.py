import numpy as np

# Берем самый первый сэмпл из вашего датасета
file_path = "data/train/sample_000000/input.npy"

try:
    data = np.load(file_path)
    print("===== АНАЛИЗ ВАШИХ NPY ДАННЫХ =====")
    print(f"Форма массива (Shape): {data.shape}")
    print(f"Тип данных (Data type): {data.dtype}")
    print("\nПервые 3 строки массива:")
    print(data[:3])
    
    if data.shape[1] > 3:
        print("\n🎉 ОТЛИЧНАЯ НОВОСТЬ: В ваших npy файлах уже зашито больше, чем просто 3D-координаты!")
    else:
        print("\n💡 В ваших npy файлах записаны только чистые 3D-координаты (X, Y, Z).")
except Exception as e:
    print(f"Не удалось прочитать файл: {e}")
