import numpy as np
from scipy.spatial.transform import Rotation

def compute_rmsd(original, reconstructed):
    # 1. Центрируем обе структуры
    centroid_orig = np.mean(original, axis=0)
    centroid_recon = np.mean(reconstructed, axis=0)
    
    orig_centered = original - centroid_orig
    recon_centered = reconstructed - centroid_recon
    
    # 2. Находим только матрицу поворота через SciPy
    rotation, _ = Rotation.align_vectors(orig_centered, recon_centered)
    
    # 3. Поворачиваем центрированную модель
    recon_aligned = rotation.apply(recon_centered)
    
    # 4. Считаем классический RMSD вручную по формуле
    diff = orig_centered - recon_aligned
    rmsd_value = np.sqrt(np.mean(np.square(diff)))
    
    return rmsd_value

def compute_per_residue_error(original, reconstructed):
    centroid_orig = np.mean(original, axis=0)
    centroid_recon = np.mean(reconstructed, axis=0)
    
    orig_centered = original - centroid_orig
    recon_centered = reconstructed - centroid_recon
    
    rotation, _ = Rotation.align_vectors(orig_centered, recon_centered)
    recon_aligned = rotation.apply(recon_centered) # Оставляем в центре для точного соответствия
    
    diff = orig_centered - recon_aligned
    distances = np.linalg.norm(diff, axis=1)
    
    return distances
