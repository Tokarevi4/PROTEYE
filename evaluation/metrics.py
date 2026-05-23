import numpy as np

def compute_rmsd(
    original,
    reconstructed
):

    diff = (
        original - reconstructed
    )

    squared = np.square(diff)

    mean_squared = np.mean(squared)

    rmsd = np.sqrt(mean_squared)

    return rmsd

def compute_per_residue_error(
    original,
    reconstructed
):

    diff = (
        original - reconstructed
    )

    distances = np.linalg.norm(
        diff,
        axis=1
    )

    return distances