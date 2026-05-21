import numpy as np


def add_coordinate_noise(
    coords,
    noise_std=0.5
):
    """
    Add Gaussian noise to coordinates.
    """

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=coords.shape
    )

    noisy_coords = coords + noise

    return noisy_coords