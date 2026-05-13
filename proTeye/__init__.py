"""
ProtEye – Generative modeling of protein conformational states.

Uses geometric deep learning and diffusion-inspired neural architectures
to model structural variability from PDB data and generate plausible
alternative conformations in 3D space.
"""

from proTeye import data, models, training, utils

__version__ = "0.1.0"
__all__ = ["data", "models", "training", "utils"]
