from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from prot_eye.pdb_parser import extract_ca_atoms


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def plot_backbone(pdb_path):

    atoms = extract_ca_atoms(pdb_path)

    coords = np.array([
        atom["coord"] for atom in atoms
    ])

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2]
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=20
    )

    ax.set_title("Protein Backbone")

    plt.show()


if __name__ == "__main__":

    plot_backbone(PDB_PATH)