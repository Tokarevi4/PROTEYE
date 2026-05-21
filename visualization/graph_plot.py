from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from prot_eye.graph_builder import build_protein_graph


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def plot_graph(pdb_path):

    graph = build_protein_graph(pdb_path)

    coords = graph["coords"]
    edges = graph["edges"]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    for edge in edges:

        i, j = edge

        x = [coords[i][0], coords[j][0]]
        y = [coords[i][1], coords[j][1]]
        z = [coords[i][2], coords[j][2]]

        ax.plot(x, y, z)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=20
    )

    ax.set_title("Protein Graph")

    plt.show()


if __name__ == "__main__":

    plot_graph(PDB_PATH)