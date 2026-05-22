from pathlib import Path
import numpy as np

from prot_eye.pdb_parser import (
    extract_ca_atoms
)


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def euclidean_distance(a, b):

    return np.linalg.norm(a - b)


def build_spatial_graph(
    pdb_path,
    radius=8.0
):

    atoms = extract_ca_atoms(
        pdb_path
    )

    coords = np.array([
        atom["coord"] for atom in atoms
    ])

    residues = [
        atom["residue"] for atom in atoms
    ]

    edges = []

    num_nodes = len(coords)

    for i in range(num_nodes):

        for j in range(num_nodes):

            if i == j:
                continue

            distance = euclidean_distance(
                coords[i],
                coords[j]
            )

            if distance < radius:

                edges.append((i, j))

    graph = {
        "coords": coords,
        "residues": residues,
        "edges": edges
    }

    return graph


if __name__ == "__main__":

    graph = build_spatial_graph(
        PDB_PATH,
        radius=8.0
    )

    print("Nodes:", len(graph["coords"]))
    print("Edges:", len(graph["edges"]))

    print("\nFirst edges:")
    print(graph["edges"][:20])