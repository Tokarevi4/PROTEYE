from pathlib import Path
import numpy as np

from prot_eye.pdb_parser import extract_ca_atoms


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def build_protein_graph(pdb_path):

    atoms = extract_ca_atoms(pdb_path)

    coords = np.array([
        atom["coord"] for atom in atoms
    ])

    residues = [
        atom["residue"] for atom in atoms
    ]

    edges = []

    for i in range(len(coords) - 1):

        edges.append((i, i + 1))
        edges.append((i + 1, i))

    graph = {
        "coords": coords,
        "residues": residues,
        "edges": edges
    }

    return graph


if __name__ == "__main__":

    graph = build_protein_graph(PDB_PATH)

    print("Nodes:", len(graph["coords"]))
    print("Edges:", len(graph["edges"]))

    print("\nFirst edges:")
    print(graph["edges"][:10])