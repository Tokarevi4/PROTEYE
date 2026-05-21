from pathlib import Path

import torch

from prot_eye.graph_builder import build_protein_graph


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


def build_graph_tensors(pdb_path):

    graph = build_protein_graph(pdb_path)

    coords = torch.tensor(
        graph["coords"],
        dtype=torch.float32
    )

    edge_index = torch.tensor(
        graph["edges"],
        dtype=torch.long
    ).t()

    return {
        "coords": coords,
        "edge_index": edge_index
    }


if __name__ == "__main__":

    tensors = build_graph_tensors(
        PDB_PATH
    )

    print("Coordinates shape:")
    print(tensors["coords"].shape)

    print("\nEdge index shape:")
    print(tensors["edge_index"].shape)