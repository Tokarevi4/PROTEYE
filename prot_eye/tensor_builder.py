from pathlib import Path

import torch

from prot_eye.graph_builder import build_protein_graph
from prot_eye.residue_features import (
    one_hot_encode_residues
)


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

    node_features = one_hot_encode_residues(
        graph["residues"]
    )

    return {
        "coords": coords,
        "edge_index": edge_index,
        "node_features": node_features
    }


if __name__ == "__main__":

    tensors = build_graph_tensors(
        PDB_PATH
    )

    print("Coordinates:")
    print(tensors["coords"].shape)

    print("\nEdges:")
    print(tensors["edge_index"].shape)

    print("\nNode features:")
    print(tensors["node_features"].shape)