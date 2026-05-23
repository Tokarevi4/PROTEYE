from pathlib import Path

import torch

from prot_eye.spatial_graph_builder import (
    build_spatial_graph
)

from prot_eye.residue_features import (
    one_hot_encode_residues
)


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1AAR.pdb"


def build_spatial_graph_tensors(
    pdb_path,
    radius=8.0
):

    graph = build_spatial_graph(
        pdb_path,
        radius=radius
    )

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

    tensors = build_spatial_graph_tensors(
        PDB_PATH
    )

    print("Coordinates:")
    print(tensors["coords"].shape)

    print("\nEdge index:")
    print(tensors["edge_index"].shape)

    print("\nNode features:")
    print(tensors["node_features"].shape)