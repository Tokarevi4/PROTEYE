from pathlib import Path

from prot_eye.spatial_tensor_builder import (
    build_spatial_graph_tensors
)

from prot_eye.edge_features import (
    build_edge_features
)


BASE_DIR = Path(__file__).resolve().parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1UBQ.pdb"


tensors = build_spatial_graph_tensors(
    PDB_PATH
)

coords = tensors["coords"]
edge_index = tensors["edge_index"]

edge_features = build_edge_features(
    coords,
    edge_index
)

print(edge_features.shape)

print("\nFirst edge feature:")
print(edge_features[0])