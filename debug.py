from prot_eye.pdb_parser import pdb_to_graph

coords, edge_index = pdb_to_graph(
    "data/sample/1AAR.pdb"
)

print("coords:", coords.shape)
print("edge_index:", edge_index.shape)

print(edge_index[:, :10])