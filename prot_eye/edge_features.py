import torch


def build_edge_features(
    coords,
    edge_index
):

    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    relative_vectors = (
        coords[target_nodes]
        - coords[source_nodes]
    )

    distances = torch.norm(
        relative_vectors,
        dim=1,
        keepdim=True
    )

    edge_features = torch.cat(
        [
            relative_vectors,
            distances
        ],
        dim=1
    )

    return edge_features