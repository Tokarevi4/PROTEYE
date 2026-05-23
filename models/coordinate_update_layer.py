import torch
import torch.nn as nn

class CoordinateUpdateLayer(nn.Module):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim
    ):
        super().__init__()

        self.edge_mlp = nn.Sequential(

            nn.Linear(
                node_dim * 2 + edge_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        node_features,
        coords,
        edge_index,
        edge_features
    ):

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        coord_updates = torch.zeros_like(
            coords
        )

        for edge_id in range(edge_index.shape[1]):

            src = source_nodes[edge_id]
            dst = target_nodes[edge_id]

            relative_vector = (
                coords[dst]
                - coords[src]
            )

            edge_input = torch.cat(
                [
                    node_features[src],
                    node_features[dst],
                    edge_features[edge_id]
                ]
            )

            weight = self.edge_mlp(
                edge_input
            )

            coord_updates[dst] += (
                relative_vector * weight
            )

        updated_coords = (
            coords + coord_updates
        )

        return updated_coords