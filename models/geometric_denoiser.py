import torch
import torch.nn as nn

from models.geometric_graph_layer import (
    GeometricGraphLayer
)

class GeometricDenoiser(nn.Module):

    def __init__(
        self,
        node_dim=20,
        edge_dim=4,
        hidden_dim=64
    ):
        super().__init__()

        self.graph_layer = (
            GeometricGraphLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim
            )
        )

        self.coord_mlp = nn.Sequential(

            nn.Linear(
                node_dim + 3,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                3
            )
        )

    def forward(
        self,
        node_features,
        noisy_coords,
        edge_index,
        edge_features
    ):

        updated_features = (
            self.graph_layer(
                node_features,
                edge_index,
                edge_features
            )
        )

        x = torch.cat(
            [
                updated_features,
                noisy_coords
            ],
            dim=1
        )

        predicted_coords = (
            self.coord_mlp(x)
        )

        return predicted_coords