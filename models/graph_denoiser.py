import torch
import torch.nn as nn

from models.graph_layer import (
    SimpleGraphLayer
)


class GraphDenoiser(nn.Module):

    def __init__(
        self,
        feature_dim=20,
        hidden_dim=64
    ):
        super().__init__()

        self.graph_layer = SimpleGraphLayer(
            feature_dim,
            hidden_dim
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                hidden_dim + 3,
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
        edge_index
    ):

        graph_features = self.graph_layer(
            node_features,
            edge_index
        )

        x = torch.cat(
            [
                graph_features,
                noisy_coords
            ],
            dim=1
        )

        predicted_coords = self.mlp(x)

        return predicted_coords