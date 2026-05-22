import torch
import torch.nn as nn


class SimpleDenoiser(nn.Module):

    def __init__(
        self,
        feature_dim=20,
        hidden_dim=64
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                feature_dim + 3,
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
        noisy_coords
    ):

        x = torch.cat(
            [
                node_features,
                noisy_coords
            ],
            dim=1
        )

        predicted_coords = self.network(x)

        return predicted_coords