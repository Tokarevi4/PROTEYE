import torch
import torch.nn as nn


class GeometricGraphLayer(nn.Module):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim
    ):
        super().__init__()

        self.message_mlp = nn.Sequential(

            nn.Linear(
                node_dim * 2 + edge_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.update_mlp = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                node_dim
            )
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_features
    ):

        num_nodes = node_features.shape[0]

        aggregated = torch.zeros(
            (
                num_nodes,
                self.update_mlp[-1].out_features
            ),
            device=node_features.device
        )

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        for edge_id in range(edge_index.shape[1]):

            src = source_nodes[edge_id]
            dst = target_nodes[edge_id]

            message_input = torch.cat(
                [
                    node_features[src],
                    node_features[dst],
                    edge_features[edge_id]
                ]
            )

            message = self.message_mlp(
                message_input
            )

            aggregated[dst] += message

        updated = self.update_mlp(
            aggregated
        )

        return updated