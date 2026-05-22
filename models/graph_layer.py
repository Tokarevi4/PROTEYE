import torch
import torch.nn as nn


class SimpleGraphLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(
        self,
        node_features,
        edge_index
    ):

        num_nodes = node_features.shape[0]

        aggregated = torch.zeros_like(
            node_features
        )

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        for src, dst in zip(
            source_nodes,
            target_nodes
        ):

            aggregated[dst] += node_features[src]

        updated = self.linear(
            aggregated
        )

        return updated