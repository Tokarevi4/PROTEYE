import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    def __init__(self, feats_dim, m_dim):
        super().__init__(aggr='mean') 
        
        # MLP для вычисления сообщений по ребрам (m_ij)
        self.edge_mlp = nn.Sequential(
            nn.Linear(feats_dim * 2 + 1, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU()
        )
        
        # MLP для вычисления весов сдвига координат (x_i)
        self.coor_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1, bias=False)
        )
        
        # MLP для обновления признаков узлов (h_i)
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, feats_dim)
        )

    def forward(self, h, pos, edge_index):
        h_new, pos_new = self.propagate(edge_index, h=h, pos=pos, size=None)
        return h_new, pos_new

    def message(self, h_i, h_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-6
        
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        coor_weights = self.coor_mlp(m_ij)
        coor_weights = torch.clamp(coor_weights, min=-10.0, max=10.0)
        
        return m_ij, rel_pos * coor_weights

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, coor_msg = inputs
        m_i = super().aggregate(m_ij, index, ptr, dim_size)
        pos_i = super().aggregate(coor_msg, index, ptr, dim_size)
        return m_i, pos_i

    def update(self, aggr_out, h, pos):
        m_i, pos_delta = aggr_out
        new_pos = pos + pos_delta 
        node_input = torch.cat([h, m_i], dim=-1)
        new_h = self.node_mlp(node_input)
        return new_h, new_pos


class ProtEyeEGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.egnn1 = EGNNLayer(feats_dim=3, m_dim=32)
        self.egnn2 = EGNNLayer(feats_dim=3, m_dim=32)
        self.head = nn.Linear(3, 3)

    def forward(self, x, pos, edge_index):
        """
        ИСПРАВЛЕНО: Явный прием тензоров вместо объекта Data.
        """
        h, pos = self.egnn1(x, pos, edge_index)
        h, pos = self.egnn2(h, pos, edge_index)
        return self.head(pos)
