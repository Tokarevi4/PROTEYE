import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing # Необходима для безопасного центрирования в батче

class EGNNLayer(MessagePassing):
    def __init__(self, feats_dim, m_dim):
        super().__init__(aggr='mean')

        # Слой для межузловых сообщений
        self.edge_mlp = nn.Sequential(
            nn.Linear(feats_dim * 2 + 1, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU()
        )

        # Слой для генерации координатных векторов сдвига
        self.coor_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1) 
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, feats_dim)
        )

    def forward(self, h, pos, edge_index):
        # Передаем координаты в propagate
        m_i, pos_delta = self.propagate(edge_index, h=h, pos=pos)
        
        # Обновляем признаки узлов
        node_input = torch.cat([h, m_i], dim=-1)
        new_h = h + self.node_mlp(node_input)
        
        # Обновляем координаты (модель учится двигать структуру)
        new_pos = pos + pos_delta
        
        return new_h, new_pos

    def message(self, h_i, h_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8

        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)

        coor_weight = self.coor_mlp(m_ij)

        rel_dir = rel_pos / (torch.sqrt(dist_sq) + 1e-6)
        coor_msg = rel_dir * coor_weight

        return m_ij, coor_msg

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, coor_msg = inputs
        m_i = super().aggregate(m_ij, index, ptr, dim_size)
        pos_delta = super().aggregate(coor_msg, index, ptr, dim_size)
        return m_i, pos_delta


class ProtEyeEGNN(nn.Module):
    def __init__(self, input_feats_dim=24, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_feats_dim, hidden_dim)

        self.egnn1 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)
        self.egnn2 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)
        self.egnn3 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)

    def forward(self, x, pos, edge_index, batch_idx=None):
        orig_pos = pos.clone()
        num_nodes = orig_pos.size(0)

        # 1. РАСЧЕТ ИНВАРИАНТОВ (ФОРМА [N, 1])
        row = edge_index[0].unsqueeze(-1)

        rel_pos = orig_pos[edge_index[0]] - orig_pos[edge_index[1]]
        edge_dists = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8)

        node_degrees = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)
        sum_neighbor_dist = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)
        max_neighbor_dist = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)

        ones = torch.ones_like(edge_dists)
        node_degrees.scatter_add_(0, row, ones)

        sum_neighbor_dist.scatter_add_(0, row, edge_dists)
        mean_neighbor_dist = sum_neighbor_dist / (node_degrees + 1e-6)

        max_neighbor_dist.scatter_reduce_(0, row, edge_dists, reduce='amax', include_self=False)

        h_dyn = torch.cat([node_degrees, mean_neighbor_dist, max_neighbor_dist], dim=-1)
        h_combined = torch.cat([x, h_dyn], dim=-1)
        h = self.embedding(h_combined)
        
        # ======================================================================
        # ИСПРАВЛЕНО: ЧИСТЫЙ PYTORCH ЦЕНТРИРОВАНИЯ БАТЧА (БЕЗ TORCH_SCATTER)
        # ======================================================================
        if batch_idx is not None:
            num_graphs = batch_idx.max().item() + 1
            
            # 1. Считаем сумму координат для каждого отдельного белка в батче
            sum_pos = torch.zeros((num_graphs, 3), dtype=pos.dtype, device=pos.device)
            sum_pos.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, 3), pos)
            
            # 2. Считаем количество атомов в каждом белке
            node_counts = torch.zeros((num_graphs, 1), dtype=pos.dtype, device=pos.device)
            ones_nodes = torch.ones((num_nodes, 1), dtype=pos.dtype, device=pos.device)
            node_counts.scatter_add_(0, batch_idx.unsqueeze(-1), ones_nodes)
            
            # 3. Находим центр масс каждого белка и вычитаем его
            graph_means = sum_pos / (node_counts + 1e-6)
            pos = pos - graph_means[batch_idx]
        else:
            center = pos.mean(dim=0, keepdim=True)
            pos = pos - center
        # ======================================================================

        # 2. ПРЯМОЙ ПРОХОД СЕТИ
        h, pos = self.egnn1(h, pos, edge_index)
        h, pos = self.egnn2(h, pos, edge_index)
        
        pos_before_last = pos.clone()
        h, pos_after_last = self.egnn3(h, pos, edge_index)
        
        predicted_noise = pos_after_last - pos_before_last
        
        return predicted_noise

