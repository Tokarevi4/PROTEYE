import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

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
        # УБИРАЕМ bias=False, даем модели свободу масштабирования
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

        # ВАЖНО: УБИРАЕМ torch.tanh() * 0.1! 
        # Ограничение tanh заставляло предсказания зануляться. 
        # Даем модели возможность генерировать реальный масштаб Гауссова шума.
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
    def __init__(self, input_feats_dim=3, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_feats_dim, hidden_dim)

        self.egnn1 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)
        self.egnn2 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)
        
        # Финальный слой Message Passing, координатный MLP которого 
        # и будет выдавать наш предсказанный вектор шума
        self.egnn3 = EGNNLayer(feats_dim=hidden_dim, m_dim=hidden_dim)

    def forward(self, pos, edge_index):
        orig_pos = pos.clone()
        num_nodes = orig_pos.size(0)

        # 1. СТАБИЛЬНЫЙ PYTORCH РАСЧЕТ ИНВАРИАНТОВ (ФОРМА [N, 1])
        # Берем строку индексов источников и расширяем до [E, 1]
        row = edge_index[0].unsqueeze(-1)

        # ИСПРАВЛЕНО: Честно вычисляем физические расстояния между соседями по графу
        rel_pos = orig_pos[edge_index[0]] - orig_pos[edge_index[1]]
        edge_dists = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8)  # Форма [E, 1]

        # Инициализируем принимающие тензоры в форме [N, 1]
        node_degrees = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)
        sum_neighbor_dist = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)
        max_neighbor_dist = torch.zeros((num_nodes, 1), dtype=torch.float32, device=pos.device)

        # Расчет степени узла (node degree)
        ones = torch.ones_like(edge_dists)
        node_degrees.scatter_add_(0, row, ones)

        # Расчет среднего расстояния до соседей
        sum_neighbor_dist.scatter_add_(0, row, edge_dists)
        mean_neighbor_dist = sum_neighbor_dist / (node_degrees + 1e-6)

        # Расчет максимального расстояния до соседей
        max_neighbor_dist.scatter_reduce_(0, row, edge_dists, reduce='amax', include_self=False)

        # Объединяем в общую матрицу динамических признаков [N, 3]
        h_dyn = torch.cat([node_degrees, mean_neighbor_dist, max_neighbor_dist], dim=-1)

        # 2. ПРЯМОЙ ПРОХОД СЕТИ
        h = self.embedding(h_dyn)
        
        center = pos.mean(dim=0, keepdim=True)
        pos = pos - center

        h, pos = self.egnn1(h, pos, edge_index)
        h, pos = self.egnn2(h, pos, edge_index)
        
        # Перед 3-м слоем запоминаем координаты
        pos_before_last = pos.clone()
        
        # 3-й слой совершает финальный эквивариантный сдвиг
        h, pos_after_last = self.egnn3(h, pos, edge_index)
        
        # Вектор шума — это чистый сдвиг координат внутри последнего Message Passing слоя
        predicted_noise = pos_after_last - pos_before_last
        
        return predicted_noise

