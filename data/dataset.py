import torch
import numpy as np
from pathlib import Path
from data.features import one_hot_encode_residues
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class ProteinDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        cutoff: float = 10.0,
        use_knn_fallback: bool = True,
        min_edges: int = 1
    ):
        self.root_dir = Path(root_dir)
        self.samples = sorted(self.root_dir.glob("sample_*"))

        self.cutoff = cutoff
        self.use_knn_fallback = use_knn_fallback
        self.min_edges = min_edges

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        x = np.load(sample_dir / "input.npy")    
        y = np.load(sample_dir / "target.npy")   

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # 1. Разделяем координаты и данные аминокислот
        
        pos = x

        h = pos.clone()
        
        # ЗАМЕНИТЕ ЭТУ СТРОКУ в зависимости от того, как устроены ваши данные:
        # Если в input.npy после 3-й колонки идут ID аминокислот (например, числа от 0 до 19):
        residues = x[:, 3:].long().squeeze(-1) 
 

        # 3. ПОЛНОСТЬЮ УДАЛИТЕ ИЛИ ЗАКОММЕНТИРУЙТЕ СЛЕДУЮЩУЮ СТРОКУ (она ломает код):
        # edge_index = radius_graph(pos, r=self.cutoff) 

        # 4. Быстрое векторное вычисление матрицы расстояний (без циклов)
        dist_matrix = torch.cdist(pos, pos, p=2)

        # Маска смежности: расстояние меньше cutoff, убираем self-loops
        adj_matrix = (dist_matrix < self.cutoff) & (~torch.eye(pos.size(0), dtype=torch.bool))

        # Превращаем матрицу смежности в формат edge_index [2, E]
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()

        # Безопасный фолбэк для изолированных белков
        if edge_index.size(1) < self.min_edges:
            if self.use_knn_fallback and pos.size(0) > 1:
                idx_i = torch.arange(pos.size(0))
                edge_index = torch.combinations(idx_i, r=2).T
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(
            x=h,
            pos=pos,
            edge_index=edge_index,
            y=y
        )


# DataLoader (production)
def create_dataloader(
    root_dir,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    cutoff=10.0
):
    dataset = ProteinDataset(
        root_dir=root_dir,
        cutoff=cutoff
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataset, loader



#  Sanity check utilities
def sanity_check_batch(batch):
    print("\n===== ProtEye SANITY CHECK =====")

    print("x shape:", batch.x.shape)
    print("pos shape:", batch.pos.shape)
    print("edge_index shape:", batch.edge_index.shape)
    print("y shape:", batch.y.shape)

    print("batch size:", batch.batch.max().item() + 1)

    # NaN / Inf checks
    print("\nNaN checks:")
    print("x:", torch.isnan(batch.x).any().item())
    print("pos:", torch.isnan(batch.pos).any().item())
    print("y:", torch.isnan(batch.y).any().item())

    # edge sanity
    print("\nEdges per node:",
          batch.edge_index.size(1) / batch.x.size(0))



if __name__ == "__main__":

    dataset, loader = create_dataloader(
        root_dir="data/train",
        batch_size=4,
        cutoff=10.0
    )

    print("Samples:", len(dataset))

    batch = next(iter(loader))

    sanity_check_batch(batch)

    sample = np.load("data/train/sample_000000/input.npy")
    print(sample.shape)