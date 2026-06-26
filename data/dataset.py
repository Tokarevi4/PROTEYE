import torch
import numpy as np
from pathlib import Path
from data.features import one_hot_encode_residues  # Импортируем вашу функцию
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

        # 1. Загружаем чистые координаты
        x_raw = np.load(sample_dir / "input.npy")    
        y_raw = np.load(sample_dir / "target.npy")   

        pos = torch.tensor(x_raw, dtype=torch.float32)
        y = torch.tensor(y_raw, dtype=torch.float32)

        # 2. Загружаем последовательность аминокислот белка
        # ПРОВЕРКА: Предполагается, что в вашей папке сэмплов лежит файл со списком остатков.
        # Например, текстовый residues.txt или сохраненный numpy-массив.
        residues_file_txt = sample_dir / "residues.txt"
        residues_file_npy = sample_dir / "residues.npy"

        if residues_file_txt.exists():
            with open(residues_file_txt, "r") as f:
                residues = f.read().splitlines()
        elif residues_file_npy.exists():
            residues = np.load(residues_file_npy).tolist()
        else:
            # ФОЛБЭК: Если типов аминокислот в данных нет вообще, временно заполняем "UNKNOWN"
            residues = ["UNK"] * pos.size(0)

        # Кодируем последовательность в One-Hot вектор признаков [N, 21]
        h = one_hot_encode_residues(residues)

        # 3. СТРОИМ ТОПОЛОГИЮ ГРАФА
        dist_matrix = torch.cdist(pos, pos, p=2)
        adj_matrix = (dist_matrix < self.cutoff) & (~torch.eye(pos.size(0), dtype=torch.bool, device=pos.device))
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()

        # Безопасный фолбэк для изолированных белков (KNN)
        if edge_index.size(1) < self.min_edges:
            if self.use_knn_fallback and pos.size(0) > 1:
                idx_i = torch.arange(pos.size(0))
                edge_index = torch.combinations(idx_i, r=2).T
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(
            x=h,               # Признаки узлов размерности [N, 21]
            pos=pos,           # Чистые координаты (будут зашумлены на обучении)
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


# Проверка корректности сборки батча
def sanity_check_batch(batch):
    print("\n===== ProtEye DATASET SANITY CHECK =====")
    print("x shape (features):", batch.x.shape)        # Должно быть [N, 21]
    print("pos shape (coords):", batch.pos.shape)      # Должно быть [N, 3]
    print("edge_index shape :", batch.edge_index.shape)
    print("y shape (target)  :", batch.y.shape)
    print("Batch size        :", batch.batch.max().item() + 1)

    print("\nNaN checks:")
    print("x:", torch.isnan(batch.x).any().item())
    print("pos:", torch.isnan(batch.pos).any().item())
    print("y:", torch.isnan(batch.y).any().item())


if __name__ == "__main__":
    dataset, loader = create_dataloader(
        root_dir="data/train",
        batch_size=4,
        cutoff=10.0
    )

    print("Total Samples:", len(dataset))
    batch = next(iter(loader))
    sanity_check_batch(batch)
