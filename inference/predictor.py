import torch
import numpy as np
from models.egnn_autoencoder import ProtEyeEGNN
from data.features import one_hot_encode_residues

class ProtEyePredictor:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Инициализируем 100-эпошную модель
        self.model = ProtEyeEGNN(input_feats_dim=24, hidden_dim=64)

        self.model.load_state_dict(
            torch.load(
                "weights/general.pt",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tensors):
        pos = tensors["coords"].to(self.device)
        edge_index = tensors["edge_index"].to(self.device)
        num_nodes = pos.size(0)

        # ======================================================================
        # ПОДГОТОВКА БИОЛОГИЧЕСКИХ ONE-HOT ПРИЗНАКОВ АМИНОКИСЛОТ
        # ======================================================================
        if "residues" in tensors:
            residues = tensors["residues"]
        elif "x" in tensors:
            residues = tensors["x"]
        else:
            residues = ["UNK"] * num_nodes

        if isinstance(residues, (list, np.ndarray)):
            x = one_hot_encode_residues(residues).to(self.device)
        elif isinstance(residues, torch.Tensor):
            x = residues.to(self.device)
            if x.ndim == 1:
                from scripts.build_dataset import decode_numeric_residues
                res_strings = decode_numeric_residues(residues.cpu().numpy())
                x = one_hot_encode_residues(res_strings).to(self.device)
        else:
            x = torch.zeros((num_nodes, 21), dtype=torch.float32, device=self.device)
            x[:, -1] = 1.0

        batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        # Модель предсказывает базовый шум
        pred_noise = self.model(
            x=x,
            pos=pos,
            edge_index=edge_index,
            batch_idx=batch_idx
        )

        # 1. Извлекаем значение ползунка шума из веб-приложения Streamlit
        slider_noise = 1.0
        for key in ["noise_std", "noise", "std", "sigma"]:
            if key in tensors:
                slider_noise = float(tensors[key].item() if isinstance(tensors[key], torch.Tensor) else tensors[key])
                break

        final_coords = pos - pred_noise * slider_noise

        return final_coords.cpu()

