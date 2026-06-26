import torch

from models.egnn_autoencoder import ProtEyeEGNN

class ProtEyePredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = ProtEyeEGNN()

        self.model.load_state_dict(
            torch.load(
                "weights/best_model_3.pt",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tensors):

        # Извлекаем зашумленные координаты
        pos = tensors["coords"].to(self.device)

        edge_index = tensors["edge_index"].to(self.device)

        # Модель возвращает предсказанный вектор шума
        pred_noise = self.model(
            pos,
            edge_index
        )

        # Восстанавливаем чистые координаты по формуле денойзинга
        final_coords = pos - pred_noise

        return final_coords.cpu()
