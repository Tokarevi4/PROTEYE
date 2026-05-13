"""
Training loop for the ConformerGenerator.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from proTeye.models.conformer import ConformerGenerator
from proTeye.training.losses import diffusion_loss


class Trainer:
    """Train a :class:`ConformerGenerator` on a protein dataset.

    Parameters
    ----------
    model :
        The model to train.
    lr :
        Learning rate for Adam.
    weight_decay :
        L2 regularisation coefficient.
    device :
        Torch device used for training.
    clip_grad_norm :
        Maximum gradient norm for clipping (disabled when ``None``).
    checkpoint_dir :
        Directory where model checkpoints are saved after each epoch.
        If *None*, no checkpoints are written.
    """

    def __init__(
        self,
        model: ConformerGenerator,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        log_every: int = 10,
    ) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        train_loader :
            DataLoader over training :class:`~proTeye.data.graph_builder.ProteinGraph` objects.
        val_loader :
            Optional DataLoader for validation.
        num_epochs :
            Number of training epochs.
        log_every :
            Print progress every this many batches.

        Returns
        -------
        history : dict with ``train_loss`` and (optionally) ``val_loss`` lists.
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, epoch, log_every)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._val_epoch(val_loader)
                history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else "  val_loss=N/A"
            print(
                f"Epoch {epoch:03d}/{num_epochs}  "
                f"train_loss={train_loss:.4f}{val_str}  "
                f"({elapsed:.1f}s)"
            )

            if self.checkpoint_dir:
                self._save_checkpoint(epoch, train_loss)

        return history

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        log_every: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, graph in enumerate(loader, 1):
            graph = self._to_device(graph)
            self.optimizer.zero_grad()

            eps_pred, eps_true = self.model(graph)
            loss = diffusion_loss(eps_pred, eps_true)
            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % log_every == 0:
                avg = total_loss / batch_idx
                print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  loss={avg:.4f}")

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for graph in loader:
            graph = self._to_device(graph)
            eps_pred, eps_true = self.model(graph)
            total_loss += diffusion_loss(eps_pred, eps_true).item()
        return total_loss / max(len(loader), 1)

    def _to_device(self, graph):
        """Move all tensors in a ProteinGraph to the training device."""
        from proTeye.data.graph_builder import ProteinGraph

        return ProteinGraph(
            node_features=graph.node_features.to(self.device),
            ca_coords=graph.ca_coords.to(self.device),
            edge_index=graph.edge_index.to(self.device),
            edge_features=graph.edge_features.to(self.device),
            aa_indices=graph.aa_indices.to(self.device),
        )

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
