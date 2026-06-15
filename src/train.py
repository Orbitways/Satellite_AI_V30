"""
train.py — Entraînement robuste avec early stopping, scheduler et gradient clipping.

Bonnes pratiques appliquées :
- Validation sur un set temporellement postérieur au train
- Early stopping sur la loss de validation (pas la loss de train)
- ReduceLROnPlateau pour adapter le learning rate
- Gradient clipping pour éviter les explosions de gradient
- Sauvegarde du meilleur checkpoint (validation loss minimale)
- Log structuré dans un fichier CSV pour analyse post-entraînement
"""

import os
import csv
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Arrête l'entraînement si la val loss ne s'améliore plus."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Retourne True si l'entraînement doit s'arrêter."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Sauvegarde en mémoire du meilleur état
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restaure les poids du meilleur checkpoint."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Meilleur modèle restauré (val_loss={self.best_loss:.6f})")


def train(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data:   Tuple[np.ndarray, np.ndarray],
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    lr_patience: int = 10,
    grad_clip: float = 1.0,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    checkpoint_path: str = "models/best_model.pth",
    log_path: str = "logs/training.csv",
) -> Dict[str, Any]:
    """
    Entraîne le modèle et retourne l'historique des métriques.

    Args:
        model          : modèle PyTorch (TCN ou GRU)
        train_data     : (X_train, y_train) arrays numpy
        val_data       : (X_val,   y_val)   arrays numpy
        epochs         : nombre max d'époques
        batch_size     : taille de batch
        lr             : learning rate initial
        weight_decay   : L2 regularisation
        patience       : early stopping patience (sur val_loss)
        lr_patience    : ReduceLROnPlateau patience
        grad_clip      : seuil de gradient clipping
        loss_type      : "huber" (robuste aux outliers) | "mse"
        huber_delta    : seuil de la Huber loss
        checkpoint_path: chemin de sauvegarde du meilleur modèle
        log_path       : chemin du CSV de log

    Returns:
        dict avec train_loss, val_loss, lr, best_val_loss, best_epoch
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    X_tr, y_tr   = _to_tensors(*train_data)
    X_val, y_val = _to_tensors(*val_data)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,      # shuffle du train loader (pas du dataset global)
        drop_last=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=0.5, verbose=False
    )
    if loss_type == "huber":
        loss_fn = nn.HuberLoss(delta=huber_delta)
    else:
        loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_epoch = 0

    # En-tête CSV
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "rmse_val", "mae_val", "lr"])

    logger.info(f"Début entraînement — {epochs} époques max, early stopping patience={patience}")
    t0 = time.time()

    for epoch in range(epochs):

        # ── Phase train ────────────────────────────────────────────────────
        model.train()
        batch_losses = []

        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        # ── Phase validation ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = loss_fn(pred_val, y_val).item()
            mae_val  = torch.mean(torch.abs(pred_val - y_val)).item()
            rmse_val = float(np.sqrt(val_loss))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Log CSV
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{rmse_val:.6f}", f"{mae_val:.6f}", f"{current_lr:.2e}"
            ])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"rmse={rmse_val:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s"
            )

        # Early stopping
        if stopper.step(val_loss, model):
            logger.info(f"Early stopping à l'époque {epoch+1} (patience={patience})")
            best_epoch = epoch + 1 - patience
            break

    stopper.restore_best(model)

    # Sauvegarde du meilleur modèle
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Meilleur modèle sauvegardé → {checkpoint_path} (val_loss={stopper.best_loss:.6f})")

    history["best_val_loss"] = stopper.best_loss
    history["best_epoch"]    = best_epoch
    return history


def _to_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
