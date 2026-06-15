"""
continual.py — Fine-tuning incrémental sur nouveaux TLE.

Deux mécanismes pour éviter le catastrophic forgetting :
1. EWC (Elastic Weight Consolidation) : pénalise la modification des poids
   importants pour les données anciennes.
2. Replay buffer : conserve un sous-ensemble de données historiques mélangées
   avec les nouvelles données lors du fine-tuning.

Référence EWC : Kirkpatrick et al., 2017 (https://arxiv.org/abs/1612.00796)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── EWC ─────────────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.

    Après l'entraînement initial, calcule la Fisher Information Matrix (FIM)
    diagonale pour chaque paramètre. Lors du fine-tuning, une pénalité
    proportionnelle à la FIM empêche les poids importants de dériver.

    Usage :
        ewc = EWC(model, train_loader)
        loss_total = loss_task + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
    ):
        self.params_mean: Dict[str, torch.Tensor] = {}
        self.fisher:      Dict[str, torch.Tensor] = {}
        self._compute(model, dataloader, device)

    def _compute(self, model: nn.Module, loader: DataLoader, device: str) -> None:
        """Calcule la FIM diagonale par rétropropagation."""
        model.eval()
        loss_fn = nn.MSELoss()

        # Sauvegarde des poids actuels (θ*)
        for name, param in model.named_parameters():
            self.params_mean[name] = param.data.clone()
            self.fisher[name]      = torch.zeros_like(param.data)

        # Accumulation des gradients au carré (FIM diagonale)
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            model.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data.clone() ** 2

        # Normalisation
        n_batches = len(loader)
        for name in self.fisher:
            self.fisher[name] /= n_batches

        logger.info("FIM calculée pour EWC")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calcule la pénalité EWC."""
        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.fisher:
                diff    = param - self.params_mean[name].detach()
                penalty = penalty + (self.fisher[name] * diff ** 2).sum()
        return penalty


# ─── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Conserve un sous-ensemble représentatif des données historiques.
    Utilisé pour mélanger ancien + nouveau lors du fine-tuning.
    """

    def __init__(self, max_ratio: float = 0.2):
        self.max_ratio = max_ratio
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def store(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stocke un sous-ensemble aléatoire des données."""
        n = len(X)
        k = max(1, int(n * self.max_ratio))
        idx = np.random.choice(n, size=k, replace=False)
        self._X = X[idx]
        self._y = y[idx]
        logger.info(f"Replay buffer : {k}/{n} échantillons conservés")

    def merge(
        self, X_new: np.ndarray, y_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fusionne les nouvelles données avec le buffer historique."""
        if self._X is None:
            return X_new, y_new

        X_merged = np.concatenate([self._X, X_new], axis=0)
        y_merged = np.concatenate([self._y, y_new], axis=0)

        # Mélange (acceptable ici car les deux sets sont déjà splittés)
        idx = np.random.permutation(len(X_merged))
        return X_merged[idx], y_merged[idx]

    def is_empty(self) -> bool:
        return self._X is None


# ─── Fine-tuning ─────────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    new_train_data: Tuple[np.ndarray, np.ndarray],
    new_val_data:   Tuple[np.ndarray, np.ndarray],
    ewc: Optional[EWC] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    ewc_lambda: float = 400.0,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    patience: int = 10,
    checkpoint_path: str = "models/best_model.pth",
) -> Dict:
    """
    Fine-tune le modèle sur de nouvelles données TLE.

    Args:
        model           : modèle à fine-tuner
        new_train_data  : (X_new, y_new) nouvelles données d'entraînement
        new_val_data    : (X_val, y_val) nouvelles données de validation
        ewc             : instance EWC (calculée sur les données précédentes)
        replay_buffer   : buffer de données historiques
        ewc_lambda      : poids de la pénalité EWC (plus grand = moins d'oubli)
        epochs          : max époques de fine-tuning
        lr              : learning rate (plus faible qu'à l'entraînement initial)
        batch_size      : taille de batch
        patience        : early stopping
        checkpoint_path : où sauvegarder le modèle affiné

    Returns:
        historique des pertes
    """
    X_new, y_new = new_train_data
    X_val, y_val = new_val_data

    # Fusion avec replay buffer
    if replay_buffer and not replay_buffer.is_empty():
        X_train, y_train = replay_buffer.merge(X_new, y_new)
        logger.info(f"Replay buffer fusionné : {len(X_train)} échantillons totaux")
    else:
        X_train, y_train = X_new, y_new

    # Tenseurs
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val,   dtype=torch.float32)
    y_v = torch.tensor(y_val,   dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": []}

    logger.info(
        f"Fine-tuning — {epochs} époques, lr={lr}, "
        f"EWC={'oui (λ='+str(ewc_lambda)+')' if ewc else 'non'}, "
        f"replay={'oui' if replay_buffer else 'non'}"
    )

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for X_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)

            # Pénalité EWC (évite de trop modifier les poids importants)
            if ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(model)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss_fn(model(X_b), y_b).item())  # loss pure

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"FT Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping fine-tuning à l'époque {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    logger.info(f"Fine-tuning terminé — meilleure val_loss={best_val_loss:.4f}")

    # Mettre à jour le replay buffer avec les nouvelles données
    if replay_buffer is not None:
        replay_buffer.store(X_new, y_new)

    return history
