"""
model.py — Modèles légers pour machine embarquée.

TCN (Temporal Convolutional Network) :
- Convolutions causales dilatées → pas de fuite temporelle
- Parallélisable (contrairement au LSTM/GRU)
- ~3× moins de paramètres qu'un LSTM équivalent
- Quantifiable INT8 sans dégradation significative

GRU léger (optionnel) :
- 1 seule couche, hidden=32
- Alternative si la plateforme supporte mieux les RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─── TCN ────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Convolution causale : n'utilise que le passé (pas le futur)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Bloc résiduel TCN avec deux convolutions causales dilatées."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)

        # Connexion résiduelle (projection si dimensions différentes)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.drop(out)
        return F.relu(out + residual)


class LightTCN(nn.Module):
    """
    TCN léger pour prédiction de résidus orbitaux.

    Entrée  : (batch, window, input_size)   — séquence d'états SGP4
    Sortie  : (batch, residual_size)        — correction Δpos prédit

    Architecture :
        Embedding → N blocs TCN dilatés → tête FC
    Les dilatations doublent à chaque couche (1, 2, 4, ...) pour
    couvrir exponentiellement plus de contexte temporel.
    """

    def __init__(
        self,
        input_size: int = 7,
        residual_size: int = 3,
        channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 32, 32]

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, residual_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C) → permuter pour Conv1d : (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]   # dernier pas temporel (causal)
        return self.head(x)


# ─── GRU léger (alternative) ────────────────────────────────────────────────

class LightGRU(nn.Module):
    """
    GRU minimaliste pour machine embarquée.

    Une seule couche, hidden=32, Dropout entre la sortie GRU et la tête FC.
    Compatible avec la quantification dynamique PyTorch (torch.qint8).
    """

    def __init__(
        self,
        input_size: int = 7,
        residual_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, residual_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(model_type: str = "tcn", **kwargs) -> nn.Module:
    """Instancie le modèle selon la configuration."""
    if model_type == "tcn":
        return LightTCN(**kwargs)
    elif model_type == "gru":
        return LightGRU(**kwargs)
    else:
        raise ValueError(f"model_type inconnu : {model_type}. Choisir 'tcn' ou 'gru'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
