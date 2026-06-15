"""
config.py — Paramètres centralisés du pipeline satellite_ai_v2.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Données ──────────────────────────────────────────────────────────
    tle_source: str = "data/sample_tle.txt"
    step_minutes: int = 5
    points_per_sat: int = 1000
    window_size: int = 20
    horizon: int = 1

    # ── Split chronologique ───────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # ── Modèle ───────────────────────────────────────────────────────────
    model_type: str = "tcn"          # "tcn" | "gru"
    input_size: int = 8              # [x, y, z, vx, vy, vz, delta_t, sat_id]
    residual_size: int = 3           # [Δx, Δy, Δz]
    tcn_channels: List[int] = field(default_factory=lambda: [32, 32, 32])
    tcn_kernel_size: int = 3
    gru_hidden: int = 32
    gru_layers: int = 1
    dropout: float = 0.1

    # ── Entraînement ─────────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    lr_patience: int = 10
    grad_clip: float = 1.0
    loss: str = "huber"              # "mse" | "huber" (robuste aux outliers)
    huber_delta: float = 1.0

    # ── Continual learning ───────────────────────────────────────────────
    ewc_lambda: float = 400.0
    replay_ratio: float = 0.2
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4

    # ── Chemins ──────────────────────────────────────────────────────────
    model_dir: str = "models"
    log_dir: str = "logs"
    scaler_path: str = "models/scalers.pkl"
    checkpoint_path: str = "models/best_model.pth"
    onnx_path: str = "models/model.onnx"
    onnx_int8_path: str = "models/model_int8.onnx"

    # ── Inférence embarquée ──────────────────────────────────────────────
    use_onnx: bool = False

    def summary(self) -> str:
        ch = str(self.tcn_channels) if self.model_type == "tcn" else f"hidden={self.gru_hidden}"
        return "\n".join([
            "── Config satellite_ai_v2 " + "─" * 30,
            f"  Modèle    : {self.model_type.upper()}  {ch}",
            f"  Input     : {self.input_size}  (x,y,z,vx,vy,vz,Δt,sat_id)",
            f"  Fenêtre   : {self.window_size} pas × {self.step_minutes} min = {self.window_size*self.step_minutes} min",
            f"  Points    : {self.points_per_sat}/satellite  (≈{self.points_per_sat*self.step_minutes/60:.0f}h)",
            f"  Split     : {self.train_ratio:.0%}/{self.val_ratio:.0%}/{1-self.train_ratio-self.val_ratio:.0%} chronologique",
            f"  Loss      : {self.loss}" + (f"  δ={self.huber_delta}" if self.loss == "huber" else ""),
            f"  Epochs    : {self.epochs}  lr={self.lr}  patience={self.patience}",
            "─" * 56,
        ])


cfg = Config()
