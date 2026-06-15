"""
predict.py — Inférence : PyTorch natif ou ONNX Runtime (embarqué).

Usage typique sur embarqué :
    predictor = OrbitalPredictor.from_onnx("models/model.onnx", scalers, sat_id=0)
    sgp4_pos  = tle_to_state(tle1, tle2, dt)[:3]
    corrected = predictor.predict_corrected(history_window, sgp4_pos)
"""

import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OrbitalPredictor:
    """
    Prédicteur orbital : SGP4 + correction IA.

    En mode ONNX (embarqué) : aucune dépendance PyTorch au runtime.
    En mode PyTorch : utilisé pendant le développement / debug.
    """

    def __init__(
        self,
        model_or_session,
        scalers: Dict,
        sat_id: int = 0,
        use_onnx: bool = False,
    ):
        self._model    = model_or_session
        self._scalers  = scalers
        self._sat_id   = sat_id
        self._use_onnx = use_onnx

    @classmethod
    def from_pytorch(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        scalers_path: str,
        sat_id: int = 0,
    ) -> "OrbitalPredictor":
        """Charge un modèle PyTorch depuis un checkpoint."""
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

        logger.info(f"Modèle PyTorch chargé : {checkpoint_path}")
        return cls(model, scalers, sat_id=sat_id, use_onnx=False)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        scalers_path: str,
        sat_id: int = 0,
    ) -> "OrbitalPredictor":
        """
        Charge un modèle ONNX Runtime (recommandé pour embarqué).
        Ne nécessite pas PyTorch à l'exécution.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime  (ou onnxruntime-gpu sur Jetson)")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

        logger.info(f"ONNX Runtime chargé : {onnx_path}")
        return cls(sess, scalers, sat_id=sat_id, use_onnx=True)

    def predict_residual(self, state_window: np.ndarray) -> np.ndarray:
        """
        Prédit le résidu Δpos = [Δx, Δy, Δz] en km.

        Args:
            state_window : (window_size, input_size) array non normalisé

        Returns:
            résidu prédit en km (espace physique)
        """
        scaler = self._scalers.get(self._sat_id)
        if scaler is None:
            raise ValueError(f"Scaler manquant pour sat_id={self._sat_id}")

        # Normalisation
        scaled = scaler.transform_states(state_window)
        x = scaled[np.newaxis].astype(np.float32)  # (1, window, features)

        if self._use_onnx:
            pred_scaled = self._model.run(None, {"input_sequence": x})[0]
        else:
            with torch.no_grad():
                pred_scaled = self._model(
                    torch.tensor(x, dtype=torch.float32)
                ).numpy()

        return scaler.inverse_residuals(pred_scaled)[0]

    def predict_corrected(
        self, state_window: np.ndarray, sgp4_position: np.ndarray
    ) -> np.ndarray:
        """
        Retourne la position corrigée = SGP4 + résidu IA.

        Args:
            state_window   : (window_size, input_size) historique d'états
            sgp4_position  : [x, y, z] position SGP4 courante (km)

        Returns:
            [x, y, z] position corrigée (km)
        """
        delta = self.predict_residual(state_window)
        return sgp4_position + delta

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Prédiction par batch (pour l'évaluation offline).
        windows : (N, window_size, input_size)
        """
        scaler = self._scalers.get(self._sat_id)
        n = len(windows)
        scaled_windows = np.array([
            scaler.transform_states(windows[i]) for i in range(n)
        ], dtype=np.float32)

        if self._use_onnx:
            preds_scaled = self._model.run(None, {"input_sequence": scaled_windows})[0]
        else:
            with torch.no_grad():
                preds_scaled = self._model(
                    torch.tensor(scaled_windows, dtype=torch.float32)
                ).numpy()

        return scaler.inverse_residuals(preds_scaled)
