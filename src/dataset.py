"""
dataset.py — Construction du dataset avec split chronologique strict.

Règles clés :
- Le split train/val/test est TOUJOURS chronologique (pas de shuffle global).
- Les fenêtres glissantes ne chevauchent JAMAIS deux satellites différents.
- Chaque satellite a son propre StandardScaler (scalers indépendants).
- Le shuffle est appliqué UNIQUEMENT à l'intérieur du set d'entraînement.
"""

import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Types d'alias
StateArray    = np.ndarray  # (N, input_size)
ResidualArray = np.ndarray  # (N, residual_size)
WindowArray   = np.ndarray  # (samples, window, features)


class SatelliteScaler:
    """
    Scalers séparés pour les états et les résidus, par satellite.
    Garantit que le scaler est fitté UNIQUEMENT sur le train set.
    """

    def __init__(self):
        self.state_scaler    = StandardScaler()
        self.residual_scaler = StandardScaler()
        self._fitted = False

    def fit(self, states: StateArray, residuals: ResidualArray) -> "SatelliteScaler":
        self.state_scaler.fit(states)
        self.residual_scaler.fit(residuals)
        self._fitted = True
        return self

    def transform_states(self, states: StateArray) -> StateArray:
        assert self._fitted, "Scaler non fitté — appeler fit() d'abord"
        return self.state_scaler.transform(states)

    def transform_residuals(self, residuals: ResidualArray) -> ResidualArray:
        assert self._fitted, "Scaler non fitté"
        return self.residual_scaler.transform(residuals)

    def inverse_residuals(self, scaled: ResidualArray) -> ResidualArray:
        return self.residual_scaler.inverse_transform(scaled)


def chronological_split(
    states: StateArray,
    residuals: ResidualArray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[StateArray, ResidualArray],
    Tuple[StateArray, ResidualArray],
    Tuple[StateArray, ResidualArray],
]:
    """
    Split temporel strict : les données sont conservées dans l'ordre
    chronologique. Aucun point futur ne contaminer le train set.
    """
    n = len(states)
    i_val  = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    train = states[:i_val],      residuals[:i_val]
    val   = states[i_val:i_test], residuals[i_val:i_test]
    test  = states[i_test:],     residuals[i_test:]

    logger.info(
        f"Split chronologique — train: {len(train[0])}, "
        f"val: {len(val[0])}, test: {len(test[0])}"
    )
    return train, val, test


def make_windows(
    states: StateArray,
    residuals: ResidualArray,
    window: int,
    horizon: int = 1,
    shuffle: bool = False,
) -> Tuple[WindowArray, np.ndarray]:
    """
    Construit des fenêtres glissantes (X, y) depuis une série chronologique.

    X[i] = states[i : i+window]        → (window, input_size)
    y[i] = residuals[i+window+horizon-1] → (residual_size,)

    Le shuffle n'est JAMAIS appliqué sur val/test.
    Sur le train set, le shuffle est appliqué après la construction des fenêtres
    pour éviter le biais de gradient — mais le split temporel est déjà fait.
    """
    X, y = [], []
    n = len(states)

    for i in range(n - window - horizon + 1):
        X.append(states[i : i + window])
        y.append(residuals[i + window + horizon - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if shuffle and len(X) > 0:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

    return X, y


def build_dataset(
    satellite_data: List[Tuple[StateArray, ResidualArray]],
    window: int = 20,
    horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[WindowArray, np.ndarray],
    Tuple[WindowArray, np.ndarray],
    Tuple[WindowArray, np.ndarray],
    Dict[int, SatelliteScaler],
]:
    """
    Construit le dataset complet en traitant chaque satellite indépendamment.

    Pour chaque satellite :
    1. Split chronologique
    2. Fit du scaler sur le train set uniquement
    3. Normalisation train/val/test
    4. Construction des fenêtres glissantes (sans chevauchement inter-satellites)

    Les jeux train/val/test sont ensuite concaténés tous satellites confondus.
    """
    scalers: Dict[int, SatelliteScaler] = {}
    train_X_all, train_y_all = [], []
    val_X_all,   val_y_all   = [], []
    test_X_all,  test_y_all  = [], []

    for sat_id, (states, residuals) in enumerate(satellite_data):

        # 1. Split chronologique
        (s_tr, r_tr), (s_val, r_val), (s_test, r_test) = chronological_split(
            states, residuals, train_ratio, val_ratio
        )

        # 2. Scaler fitté UNIQUEMENT sur le train
        scaler = SatelliteScaler()
        scaler.fit(s_tr, r_tr)
        scalers[sat_id] = scaler

        # 3. Normalisation
        s_tr_n   = scaler.transform_states(s_tr)
        s_val_n  = scaler.transform_states(s_val)
        s_test_n = scaler.transform_states(s_test)
        r_tr_n   = scaler.transform_residuals(r_tr)
        r_val_n  = scaler.transform_residuals(r_val)
        r_test_n = scaler.transform_residuals(r_test)

        # 4. Fenêtres glissantes (shuffle seulement sur train)
        X_tr, y_tr     = make_windows(s_tr_n,   r_tr_n,   window, horizon, shuffle=True)
        X_val, y_val   = make_windows(s_val_n,  r_val_n,  window, horizon, shuffle=False)
        X_test, y_test = make_windows(s_test_n, r_test_n, window, horizon, shuffle=False)

        if len(X_tr) == 0:
            logger.warning(f"SAT {sat_id} : pas assez de données pour le window.")
            continue

        train_X_all.append(X_tr);   train_y_all.append(y_tr)
        val_X_all.append(X_val);     val_y_all.append(y_val)
        test_X_all.append(X_test);   test_y_all.append(y_test)

        logger.info(
            f"SAT {sat_id} → train {len(X_tr)}, val {len(X_val)}, test {len(X_test)} fenêtres"
        )

    # Concaténation multi-satellites
    def stack(lst):
        return np.concatenate(lst, axis=0) if lst else np.array([])

    train = stack(train_X_all), stack(train_y_all)
    val   = stack(val_X_all),   stack(val_y_all)
    test  = stack(test_X_all),  stack(test_y_all)

    logger.info(
        f"Dataset final — train: {len(train[0])}, val: {len(val[0])}, test: {len(test[0])}"
    )
    return train, val, test, scalers


def save_scalers(scalers: Dict[int, SatelliteScaler], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers sauvegardés → {path}")


def load_scalers(path: str) -> Dict[int, SatelliteScaler]:
    with open(path, "rb") as f:
        return pickle.load(f)
