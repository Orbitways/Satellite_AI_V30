"""
sgp4_utils.py — Calcul SGP4 et génération de résidus.

Deux modes de résidus :
  1. simulate_residual()        → résidus simulés (développement / tests)
  2. differential_tle_residual() → résidus RÉELS depuis deux époques TLE
                                   (production — source : Celestrak toutes ~2h)

Le vecteur d'état retourné est [x, y, z, vx, vy, vz] en km et km/s
dans le référentiel ECI (Earth-Centered Inertial).
"""

import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)

# Perturbations physiques simulées (ordre de grandeur réaliste pour LEO)
RESIDUAL_SCALE_KM = {
    "drag":        0.5,   # traînée atmosphérique (principal en LEO)
    "solar":       0.1,   # pression de radiation solaire
    "gravity":     0.08,  # harmoniques gravitationnels (J3, J4+)
    "third_body":  0.02,  # attractions Lune/Soleil
}


def tle_to_state(tle1: str, tle2: str, dt: datetime) -> Optional[np.ndarray]:
    """
    Calcule le vecteur d'état SGP4 [x, y, z, vx, vy, vz] pour un instant dt.
    Retourne None si SGP4 signale une erreur (orbite dégénérée, etc.).
    """
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)

    if e != 0:
        logger.debug(f"SGP4 erreur {e} pour dt={dt}")
        return None

    return np.array(list(r) + list(v), dtype=np.float64)


def simulate_residual(state: np.ndarray, t_seconds: float,
                      sat_id: int = 0, seed: int = 42) -> np.ndarray:
    """
    Simule un résidu physiquement cohérent Δpos = [Δx, Δy, Δz] en km.

    En production : remplacer par des résidus calculés depuis des données
    de mesure réelles (TLE précis - TLE opérationnel, ou ranging radar).

    Le résidu croît avec le temps (dérive), inclut une composante périodique
    orbitale, et varie par satellite (seed différente).
    """
    rng = np.random.default_rng(seed=seed + int(t_seconds / 300))

    # Dérive temporelle (drag atmosphérique dominant)
    t_hours = t_seconds / 3600.0
    drag_drift = RESIDUAL_SCALE_KM["drag"] * (1 - np.exp(-t_hours / 48.0))

    # Composante périodique (période orbitale ~90 min pour LEO)
    orbital_period_s = 5400.0
    phase = 2 * np.pi * t_seconds / orbital_period_s
    periodic = 0.05 * np.array([np.sin(phase), np.cos(phase), np.sin(phase / 2)])

    # Bruit de mesure (blanc)
    noise = rng.normal(0, 0.02, size=3)

    # Résidu total sur la position [Δx, Δy, Δz]
    residual = np.array([
        drag_drift * rng.normal(0.6, 0.1),
        drag_drift * rng.normal(0.3, 0.1),
        drag_drift * rng.normal(0.1, 0.05),
    ]) + periodic + noise

    return residual.astype(np.float64)


def generate_trajectory(
    tle1: str, tle2: str, sat_id: int,
    n_points: int = 1000, step_minutes: int = 5,
    start: Optional[datetime] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère la trajectoire SGP4 et les résidus simulés pour un satellite.

    Retourne :
        states   : array (N, 8) = [x, y, z, vx, vy, vz, delta_t, sat_id]
        residuals: array (N, 3) = [Δx, Δy, Δz] (cible de l'IA)

    Le sat_id est encodé directement dans le vecteur d'état pour que l'IA
    distingue les satellites sans mélanger leurs fenêtres glissantes.
    """
    if start is None:
        start = datetime.now(timezone.utc).replace(tzinfo=None)

    states_list = []
    residuals_list = []

    for i in range(n_points):
        dt = start + timedelta(minutes=i * step_minutes)
        t_seconds = i * step_minutes * 60.0

        state = tle_to_state(tle1, tle2, dt)
        if state is None:
            continue

        residual = simulate_residual(state, t_seconds, sat_id=sat_id, seed=sat_id * 1000)

        # Ajouter delta_t normalisé et sat_id au vecteur d'état
        extended = np.append(state, [t_seconds, float(sat_id)])
        states_list.append(extended)
        residuals_list.append(residual)

    if not states_list:
        raise ValueError(f"Aucun état valide généré pour le satellite {sat_id}")

    states = np.array(states_list, dtype=np.float64)
    residuals = np.array(residuals_list, dtype=np.float64)

    logger.info(f"[SAT {sat_id}] {len(states)} points générés")
    return states, residuals


# ─── Résidus réels par TLE différentiels ─────────────────────────────────────

def differential_tle_residual(
    tle1_old: str, tle2_old: str,
    tle1_new: str, tle2_new: str,
    dt: datetime,
) -> Optional[np.ndarray]:
    """
    Calcule le résidu RÉEL entre deux époques TLE pour un même satellite.

    Principe (Approche A — sans radar) :
        position_ref  = SGP4(TLE_nouveau, t)   ← "vérité terrain" (TLE récent)
        position_pred = SGP4(TLE_ancien,  t)   ← prédiction depuis l'ancien TLE
        résidu        = position_ref - position_pred  (km)

    Celestrak publie de nouveaux TLE toutes ~2h pour l'ISS, ~12h pour LEO général.
    En ingérant ces TLE successifs, on peut construire un dataset de résidus réels
    sans aucun équipement radar.

    Usage typique :
        # Charger deux TLE successifs depuis Celestrak
        residual = differential_tle_residual(tle1_t0, tle2_t0, tle1_t1, tle2_t1, dt=now)
        # residual est le vecteur Δpos [Δx, Δy, Δz] en km

    Retourne None si l'un des deux SGP4 échoue.
    """
    pos_old = tle_to_state(tle1_old, tle2_old, dt)
    pos_new = tle_to_state(tle1_new, tle2_new, dt)

    if pos_old is None or pos_new is None:
        return None

    # Résidu position uniquement [Δx, Δy, Δz]
    return (pos_new[:3] - pos_old[:3]).astype(np.float64)


def build_real_residuals_from_tle_history(
    tle_history: list,   # liste de (name, tle1, tle2) ordonnée chronologiquement
    sat_id: int = 0,
    step_minutes: int = 5,
    n_points_per_pair: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un dataset de résidus RÉELS depuis un historique de TLE successifs.

    Pour chaque paire de TLE consécutifs (t_i, t_{i+1}) :
    - On propage les deux depuis t_i jusqu'à t_{i+1}
    - Le résidu à chaque pas = SGP4(nouveau TLE) - SGP4(ancien TLE)

    C'est la méthode de production recommandée pour entraîner le modèle
    sur de vraies perturbations (drag, pression solaire, etc.)

    Args:
        tle_history          : liste de TLE ordonnés par date d'époque
        sat_id               : identifiant satellite (pour le vecteur d'état)
        step_minutes         : résolution temporelle
        n_points_per_pair    : points générés entre chaque paire TLE

    Returns:
        states    : (N, 8) — vecteurs d'état SGP4
        residuals : (N, 3) — résidus réels [Δx, Δy, Δz] en km
    """
    from tle_fetcher import get_tle_epoch

    if len(tle_history) < 2:
        raise ValueError("Il faut au moins 2 TLE pour calculer des résidus différentiels.")

    all_states, all_residuals = [], []

    for i in range(len(tle_history) - 1):
        _, tle1_old, tle2_old = tle_history[i]
        _, tle1_new, tle2_new = tle_history[i + 1]

        epoch_old = get_tle_epoch(tle1_old)
        if epoch_old is None:
            continue

        for j in range(n_points_per_pair):
            dt = epoch_old + timedelta(minutes=j * step_minutes)
            t_s = j * step_minutes * 60.0

            state = tle_to_state(tle1_old, tle2_old, dt)
            residual = differential_tle_residual(tle1_old, tle2_old, tle1_new, tle2_new, dt)

            if state is None or residual is None:
                continue

            extended = np.append(state, [t_s, float(sat_id)])
            all_states.append(extended)
            all_residuals.append(residual)

    if not all_states:
        raise ValueError("Aucun résidu calculable depuis l'historique TLE fourni.")

    states    = np.array(all_states,    dtype=np.float64)
    residuals = np.array(all_residuals, dtype=np.float64)
    logger.info(
        f"[SAT {sat_id}] {len(states)} résidus réels générés depuis "
        f"{len(tle_history)-1} paires TLE"
    )
    return states, residuals
