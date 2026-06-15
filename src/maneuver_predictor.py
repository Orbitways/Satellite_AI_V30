"""
maneuver_predictor.py — Détection et prédiction de manœuvres orbitales.

Détection : résidus différentiels TLE > seuil → manœuvre confirmée
Prédiction : modèle de régression simple sur l'historique des manœuvres
  - Quand : régularité temporelle (période entre manœuvres)
  - ΔV   : amplitude basée sur l'historique des ΔV précédents

Le modèle de prédiction est intentionnellement simple (pas de réseau de neurones)
pour être exploitable sans données d'entraînement massives.
"""

import os, sys, json, logging, math
import numpy as np
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
logger = logging.getLogger(__name__)


def detect_maneuvers_from_db(days: int = 7, min_dv_ms: float = 10.0,
                              emit=None) -> list:
    """
    Analyse l'historique TLE en base et détecte les manœuvres sur `days` jours.

    Méthode :
    1. Pour chaque satellite ayant ≥ 2 TLE sur la période
    2. Calculer le résidu différentiel entre TLE consécutifs
    3. Si |résidu| > seuil → manœuvre probable

    Retourne la liste des manœuvres détectées.
    """
    from tle_database import get_connection, init_db, store_maneuver
    from tle_fetcher import _validate_tle

    init_db()
    conn = get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Satellites avec ≥ 2 TLE récents en LEO
    rows = conn.execute("""
        SELECT norad_id, name, COUNT(*) as n
        FROM tle_records
        WHERE ingested_at >= ? AND orbit_class = 'LEO'
        GROUP BY norad_id HAVING n >= 2
        ORDER BY n DESC
    """, (cutoff,)).fetchall()
    conn.close()

    if emit: emit(f"Analyse de {len(rows)} satellites LEO...", 10)

    detections = []
    n_total = len(rows)

    for idx, row in enumerate(rows):
        norad_id = row["norad_id"]
        name     = row["name"]

        if emit and idx % 50 == 0:
            pct = 10 + int(idx / n_total * 75)
            emit(f"Analyse {idx+1}/{n_total} satellites...", pct)

        try:
            det = _analyze_satellite(norad_id, name, cutoff)
            if det:
                detections.extend(det)
                # Stocker en base
                for d in det:
                    store_maneuver(
                        norad_id=d["norad_id"],
                        name=d["name"],
                        delta_v_ms=d["delta_v_ms"],
                        residual_m=d["residual_m"],
                        epoch_before=d["epoch_before"],
                        epoch_after=d["epoch_after"],
                        event_type=d["event_type"],
                        severity=d["severity"],
                    )
        except Exception as e:
            logger.debug(f"Erreur analyse {norad_id}: {e}")

    # Trier par ΔV décroissant
    detections.sort(key=lambda x: x["delta_v_ms"], reverse=True)
    msg = f"✓ {len(detections)} manœuvres détectées sur {days} jours"
    logger.info(msg)
    if emit: emit(msg, 90)
    return detections


def _analyze_satellite(norad_id: str, name: str, cutoff: str) -> list:
    """Analyse les TLE d'un satellite pour détecter les manœuvres."""
    from tle_database import get_connection
    from sgp4_utils import differential_tle_residual
    from tle_fetcher import get_tle_epoch

    conn = get_connection()
    records = conn.execute("""
        SELECT tle1, tle2, epoch FROM tle_records
        WHERE norad_id = ? AND ingested_at >= ?
        ORDER BY epoch ASC
    """, (norad_id, cutoff)).fetchall()
    conn.close()

    if len(records) < 2:
        return []

    detections = []
    for i in range(len(records) - 1):
        r0, r1 = records[i], records[i+1]
        l1o, l2o = r0["tle1"], r0["tle2"]
        l1n, l2n = r1["tle1"], r1["tle2"]

        epoch = get_tle_epoch(l1o)
        if not epoch:
            continue

        # Résidu différentiel à l'époque du TLE le plus récent
        dr = differential_tle_residual(l1o, l2o, l1n, l2n, epoch)
        if dr is None:
            continue

        residual_m = float(np.linalg.norm(dr)) * 1000  # km → m

        # Seuil de détection
        if residual_m < 200:
            continue

        # Estimer ΔV depuis le résidu
        # Approximation : ΔV ≈ résidu / Δt (orbite circulaire)
        epoch_next = get_tle_epoch(l1n)
        dt_h = ((epoch_next - epoch).total_seconds() / 3600) if epoch_next else 6.0
        delta_v_ms = _estimate_dv(residual_m, dt_h)

        # Classification
        if residual_m > 50000:
            event_type, severity = "maneuver", "CRITIQUE"
        elif residual_m > 5000:
            event_type, severity = "maneuver", "MAJEUR"
        elif residual_m > 500:
            event_type, severity = "maneuver", "MODERE"
        else:
            event_type, severity = "anomaly", "FAIBLE"

        detections.append({
            "norad_id":     norad_id,
            "name":         name,
            "epoch_before": r0["epoch"],
            "epoch_after":  r1["epoch"],
            "residual_m":   round(residual_m, 1),
            "delta_v_ms":   round(delta_v_ms, 3),
            "dt_hours":     round(dt_h, 1),
            "event_type":   event_type,
            "severity":     severity,
        })

    return detections


def _estimate_dv(residual_m: float, dt_hours: float) -> float:
    """
    Estimation ΔV depuis le résidu différentiel.
    Approximation impulsionnelle : ΔV ≈ Δr / (2 × Δt_orbital / π)
    Valide pour manœuvres dans le plan orbital.
    """
    if dt_hours <= 0:
        return 0.0
    T_orbit_h = 1.5  # période typique LEO
    delta_v = residual_m / (2 * dt_hours * 3600 / math.pi)
    return max(0.0, min(delta_v, 500.0))  # cap à 500 m/s


def predict_maneuvers(emit=None) -> list:
    """
    Prédit les prochaines manœuvres depuis l'historique.

    Modèle :
    - Pour chaque satellite ayant ≥ 3 manœuvres historiques
    - Calculer la période médiane entre manœuvres
    - Prédire la prochaine à t_dernier + période_médiane
    - ΔV prédit = médiane des ΔV historiques
    - Confiance = 1 / (1 + CV) où CV est le coeff. de variation
    """
    from tle_database import get_connection, store_prediction

    conn = get_connection()
    # Satellites avec ≥ 3 manœuvres confirmées
    rows = conn.execute("""
        SELECT norad_id, name, COUNT(*) as n,
               AVG(delta_v_ms) as avg_dv,
               MAX(detected_at) as last_at
        FROM maneuvers
        WHERE event_type='maneuver'
        GROUP BY norad_id HAVING n >= 3
        ORDER BY n DESC
    """).fetchall()

    predictions = []
    now = datetime.now(timezone.utc)

    for row in rows:
        norad_id = row["norad_id"]
        mans = conn.execute("""
            SELECT detected_at, delta_v_ms FROM maneuvers
            WHERE norad_id=? AND event_type='maneuver'
            ORDER BY detected_at ASC
        """, (norad_id,)).fetchall()

        if len(mans) < 3:
            continue

        # Périodes entre manœuvres
        dates = []
        for m in mans:
            try:
                dt = datetime.fromisoformat(m["detected_at"].replace("Z","+00:00"))
                dates.append(dt)
            except:
                pass

        if len(dates) < 2:
            continue

        intervals_days = [(dates[i+1]-dates[i]).total_seconds()/86400
                          for i in range(len(dates)-1)]
        dvs = [m["delta_v_ms"] for m in mans if m["delta_v_ms"]]

        period_med = float(np.median(intervals_days))
        dv_med     = float(np.median(dvs)) if dvs else 0.0
        cv         = float(np.std(intervals_days) / (np.mean(intervals_days)+1e-6))
        confidence = round(max(0.05, min(0.95, 1.0 / (1.0 + cv))), 3)

        # Prochaine manœuvre prédite
        next_dt = dates[-1] + timedelta(days=period_med)
        if next_dt < now:
            next_dt = now + timedelta(days=period_med * 0.5)

        pred = {
            "norad_id":          norad_id,
            "name":              row["name"],
            "predicted_epoch":   next_dt.isoformat(),
            "days_until":        round((next_dt - now).total_seconds() / 86400, 1),
            "delta_v_ms_pred":   round(dv_med, 2),
            "confidence":        confidence,
            "n_historical":      len(mans),
            "period_days_med":   round(period_med, 1),
        }
        predictions.append(pred)

        store_prediction(
            norad_id=norad_id, name=row["name"],
            predicted_epoch=next_dt.isoformat(),
            delta_v_ms_pred=dv_med, confidence=confidence,
            notes=f"Basé sur {len(mans)} manœuvres, période médiane {period_med:.1f}j"
        )

    conn.close()
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    if emit: emit(f"✓ {len(predictions)} prédictions générées", 100)
    return predictions
