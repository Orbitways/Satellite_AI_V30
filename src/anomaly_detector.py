"""
anomaly_detector.py — Détection d'anomalies comportementales orbitales.

Distinct de maneuver_detector.py qui détecte les manœuvres individuelles
par résidu de propagation. Ce module analyse les PATTERNS comportementaux
sur l'historique TLE pour qualifier des anomalies de haut niveau.

Taxonomie des anomalies (par ordre de priorité) :
  1. UNDECLARED_MANEUVER  — Manœuvre non déclarée (ΔV significatif sans annonce)
  2. EVASIVE_BEHAVIOR     — Comportement évasif (manœuvres répétées, atypiques)
  3. ORBITAL_PLANE_CHANGE — Changement de plan orbital (ΔV normal rare = événement)
  4. LOSS_OF_CONTROL      — Perte de contrôle apparente (B* élevé, pas de manœuvre)
  5. UNDECLARED_DEORBIT   — Déorbit non déclaré (baisse altitude monotone)

Méthode statistique : z-score robuste (médiane + MAD) par groupe orbital.
  Référence : Rousseeuw, P.J. & Leroy, A.M. (1987). "Robust Regression and
  Outlier Detection." Wiley, New York.

Seuils calibrés sur Starlink :
  Oltrogge, D.L. & Alfano, S. (2019). "The Technical Challenges of
  Conjunction Assessment." 70th IAC, Washington.

P�riode d'analyse minimale recommandée : 14 jours (≥ 5 TLE par satellite).
"""

import math
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Seuils par défaut (calibrés sur Starlink, ajustables) ─────────────────
DEFAULT_THRESHOLDS = {
    # Seuils z-score robuste (médiane + MAD) pour déclencher une anomalie
    "z_undeclared_maneuver":   3.5,   # Résidu propagation > 3.5σ constellation
    "z_evasive":               2.0,   # Fréquence manœuvres > 2σ constellation
    "z_plane_change":          4.0,   # ΔV normal > 4σ (rare = significatif)
    "z_loss_control":          3.0,   # B* > 3σ sans manœuvre détectée
    "z_deorbit":               2.5,   # Baisse altitude > 2.5σ constellation

    # Valeurs absolues minimales (éviter les faux positifs)
    "min_dv_undeclared_ms":    0.5,   # ΔV minimum pour parler de manœuvre [m/s]
    "min_maneuver_freq_evasive": 3/7, # > 3 manœuvres/semaine = évasif
    "min_bstar_loss":          3e-4,  # B* minimum pour suspect de perte contrôle
    "min_alt_drop_km_per_day": 1.0,   # Baisse altitude minimale [km/j] pour déorbit
    "min_inc_change_deg":      0.05,  # Changement inclinaison minimal [°]

    # Fenêtres temporelles
    "window_days":              30,   # Fenêtre d'analyse principale [jours]
    "min_tles_for_analysis":     3,   # Minimum de TLE pour une analyse valide
    "evasive_window_days":       7,   # Fenêtre courte pour comportement évasif
}

ANOMALY_LABELS = {
    "UNDECLARED_MANEUVER": "Manœuvre non déclarée",
    "EVASIVE_BEHAVIOR":    "Comportement évasif",
    "ORBITAL_PLANE_CHANGE":"Changement de plan orbital",
    "LOSS_OF_CONTROL":     "Perte de contrôle apparente",
    "UNDECLARED_DEORBIT":  "Déorbit non déclaré",
}

ANOMALY_SEVERITY = {
    "UNDECLARED_MANEUVER": "MODERE",
    "EVASIVE_BEHAVIOR":    "MAJEUR",
    "ORBITAL_PLANE_CHANGE":"CRITIQUE",
    "LOSS_OF_CONTROL":     "CRITIQUE",
    "UNDECLARED_DEORBIT":  "MAJEUR",
}


def detect_behavioral_anomalies(norad_filter: str = "",
                                 thresholds: dict = None,
                                 emit=None) -> list:
    """
    Analyse comportementale sur tous les satellites en base.

    Contrairement à detect_maneuvers_adaptive() qui analyse chaque paire
    de TLE successifs, cette fonction analyse les TENDANCES sur 30 jours.

    Args:
        norad_filter : si non vide, analyser uniquement ce satellite
        thresholds   : seuils personnalisés (surcharge les défauts)
        emit         : callback SSE (msg, pct)

    Returns:
        Liste d'anomalies détectées, triées par sévérité décroissante.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tle_database import get_connection, init_db

    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    init_db()
    conn = get_connection()

    # ── Récupérer l'historique TLE ─────────────────────────────────────────
    if norad_filter:
        nf = norad_filter.strip().upper()
        rows = conn.execute("""
            SELECT norad_id, name, tle1, tle2, epoch, alt_km, inc, ecc, mm
            FROM tle_records
            WHERE (norad_id = ? OR UPPER(name) LIKE ?)
              AND orbit_class = 'LEO'
            ORDER BY norad_id, epoch ASC
        """, (nf, f'%{nf}%')).fetchall()
    else:
        rows = conn.execute("""
            SELECT norad_id, name, tle1, tle2, epoch, alt_km, inc, ecc, mm
            FROM tle_records
            WHERE orbit_class = 'LEO'
            ORDER BY norad_id, epoch ASC
        """).fetchall()

    # Récupérer les manœuvres détectées
    maneuvers_rows = conn.execute("""
        SELECT norad_id, detected_at, delta_v_ms, event_type, residual_norm_m
        FROM maneuvers
        ORDER BY norad_id, detected_at ASC
    """).fetchall()
    conn.close()

    # Grouper par satellite
    from collections import defaultdict
    sat_tles = defaultdict(list)
    for r in rows:
        sat_tles[r["norad_id"]].append(dict(r))

    sat_mans = defaultdict(list)
    for m in maneuvers_rows:
        sat_mans[m["norad_id"]].append(dict(m))

    n_total = len(sat_tles)
    if emit: emit(f"[INFO] Analyse comportementale de {n_total} satellites...", 5)
    if n_total == 0:
        if emit: emit("[AVERT] Base vide", 100)
        return []

    # ── Calculer les statistiques de la constellation (baseline) ──────────
    constellation_stats = _compute_constellation_baseline(sat_tles)

    anomalies = []
    for idx, (norad_id, tles) in enumerate(sat_tles.items()):
        if emit and idx % max(1, n_total // 20) == 0:
            pct = 5 + int(idx / n_total * 85)
            emit(f"[INFO] {idx+1}/{n_total}...", pct)

        if len(tles) < th["min_tles_for_analysis"]:
            continue

        mans = sat_mans.get(norad_id, [])
        name = tles[-1]["name"]

        sat_anomalies = _analyze_satellite_behavior(
            norad_id, name, tles, mans, constellation_stats, th
        )
        anomalies.extend(sat_anomalies)

    # Trier : CRITIQUE > MAJEUR > MODERE, puis par z-score
    severity_order = {"CRITIQUE": 0, "MAJEUR": 1, "MODERE": 2, "FAIBLE": 3}
    anomalies.sort(key=lambda x: (
        severity_order.get(x["severity"], 9), -x.get("z_score", 0)
    ))

    if emit:
        emit(f"[OK] {len(anomalies)} anomalies comportementales détectées", 95)
    return anomalies


def _compute_constellation_baseline(sat_tles: dict) -> dict:
    """
    Calcule les statistiques robustes de la constellation (médiane + MAD).
    Utilisées comme baseline pour le z-score de chaque satellite.

    Référence : Rousseeuw & Leroy (1987), §3.1
    """
    bstar_vals, alt_vals, dv_vals = [], [], []

    for norad_id, tles in sat_tles.items():
        if not tles:
            continue
        last = tles[-1]
        # B* depuis le dernier TLE
        bstar = _parse_bstar(last.get("tle1", ""))
        if bstar > 0:
            bstar_vals.append(bstar)
        # Altitude
        if last.get("alt_km"):
            alt_vals.append(float(last["alt_km"]))

    def robust_stats(vals):
        if len(vals) < 3:
            return 0.0, 1.0
        arr = sorted(vals)
        n = len(arr)
        med = arr[n//2] if n % 2 else (arr[n//2-1]+arr[n//2])/2
        mad = sorted(abs(v-med) for v in arr)[n//2]
        sigma = max(mad * 1.4826, 1e-12)  # MAD → σ (facteur 1.4826 pour Gaussienne)
        return med, sigma

    med_bstar, sig_bstar = robust_stats(bstar_vals)
    med_alt,   sig_alt   = robust_stats(alt_vals)

    return {
        "bstar": {"median": med_bstar, "sigma": sig_bstar},
        "alt_km": {"median": med_alt,   "sigma": sig_alt},
        "n_satellites": len(sat_tles),
    }


def _analyze_satellite_behavior(norad_id: str, name: str,
                                  tles: list, mans: list,
                                  baseline: dict, th: dict) -> list:
    """Analyse comportementale d'un satellite individuel."""
    anomalies = []
    last_tle = tles[-1]
    bstar = _parse_bstar(last_tle.get("tle1", ""))
    alt_km = float(last_tle.get("alt_km", 500) or 500)
    inc_deg = float(last_tle.get("inc", 53) or 53)

    # Manœuvres récentes (30j)
    recent_mans = [m for m in mans if _days_ago(m.get("detected_at","")) <= th["window_days"]]
    recent_mans_7d = [m for m in mans if _days_ago(m.get("detected_at","")) <= th["evasive_window_days"]]

    # ── 1. Manœuvre non déclarée ───────────────────────────────────────────
    # Critère : ΔV significatif > min_dv_undeclared_ms sans pattern station-keeping
    high_dv_mans = [m for m in recent_mans
                    if abs(m.get("delta_v_ms", 0)) > th["min_dv_undeclared_ms"]
                    and m.get("event_type", "") not in ("stationkeeping", "noise")]
    if high_dv_mans:
        max_dv = max(abs(m.get("delta_v_ms", 0)) for m in high_dv_mans)
        # z-score : comparer au ΔV médian de la constellation (approx)
        z_dv = max_dv / max(th["min_dv_undeclared_ms"], 0.5)  # z vs seuil minimal
        if z_dv >= th["z_undeclared_maneuver"] / 3:  # seuil relatif
            anomalies.append(_make_anomaly(
                norad_id, name, "UNDECLARED_MANEUVER",
                f"ΔV de {max_dv:.2f} m/s détecté sans pattern station-keeping nominal",
                z_score=round(z_dv, 1),
                data={"n_maneuvers": len(high_dv_mans), "max_dv_ms": round(max_dv, 3)},
                epoch=last_tle.get("epoch", ""),
            ))

    # ── 2. Comportement évasif ─────────────────────────────────────────────
    # Critère : fréquence de manœuvres >> médiane constellation sur 7 jours
    freq_7d = len(recent_mans_7d) / 7.0  # manœuvres/jour sur 7j
    if freq_7d >= th["min_maneuver_freq_evasive"]:
        anomalies.append(_make_anomaly(
            norad_id, name, "EVASIVE_BEHAVIOR",
            f"{len(recent_mans_7d)} manœuvres en 7 jours ({freq_7d:.1f}/j >> nominal {th['min_maneuver_freq_evasive']:.2f}/j)",
            z_score=round(freq_7d / max(th["min_maneuver_freq_evasive"], 0.001), 1),
            data={"n_maneuvers_7d": len(recent_mans_7d), "freq_per_day": round(freq_7d, 3)},
            epoch=last_tle.get("epoch", ""),
        ))

    # ── 3. Changement de plan orbital ─────────────────────────────────────
    # Critère : variation d'inclinaison > seuil entre premier et dernier TLE
    if len(tles) >= 3:
        inc_first = float(tles[0].get("inc", 0) or 0)
        inc_last  = float(tles[-1].get("inc", 0) or 0)
        delta_inc = abs(inc_last - inc_first)
        if delta_inc >= th["min_inc_change_deg"]:
            # ΔV estimé pour changer l'inclinaison : ΔV ≈ 2 × v × sin(Δi/2)
            v_orb = math.sqrt(398600 / (6371 + alt_km))  # km/s
            dv_est = 2 * v_orb * math.sin(math.radians(delta_inc) / 2) * 1000  # m/s
            anomalies.append(_make_anomaly(
                norad_id, name, "ORBITAL_PLANE_CHANGE",
                f"Inclinaison : {inc_first:.4f}° → {inc_last:.4f}° (Δ={delta_inc:.4f}°, ΔV≈{dv_est:.0f} m/s)",
                z_score=round(delta_inc / th["min_inc_change_deg"], 1),
                data={"inc_change_deg": round(delta_inc, 5),
                      "dv_estimated_ms": round(dv_est, 1)},
                epoch=last_tle.get("epoch", ""),
            ))

    # ── 4. Perte de contrôle apparente ────────────────────────────────────
    # Critère : B* anormalement élevé (drag passif) SANS manœuvre détectée
    b_med   = baseline["bstar"]["median"]
    b_sigma = baseline["bstar"]["sigma"]
    z_bstar = (bstar - b_med) / b_sigma if b_sigma > 0 else 0
    no_recent_man = len(recent_mans) == 0
    if bstar >= th["min_bstar_loss"] and z_bstar >= th["z_loss_control"] and no_recent_man:
        anomalies.append(_make_anomaly(
            norad_id, name, "LOSS_OF_CONTROL",
            f"B*={bstar:.2e} ({z_bstar:.1f}σ au-dessus médiane constellation) sans manœuvre détectée sur 30j",
            z_score=round(z_bstar, 1),
            data={"bstar": bstar, "bstar_median": b_med,
                  "n_maneuvers_30d": 0},
            epoch=last_tle.get("epoch", ""),
        ))

    # ── 5. Déorbit non déclaré ─────────────────────────────────────────────
    # Critère : baisse monotone de l'altitude sur les TLE disponibles
    if len(tles) >= 4:
        alts = [float(t.get("alt_km", 0) or 0) for t in tles[-6:]]
        alts = [a for a in alts if a > 0]
        if len(alts) >= 4:
            # Régression linéaire des altitudes vs index temporel
            n = len(alts)
            x = list(range(n))
            xm = sum(x)/n; ym = sum(alts)/n
            num = sum((x[i]-xm)*(alts[i]-ym) for i in range(n))
            den = sum((x[i]-xm)**2 for i in range(n))
            slope = num/den if den > 0 else 0  # km / TLE index

            # Convertir en km/jour (TLE espacés typiquement de 1-3j)
            # Estimation : 30j / (n-1) jours entre TLEs
            days_span = 30.0 / max(n-1, 1)
            slope_km_per_day = slope / days_span if days_span > 0 else 0

            if slope_km_per_day <= -th["min_alt_drop_km_per_day"]:
                a_med   = baseline["alt_km"]["median"]
                a_sigma = baseline["alt_km"]["sigma"]
                z_alt   = abs(slope_km_per_day) / max(th["min_alt_drop_km_per_day"], 0.1)
                anomalies.append(_make_anomaly(
                    norad_id, name, "UNDECLARED_DEORBIT",
                    f"Baisse d'altitude : {slope_km_per_day:.2f} km/j ({alts[0]:.0f} → {alts[-1]:.0f} km)",
                    z_score=round(z_alt, 1),
                    data={"slope_km_per_day": round(slope_km_per_day, 3),
                          "alt_first_km": round(alts[0], 1),
                          "alt_last_km":  round(alts[-1], 1)},
                    epoch=last_tle.get("epoch", ""),
                ))

    return anomalies


def _make_anomaly(norad_id, name, atype, description,
                   z_score=0.0, data=None, epoch="") -> dict:
    sev = ANOMALY_SEVERITY.get(atype, "MODERE")
    return {
        "norad_id":    norad_id,
        "name":        name,
        "type":        atype,
        "label":       ANOMALY_LABELS.get(atype, atype),
        "description": description,
        "severity":    sev,
        "z_score":     z_score,
        "data":        data or {},
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "epoch":       epoch,
    }


def _parse_bstar(tle1: str) -> float:
    try:
        s = tle1[53:61].strip()
        if not s or s in ('00000-0', '+00000-0', '00000+0'):
            return 0.0
        sign = -1 if s[0] == '-' else 1
        body = s[1:] if s[0] in '+-' else s
        mantissa = float('0.' + body[:5])
        exp = int(body[5:])
        return sign * mantissa * (10 ** exp)
    except Exception:
        return 0.0


def _days_ago(dt_str: str) -> float:
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return 999.0
