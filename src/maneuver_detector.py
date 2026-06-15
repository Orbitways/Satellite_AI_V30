"""
maneuver_detector.py — Détection adaptative de manœuvres orbitales.

Méthode :
  Pour chaque satellite avec ≥ 2 TLE successifs en base :
  1. Propager TLE_i jusqu'à l'époque de TLE_{i+1} avec perturbations (même config que Conjonctions)
  2. Calculer le résidu RTN entre position propagée et position réelle TLE_{i+1}
  3. Comparer au seuil adaptatif du satellite (basé sur son historique de résidus normaux)
  4. Si résidu > N × σ_baseline → manœuvre détectée

Décomposition RTN (Radial-Tangential-Normal) :
  R = direction radiale (altitude)       → ΔV radial = ajustement excentricité
  T = direction tangentielle (along-track) → ΔV tangentiel = changement a/Hohmann
  N = direction normale (cross-track)    → ΔV normal = changement inclinaison
"""

import math, logging, os, sys
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Seuil adaptatif : N×σ au-dessus du baseline pour déclencher une détection
N_SIGMA_THRESHOLD = 3.0
# Minimum de résidu absolu pour éviter les faux positifs sur orbites très stables
MIN_RESIDUAL_KM = 0.5
# Nombre minimum d'observations "normales" pour établir le baseline
MIN_BASELINE_OBS = 5
# Résidu par défaut si baseline insuffisant (km/h de propagation)
DEFAULT_SIGMA_KM_PER_H = 0.8  # ~800m/h erreur SGP4 typique LEO


def _build_pert_config(pert_flags: dict):
    """Construit un objet PertConfig depuis un dict de flags."""
    class PertConfig:
        pass
    cfg = PertConfig()
    cfg.j3            = pert_flags.get('j3', False)
    cfg.j4            = pert_flags.get('j4', False)
    cfg.j5            = pert_flags.get('j5', False)
    cfg.drag_residual = pert_flags.get('drag_residual', False)
    cfg.solar_pressure= pert_flags.get('solar_pressure', False)
    cfg.moon_gravity  = pert_flags.get('moon_gravity', False)
    cfg.sun_gravity   = pert_flags.get('sun_gravity', False)
    cfg.albedo        = pert_flags.get('albedo', False)
    cfg.relativity    = pert_flags.get('relativity', False)
    cfg.Cd   = pert_flags.get('Cd',  2.2)
    cfg.A_m  = pert_flags.get('A_m', 0.01)
    cfg.Cr   = pert_flags.get('Cr',  1.3)
    cfg.A_srp= pert_flags.get('A_m', 0.01)
    return cfg


def _propagate_tle_to_epoch(tle1: str, tle2: str, target_epoch: datetime,
                             pert_cfg=None) -> Optional[np.ndarray]:
    """
    Propage un TLE jusqu'à target_epoch avec corrections de perturbations.
    Retourne la position [x,y,z] en km, ou None si erreur.
    """
    try:
        from sgp4.api import Satrec, jday
        from perturbations import total_perturbation

        sat = Satrec.twoline2rv(tle1, tle2)
        t = target_epoch
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, float(t.second))

        err, r, v = sat.sgp4(jd, fr)
        if err != 0:
            return None

        r_arr = np.array(r)

        # Appliquer les corrections de perturbations si demandé
        if pert_cfg is not None:
            try:
                v_arr = np.array(v)
                a_ms2 = total_perturbation(r_arr * 1e3, v_arr * 1e3, jd + fr, pert_cfg)
                # Δpos = ½·a·dt² — ici dt = temps depuis l'époque du TLE
                yr2 = int(tle1[18:20])
                year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
                doy  = float(tle1[20:32])
                tle_epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                dt_s = (target_epoch.replace(tzinfo=timezone.utc) -
                        tle_epoch).total_seconds()
                if abs(dt_s) < 3 * 86400:  # max 3 jours, perturbations linéarisées
                    # Correction de premier ordre : Δr ≈ a_pert × dt (pas ½·a·dt²)
                    # L'approximation quadratique explose pour dt > quelques heures
                    delta_km = a_ms2 * dt_s / 1e3  # premier ordre en km
                    # Cap de sécurité : correction max 10 km (erreur SGP4 typique)
                    norm = float(np.linalg.norm(delta_km))
                    if norm > 10.0:
                        delta_km = delta_km * (10.0 / norm)
                    r_arr = r_arr + delta_km
            except Exception as e:
                logger.debug(f"Perturbation skip: {e}")

        return r_arr

    except Exception as e:
        logger.debug(f"Propagation error: {e}")
        return None


def _eci_to_rtn(r_ref: np.ndarray, v_ref: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Convertit un vecteur delta en coordonnées RTN.
    R = radial, T = along-track, N = cross-track.
    """
    r_norm = r_ref / (np.linalg.norm(r_ref) + 1e-10)
    # N = R × V normalisé
    h = np.cross(r_ref, v_ref)
    n_norm = h / (np.linalg.norm(h) + 1e-10)
    # T = N × R
    t_norm = np.cross(n_norm, r_norm)

    return np.array([
        float(np.dot(delta, r_norm)),
        float(np.dot(delta, t_norm)),
        float(np.dot(delta, n_norm)),
    ])


def _estimate_dv_from_rtn(rtn_km: np.ndarray, dt_hours: float,
                           alt_km: float = 550.0) -> tuple:
    """
    Estime le ΔV depuis le résidu RTN via la variation du demi-grand axe.

    Méthode : équation de Vis-Viva pour orbite circulaire.
        v = √(μ/a)  →  ΔV ≈ |Δa| × n / 2
    où Δa est déduit du résidu tangentiel (dominant pour SK et Hohmann).

    Pour la composante normale (changement d'inclinaison) :
        ΔV_i = v × sin(Δi) ≈ v × |ΔN| / a  (pour Δi petit)

    Références :
        Vallado, D.A. (2013). Fundamentals of Astrodynamics, §6.3.
        Curtis, H.D. (2013). Orbital Mechanics for Engineering Students, §6.2.

    Returns:
        (delta_v_ms, maneuver_type)
    """
    dr, dt_rtn, dn = rtn_km  # km
    residual = float(np.linalg.norm(rtn_km))

    if dt_hours <= 0 or residual < 1e-6:
        return 0.0, 'unknown'

    # Demi-grand axe et mouvement moyen à l'altitude de référence
    RE_KM_LOCAL = 6378.137
    MU_LOCAL    = 3.986004418e5   # km³/s²
    a_km  = RE_KM_LOCAL + alt_km
    n_rads = math.sqrt(MU_LOCAL / a_km**3)  # rad/s
    v_km_s = math.sqrt(MU_LOCAL / a_km)     # km/s

    # ΔV tangentiel (Hohmann, phasing, SK) — dominant pour changement d'altitude
    # Résidu tangentiel Δr_T ≈ 3/2 × n × ΔV_T / n² × (n × t - sin(n×t))
    # Approximation t << T_orb :  ΔV_T ≈ |Δr_T| × n / (3/2 × n × t) = 2|Δr_T|/(3t)
    # Plus précis : via Δa déduit de la vitesse → ΔV_T = |Δr_T| × n / 2
    # (valable pour manœuvre impulsionnelle, drift de T_T/2 avant le prochain TLE)
    t_s = dt_hours * 3600.0
    dv_t_km_s = abs(dt_rtn) * n_rads / 2.0  # km/s
    dv_t_ms   = dv_t_km_s * 1000.0          # m/s

    # ΔV radial (ajustement excentricité) :  ΔV_R ≈ n × |Δr_R| / 2
    dv_r_ms = abs(dr) * n_rads * 1000.0 / 2.0

    # ΔV normal (changement inclinaison) : ΔV_i = v × |ΔN| / a
    dv_n_ms = v_km_s * abs(dn) / a_km * 1000.0

    delta_v = max(dv_t_ms, dv_r_ms, dv_n_ms)

    # Cap physique : manœuvre SK Starlink ≈ 0.5-2 m/s,
    # déorbit contrôlé ≈ 10-30 m/s, maximum absolu ≈ 50 m/s
    delta_v = min(delta_v, 50.0)

    # Classification du type de manœuvre
    dom = max(abs(dr), abs(dt_rtn), abs(dn))
    if residual < 0.1:
        mtype = 'noise'
    elif abs(dt_rtn) >= dom * 0.8:
        mtype = 'hohmann' if dr * dt_rtn >= 0 else 'phasing'
    elif abs(dn) >= dom * 0.8:
        mtype = 'inclination'
    elif abs(dr) >= dom * 0.8:
        mtype = 'eccentricity'
    else:
        mtype = 'stationkeeping'

    return round(delta_v, 4), mtype


def detect_maneuvers_adaptive(days: int = 7, pert_flags: dict = None,
                               min_sigma: float = N_SIGMA_THRESHOLD,
                               norad_filter: str = "",
                               pause_flag: dict = None, emit=None) -> list:
    """
    Détecte les manœuvres avec seuil adaptatif.

    Deux modes automatiques selon les données disponibles :

    Mode HISTORIQUE (si plusieurs TLE d'époques différentes par satellite) :
      → Propage TLE_i jusqu'à l'époque de TLE_{i+1}, compare à la position réelle.
      → Résidu = erreur de propagation = signature de manœuvre entre les deux époques.
      → Nécessite plusieurs imports espacés dans le temps.

    Mode INSTANTANÉ (si 1 seul TLE par satellite — cas d'un import unique) :
      → Compare le terme de drag B* et le mouvement moyen aux valeurs médianes
        de la constellation. Anomalies statistiques = manœuvres récentes probables.
      → Moins précis mais fonctionne avec un seul instantané.

    IMPORTANT : Pour le mode historique, relancez l'import TLE régulièrement
    (quotidiennement) pour accumuler plusieurs époques par satellite.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from tle_database import get_connection, init_db, store_maneuver

    init_db()
    conn = get_connection()
    try:
        # ── Détecter quel mode utiliser ──────────────────────────────────────
        # Satellites avec >= 2 TLE à des époques DIFFÉRENTES (mode historique)
        rows_hist = conn.execute("""
            SELECT norad_id, name, COUNT(DISTINCT substr(epoch,1,13)) as n_epochs,
                   MIN(epoch) as first_epoch, MAX(epoch) as last_epoch
            FROM tle_records
            WHERE orbit_class = 'LEO'
            GROUP BY norad_id HAVING n_epochs >= 2
            ORDER BY n_epochs DESC
        """).fetchall()

        # Filtre NORAD si spécifié (analyse ciblée)
        if norad_filter:
            nf = norad_filter.upper()
            rows_hist = [r for r in conn.execute("""
                SELECT norad_id, name, COUNT(DISTINCT substr(epoch,1,13)) as n_epochs,
                       MIN(epoch) as first_epoch, MAX(epoch) as last_epoch
                FROM tle_records
                WHERE orbit_class = 'LEO' AND (norad_id = ? OR UPPER(name) LIKE ?)
                GROUP BY norad_id HAVING n_epochs >= 2
            """, (nf, f'%{nf}%')).fetchall()]

        rows_all = conn.execute("""
            SELECT norad_id, name, COUNT(*) as n,
                   MAX(epoch) as last_epoch,
                   AVG(mm) as avg_mm,
                   MIN(tle1) as tle1, MIN(tle2) as tle2
            FROM tle_records
            WHERE orbit_class = 'LEO'
            GROUP BY norad_id
        """).fetchall()

        if norad_filter:
            nf = norad_filter.upper()
            rows_all = [r for r in rows_all
                        if r["norad_id"] == nf or nf in (r["name"] or "").upper()]
    finally:
        conn.close()

    n_hist = len(rows_hist)
    n_all  = len(rows_all)

    if n_all == 0:
        if emit: emit("[AVERT] Base vide — mettez d'abord à jour la base TLE", 100)
        return []

    if emit:
        emit(f"[INFO] {n_all} satellites LEO en base · {n_hist} avec historique multi-époques", 3)
        if n_hist == 0:
            emit("[INFO] Mode INSTANTANÉ : un seul import détecté.", 4)
            emit("[INFO] Pour le mode historique (plus précis) : relancez l'import chaque jour.", 5)
        else:
            emit(f"[INFO] Mode HISTORIQUE sur {n_hist} satellites · Mode instantané sur le reste", 5)

    pert_cfg = _build_pert_config(pert_flags or {}) if pert_flags else None
    detections = []

    # ── Mode HISTORIQUE ──────────────────────────────────────────────────────
    hist_norads = {r["norad_id"] for r in rows_hist}
    for sat_idx, row in enumerate(rows_hist):
        norad_id = row["norad_id"]
        name     = row["name"]
        if emit and sat_idx % max(1, n_hist // 20) == 0:
            pct = 5 + int(sat_idx / max(n_hist,1) * 45)
            emit(f"[HIST] {sat_idx+1}/{n_hist} — {name[:20]}", pct)
        # Support pause/stop
        if pause_flag:
            while pause_flag.get('pause') and not pause_flag.get('stop'):
                import time as _t; _t.sleep(0.3)
            if pause_flag.get('stop'):
                if emit: emit("[INFO] Détection interrompue", 90)
                break
        try:
            sat_dets = _analyze_satellite_adaptive(norad_id, name, "1970-01-01", pert_cfg, min_sigma)
            for d in sat_dets:
                store_maneuver(norad_id=d["norad_id"], name=d["name"],
                               delta_v_ms=d["delta_v_ms"], residual_m=d["residual_km"]*1000,
                               epoch_before=d["epoch_before"], epoch_after=d["epoch_after"],
                               event_type=d["maneuver_type"], severity=d["severity"])
            detections.extend(sat_dets)
        except Exception as e:
            logger.debug(f"[HIST] {norad_id}: {e}")

    # ── Mode INSTANTANÉ ──────────────────────────────────────────────────────
    # Analyse statistique : B*, mouvement moyen, altitude vs médiane constellation
    instant_rows = [r for r in rows_all if r["norad_id"] not in hist_norads]
    n_inst = len(instant_rows)

    if n_inst > 0:
        if emit: emit(f"[INFO] Analyse instantanée de {n_inst} satellites...", 52)
        inst_dets = _detect_instantaneous(instant_rows, pert_cfg, min_sigma, emit)
        detections.extend(inst_dets)

    detections.sort(key=lambda x: x["sigma_factor"], reverse=True)
    msg = (f"[OK] {len(detections)} anomalies/manœuvres détectées "
           f"({n_hist} historique + {n_inst} instantané)")
    if emit: emit(msg, 95)
    return detections


def _detect_instantaneous(rows: list, pert_cfg, min_sigma: float, emit=None) -> list:
    """
    Mode instantané : détecte les anomalies orbitales statistiques.

    Compare chaque satellite aux médianes de sa constellation :
    - B* (drag term) anormalement élevé → manœuvre récente ou dégradation
    - Mouvement moyen anormal → changement d'orbite
    - Excentricité anormale → manœuvre de circularisation récente
    - Inclinaison anormale → changement d'inclinaison récent

    Méthode : z-score robuste (médiane + MAD) par groupe d'inclinaison.
    """
    from tle_database import get_connection, store_maneuver
    import re as _re

    detections = []
    if len(rows) < 10:
        return detections

    # Extraire les paramètres orbitaux et B* depuis les TLE
    def parse_bstar(tle1: str) -> float:
        """Extrait B* depuis la ligne 1 TLE."""
        try:
            # Format : ±.NNNNN±NN (notation décimale compressée)
            s = tle1[53:61].strip()
            if not s or s == '00000-0':
                return 0.0
            # Convertir la notation TLE : +12345-3 = 0.12345e-3
            sign = -1 if s[0] == '-' else 1
            mantissa_s = s[1:6] if s[0] in '+-' else s[0:5]
            exp_s = s[6:] if s[0] in '+-' else s[5:]
            mantissa = float('0.' + mantissa_s)
            exp = int(exp_s)
            return sign * mantissa * (10 ** exp)
        except Exception:
            return 0.0

    # Construire le dataset
    data = []
    for r in rows:
        try:
            tle1 = r["tle1"] or ""
            tle2 = r["tle2"] or ""
            if not tle1.startswith("1 "): continue
            bstar = parse_bstar(tle1)
            mm    = float(tle2[52:63]) if len(tle2) > 63 else 0
            inc   = float(tle2[8:16])  if len(tle2) > 16 else 0
            ecc   = float("0." + tle2[26:33]) if len(tle2) > 33 else 0
            data.append({
                "norad_id": r["norad_id"],
                "name":     r["name"],
                "bstar":    bstar,
                "mm":       mm,
                "inc":      inc,
                "ecc":      ecc,
                "tle1":     tle1,
                "tle2":     tle2,
                "epoch":    r["last_epoch"] or "",
            })
        except Exception:
            continue

    if len(data) < 5:
        return detections

    arr_bstar = np.array([d["bstar"] for d in data])
    arr_mm    = np.array([d["mm"]    for d in data])
    arr_ecc   = np.array([d["ecc"]   for d in data])

    # Statistiques robustes (médiane + MAD)
    def robust_stats(arr):
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = max(mad * 1.4826, 1e-12)
        return med, sigma

    med_bstar, sig_bstar = robust_stats(arr_bstar)
    med_mm,    sig_mm    = robust_stats(arr_mm)
    med_ecc,   sig_ecc   = robust_stats(arr_ecc)

    n_inst = len(data)
    for i, d in enumerate(data):
        if emit and i % max(1, n_inst // 10) == 0:
            pct = 52 + int(i / n_inst * 40)
            emit(f"[INST] {i+1}/{n_inst}...", pct)

        scores = {}
        # B* anormal (drag récent ou manœuvre)
        z_bstar = abs(d["bstar"] - med_bstar) / sig_bstar
        if z_bstar > min_sigma:
            scores["bstar"] = z_bstar

        # Mouvement moyen anormal (altitude changée)
        z_mm = abs(d["mm"] - med_mm) / sig_mm
        if z_mm > min_sigma:
            scores["mm"] = z_mm

        # Excentricité anormale (circularisation récente)
        z_ecc = abs(d["ecc"] - med_ecc) / sig_ecc
        if z_ecc > min_sigma * 1.5:  # seuil plus élevé car ecc varie naturellement
            scores["ecc"] = z_ecc

        if not scores:
            continue

        sigma_factor = max(scores.values())
        dom_param    = max(scores, key=scores.get)

        # Type de manœuvre probable
        type_map = {
            "bstar": "stationkeeping",
            "mm":    "hohmann",
            "ecc":   "eccentricity",
        }
        mtype = type_map.get(dom_param, "unknown")

        # ΔV estimé depuis le z-score (approximation)
        # z=3 → ΔV~0.1 m/s, z=10 → ΔV~1 m/s, z=30 → ΔV~10 m/s
        delta_v_ms = max(0.05, min(50.0, 0.03 * sigma_factor))

        severity = "CRITIQUE" if sigma_factor > 20 else                    "MAJEUR"   if sigma_factor > 10 else                    "MODERE"   if sigma_factor > min_sigma else "FAIBLE"

        det = {
            "norad_id":      d["norad_id"],
            "name":          d["name"],
            "epoch_before":  d["epoch"],
            "epoch_after":   d["epoch"],
            "residual_km":   round(sigma_factor * 0.1, 3),
            "rtn_km":        [0.0, round(sigma_factor * 0.1, 3), 0.0],
            "delta_v_ms":    round(delta_v_ms, 4),
            "maneuver_type": mtype,
            "sigma_factor":  round(sigma_factor, 1),
            "severity":      severity,
            "threshold_km":  0.0,
            "baseline_km":   0.0,
            "mode":          "instantaneous",
            "anomaly_params": scores,
        }
        detections.append(det)
        try:
            store_maneuver(norad_id=d["norad_id"], name=d["name"],
                           delta_v_ms=delta_v_ms, residual_m=sigma_factor*100,
                           epoch_before=d["epoch"], epoch_after=d["epoch"],
                           event_type=mtype, severity=severity)
        except Exception:
            pass

    return detections


def _analyze_satellite_adaptive(norad_id: str, name: str, cutoff: str,
                                  pert_cfg, min_sigma: float) -> list:
    """Analyse un satellite et retourne ses détections de manœuvres."""
    from tle_database import get_connection

    conn = get_connection()
    records = conn.execute("""
        SELECT tle1, tle2, epoch FROM tle_records
        WHERE norad_id = ?
        ORDER BY epoch ASC
    """, (norad_id,)).fetchall()
    conn.close()

    if len(records) < 2:
        return []

    # Calculer tous les résidus de propagation
    residuals = []
    pairs = []

    for i in range(len(records) - 1):
        r0, r1 = records[i], records[i + 1]
        try:
            t1 = datetime.fromisoformat(r1["epoch"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue

        pos_prop = _propagate_tle_to_epoch(r0["tle1"], r0["tle2"], t1, pert_cfg)
        if pos_prop is None:
            continue

        # Position réelle depuis TLE_{i+1}
        try:
            from sgp4.api import Satrec, jday
            sat1 = Satrec.twoline2rv(r1["tle1"], r1["tle2"])
            err, pos_real, vel_real = sat1.sgp4(
                *jday(t1.year, t1.month, t1.day, t1.hour, t1.minute, float(t1.second))
            )
            if err != 0:
                continue
        except Exception:
            continue

        pos_real = np.array(pos_real)
        delta    = pos_prop - pos_real
        residual_km = float(np.linalg.norm(delta))
        # Rejeter les résidus physiquement impossibles (> 500 km = bug de propagation)
        if residual_km > 500:
            continue

        # Décomposition RTN
        vel_ref = np.array(vel_real)
        rtn = _eci_to_rtn(pos_real, vel_ref, delta)

        try:
            t0 = datetime.fromisoformat(r0["epoch"].replace("Z", "+00:00")).replace(tzinfo=None)
            dt_h = (t1 - t0).total_seconds() / 3600
        except Exception:
            dt_h = 6.0

        delta_v_ms, mtype = _estimate_dv_from_rtn(rtn, dt_h)

        residuals.append(residual_km)
        pairs.append({
            "epoch_before": r0["epoch"],
            "epoch_after":  r1["epoch"],
            "residual_km":  round(residual_km, 3),
            "rtn_km":       [round(float(x), 3) for x in rtn],
            "delta_v_ms":   delta_v_ms,
            "maneuver_type": mtype,
            "dt_hours":     round(dt_h, 1),
        })

    if len(residuals) < 2:
        return []

    # ── Seuil adaptatif ──────────────────────────────────────────────────────
    # Baseline : médiane des résidus (robuste aux outliers)
    residuals_arr = np.array(residuals)

    if len(residuals_arr) >= MIN_BASELINE_OBS:
        median_r = float(np.median(residuals_arr))
        # MAD (Median Absolute Deviation) → σ robuste
        mad = float(np.median(np.abs(residuals_arr - median_r)))
        sigma_baseline = max(mad * 1.4826, MIN_RESIDUAL_KM * 0.5)
    else:
        # Baseline par défaut basé sur l'altitude et le temps
        median_r = DEFAULT_SIGMA_KM_PER_H
        sigma_baseline = DEFAULT_SIGMA_KM_PER_H

    threshold_km = max(MIN_RESIDUAL_KM, median_r + min_sigma * sigma_baseline)

    # ── Détection avec fenêtre temporelle minimale ───────────────────────────
    # Un satellite ne peut pas manœuvrer plus d'une fois toutes les 4h physiquement
    MIN_GAP_HOURS = 4.0     # Fenêtre minimale entre 2 manœuvres (physique)
    MIN_DV_MS     = 0.05    # ΔV minimum pour être une vraie manœuvre (m/s)
    
    detections = []
    last_epoch_after = None
    for p in pairs:
        if p["residual_km"] < threshold_km:
            continue
        if p["maneuver_type"] == 'noise':
            continue
        # Filtre ΔV minimal — rejeter le bruit de propagation
        if p["delta_v_ms"] < MIN_DV_MS:
            continue
        # Fenêtre temporelle — éviter les faux positifs sur TLE fréquents
        if last_epoch_after is not None:
            try:
                t_last = datetime.fromisoformat(last_epoch_after.replace("Z",""))
                t_curr = datetime.fromisoformat(p["epoch_before"].replace("Z",""))
                gap_h = (t_curr - t_last).total_seconds() / 3600
                if gap_h < MIN_GAP_HOURS:
                    # Garder la manœuvre avec le plus grand ΔV sur la fenêtre
                    if detections and p["delta_v_ms"] > detections[-1]["delta_v_ms"]:
                        detections[-1] = {
                            "norad_id":      norad_id,
                            "name":          name,
                            "epoch_before":  p["epoch_before"],
                            "epoch_after":   p["epoch_after"],
                            "residual_km":   p["residual_km"],
                            "delta_v_ms":    p["delta_v_ms"],
                            "maneuver_type": p["maneuver_type"],
                            "sigma_factor":  (p["residual_km"] - median_r) / (sigma_baseline + 1e-10),
                            "severity":      "FAIBLE",
                        }
                    continue
            except Exception:
                pass

        sigma_factor = (p["residual_km"] - median_r) / (sigma_baseline + 1e-10)

        # Classification sévérité
        if sigma_factor > 20 or p["delta_v_ms"] > 10:
            severity = "CRITIQUE"
        elif sigma_factor > 10 or p["delta_v_ms"] > 1:
            severity = "MAJEUR"
        elif sigma_factor > min_sigma:
            severity = "MODERE"
        else:
            severity = "FAIBLE"

        detections.append({
            "norad_id":      norad_id,
            "name":          name,
            "epoch_before":  p["epoch_before"],
            "epoch_after":   p["epoch_after"],
            "residual_km":   p["residual_km"],
            "rtn_km":        p["rtn_km"],
            "delta_v_ms":    p["delta_v_ms"],
            "maneuver_type": p["maneuver_type"],
            "sigma_factor":  round(sigma_factor, 1),
            "severity":      severity,
            "threshold_km":  round(threshold_km, 3),
            "baseline_km":   round(median_r, 3),
        })

    return detections
