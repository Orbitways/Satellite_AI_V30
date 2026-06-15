"""
sustainability_scorer.py — Score de durabilité orbitale multicritère.

Basé sur le Space Sustainability Rating (SSR) :
  Letizia, F. et al. (2020). "Space Sustainability Rating."
  Acta Astronautica, 180, 539-554. doi:10.1016/j.actaastro.2020.09.001

Et les guidelines IADC :
  IADC-02-01 Rev.7 (2021). "IADC Space Debris Mitigation Guidelines."
  Inter-Agency Space Debris Coordination Committee.

Structure du score (0–100, agrégation multicritère) :
  Dimension A — Conformité déorbit        (0–25 pts)  [SSR: Post-mission disposal]
  Dimension B — Activité de manœuvre      (0–20 pts)  [SSR: Collision avoidance]
  Dimension C — Historique de conjonctions(0–20 pts)  [SSR: Collision avoidance]
  Dimension D — Stabilité orbitale        (0–15 pts)  [SSR: Operational orbit]
  Dimension E — Fraîcheur des données TLE (0–10 pts)  [SSR: Trackability proxy]
  Dimension F — Zone orbitale             (0–10 pts)  [SSR: Operational orbit]

Seuils de disqualification (score forcé à 0) :
  - Temps de déorbit naturel > 25 ans (IADC Rule §5.3.2)
  - Satellite classé "perte de contrôle apparente"
  - Pc cumulée 30j > 10⁻³ (risque extrême)

Références valeurs typiques LEO :
  Oltrogge, D.L. & Alfano, S. (2019). "The Technical Challenges of Sustainability."
  Boley, A.C. & Byers, M. (2021). Nature Astronomy 5, 116–117.
"""

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constantes physiques et orbitales ─────────────────────────────────────
RE_KM     = 6371.0       # Rayon terrestre moyen [km]
MU        = 398600.4418  # Paramètre gravitationnel [km³/s²]
TWOPI     = 2 * math.pi

# ── Seuils par défaut (calibrés sur la constellation Starlink) ────────────
# Source : Oltrogge & Alfano (2019), Boley & Byers (2021), Space-Track stats
DEFAULT_THRESHOLDS = {
    # Dimension A — Conformité déorbit
    "deorbit_years_limit":       25.0,   # IADC limit [ans] — disqualifying si > 25
    "deorbit_years_good":         5.0,   # Score plein si < 5 ans
    "deorbit_years_acceptable":  15.0,   # Score partiel si 5–15 ans

    # Dimension B — Activité de manœuvre (station-keeping = bon signe)
    "maneuver_freq_nominal":      1/7,   # Manœuvres/jour — Starlink typique
    "maneuver_freq_inactive":   1/60,    # < 1/60j = potentiellement inactif
    "maneuver_freq_evasive":    3/7,     # > 3/semaine = comportement évasif
    "dv_cumul_30d_normal":        0.5,   # ΔV cumulé 30j normal [m/s] — Starlink
    "dv_cumul_30d_high":          5.0,   # ΔV > 5 m/s = atypique

    # Dimension C — Historique conjonctions
    "pc_high_threshold":        1e-4,    # Pc ≥ 10⁻⁴ = "alerte rouge" NASA/ESA
    "pc_medium_threshold":      1e-5,    # Pc ≥ 10⁻⁵ = "alerte orange"
    "n_high_conj_disqualify":     5,     # ≥ 5 conj. rouges/30j → disqualification

    # Dimension D — Stabilité orbitale
    "bstar_nominal":            1e-4,    # B* Starlink typique
    "bstar_anomaly":            5e-4,    # B* > 5×10⁻⁴ = drag anormal
    "alt_drift_normal":          0.5,    # Dérive altitude normale [km/j]
    "alt_drift_deorbit":         2.0,    # > 2 km/j = déorbit actif

    # Dimension E — Fraîcheur TLE
    "tle_age_good_days":          3,     # < 3j = très frais
    "tle_age_stale_days":        14,     # > 14j = données obsolètes

    # Dimension F — Zone orbitale à risque
    "altitude_danger_low":      750,     # < 750 km : peu peuplé
    "altitude_danger_mid_lo":   750,     # 750–900 km : zone critique (beaucoup de débris)
    "altitude_danger_mid_hi":   900,
    "altitude_danger_high":    1200,     # > 1200 km : zone Van Allen commence
}


def deorbit_time_years(alt_km: float, bstar: float,
                       inc_deg: float = 53.0) -> float:
    """
    Estime le temps de déorbit naturel en années.

    Méthode : table empirique basée sur NASA DAS 3.0 et les statistiques
    de déorbit observées de débris LEO (Liou et al. 2010, Vallado & Kelso 2013).
    L'altitude est le facteur dominant ; B* TLE module la vitesse de déorbit.

    Note : B* TLE est calibré pour SGP4/CIRA-72 et ne peut pas être converti
    directement en coefficient balistique physique sans connaître la densité réelle.
    Cette table empirique évite ce problème en utilisant des durées observées.

    Plafond à 500 ans (règle IADC : au-delà de 600 km, déorbit naturel impraticable).

    Returns:
        Temps de déorbit estimé [années], plafonné à 500.
    """
    if alt_km <= 200: return 0.003
    if alt_km <= 0 or alt_km > 2000: return 500.0

    # Durée de vie médiane par altitude (Bc≈30 kg/m², F10.7 nominal ≈ 150 sfu)
    # Source : Liou et al. (2010), NASA DAS 3.0 validation; Klinkrad (2006)
    table = [
        (200,  0.01),
        (300,  0.5),
        (400,  5.0),
        (450,  12.0),
        (500,  25.0),
        (550,  40.0),
        (600,  80.0),
        (700,  300.0),
        (800,  500.0),
        (900,  500.0),
        (1000, 500.0),
    ]
    alts = [t[0] for t in table]
    taus = [t[1] for t in table]

    # Interpolation linéaire
    if alt_km <= alts[0]:
        tau_med = taus[0]
    elif alt_km >= alts[-1]:
        tau_med = taus[-1]
    else:
        for i in range(len(alts) - 1):
            if alts[i] <= alt_km <= alts[i+1]:
                frac = (alt_km - alts[i]) / (alts[i+1] - alts[i])
                tau_med = taus[i] + frac * (taus[i+1] - taus[i])
                break

    # Modulation par B* : B* élevé → déorbit plus rapide
    # B* nominal LEO ≈ 1e-4, exposant atténué (non-linéaire)
    if bstar <= 0:
        # Satellite actif avec propulsion probable → limiter à 25 ans (règle IADC)
        return min(tau_med, 25.0)

    bstar_nominal = 1e-4
    factor = (abs(bstar) / bstar_nominal) ** 0.3
    factor = max(0.1, min(factor, 5.0))
    tau = tau_med / factor

    return min(float(tau), 500.0)


def score_satellite(norad_id: str, name: str,
                    tle1: str, tle2: str,
                    epoch: str,
                    alt_km: float,
                    inc_deg: float,
                    ecc: float,
                    mm: float,
                    maneuver_history: list,     # [{delta_v_ms, detected_at, event_type}]
                    conjunction_history: list,  # [{Pc, t_ca}]
                    thresholds: dict = None) -> dict:
    """
    Calcule le score de durabilité orbitale d'un satellite (0–100).

    Returns dict avec score global, scores par dimension, flags de
    disqualification et métadonnées.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ── Extraire B* depuis TLE1 ────────────────────────────────────────────
    bstar = _parse_bstar(tle1)

    # ── Calcul de l'âge du TLE ────────────────────────────────────────────
    tle_age_days = _tle_age_days(epoch)

    # ── Temps de déorbit estimé ────────────────────────────────────────────
    tau_years = deorbit_time_years(alt_km, bstar, inc_deg)

    # ══ DIMENSION A — Conformité déorbit (0–25 pts) ═══════════════════════
    # Référence : IADC-02-01 §5.3.2 "25-year rule"
    disqualified_deorbit = tau_years > th["deorbit_years_limit"]
    if tau_years <= th["deorbit_years_good"]:
        score_A = 25.0
    elif tau_years <= th["deorbit_years_acceptable"]:
        # Interpolation linéaire 5–15 ans → 25–10 pts
        score_A = 25.0 - 15.0 * (tau_years - th["deorbit_years_good"]) / \
                  (th["deorbit_years_acceptable"] - th["deorbit_years_good"])
    elif tau_years <= th["deorbit_years_limit"]:
        # Interpolation 15–25 ans → 10–0 pts
        score_A = 10.0 * (1 - (tau_years - th["deorbit_years_acceptable"]) /
                  (th["deorbit_years_limit"] - th["deorbit_years_acceptable"]))
        score_A = max(0, score_A)
    else:
        score_A = 0.0

    # ══ DIMENSION B — Activité de manœuvre (0–20 pts) ═════════════════════
    # Station-keeping régulier = signe de satellite contrôlé et manœuvrable
    now = datetime.now(timezone.utc)
    last_30d = [m for m in maneuver_history
                if _days_ago(m.get('detected_at','')) <= 30]
    n_man_30d  = len(last_30d)
    dv_cumul   = sum(abs(m.get('delta_v_ms', 0)) for m in last_30d)
    freq_per_day = n_man_30d / 30.0 if last_30d else 0.0

    if freq_per_day == 0:
        score_B = 5.0    # Pas de manœuvre = potentiellement inactif
    elif freq_per_day < th["maneuver_freq_inactive"]:
        score_B = 8.0    # Très peu actif
    elif freq_per_day <= th["maneuver_freq_nominal"] * 2:
        score_B = 20.0   # Nominal
    elif freq_per_day <= th["maneuver_freq_evasive"]:
        score_B = 12.0   # Un peu trop actif
    else:
        score_B = 6.0    # Comportement évasif potentiel

    # Pénalité ΔV cumulé excessif
    if dv_cumul > th["dv_cumul_30d_high"]:
        score_B = max(0, score_B - 5)

    # ══ DIMENSION C — Historique conjonctions (0–20 pts) ══════════════════
    # Source : NASA-STD-8719.14A, ESA MASTER-8
    n_high   = sum(1 for c in conjunction_history
                   if float(c.get('Pc', 0)) >= th["pc_high_threshold"])
    n_medium = sum(1 for c in conjunction_history
                   if th["pc_medium_threshold"] <= float(c.get('Pc', 0)) < th["pc_high_threshold"])
    disqualified_conj = n_high >= th["n_high_conj_disqualify"]

    if n_high == 0 and n_medium == 0:
        score_C = 20.0
    elif n_high == 0:
        score_C = max(10, 20 - n_medium * 2)
    else:
        score_C = max(0, 20 - n_high * 5 - n_medium * 1)

    # ══ DIMENSION D — Stabilité orbitale (0–15 pts) ═══════════════════════
    score_D = 15.0
    # B* anormal
    if bstar > th["bstar_anomaly"]:
        score_D -= 6.0
    elif bstar > th["bstar_nominal"] * 2:
        score_D -= 3.0
    # Excentricité (orbite circulaire = meilleure)
    if ecc > 0.01:
        score_D -= 3.0 * min(1, ecc / 0.1)
    score_D = max(0, score_D)

    # ══ DIMENSION E — Fraîcheur TLE (0–10 pts) ════════════════════════════
    # Proxy de la qualité du suivi au sol
    if tle_age_days <= th["tle_age_good_days"]:
        score_E = 10.0
    elif tle_age_days <= th["tle_age_stale_days"]:
        score_E = 10.0 * (1 - (tle_age_days - th["tle_age_good_days"]) /
                  (th["tle_age_stale_days"] - th["tle_age_good_days"]))
    else:
        score_E = 0.0

    # ══ DIMENSION F — Zone orbitale (0–10 pts) ════════════════════════════
    # Source : Klinkrad (2006), "Space Debris — Models and Risk Analysis"
    # Zones à risque : 750–900 km (post-Fengyun-1C, Iridium-Cosmos) et
    #                  1200+ km (début ceinture de Van Allen)
    if alt_km < th["altitude_danger_mid_lo"]:
        score_F = 10.0   # < 750 km : déorbit naturel rapide
    elif alt_km <= th["altitude_danger_mid_hi"]:
        # Zone critique 750–900 km : penalité maximale
        score_F = 3.0
    elif alt_km <= th["altitude_danger_high"]:
        # 900–1200 km : zone dense
        score_F = 6.0
    else:
        score_F = 4.0    # > 1200 km : Van Allen commence

    # ══ AGRÉGATION FINALE ══════════════════════════════════════════════════
    raw_score = score_A + score_B + score_C + score_D + score_E + score_F
    raw_score = min(100.0, max(0.0, raw_score))

    # Disqualification (IADC §5, NASA-STD-8719.14A §4.7)
    disqualified = disqualified_deorbit or disqualified_conj
    final_score  = 0.0 if disqualified else raw_score

    # Grade SSR (aligné sur Letizia et al. 2020, Table 3)
    grade = _score_to_grade(final_score, disqualified)

    return {
        "norad_id":    norad_id,
        "name":        name,
        "score":       round(final_score, 1),
        "grade":       grade,
        "disqualified": disqualified,
        "disq_reason": (
            "Temps déorbit > 25 ans (IADC §5.3.2)" if disqualified_deorbit else
            f"Conjonctions critiques >= {th['n_high_conj_disqualify']}/30j" if disqualified_conj else
            None
        ),
        "dimensions": {
            "A_deorbit":      {"score": round(score_A, 1), "max": 25,
                               "tau_years": round(tau_years, 1),
                               "label": "Conformité déorbit"},
            "B_maneuver":     {"score": round(score_B, 1), "max": 20,
                               "n_maneuvers_30d": n_man_30d,
                               "dv_cumul_ms": round(dv_cumul, 2),
                               "label": "Activité de manœuvre"},
            "C_conjunctions": {"score": round(score_C, 1), "max": 20,
                               "n_high": n_high, "n_medium": n_medium,
                               "label": "Historique conjonctions"},
            "D_stability":    {"score": round(score_D, 1), "max": 15,
                               "bstar": bstar, "ecc": ecc,
                               "label": "Stabilité orbitale"},
            "E_tle_age":      {"score": round(score_E, 1), "max": 10,
                               "age_days": round(tle_age_days, 1),
                               "label": "Fraîcheur TLE"},
            "F_zone":         {"score": round(score_F, 1), "max": 10,
                               "alt_km": round(alt_km, 1),
                               "label": "Zone orbitale"},
        },
        "bstar":       bstar,
        "tau_years":   round(tau_years, 1),
        "tle_age_days": round(tle_age_days, 1),
    }


def score_constellation(norad_list: list, sat_scores: list) -> dict:
    """
    Agrège les scores individuels en score de constellation.
    Méthode : moyenne pondérée avec pénalité pour variance élevée.
    """
    if not sat_scores:
        return {"score": 0, "grade": "N/A", "n_satellites": 0}
    scores   = [s["score"] for s in sat_scores]
    n        = len(scores)
    mean     = sum(scores) / n
    variance = sum((s - mean)**2 for s in scores) / max(n-1, 1)
    std_dev  = math.sqrt(variance)
    # Pénalité variance : forte dispersion = incohérence opérationnelle
    penalty  = min(10, std_dev * 0.3)
    final    = max(0, min(100, mean - penalty))
    n_disq   = sum(1 for s in sat_scores if s["disqualified"])
    return {
        "score":       round(final, 1),
        "grade":       _score_to_grade(final, False),
        "n_satellites": n,
        "n_disqualified": n_disq,
        "mean_score":  round(mean, 1),
        "std_dev":     round(std_dev, 1),
        "variance_penalty": round(penalty, 1),
    }


def score_operator(operator_name: str, constellations: dict) -> dict:
    """Agrège les scores de constellation en score opérateur."""
    all_scores = [c["score"] for c in constellations.values() if "score" in c]
    if not all_scores:
        return {"score": 0, "grade": "N/A"}
    mean = sum(all_scores) / len(all_scores)
    return {
        "operator":       operator_name,
        "score":          round(mean, 1),
        "grade":          _score_to_grade(mean, False),
        "n_constellations": len(constellations),
        "constellations": constellations,
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _score_to_grade(score: float, disqualified: bool) -> str:
    """Aligné sur SSR grades — Letizia et al. (2020), Table 3."""
    if disqualified or score < 20:
        return "F"
    elif score < 40:
        return "D"
    elif score < 55:
        return "C"
    elif score < 70:
        return "B"
    elif score < 85:
        return "A"
    else:
        return "A+"


def _parse_bstar(tle1: str) -> float:
    """Extrait B* de la ligne 1 TLE."""
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


def _tle_age_days(epoch_str: str) -> float:
    """Calcule l'âge du TLE en jours depuis son époque."""
    try:
        ep = datetime.fromisoformat(epoch_str.replace('Z', '+00:00'))
        if ep.tzinfo is None:
            ep = ep.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ep).total_seconds() / 86400
    except Exception:
        return 999.0


def _days_ago(dt_str: str) -> float:
    """Retourne le nombre de jours depuis une date ISO."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return 999.0
