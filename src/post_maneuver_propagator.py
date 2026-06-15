"""
post_maneuver_propagator.py — Génération de TLE synthétiques post-manœuvre.

Pour chaque prédiction de manœuvre, applique le ΔV prédit à l'orbite actuelle
et génère un TLE synthétique représentant l'orbite post-manœuvre probable.

Méthode : approximation impulsionnelle dans le repère RTN.
  - Hohmann/Phasing → ΔV tangentiel → changement de demi-grand axe
  - Inclination     → ΔV normal     → changement d'inclinaison
  - Eccentricity    → ΔV radial     → changement d'excentricité
  - Stationkeeping  → ΔV mixte      → correction d'orbite

Référence : Vallado, D. A. (2013). Fundamentals of Astrodynamics.
"""

import math, logging
import numpy as np

logger = logging.getLogger(__name__)

# Constantes orbitales
MU_KM3_S2 = 398600.4418   # GM Terre [km³/s²]
RE_KM      = 6378.137      # Rayon équatorial [km]
J2         = 1.08262668e-3 # Coefficient harmonique J2


def _elements_from_tle(tle1: str, tle2: str) -> dict | None:
    """Extrait les éléments orbitaux moyens depuis un TLE."""
    try:
        from sgp4.api import Satrec, jday
        from datetime import datetime, timezone

        sat = Satrec.twoline2rv(tle1, tle2)
        now = datetime.now(timezone.utc)
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, float(now.second))
        err, r, v = sat.sgp4(jd, fr)
        if err != 0:
            return None

        r_arr = np.array(r)
        v_arr = np.array(v)
        r_mag = float(np.linalg.norm(r_arr))
        v_mag = float(np.linalg.norm(v_arr))

        # Éléments orbitaux classiques
        # Demi-grand axe
        eps = v_mag**2 / 2 - MU_KM3_S2 / r_mag
        a   = -MU_KM3_S2 / (2 * eps)

        # Excentricité
        e_vec = ((v_mag**2 - MU_KM3_S2/r_mag) * r_arr - np.dot(r_arr, v_arr) * v_arr) / MU_KM3_S2
        e = float(np.linalg.norm(e_vec))

        # Inclinaison
        h_vec = np.cross(r_arr, v_arr)
        h_mag = float(np.linalg.norm(h_vec))
        inc   = math.degrees(math.acos(max(-1, min(1, h_vec[2] / (h_mag + 1e-10)))))

        # Mouvement moyen [rad/s]
        n_rad_s = math.sqrt(MU_KM3_S2 / max(a, RE_KM)**3)
        mm_rev_day = n_rad_s * 86400 / (2 * math.pi)

        return {
            "a_km": a, "e": e, "inc_deg": inc,
            "mm_rev_day": mm_rev_day,
            "r_km": r_arr, "v_km_s": v_arr,
            "r_mag": r_mag, "v_mag": v_mag,
            "h_vec": h_vec,
        }
    except Exception as ex:
        logger.debug(f"elements_from_tle: {ex}")
        return None


def apply_delta_v(tle1: str, tle2: str,
                  dv_ms: float, maneuver_type: str,
                  norad_id: str = "99999") -> tuple[str, str] | None:
    """
    Applique un ΔV impulsionnel à l'orbite et retourne un TLE synthétique.

    dv_ms : amplitude du ΔV en m/s
    maneuver_type : 'hohmann', 'phasing', 'inclination', 'eccentricity', 'stationkeeping'

    Retourne (tle1_synth, tle2_synth) ou None si erreur.
    """
    elems = _elements_from_tle(tle1, tle2)
    if elems is None:
        return None

    dv_km_s = dv_ms / 1000.0
    a   = elems["a_km"]
    e   = elems["e"]
    inc = elems["inc_deg"]
    mm  = elems["mm_rev_day"]
    r   = elems["r_mag"]
    v   = elems["v_mag"]

    # ── Application du ΔV selon le type de manœuvre ──────────────────────────
    a_new   = a
    e_new   = e
    inc_new = inc
    mm_new  = mm

    if maneuver_type in ("hohmann", "phasing", "stationkeeping"):
        # ΔV tangentiel → changement de vitesse → changement de demi-grand axe
        # Vis-viva : v² = GM(2/r - 1/a) → dv tangentiel change a
        v_new = v + dv_km_s
        # Nouveau demi-grand axe depuis l'énergie
        eps_new = v_new**2 / 2 - MU_KM3_S2 / r
        if eps_new >= 0:
            # Orbite hyperbolique — cap à LEO haute
            a_new = RE_KM + 2000
        else:
            a_new = -MU_KM3_S2 / (2 * eps_new)
        a_new = max(RE_KM + 200, min(RE_KM + 40000, a_new))

        # Nouveau mouvement moyen
        mm_new = math.sqrt(MU_KM3_S2 / a_new**3) * 86400 / (2 * math.pi)

        # Excentricité approximativement inchangée pour un Hohmann circulaire
        if maneuver_type == "hohmann":
            e_new = max(0.0, e * (a / a_new))

    elif maneuver_type == "inclination":
        # ΔV normal → changement d'inclinaison
        # Δi = ΔV_N / V_orb (petits angles)
        v_orb = math.sqrt(MU_KM3_S2 / a)  # vitesse orbitale circulaire
        delta_inc_rad = dv_km_s / (v_orb + 1e-10)
        delta_inc_deg = math.degrees(delta_inc_rad)
        inc_new = max(0, min(180, inc + delta_inc_deg))

    elif maneuver_type == "eccentricity":
        # ΔV radial → changement d'excentricité
        # Approximation : Δe ≈ ΔV_R × 2 / (n × a)
        n = math.sqrt(MU_KM3_S2 / a**3)
        delta_e = 2 * dv_km_s / (n * a + 1e-10)
        e_new = max(0.0, min(0.9, e + delta_e))

    # ── Construction du TLE synthétique ──────────────────────────────────────
    return _build_synthetic_tle(tle1, tle2, a_new, e_new, inc_new, mm_new, norad_id)


def _build_synthetic_tle(tle1_orig: str, tle2_orig: str,
                          a_new: float, e_new: float,
                          inc_new: float, mm_new: float,
                          norad_id: str) -> tuple[str, str]:
    """
    Construit un TLE synthétique en modifiant les éléments orbitaux.
    Conserve les autres éléments (RAAN, argument du périgée, anomalie moyenne)
    depuis le TLE original.
    """
    # Extraire les champs du TLE original
    inc_str  = f"{inc_new:8.4f}"
    raan     = tle2_orig[17:25].strip()   # RAAN inchangé
    ecc_str  = f"{e_new:>7.7f}".replace("0.", "").replace(".", "")[:7]
    argp     = tle2_orig[34:42].strip()   # Arg. périgée inchangé
    ma       = tle2_orig[43:51].strip()   # Anomalie moyenne inchangée
    mm_str   = f"{mm_new:11.8f}"

    # Révolution (inchangé)
    rev_str  = tle2_orig[63:68].strip() if len(tle2_orig) > 68 else "00001"

    # Reconstruire TLE1 (champ BSTAR inchangé)
    bstar    = tle2_orig[53:61] if len(tle2_orig) > 61 else "00000-0"
    epoch    = tle1_orig[18:32]  # Garder l'époque originale
    norad_f  = norad_id.rjust(5)

    line1 = f"1 {norad_f}U SYNTH    {epoch}  .00000000  00000-0  {tle1_orig[54:61]} 0  9990"
    line2 = f"2 {norad_f} {inc_str} {raan.rjust(8)} {ecc_str} {argp.rjust(8)} {ma.rjust(8)} {mm_str}   {rev_str.rjust(5)}0"

    # Ajouter les checksums
    line1 = line1[:68] + str(_tle_checksum(line1[:68]))
    line2 = line2[:68] + str(_tle_checksum(line2[:68]))

    return line1[:69], line2[:69]


def _tle_checksum(line: str) -> int:
    """Calcule le checksum d'une ligne TLE."""
    total = 0
    for c in line[:68]:
        if c.isdigit():
            total += int(c)
        elif c == '-':
            total += 1
    return total % 10


def generate_post_maneuver_tles(predictions: list) -> list:
    """
    Pour chaque prédiction de manœuvre, génère un TLE synthétique post-manœuvre.

    Retourne une liste de tuples (name_synth, tle1_synth, tle2_synth, prediction)
    prêts à être injectés dans run_conjunction_analysis() comme extra_tles.
    """
    results = []
    for pred in predictions:
        if not pred.get("tle1") or not pred.get("tle2"):
            continue
        if pred.get("p_7d", 0) < 0.3:
            continue

        try:
            synth = apply_delta_v(
                tle1=pred["tle1"],
                tle2=pred["tle2"],
                dv_ms=pred.get("dv_pred_ms", 5.0),
                maneuver_type=pred.get("type_pred", "stationkeeping"),
                norad_id=f"S{pred['norad_id'][:4]}",
            )
            if synth is None:
                continue

            tle1_s, tle2_s = synth
            name_synth = f"[PRED] {pred['name'][:16]}"

            results.append({
                "name":       name_synth,
                "tle1":       tle1_s,
                "tle2":       tle2_s,
                "norad_id":   pred["norad_id"],
                "p_7d":       pred["p_7d"],
                "dv_ms":      pred.get("dv_pred_ms", 0),
                "type_pred":  pred.get("type_pred", "unknown"),
                "original":   pred,
            })
        except Exception as e:
            logger.debug(f"TLE synthétique {pred.get('norad_id')}: {e}")

    logger.info(f"{len(results)} TLE synthétiques générés")
    return results
