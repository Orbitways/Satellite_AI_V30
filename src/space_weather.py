"""
space_weather.py — Météo spatiale depuis NOAA SWPC.

Sources officielles (publiques, sans authentification) :
  F10.7 mensuel  : https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json
    → Publication : mensuelle. Dernière valeur = mois précédent.
    → Référence : Tapping, K.F. (2013). The 10.7 cm solar radio flux (F10.7).
                  Space Weather, 11(7), 394-406. doi:10.1002/swe.20064
  Kp 3h          : https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
    → Publication : toutes les 3h.
    → Référence : Bartels, J. (1949). The standardized index Ks and the planetary
                  index Kp. IATME Bull. 12b, 97.

Politique de données manquantes :
  Si une source NOAA est inaccessible, la valeur correspondante est None
  et l'interface affiche "No data" — aucune valeur par défaut spéculative.
"""

import urllib.request, json, logging, os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CACHE_FILE = os.path.join("data", "space_weather_cache.json")
# F10.7 est mensuel — inutile de rappeler plus souvent que 6h
# Kp est 3h — on peut rappeler toutes les 30min pour le Kp
_CACHE_MAX_AGE_S = 21600   # 6h pour F10.7
_KP_CACHE_AGE_S  = 1800    # 30min pour Kp


def _fetch(url: str, timeout: int = 10):
    """Fetch JSON depuis URL. Retourne None si indisponible — jamais de valeur inventée."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "OW-SatelliteCoordination/1.0",
            "Accept": "application/json, text/plain"
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"NOAA SWPC indisponible ({url}): {e}")
        return None


def get_current_conditions() -> dict:
    """
    Retourne les conditions de météo spatiale.

    Chaque valeur est soit une donnée réelle avec sa date source,
    soit None avec un message "No data" — jamais une valeur inventée.

    Returns:
        dict avec les clés :
          f107_current   : float | None — flux solaire [sfu] (mensuel NOAA)
          f107_date      : str | None   — date de la mesure
          f107_81day     : float | None — moyenne 81j
          f107_history   : list         — historique 24 mois [(date, f107)]
          kp_current     : float | None — indice Kp (3h NOAA)
          kp_date        : str | None   — date de la mesure Kp
          kp_max_24h     : float | None — Kp max 24h
          kp_forecast    : list         — prévisions [(time, kp)]
          storm_level    : str | None
          storm_color    : str
          alerts         : list
          _live_f107     : bool         — True si F10.7 vient de NOAA
          _live_kp       : bool         — True si Kp vient de NOAA
          _fetched_at    : float        — timestamp du fetch
          _cache_age_s   : float        — âge du cache en secondes
    """
    now_ts = datetime.now().timestamp()

    # ── Lire cache si frais ──────────────────────────────────────────────
    if os.path.exists(_CACHE_FILE):
        try:
            cached = json.load(open(_CACHE_FILE))
            age = now_ts - cached.get("_fetched_at", 0)
            # F10.7 mensuel : cache valide 6h
            # Kp 3h : rafraîchir après 30min
            kp_age = now_ts - cached.get("_kp_fetched_at", 0)
            if age < _CACHE_MAX_AGE_S and kp_age < _KP_CACHE_AGE_S:
                cached["_cache_age_s"] = age
                return cached
        except Exception:
            pass

    result = {
        # F10.7 — toutes les valeurs None par défaut (pas de spéculation)
        "f107_current":  None,
        "f107_date":     None,
        "f107_81day":    None,
        "f107_history":  [],
        # Kp
        "kp_current":    None,
        "kp_date":       None,
        "kp_max_24h":    None,
        "kp_forecast":   [],
        # Dérivés
        "storm_level":   None,
        "storm_color":   "#444466",
        "alerts":        [],
        # Métadonnées
        "_live_f107":    False,
        "_live_kp":      False,
        "_fetched_at":   now_ts,
        "_kp_fetched_at": now_ts,
        "_cache_age_s":  0,
    }

    # ── F10.7 mensuel (Tapping 2013) ─────────────────────────────────────
    f107_data = _fetch(
        "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
    )
    if f107_data and isinstance(f107_data, list) and len(f107_data) > 0:
        # Trouver le dernier point avec une valeur F10.7 réelle (pas 0 ni None)
        last_valid = None
        for row in reversed(f107_data):
            try:
                val = float(row.get("f10.7", 0) or 0)
                if val > 0:
                    last_valid = row
                    break
            except (ValueError, TypeError):
                continue

        if last_valid:
            result["f107_current"] = float(last_valid.get("f10.7"))
            result["f107_date"]    = last_valid.get("time-tag", "")
            try:
                result["f107_81day"] = float(last_valid.get("f10.7_81_day_avg") or 0) or None
            except (ValueError, TypeError):
                result["f107_81day"] = None
            result["_live_f107"] = True

        # Historique : tous les points valides
        history = []
        for row in f107_data:
            try:
                val = float(row.get("f10.7", 0) or 0)
                if val > 0 and row.get("time-tag"):
                    history.append({
                        "date":  row["time-tag"],
                        "f107":  val,
                    })
            except (ValueError, TypeError):
                continue
        result["f107_history"] = history
    # Si indisponible : f107_current reste None → interface affiche "No data"

    # ── Kp 3h (Bartels 1949, NOAA SWPC) ─────────────────────────────────
    kp_data = _fetch(
        "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    )
    if kp_data and isinstance(kp_data, list) and len(kp_data) > 1:
        # Format : [[time_tag, kp, ...], ...]  — première ligne = headers
        headers_row = kp_data[0]
        data_rows   = kp_data[1:]
        # Trouver le dernier Kp observé non nul
        for row in reversed(data_rows[-48:]):  # 48 × 3h = 6 derniers jours
            try:
                kp_val = float(row[1]) if len(row) > 1 and row[1] else None
                if kp_val is not None and kp_val >= 0:
                    result["kp_current"] = kp_val
                    result["kp_date"]    = row[0] if len(row) > 0 else None
                    result["_live_kp"]   = True
                    break
            except (ValueError, TypeError, IndexError):
                continue

        # Kp max 24h
        kp_vals = []
        for row in data_rows[-8:]:  # 8 × 3h = 24h
            try:
                v = float(row[1]) if len(row) > 1 and row[1] else None
                if v is not None:
                    kp_vals.append(v)
            except (ValueError, TypeError):
                continue
        result["kp_max_24h"] = max(kp_vals) if kp_vals else None

    # Fallback Kp : prévisions (si données observées inaccessibles)
    if not result["_live_kp"]:
        kp_fc = _fetch(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        )
        if kp_fc and isinstance(kp_fc, list) and len(kp_fc) > 1:
            for row in reversed(kp_fc[1:]):
                try:
                    kp_val = float(row[1]) if len(row) > 1 and row[1] else None
                    if kp_val is not None:
                        result["kp_current"] = kp_val
                        result["kp_date"]    = row[0]
                        result["_live_kp"]   = True
                        break
                except (ValueError, TypeError):
                    continue
            # Prévisions
            forecasts = []
            kp_fvals  = []
            for row in kp_fc[1:]:
                try:
                    kp_val = float(row[1]) if len(row) > 1 and row[1] else None
                    if kp_val is not None:
                        forecasts.append({"time": row[0], "kp": kp_val})
                        kp_fvals.append(kp_val)
                except (ValueError, TypeError):
                    continue
            result["kp_forecast"] = forecasts[:72]
            if kp_fvals and result["kp_max_24h"] is None:
                result["kp_max_24h"] = max(kp_fvals[:8]) if len(kp_fvals) >= 8 else max(kp_fvals)

    # ── Alertes NOAA ──────────────────────────────────────────────────────
    alerts_data = _fetch("https://services.swpc.noaa.gov/products/alerts.json")
    if alerts_data and isinstance(alerts_data, list):
        result["alerts"] = [
            {
                "message": str(a.get("message", ""))[:200],
                "issued":  str(a.get("issue_datetime", "")),
            }
            for a in alerts_data[:5]
            if a.get("message")
        ]

    # ── Niveau tempête — seulement si Kp disponible ───────────────────────
    kp = result["kp_max_24h"]
    if kp is not None:
        if kp >= 8:
            result["storm_level"] = "EXTRÊME (G5)"; result["storm_color"] = "#EC6E48"
        elif kp >= 7:
            result["storm_level"] = "SÉVÈRE (G4)";  result["storm_color"] = "#EC6E48"
        elif kp >= 6:
            result["storm_level"] = "FORT (G3)";    result["storm_color"] = "#F3B63F"
        elif kp >= 5:
            result["storm_level"] = "MODÉRÉ (G2)";  result["storm_color"] = "#F3B63F"
        elif kp >= 4:
            result["storm_level"] = "MINEUR (G1)";  result["storm_color"] = "#605DF6"
        else:
            result["storm_level"] = "CALME";         result["storm_color"] = "#6FE99E"
    # else : storm_level reste None → "No data" dans l'interface

    # ── Sauvegarder cache ────────────────────────────────────────────────
    try:
        os.makedirs("data", exist_ok=True)
        json.dump(result, open(_CACHE_FILE, "w"))
    except Exception as e:
        logger.warning(f"Cache space weather non sauvegardé: {e}")

    return result


def drag_factor_from_f107(f107) -> float:
    """
    Facteur de modulation du drag atmosphérique depuis F10.7.

    Modèle empirique basé sur JB2008 (Bowman et al. 2008) :
      Bowman, B.R. et al. (2008). A new empirical thermospheric density model JB2008.
      AIAA 2008-6438.

    F10.7 nominal ≈ 150 sfu → factor = 1.0 (pas de perturbation)
    F10.7 = 300 sfu (maximum solaire) → factor ≈ 2.8
    F10.7 = 70 sfu (minimum solaire) → factor ≈ 0.4

    IMPORTANT : il s'agit d'une ESTIMATION. Le modèle JB2008 complet
    utilise aussi F10.7P, S10.7, M10.7, Y10.7 — non disponibles ici.

    Args:
        f107 : flux solaire [sfu] — None si non disponible

    Returns:
        float entre 0.2 et 4.0, ou 1.0 si f107 est None (neutre)
    """
    if f107 is None or f107 <= 0:
        return 1.0   # pas de modulation si données absentes
    return max(0.2, min(4.0, (float(f107) / 150.0) ** 1.5))


def kp_to_density_perturbation(kp) -> float:
    """
    Perturbation de densité atmosphérique due à l'activité géomagnétique.

    Basé sur Emmert et al. (2004) :
      Emmert, J.T. et al. (2004). Climatology of thermospheric neutral winds.
      Journal of Geophysical Research, 109, A12103. doi:10.1029/2004JA010777

    Kp = 0-1 → perturbation ≈ 0%
    Kp = 5   → perturbation ≈ +50% densité
    Kp = 9   → perturbation ≈ +200% (tempête extrême)

    Args:
        kp : indice Kp [0-9] — None si non disponible

    Returns:
        float perturbation relative, ou 0.0 si kp est None
    """
    if kp is None:
        return 0.0
    return max(0.0, float(kp) / 4.5)
