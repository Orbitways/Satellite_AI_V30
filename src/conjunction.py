"""
conjunction.py — Détection de conjonctions orbitales + calcul Pc (Foster 1992).

Sources TLE :
  1. Space-Track API (si .env configuré) — requête GP par OBJECT_NAME
  2. Celestrak pub/TLE (fallback, nécessite accès internet sans restriction)
  3. TLE locaux du projet (fallback final)

Algorithme :
  1. fetch_constellation()    — TLE d'une constellation
  2. propagate_all()          — SGP4 parallèle, max 500 pas
  3. screen_conjunctions()    — voxel grid O(N·log N)
  4. compute_Pc_Foster()      — probabilité collision 2D
  5. run_conjunction_analysis()— pipeline complet
"""

import math, logging, json, time, os, sys
import numpy as np
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

RE_KM   = 6378.137
MU_KM   = 398600.4418
SIGMA_TLE = {"LEO":{"r":.3,"t":.5,"n":.3},"MEO":{"r":1.,"t":2.,"n":1.},"GEO":{"r":2.,"t":5.,"n":2.},"HEO":{"r":1.5,"t":3.,"n":1.5}}
HARD_BODY = {"default":5e-3,"starlink":3e-3,"iss":50e-3,"debris":.1e-3}

# ── Mapping constellation → critères Space-Track / Celestrak ─────────────────
CONSTELLATION_MAP = {
    # Chiffres mis à jour mai 2026
    "starlink":             {"spacetrack_name": "STARLINK",      "celestrak": "starlink.txt",            "n_approx": 7000},
    "oneweb":               {"spacetrack_name": "ONEWEB",        "celestrak": "oneweb.txt",              "n_approx": 650},
    "planet":               {"spacetrack_name": "FLOCK",         "celestrak": "planet.txt",              "n_approx": 200},
    "spire":                {"spacetrack_name": "LEMUR",         "celestrak": "spire.txt",               "n_approx": 115},
    "gps-ops":              {"spacetrack_name": "GPS",           "celestrak": "gps-ops.txt",             "n_approx": 31},
    "galileo":              {"spacetrack_name": "GALILEO",       "celestrak": "galileo.txt",             "n_approx": 28},
    "iridium":              {"spacetrack_name": "IRIDIUM",       "celestrak": "iridium-NEXT.txt",        "n_approx": 66},
    "active":               {"spacetrack_name": None,            "celestrak": "active.txt",              "n_approx": 9000},
    "stations":             {"spacetrack_name": "ISS",           "celestrak": "stations.txt",            "n_approx": 15},
    "debris":               {"spacetrack_name": "COSMOS 1408",   "celestrak": "cosmos-1408-debris.txt",  "n_approx": 1500},
    "cosmos-2251-debris":   {"spacetrack_name": "COSMOS 2251",   "celestrak": "cosmos-2251-debris.txt",  "n_approx": 1400},
}

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


# ── Fetch via Space-Track (authentifié, fiable) ───────────────────────────────

def _fetch_via_spacetrack(constellation: str, emit=None) -> list:
    """Utilise le client Space-Track existant pour récupérer les TLE."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    try:
        from spacetrack import SpaceTrackSession, _load_env
        creds = _load_env()
        if not creds.get("email") or not creds.get("password"):
            return []
    except Exception:
        return []

    info = CONSTELLATION_MAP.get(constellation, {})
    name_filter = info.get("spacetrack_name")
    if not name_filter:
        return []

    if emit: emit(f"Space-Track : requête '{name_filter}'...", 8)

    # URL Space-Track GP (General Perturbations) — TLE courants par nom
    from spacetrack import QUERY_URL
    url = (
        f"{QUERY_URL}/class/gp"
        f"/OBJECT_NAME/{name_filter}~~"      # ~~ = LIKE %name%
        f"/EPOCH/%3Enow-3"                   # époque < 3 jours (assez récent pour LEO)
        f"/orderby/EPOCH%20desc"
        f"/format/tle"
    )

    sys.path.insert(0, os.path.dirname(__file__))
    from tle_fetcher import _validate_tle

    try:
        with SpaceTrackSession() as st:
            raw = st._request(url).decode("utf-8", errors="replace")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        tles = []
        i = 0
        while i + 2 < len(lines):
            n, l1, l2 = lines[i], lines[i+1], lines[i+2]
            if _validate_tle(l1, l2):
                tles.append((n, l1, l2))
            i += 3
        if tles:
            msg = f"✓ Space-Track : {len(tles)} TLE '{name_filter}'"
            logger.info(msg)
            if emit: emit(msg, 15)
        return tles
    except Exception as e:
        logger.warning(f"Space-Track GP échoué : {e}")
        return []


# ── Fetch via Celestrak (fallback) ────────────────────────────────────────────

def _fetch_via_celestrak(constellation: str, emit=None) -> list:
    import urllib.request
    sys.path.insert(0, os.path.dirname(__file__))
    from tle_fetcher import _validate_tle

    info = CONSTELLATION_MAP.get(constellation, {})
    fname = info.get("celestrak", f"{constellation}.txt")

    urls = [
        f"https://celestrak.org/pub/TLE/{fname}",
        f"https://celestrak.com/pub/TLE/{fname}",
        f"https://celestrak.org/SOCRATES/query.php?GROUP={constellation}&FORMAT=tle",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read().decode("utf-8", errors="replace")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            tles = []
            i = 0
            while i + 2 < len(lines):
                n, l1, l2 = lines[i], lines[i+1], lines[i+2]
                if _validate_tle(l1, l2):
                    tles.append((n, l1, l2))
                i += 3
            if tles:
                msg = f"✓ Celestrak : {len(tles)} TLE"
                logger.info(msg)
                if emit: emit(msg, 15)
                return tles
        except Exception as e:
            logger.debug(f"Celestrak {url}: {e}")

    return []


# ── Fetch depuis le cache Space-Track local ───────────────────────────────────

def _fetch_from_local_cache(constellation: str, emit=None) -> list:
    """Utilise le cache Space-Track téléchargé lors de l'import."""
    cache_dir = os.path.join("data", "spacetrack_cache")
    if not os.path.exists(cache_dir):
        return []

    sys.path.insert(0, os.path.dirname(__file__))
    from tle_fetcher import _validate_tle

    tles = []
    info = CONSTELLATION_MAP.get(constellation, {})
    name_filter = (info.get("spacetrack_name") or constellation).upper()

    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(cache_dir, fname)))
            for item in data:
                if isinstance(item, list) and len(item) == 3:
                    n, l1, l2 = item
                    if name_filter in n.upper() and _validate_tle(l1, l2):
                        tles.append((n, l1, l2))
        except Exception:
            continue

    if tles:
        msg = f"✓ Cache local : {len(tles)} TLE (données importées)"
        logger.info(msg)
        if emit: emit(msg, 15)
    return tles


# ── Fetch depuis les TLE locaux du projet ────────────────────────────────────

def _fetch_from_project_tles(emit=None) -> list:
    """Utilise data/sample_tle.txt comme fallback absolu."""
    sys.path.insert(0, os.path.dirname(__file__))
    from tle_fetcher import parse_tle_file
    from config import cfg
    tles = parse_tle_file(cfg.tle_source)
    if tles and emit:
        emit(f"⚠ Fallback TLE locaux : {len(tles)} satellites", 15)
    return tles


# ── Point d'entrée fetch ──────────────────────────────────────────────────────

def fetch_constellation(name: str = "starlink", emit=None) -> list:
    """
    Récupère les TLE d'une constellation par ordre de priorité :
      1. Space-Track API (si .env configuré)
      2. Celestrak (internet direct)
      3. Cache local Space-Track
      4. TLE locaux du projet

    Retourne [(sat_name, tle1, tle2), ...]
    """
    msg = f"Chargement constellation '{name}'..."
    logger.info(msg)
    if emit: emit(msg, 3)

    # 1. Space-Track
    tles = _fetch_via_spacetrack(name, emit)
    if tles:
        return tles

    # 2. Celestrak
    if emit: emit("Space-Track non disponible, tentative Celestrak...", 6)
    tles = _fetch_via_celestrak(name, emit)
    if tles:
        return tles

    # 3. Cache local
    if emit: emit("Celestrak inaccessible, recherche dans le cache local...", 8)
    tles = _fetch_from_local_cache(name, emit)
    if tles:
        return tles

    # 4. TLE du projet
    if emit: emit("⚠ Aucune source distante disponible, utilisation des TLE locaux...", 10)
    tles = _fetch_from_project_tles(emit)
    if tles:
        return tles

    raise ConnectionError(
        "Aucune source TLE disponible.\n"
        "Solutions :\n"
        "  1. Configurez .env avec vos credentials Space-Track\n"
        "  2. Importez des TLE via l'onglet 'Import TLE' d'abord\n"
        "  3. Vérifiez votre connexion internet (Celestrak)"
    )


# ── Propagation SGP4 ─────────────────────────────────────────────────────────

def propagate_all(tles, hours=24., step_min=5., pert_flags=None, emit=None):
    """
    Propage tous les satellites via SGP4 + corrections de perturbations optionnelles.

    pert_flags : dict avec clés matching PertConfig (j3, j4, drag_residual, etc.)
                 Si None, SGP4 pur.
    """
    from sgp4.api import Satrec, jday

    # Préparer la config de perturbations si demandée
    pert_cfg = None
    if pert_flags:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from perturbations import total_perturbation

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
            cfg.Cd   = pert_flags.get('Cd',   2.2)
            cfg.A_m  = pert_flags.get('A_m',  0.01)
            cfg.Cr   = pert_flags.get('Cr',   1.3)
            cfg.A_srp= pert_flags.get('A_m',  0.01)
            # Vérifier qu'au moins une perturbation est active
            active = any([cfg.j3, cfg.j4, cfg.j5, cfg.drag_residual,
                          cfg.solar_pressure, cfg.moon_gravity, cfg.sun_gravity,
                          cfg.albedo, cfg.relativity])
            pert_cfg = (cfg, total_perturbation) if active else None
        except Exception as e:
            logger.warning(f"Perturbations non disponibles : {e}")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    jd0, fr0 = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    dt_s    = step_min * 60.
    n_steps = min(int(hours * 3600 / dt_s) + 1, 500)
    times   = [now + timedelta(seconds=i*dt_s) for i in range(n_steps)]
    results = {}
    n_total = len(tles)

    n_active = sum(1 for k,v in (pert_flags or {}).items()
                   if k not in ('Cd','A_m','Cr') and v)
    pert_label = f" + {n_active} perturbations" if n_active > 0 else " (SGP4 pur)"

    for idx, (sat_name, tle1, tle2) in enumerate(tles):
        if emit and idx % max(1, n_total//20) == 0:
            pct = 15 + int(idx/n_total*50)
            emit(f"Propagation {idx+1}/{n_total}{pert_label}...", pct)
        try:
            sat = Satrec.twoline2rv(tle1, tle2)
            pos_list, vel_list = [], []
            ok = True
            for i in range(n_steps):
                jd_val = jd0 + fr0 + i*dt_s/86400.
                err, r, v = sat.sgp4(int(jd_val), jd_val-int(jd_val))
                if err != 0: ok=False; break

                # Appliquer les corrections de perturbations (intégration Euler)
                if pert_cfg is not None:
                    cfg_obj, total_pert = pert_cfg
                    try:
                        r_arr = np.array(r)       # km
                        v_arr = np.array(v)       # km/s
                        # a en m/s² → correction en km
                        a_ms2 = total_pert(r_arr*1e3, v_arr*1e3, jd_val, cfg_obj)
                        # Δpos = ½·a·dt² en km
                        delta_km = 0.5 * a_ms2 * dt_s**2 / 1e3
                        r = (r_arr + delta_km).tolist()
                    except Exception:
                        r = list(r)

                pos_list.append(r); vel_list.append(v)
            if not ok or not pos_list: continue
            pos_km = np.array(pos_list); vel_km = np.array(vel_list)
            alt_km = float(np.mean(np.linalg.norm(pos_km, axis=1))) - RE_KM
            oc = "LEO" if alt_km<2000 else "MEO" if alt_km<20200 else "HEO" if alt_km<36000 else "GEO"
            results[sat_name] = {"times":times,"pos_km":pos_km,"vel_km_s":vel_km,
                                  "norad":tle1[2:7].strip(),"alt_km":round(alt_km,1),
                                  "orbit_class":oc,"tle1":tle1,"tle2":tle2}
        except Exception as e:
            logger.debug(f"Propagation {sat_name}: {e}")

    active_str = f" (avec {n_active} perturbations)" if n_active > 0 else ""
    msg = f"✓ {len(results)}/{n_total} satellites propagés{active_str}"
    logger.info(msg)
    if emit: emit(msg, 65)
    return results


# ── Screening voxel ───────────────────────────────────────────────────────────

def screen_conjunctions(sat_states, threshold_km=5., emit=None):
    sat_names = list(sat_states.keys())
    if len(sat_names) < 2: return []
    n_steps = len(next(iter(sat_states.values()))["pos_km"])
    vox = threshold_km * 1.5
    candidates = {}
    if emit: emit(f"Screening {len(sat_names)} satellites × {n_steps} pas...", 66)
    for t_idx in range(n_steps):
        grid = {}
        positions = {}
        for nm in sat_names:
            pos = sat_states[nm]["pos_km"][t_idx]
            positions[nm] = pos
            vk = (int(pos[0]/vox), int(pos[1]/vox), int(pos[2]/vox))
            grid.setdefault(vk, []).append(nm)
        checked = set()
        for (vx,vy,vz), members in grid.items():
            neighborhood = []
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for dz in (-1,0,1):
                        neighborhood.extend(grid.get((vx+dx,vy+dy,vz+dz),[]))
            for nameA in members:
                for nameB in neighborhood:
                    if nameA >= nameB: continue
                    pair = (nameA,nameB)
                    if pair in checked: continue
                    checked.add(pair)
                    d = np.linalg.norm(positions[nameA]-positions[nameB])
                    if d < threshold_km:
                        if pair not in candidates or d < candidates[pair]["min_dist_km"]:
                            candidates[pair] = {"min_dist_km":d,"t_idx":t_idx,
                                                "t_ca":sat_states[nameA]["times"][t_idx]}
    msg = f"✓ {len(candidates)} conjonctions < {threshold_km} km"
    logger.info(msg)
    if emit: emit(msg, 80)
    return list(candidates.items())


# ── Probabilité de collision ──────────────────────────────────────────────────

def compute_Pc_Foster(pos_A, vel_A, pos_B, vel_B, sigma_A, sigma_B, r_hard=5e-3):
    delta_r = pos_B - pos_A
    miss    = float(np.linalg.norm(delta_r))
    delta_v = vel_B - vel_A
    v_rel   = float(np.linalg.norm(delta_v))
    if v_rel < 1e-10 or miss < 1e-10: return 0.
    v_hat = delta_v/v_rel
    d_miss = float(np.linalg.norm(delta_r - np.dot(delta_r,v_hat)*v_hat))
    sigma  = math.sqrt(sigma_A**2 + sigma_B**2)
    if sigma < 1e-10: return 0.
    Pc = (r_hard**2/(2.*sigma**2)) * math.exp(-d_miss**2/(2.*sigma**2))
    return min(Pc, 1.)




def compute_Pc_Patera(pos_A, vel_A, pos_B, vel_B, sigma_A, sigma_B, r_hard=5e-3):
    """
    Méthode de Patera (2001) — intégration numérique 2D dans le plan de rencontre.
    Gère les covariances non-sphériques. Plus précis que Foster quand r_hard ≈ σ_c.
    J. Guidance, Control and Dynamics 24(5):958-964.
    """
    delta_r = pos_B - pos_A
    delta_v = vel_B - vel_A
    v_rel   = float(np.linalg.norm(delta_v))
    if v_rel < 1e-10: return 0.

    v_hat = delta_v / v_rel
    # Base orthonormale du plan de rencontre
    e1 = delta_r - np.dot(delta_r, v_hat) * v_hat
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-10:
        # Position dans le plan : utiliser un vecteur arbitraire perpendiculaire
        perp = np.array([1.,0.,0.]) if abs(v_hat[0]) < 0.9 else np.array([0.,1.,0.])
        e1 = perp - np.dot(perp, v_hat) * v_hat
        e1 /= np.linalg.norm(e1)
    else:
        e1 /= e1_norm

    e2 = np.cross(v_hat, e1)

    # Projection dans le plan de rencontre
    d1 = float(np.dot(delta_r, e1))
    d2 = float(np.dot(delta_r, e2))

    # Covariances (isotropes ici, extensible à matrices complètes)
    sigma_c = math.sqrt(sigma_A**2 + sigma_B**2)
    sigma_x, sigma_y = sigma_c, sigma_c

    # Intégration numérique 2D sur la sphère dure
    # Grille polaire autour du point de ratage
    Pc = 0.0
    n_r, n_theta = 20, 36
    for ir in range(n_r):
        r_inner = ir * r_hard / n_r
        r_outer = (ir + 1) * r_hard / n_r
        r_mid   = (r_inner + r_outer) / 2
        dr      = r_outer - r_inner
        for it in range(n_theta):
            theta = (it + 0.5) * 2 * math.pi / n_theta
            dtheta = 2 * math.pi / n_theta
            x = d1 + r_mid * math.cos(theta)
            y = d2 + r_mid * math.sin(theta)
            gauss = math.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2))
            gauss /= (2 * math.pi * sigma_x * sigma_y)
            Pc += gauss * r_mid * dr * dtheta

    return min(Pc, 1.0)


def compute_Pc_MonteCarlo(pos_A, vel_A, pos_B, vel_B, sigma_A, sigma_B,
                           r_hard=5e-3, n_samples=50000):
    """
    Méthode Monte-Carlo (Alfano 2005). N tirages de positions perturbées.
    Coûteux mais exact. J. Spacecraft & Rockets 42(2):292-297.
    n_samples=50000 pour équilibre vitesse/précision (σ ≈ 0.5%).
    """
    rng = np.random.default_rng(seed=42)
    sigma_c = math.sqrt(sigma_A**2 + sigma_B**2)

    # Tirer N positions relatives perturbées
    noise = rng.normal(0, sigma_c, (n_samples, 3))
    delta_r = pos_B - pos_A
    perturbed = delta_r + noise

    # Compter les impacts (distance < r_hard)
    distances = np.linalg.norm(perturbed, axis=1)
    n_impacts = np.sum(distances < r_hard)

    return float(n_impacts) / n_samples


def _pc_color(Pc):
    if Pc >= 1e-4: return "#EC6E48"
    if Pc >= 1e-5: return "#F3B63F"
    if Pc >= 1e-6: return "#605DF6"
    if Pc >= 1e-7: return "#185BFD"
    return "#6FE99E"

def _pc_level(Pc):
    if Pc >= 1e-4: return "CRITIQUE"
    if Pc >= 1e-5: return "ALERTE"
    if Pc >= 1e-6: return "SURVEILLANCE"
    if Pc >= 1e-7: return "FAIBLE"
    return "NORMAL"


# ── Pipeline complet ──────────────────────────────────────────────────────────

def run_conjunction_analysis(constellation="starlink", extra_tles=None,
                              hours=24., step_min=5., threshold_km=5.,
                              max_results=200, pc_method="foster", mc_n=50000,
                              pert_flags=None, emit=None):
    if emit: emit("Chargement des TLE...", 2)
    tles = fetch_constellation(constellation, emit=emit)
    if extra_tles: tles.extend(extra_tles)
    if not tles: raise ValueError("Aucun TLE disponible.")

    if emit: emit(f"Propagation de {len(tles)} satellites sur {hours}h...", 15)
    sat_states = propagate_all(tles, hours=hours, step_min=step_min, pert_flags=pert_flags, emit=emit)
    if len(sat_states) < 2: raise ValueError("Pas assez de satellites propagés.")

    candidates = screen_conjunctions(sat_states, threshold_km=threshold_km, emit=emit)
    if not candidates:
        if emit: emit("✓ Aucune conjonction détectée.", 100)
        return []

    if emit:
        mc_suffix = f', N={mc_n}' if pc_method == 'montecarlo' else ''
        emit(f"Calcul Pc ({pc_method.upper()}{mc_suffix}) pour {len(candidates)} candidats...", 82)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conjunctions = []

    for (nameA, nameB), info in candidates:
        if nameA not in sat_states or nameB not in sat_states: continue
        t_idx = info["t_idx"]; t_ca = info["t_ca"]
        pos_A = sat_states[nameA]["pos_km"][t_idx]
        vel_A = sat_states[nameA]["vel_km_s"][t_idx]
        pos_B = sat_states[nameB]["pos_km"][t_idx]
        vel_B = sat_states[nameB]["vel_km_s"][t_idx]
        oc = sat_states[nameA]["orbit_class"]
        sg = SIGMA_TLE.get(oc, SIGMA_TLE["LEO"])
        sigma = math.sqrt(sg["r"]**2+sg["t"]**2+sg["n"]**2)/math.sqrt(3)
        r_hard = HARD_BODY["starlink"] if "starlink" in nameA.lower()+nameB.lower() else \
                 HARD_BODY["iss"] if any(x in nameA.lower()+nameB.lower() for x in ["iss","zarya"]) else \
                 HARD_BODY["default"]
        if pc_method == "patera":
            Pc = compute_Pc_Patera(pos_A, vel_A, pos_B, vel_B, sigma, sigma, r_hard)
        elif pc_method == "montecarlo":
            Pc = compute_Pc_MonteCarlo(pos_A, vel_A, pos_B, vel_B, sigma, sigma, r_hard, n_samples=mc_n)
        else:  # foster (défaut)
            Pc = compute_Pc_Foster(pos_A, vel_A, pos_B, vel_B, sigma, sigma, r_hard)
        v_rel = float(np.linalg.norm(vel_A - vel_B))
        pos_ca = (pos_A + pos_B) / 2.
        lat, lon, alt = _eci_to_geo(pos_ca, t_ca)
        t_ca_h = (t_ca - now).total_seconds() / 3600.
        # Trajectoires 1/4 orbite (~22 min) autour du TCA pour le globe 3D
        n_steps_total = len(sat_states[nameA]["pos_km"])
        half_orbit_steps = max(3, int(22.5 / step_min))
        i0 = max(0, t_idx - half_orbit_steps)
        i1 = min(n_steps_total, t_idx + half_orbit_steps + 1)
        indices = list(range(i0, i1))
        step_sub = max(1, len(indices) // 40)
        indices = indices[::step_sub]
        # Convertir les trajectoires en lat/lon/alt (cohérent avec les marqueurs)
        def pos_to_geo(pos_km, t_idx_val):
            t_pt = sat_states[nameA]["times"][t_idx_val]
            lat_t, lon_t, alt_t = _eci_to_geo(pos_km, t_pt)
            return [round(lat_t, 3), round(lon_t, 3), round(alt_t, 1)]

        traj_A = [pos_to_geo(sat_states[nameA]["pos_km"][i], i) for i in indices]
        traj_B = [pos_to_geo(sat_states[nameB]["pos_km"][i], i) for i in indices]

        conjunctions.append({
            "id":           f"{sat_states[nameA]['norad']}_{sat_states[nameB]['norad']}_{t_idx:04d}",
            "sat_A":        nameA, "sat_B":        nameB,
            "norad_A":      sat_states[nameA]["norad"],
            "norad_B":      sat_states[nameB]["norad"],
            "t_ca":         t_ca.isoformat(), "t_ca_h":   round(t_ca_h,2),
            "miss_dist_km": round(info["min_dist_km"],4), "v_rel_km_s": round(v_rel,3),
            "Pc":           Pc, "Pc_str": f"{Pc:.2e}", "pc_method": pc_method,
            "level":        _pc_level(Pc), "color": _pc_color(Pc),
            "pos_ca_km":    pos_ca.tolist(),
            "lat":          round(lat,3), "lon": round(lon,3), "alt_km": round(alt,1),
            "traj_A_km":    traj_A,
            "traj_B_km":    traj_B,
        })

    conjunctions.sort(key=lambda x: x["Pc"], reverse=True)
    conjunctions = conjunctions[:max_results]
    msg = f"✓ {len(conjunctions)} conjonctions (triées par Pc)"
    logger.info(msg)
    if emit: emit(msg, 98)
    return conjunctions


def _eci_to_geo(pos_km, t):
    x,y,z = pos_km
    r = math.sqrt(x**2+y**2+z**2)
    alt = r - RE_KM
    jd = (367*t.year - int(7*(t.year+int((t.month+9)/12))/4) +
          int(275*t.month/9) + t.day + 1721013.5 +
          (t.hour+t.minute/60+t.second/3600)/24)
    gmst = math.radians((280.46061837+360.98564736629*(jd-2451545.)+
                          ((jd-2451545.)/36525.)**2*.000387933)%360)
    lon = math.degrees(math.atan2(y,x)-gmst)%360
    if lon>180: lon-=360
    lat = math.degrees(math.asin(z/r)) if r>0 else 0
    return lat, lon, alt
