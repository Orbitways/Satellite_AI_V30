#!/usr/bin/env python3
"""
make_differential_dataset.py — Génération de résidus différentiels TLE.

Modes :
  simulate    TLE synthétiques offline (test sans internet)
  spacetrack  Vraies données depuis Space-Track.org (credentials .env requis)
  celestrak   TLE courants depuis Celestrak (sans compte, limité)

Usage :
  python scripts/make_differential_dataset.py --mode spacetrack
  python scripts/make_differential_dataset.py --mode spacetrack --days 7 --finetune
  python scripts/make_differential_dataset.py --mode simulate
"""

import sys, os, argparse, json, logging
import numpy as np
from datetime import datetime, timedelta, timezone

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tle_fetcher import parse_tle_file, _validate_tle, _checksum, get_tle_epoch
from sgp4_utils import tle_to_state, differential_tle_residual
from config import cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─── Évolution TLE synthétique ────────────────────────────────────────────────

def evolve_tle(tle1, tle2, delta_days, drag_factor=1.0):
    """Génère un TLE futur avec drag + précession J2."""
    mu_km=398600.4418; RE=6378.137; J2=1.08262668e-3
    inc=float(tle2[8:16]); raan=float(tle2[17:25])
    ecc=float("0."+tle2[26:33]); aop=float(tle2[34:42])
    ma=float(tle2[43:51]); mm=float(tle2[52:63]); rev=int(tle2[63:68])
    T_s=86400./mm; a_km=(mu_km*(T_s/(2*np.pi))**2)**(1/3); alt=a_km-6371.
    rho_tab=[(200,2.5e-10),(300,1.9e-11),(400,2.8e-12),(500,5.2e-13),(600,1.1e-13)]
    rho=rho_tab[-1][1]
    for h0,r0 in rho_tab:
        if alt<h0: rho=r0; break
    rho*=drag_factor
    v_km_s=np.sqrt(mu_km/a_km)
    da=-0.5*0.022*drag_factor*rho*v_km_s**2*86400*1e-3
    a_new=max(a_km+da*delta_days,6471.); mm_new=86400./(2*np.pi*np.sqrt(a_new**3/mu_km))
    n=2*np.pi/T_s; e2=1-ecc**2
    dO=np.degrees(-1.5*J2*(RE/a_km)**2*n*np.cos(np.radians(inc))/e2**2)*86400
    dw=np.degrees(1.5*J2*(RE/a_km)**2*n*(2-2.5*np.sin(np.radians(inc))**2)/e2**2)*86400
    raan_new=(raan+dO*delta_days)%360; aop_new=(aop+dw*delta_days)%360
    ma_new=(ma+mm_new*delta_days*360)%360
    ecc_new=max(1e-6,ecc*(1-1e-5*delta_days*drag_factor))
    rev_new=rev+int(mm_new*delta_days)
    epoch_old=get_tle_epoch(tle1)
    if epoch_old:
        ep=epoch_old+timedelta(days=delta_days); yr=ep.year%100
        doy=ep.timetuple().tm_yday
        frac=(ep.hour*3600+ep.minute*60+ep.second)/86400.
        ep_str=f"{yr:02d}{doy+frac:012.8f}"
    else: ep_str=tle1[18:32]
    l1=tle1[:18]+f"{ep_str:<14}"+tle1[32:68]+"0"
    l1=l1[:68]+str(_checksum(l1))
    ecc_str=f"{ecc_new:.7f}"[2:]
    l2=(tle2[:8]+f"{inc:8.4f} "+f"{raan_new:8.4f} "+f"{ecc_str} "+
        f"{aop_new:8.4f} "+f"{ma_new:8.4f} "+f"{mm_new:11.8f}"+f"{rev_new:5d}"+"0")
    l2=l2[:68]+str(_checksum(l2))
    return l1, l2


def generate_synthetic_history(tle1, tle2, name, n_epochs=12, interval_hours=6., drag_factor=1.):
    history=[(name,tle1,tle2)]; cur1,cur2=tle1,tle2
    for i in range(1,n_epochs):
        try:
            n1,n2=evolve_tle(cur1,cur2,interval_hours/24.,drag_factor)
            if _validate_tle(n1,n2): history.append((name,n1,n2)); cur1,cur2=n1,n2
            else: break
        except Exception as e: logger.warning(f"Évolution échouée {i}: {e}"); break
    return history


# ─── Calcul des résidus différentiels ────────────────────────────────────────

def compute_residuals(history, step_minutes=5, points_per_pair=60, sat_id=0, emit=None):
    """
    Calcule les résidus réels depuis l'historique TLE.
    résidu(t) = SGP4(TLE_suivant, t) − SGP4(TLE_courant, t)
    """
    all_states, all_res = [], []
    n_pairs = len(history) - 1

    for i in range(n_pairs):
        _, l1o, l2o = history[i]
        _, l1n, l2n = history[i+1]
        epoch = get_tle_epoch(l1o)
        if not epoch: continue

        for j in range(points_per_pair):
            t   = epoch + timedelta(minutes=j*step_minutes)
            t_s = float(j*step_minutes*60)
            st  = tle_to_state(l1o, l2o, t)
            dr  = differential_tle_residual(l1o, l2o, l1n, l2n, t)
            if st is None or dr is None: continue
            all_states.append(np.append(st, [t_s, float(sat_id)]))
            all_res.append(dr)

        if emit and n_pairs > 0:
            pct = int((i+1)/n_pairs*100)
            emit(f"  Paire {i+1}/{n_pairs} calculée ({pct}%)", pct)

    if not all_states:
        return np.zeros((0,8)), np.zeros((0,3))
    return np.array(all_states), np.array(all_res)


# ─── Sources de données ───────────────────────────────────────────────────────

def fetch_spacetrack(days=7, emit=None):
    """Télécharge l'historique TLE depuis Space-Track.org."""
    from spacetrack import fetch_all_satellites
    return fetch_all_satellites(days=days, emit=lambda m: emit(m) if emit else None)


def fetch_celestrak_history(norad_id, emit=None):
    """Celestrak : TLE courant seulement (pas d'historique)."""
    import urllib.request
    url = f"https://celestrak.org/SOCRATES/query.php?CATALOG={norad_id}&FORMAT=tle"
    if emit: emit(f"Celestrak → NORAD {norad_id}...")
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = [l.strip() for l in r.read().decode().splitlines() if l.strip()]
    hist = []
    for i in range(0, len(lines)-2, 3):
        if _validate_tle(lines[i+1], lines[i+2]):
            hist.append((lines[i], lines[i+1], lines[i+2]))
    return hist


# ─── Pipeline principal ───────────────────────────────────────────────────────

NORAD_MAP = {"ISS":"25544","STARLINK":"44713","NOAA18":"28654","TERRA":"25994"}


def run_pipeline(
    mode:      str = "spacetrack",
    tle_path:  str = None,
    days:      int = 7,
    epochs:    int = 12,
    interval:  float = 6.0,
    output:    str = "data/differential_dataset.npz",
    finetune:  bool = False,
    emit:      callable = None,
) -> dict:
    """
    Pipeline complet de génération du dataset.
    Utilisable depuis CLI ou depuis l'API web (emit = callback SSE).

    Retourne un rapport dict.
    """
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tle_source = tle_path or cfg.tle_source

    def log(msg, pct=None):
        logger.info(msg)
        if emit: emit(msg, pct)

    log(f"=== DATASET DIFFÉRENTIEL TLE — mode={mode} ===")

    tles_base = parse_tle_file(tle_source)
    if not tles_base:
        raise ValueError(f"Aucun TLE valide dans {tle_source}")

    all_X, all_y = [], []
    report = {
        "mode":       mode,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "satellites": [],
    }

    # ── Récupération des historiques ──────────────────────────────────────────
    if mode == "spacetrack":
        log("Connexion à Space-Track.org...")
        norad_histories = fetch_spacetrack(days=days, emit=emit)
        # Indexer par nom de satellite
        histories_by_name = {}
        for norad_id, hist in norad_histories.items():
            from spacetrack import SATELLITES as ST_SATS
            name = ST_SATS.get(norad_id, f"NORAD-{norad_id}")
            histories_by_name[name] = hist
    else:
        histories_by_name = None  # sera calculé par satellite

    # ── Traitement par satellite ──────────────────────────────────────────────
    n_sats = len(tles_base)
    for sat_idx, (name, tle1, tle2) in enumerate(tles_base):
        pct_base = int(sat_idx / n_sats * 80)
        log(f"\n[{sat_idx+1}/{n_sats}] {name}", pct_base)

        mm = float(tle2[52:63])
        drag = 1.5 if mm > 15.4 else 0.8

        if mode == "spacetrack":
            # Chercher dans les données Space-Track
            history = histories_by_name.get(name, [])
            if len(history) < 2:
                log(f"  ⚠ Pas assez de TLE Space-Track pour {name} → fallback simulate")
                history = generate_synthetic_history(tle1, tle2, name, epochs, interval, drag)

        elif mode == "celestrak":
            key = name.split()[0].upper()
            norad = NORAD_MAP.get(key)
            if norad:
                try:
                    history = fetch_celestrak_history(norad, emit=emit)
                except Exception as e:
                    log(f"  ⚠ Celestrak échoué ({e}) → simulate")
                    history = generate_synthetic_history(tle1, tle2, name, epochs, interval, drag)
            else:
                history = generate_synthetic_history(tle1, tle2, name, epochs, interval, drag)

        else:  # simulate
            history = generate_synthetic_history(tle1, tle2, name, epochs, interval, drag)

        log(f"  {len(history)} époques TLE disponibles")

        X, y = compute_residuals(
            history, cfg.step_minutes, 60,
            sat_id=sat_idx,
            emit=lambda m, p=None: log(m, pct_base + int((p or 0) * 0.15)) if p else log(m),
        )

        if len(X) == 0:
            log(f"  ⚠ Aucun résidu calculable pour {name}")
            continue

        all_X.append(X); all_y.append(y)
        norms = np.linalg.norm(y, axis=1) * 1000
        sat_report = {
            "name":           name,
            "n_tle_epochs":   len(history),
            "n_residuals":    int(len(X)),
            "mean_residual_m": round(float(np.mean(norms)), 1),
            "max_residual_m":  round(float(np.max(norms)), 1),
        }
        report["satellites"].append(sat_report)
        log(f"  ✓ {len(X)} résidus | moy={np.mean(norms):.1f}m max={np.max(norms):.1f}m", pct_base + 15)

    if not all_X:
        raise RuntimeError("Aucun résidu généré pour aucun satellite.")

    Xf = np.concatenate(all_X); yf = np.concatenate(all_y)
    np.savez(output, states=Xf, residuals=yf)

    mean_m = float(np.mean(np.linalg.norm(yf, axis=1)) * 1000)
    report["n_total_residuals"] = int(len(Xf))
    report["mean_residual_m"]   = round(mean_m, 1)
    report["output"]            = output

    log(f"\n✓ Dataset sauvegardé → {output}", 90)
    log(f"  {len(Xf)} résidus | moy global={mean_m:.1f}m")

    # Rapport JSON
    report_path = "data/differential_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"  Rapport → {report_path}")

    # Fine-tuning optionnel
    if finetune:
        log("\nDémarrage du fine-tuning EWC...", 92)
        import subprocess
        r = subprocess.run(
            [sys.executable, "main.py", "--mode", "finetune", "--tle", tle_source],
            capture_output=True, text=True, cwd=ROOT,
        )
        if r.returncode == 0:
            log("✓ Fine-tuning terminé", 99)
        else:
            log(f"⚠ Fine-tuning erreur : {r.stderr[-200:]}")

    log("=== TERMINÉ ===", 100)
    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dataset de résidus différentiels TLE")
    ap.add_argument("--mode", choices=["simulate","spacetrack","celestrak"], default="spacetrack")
    ap.add_argument("--tle",      default=cfg.tle_source)
    ap.add_argument("--days",     type=int,   default=7,   help="Jours d'historique (mode spacetrack)")
    ap.add_argument("--epochs",   type=int,   default=12,  help="Époques synthétiques (mode simulate)")
    ap.add_argument("--interval", type=float, default=6.0, help="Heures entre TLE (mode simulate)")
    ap.add_argument("--output",   default="data/differential_dataset.npz")
    ap.add_argument("--finetune", action="store_true")
    args = ap.parse_args()

    run_pipeline(
        mode=args.mode, tle_path=args.tle, days=args.days,
        epochs=args.epochs, interval=args.interval,
        output=args.output, finetune=args.finetune,
    )


if __name__ == "__main__":
    main()
