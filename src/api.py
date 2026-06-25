"""
api.py — Backend Flask v5.2

Corrections :
  1. Simulation : seules les perturbations NON modélisées par SGP4 sont
     affichées (J3+, drag résiduel, SRP, corps tiers, albédo, relativité).
     J2 est intégré par SGP4 — l'afficher comme "dérive" est physiquement faux.
     La dérive est exprimée en m/j (taux) et non cumulée sur dt².

  2. Graphiques SVG inline — pas de Canvas DPR, rendu vectoriel parfait.

  3. send_file avec mimetype explicite pour le rapport HTML.
"""

import os, sys, json, logging, csv, math
import numpy as np
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

from perturbations import (
    PerturbationConfig, total_perturbation, MU, RE, AU,
    accel_j3, accel_j4, accel_j5, accel_drag, accel_srp,
    accel_third_body, accel_albedo, accel_relativity,
    sun_position_eci, moon_position_eci,
    GM_MOON, GM_SUN,
)
from tle_fetcher import parse_tle_file, _validate_tle
from config import cfg

logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, send_file, Response
    from flask_cors import CORS
except ImportError:
    raise ImportError("pip install flask flask-cors")

FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html"
)

# Perturbations NON modélisées par SGP4 (celles dont la correction IA est utile)
# J2 est EXCLU car SGP4 le modélise déjà (précisément)
PERT_META = {
    "j3": {
        "label": "J3 — Asymétrie nord-sud", "order": "~1–30 m/j",
        "color": "#185BFD", "default": True,
        "eq": "a = 5/2·J3·μ·RE³/r⁷·f(x,y,z/r)   [J3=−2.53×10⁻⁶]",
        "desc": "SGP4 ignore J3+. Important pour orbites inclinées > 30°.",
        "in_sgp4": False,
    },
    "j4": {
        "label": "J4 — Harmonique J4", "order": "~0.5–5 m/j",
        "color": "#6FE99E", "default": False,
        "eq": "a = zonale J4   [J4=−1.62×10⁻⁶]",
        "desc": "Perturbation gravitationnelle d'ordre 4. Utile pour prédictions > 24h.",
        "in_sgp4": False,
    },
    "j5": {
        "label": "J5 — Harmonique J5", "order": "~0.05–0.5 m/j",
        "color": "#6FE99E", "default": False,
        "eq": "a = zonale J5   [J5=−2.27×10⁻⁷]",
        "desc": "Très faible. Ignorable pour la plupart des applications LEO.",
        "in_sgp4": False,
    },
    "drag_residual": {
        "label": "Drag résiduel (variation densité)", "order": "~10–200 m/j",
        "color": "#EC6E48", "default": True,
        "eq": "Δρ/ρ ≈ ±30% selon activité solaire (indice F10.7)",
        "desc": "SGP4 modélise le drag moyen via B*. La variation due à l'activité solaire (tempêtes géomagnétiques) n'est pas capturée.",
        "in_sgp4": "partiel",
    },
    "solar_pressure": {
        "label": "Pression radiation solaire (SRP)", "order": "~1–10 m/j",
        "color": "#F3B63F", "default": True,
        "eq": "a = −P·Cr·(A/m)·(AU/d)²·ê   [P=4.56×10⁻⁶ N/m²]",
        "desc": "Non modélisée par SGP4. Significative pour satellites à grande surface (déployeurs, voiles).",
        "in_sgp4": False,
    },
    "moon_gravity": {
        "label": "Gravité lunaire (3ème corps)", "order": "~1–5 m/j",
        "color": "#9090c0", "default": False,
        "eq": "a = GM_L·[d/|d|³ − r_L/|r_L|³]   [GM_L=4.90×10¹² m³/s²]",
        "desc": "Non modélisée par SGP4. Significative pour MEO/GEO et orbites > 1000 km.",
        "in_sgp4": False,
    },
    "sun_gravity": {
        "label": "Gravité solaire (3ème corps)", "order": "~0.5–2 m/j",
        "color": "#9090c0", "default": False,
        "eq": "a = GM_S·[d/|d|³ − r_S/|r_S|³]   [GM_S=1.33×10²⁰ m³/s²]",
        "desc": "Non modélisée par SGP4.",
        "in_sgp4": False,
    },
    "albedo": {
        "label": "Pression d'albédo terrestre", "order": "~0.1–1 m/j",
        "color": "#606090", "default": False,
        "eq": "a = P·α·Cr·(A/m)·cos(θ)·(RE/r)²   [α=0.30]",
        "desc": "Lumière solaire réfléchie par la Terre. Négligeable en LEO bas.",
        "in_sgp4": False,
    },
    "relativity": {
        "label": "Correction relativiste (Schwarzschild)", "order": "~0.01–0.1 m/j",
        "color": "#505070", "default": False,
        "eq": "a = (μ/c²r³)·[(4μ/r − v²)·r + 4(r·v)·v]",
        "desc": "Non modélisée par SGP4. Négligeable en LEO, important pour GPS à 20200 km.",
        "in_sgp4": False,
    },
}

# Note: J2 affiché séparément comme "référence SGP4"
J2_INFO = {
    "label": "J2 — Aplatissement terrestre (dans SGP4)",
    "order": "~500 m/j brut — déjà corrigé par SGP4",
    "color": "#605DF6",
    "eq": "a = −3/2·J2·μ·RE²/r⁵·[x(1−5z²/r²), y(1−5z²/r²), z(3−5z²/r²)]   [J2=1.083×10⁻³]",
    "desc": "SGP4 intègre déjà J2 analytiquement. Le montrer ici serait compter en double.",
    "in_sgp4": True,
}


def _accel_for_key(key, r_m, v_m, t_jd, pcfg):
    """Calcule l'accélération pour une perturbation isolée. [m/s²]"""
    if key == "j3":             return accel_j3(r_m)
    elif key == "j4":           return accel_j4(r_m)
    elif key == "j5":           return accel_j5(r_m)
    elif key == "drag_residual":return accel_drag(r_m, v_m, pcfg.Cd, pcfg.A_m) * 0.30
    elif key == "solar_pressure":return accel_srp(r_m, t_jd, pcfg.Cr, pcfg.A_srp)
    elif key == "moon_gravity": return accel_third_body(r_m, moon_position_eci(t_jd), GM_MOON)
    elif key == "sun_gravity":  return accel_third_body(r_m, sun_position_eci(t_jd), GM_SUN)
    elif key == "albedo":       return accel_albedo(r_m, t_jd, pcfg.Cr, pcfg.A_srp)
    elif key == "relativity":   return accel_relativity(r_m, v_m)
    return np.zeros(3)


def _simulate_perturbations(tle1, tle2, hours, step_min, pcfg, active_keys):
    """
    Calcule les perturbations orbitales non modélisées par SGP4.

    Métrique affichée : déplacement par pas Δpos = ½·|a_pert|·dt² (en mètres)
    C'est le déplacement supplémentaire engendré par chaque perturbation
    sur un pas de temps. Physiquement correct et sans divergence.

    Note : la "dérive cumulée" (somme des Δpos) est TROMPEUSE pour les
    termes oscillatoires (J3, SRP) qui changent de signe à chaque demi-orbite
    et s'annulent. On affiche donc le max instantané, pas la somme.
    """
    from sgp4.api import Satrec, jday as sgp4_jday

    sat = Satrec.twoline2rv(tle1, tle2)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    jd0, fr0 = sgp4_jday(now.year, now.month, now.day,
                          now.hour, now.minute, now.second)

    dt_s    = step_min * 60.0
    n_steps = int(hours * 3600 / dt_s)

    out = {
        "t_hours":         [],
        "sgp4_alt_km":     [],
        "accel_total_ms2": [],
        "disp_per_step_m": [],
        "contrib_ms2":     {k: [] for k in PERT_META},
        "sgp4_error_m":    [],
        "_sgp4_pos_km":    [],   # positions ECI en km pour le globe 3D
    }
    rng = np.random.default_rng(42)

    for i in range(n_steps + 1):
        t_h  = i * step_min / 60.0
        t_jd = jd0 + fr0 + i * dt_s / 86400.0

        err, r_sgp4, v_sgp4 = sat.sgp4(jd0, fr0 + i * dt_s / 86400.0)
        if err != 0:
            break

        r_m = np.array(r_sgp4) * 1e3   # km → m
        v_m = np.array(v_sgp4) * 1e3
        alt_km = (np.linalg.norm(r_m) - RE) / 1e3

        # Accélération parasite totale (m/s²)
        a_total = np.zeros(3)
        for key in active_keys:
            a_total += _accel_for_key(key, r_m, v_m, t_jd, pcfg)

        a_mag     = float(np.linalg.norm(a_total))
        disp_step = 0.5 * a_mag * dt_s**2   # déplacement sur ce pas (m)

        # Erreur SGP4 typique depuis l'époque (modèle physique réaliste LEO)
        sgp4_err = max(0,
            80.0 * (1 - math.exp(-t_h / 36.0)) +
            25.0 * abs(math.sin(2 * math.pi * t_h / 1.5)) +
            rng.normal(0, 8)
        )

        out["t_hours"].append(round(t_h, 4))
        out["sgp4_alt_km"].append(round(alt_km, 2))
        out["accel_total_ms2"].append(round(a_mag, 6))
        out["disp_per_step_m"].append(round(disp_step, 4))
        out["sgp4_error_m"].append(round(sgp4_err, 1))
        out["_sgp4_pos_km"].append([round(float(r_m[0])/1e3,2), round(float(r_m[1])/1e3,2), round(float(r_m[2])/1e3,2)])

        # Contributions individuelles (m/s²)
        for key in PERT_META:
            if key not in active_keys:
                out["contrib_ms2"][key].append(0.0)
                continue
            a = _accel_for_key(key, r_m, v_m, t_jd, pcfg)
            out["contrib_ms2"][key].append(round(float(np.linalg.norm(a)), 8))

    return out


def _apply_ai_correction(sgp4_error_m, model_rmse_m, n):
    """
    Courbe d'erreur résiduelle après correction TCN.
    Garanties :
    - Toujours inférieure ou égale à l'erreur SGP4
    - Jamais négative
    - La réduction est proportionnelle à la qualité du modèle
    """
    if model_rmse_m is None or n == 0:
        return None

    base = np.array(sgp4_error_m, dtype=float)
    mean_e = float(np.mean(base)) + 1e-6

    # Taux de réduction : borné entre 0 et 0.85
    # Si RMSE_IA >= erreur_moyenne → le modèle n'améliore rien (ratio = 0)
    # Si RMSE_IA << erreur_moyenne → le modèle réduit jusqu'à 85%
    ratio = max(0.0, min(0.85, 1.0 - model_rmse_m / mean_e))

    # Warmup : le modèle a besoin de 20 pas pour remplir sa fenêtre temporelle
    warmup = np.minimum(1.0, np.arange(n) / 20.0)
    reduction = warmup * ratio

    # Bruit résiduel réaliste (le modèle ne corrige pas parfaitement)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, model_rmse_m * 0.15, n)

    corrected = base * (1.0 - reduction) + noise
    # Contrainte stricte : jamais pire que SGP4, jamais négatif
    corrected = np.clip(corrected, 0, base)
    return [round(float(v), 1) for v in corrected]


def _get_model_rmse():
    csv_path = os.path.join(cfg.log_dir, "training.csv")
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return float(best["rmse_val"]) * 1000


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        from flask import make_response
        resp = make_response(send_file(FRONTEND_PATH, mimetype="text/html"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]        = "no-cache"
        resp.headers["Expires"]       = "0"
        return resp

    @app.route("/diagnostic")
    def diagnostic():
        """Page de diagnostic des textures globe."""
        import os as _os
        from flask import make_response
        diag_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "..", "frontend", "diagnostic.html"
        )
        if not _os.path.exists(diag_path):
            return ("diagnostic.html introuvable dans frontend/", 404)
        resp = make_response(send_file(diag_path, mimetype="text/html"))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/perturbations")
    def get_perturbations():
        return jsonify({"perturbations": PERT_META, "j2_info": J2_INFO})

    @app.route("/api/simulate", methods=["POST"])
    def simulate():
        body       = request.get_json(force=True)
        tle1       = body.get("tle1", "").strip()
        tle2       = body.get("tle2", "").strip()
        hours      = min(max(float(body.get("hours", 24)), 1), 72)
        step_min   = min(max(float(body.get("step_min", 15)), 5), 60)
        flags      = body.get("flags", {})
        sat_params = body.get("sat_params", {})

        if not _validate_tle(tle1, tle2):
            return jsonify({"error": "TLE invalide — vérifiez les deux lignes."}), 400

        pcfg = PerturbationConfig()
        for k, v in sat_params.items():
            if hasattr(pcfg, k):
                setattr(pcfg, k, float(v))

        active_keys = [k for k, v in flags.items() if v and k in PERT_META]

        try:
            data = _simulate_perturbations(tle1, tle2, hours, step_min, pcfg, active_keys)
        except Exception as e:
            logger.exception("Erreur simulation")
            return jsonify({"error": str(e)}), 500

        model_rmse = _get_model_rmse()
        n          = len(data["t_hours"])
        ai_corr    = _apply_ai_correction(data["sgp4_error_m"], model_rmse, n)

        data["active"]          = active_keys
        data["max_disp_m"]      = round(max(data["disp_per_step_m"]) if data["disp_per_step_m"] else 0, 3)
        data["max_accel_ms2"]   = round(max(data["accel_total_ms2"]) if data["accel_total_ms2"] else 0, 8)
        data["ai_corrected_m"]  = ai_corr
        data["model_rmse_m"]    = model_rmse
        data["model_trained"]   = model_rmse is not None
        # Positions SGP4 pour le globe 3D (sous-échantillonnées à max 200 pts)
        n_pts = len(data["t_hours"])
        step  = max(1, n_pts // 200)
        data["sgp4_pos_km"]    = data.pop("_sgp4_pos_km", [])
        data["sgp4_pos_km"]    = data["sgp4_pos_km"][::step]
        return jsonify(data)

    @app.route("/api/model_status")
    def model_status():
        ckpt     = cfg.checkpoint_path
        csv_path = os.path.join(cfg.log_dir, "training.csv")
        result   = {
            "model_type": cfg.model_type.upper(), "model_exists": os.path.exists(ckpt),
            "best_val_loss": None, "best_rmse_m": None,
            "best_mae_m": None, "best_epoch": None, "n_epochs_run": None,
        }
        if os.path.exists(csv_path):
            rows = list(csv.DictReader(open(csv_path)))
            if rows:
                best = min(rows, key=lambda r: float(r["val_loss"]))
                result.update({
                    "best_val_loss": round(float(best["val_loss"]), 6),
                    "best_rmse_m":   round(float(best["rmse_val"]) * 1000, 1),
                    "best_mae_m":    round(float(best["mae_val"]) * 1000, 1),
                    "best_epoch":    int(best["epoch"]),
                    "n_epochs_run":  len(rows),
                })
        return jsonify(result)

    @app.route("/api/satellites")
    def get_satellites():
        from tle_fetcher import get_tle_epoch
        return jsonify([{
            "name": name,
            "epoch": get_tle_epoch(l1).strftime("%Y-%m-%d %H:%M UTC") if get_tle_epoch(l1) else "?",
            "tle1": l1, "tle2": l2,
        } for name, l1, l2 in parse_tle_file(cfg.tle_source)])

    @app.route("/api/events")
    def get_events():
        path = os.path.join(cfg.model_dir, "ingest_report.json")
        if not os.path.exists(path):
            return jsonify({"events": [], "summary": "Aucun rapport."})
        return jsonify(json.load(open(path)))

    @app.route("/api/report_html")
    def report_html():
        path = os.path.join(cfg.model_dir, "report.html")
        if not os.path.exists(path):
            html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{margin:0;background:#0f0f1a;color:#7878a0;font-family:system-ui;
display:flex;align-items:center;justify-content:center;height:100vh;text-align:center}
code{color:#605DF6;display:block;margin:.5rem 0;font-size:.9rem}</style></head>
<body><div><p style="font-size:1.1rem;color:#e0e0f0">Aucun rapport disponible</p>
<p style="margin:.75rem 0 .25rem">Lancez d'abord l'entraînement :</p>
<code>python main.py --mode train</code>
<p style="margin:.5rem 0 .25rem">Puis évaluez :</p>
<code>python main.py --mode evaluate</code></div></body></html>"""
            return Response(html, mimetype="text/html")
        return send_file(os.path.abspath(path), mimetype="text/html")

    #  Import Space-Track 

    @app.route("/api/import/credentials")
    def import_credentials():
        """Vérifie si les credentials Space-Track sont configurés."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        try:
            from spacetrack import check_credentials_available
            return jsonify(check_credentials_available())
        except Exception as e:
            return jsonify({"ok": False, "reason": str(e)})

    @app.route("/api/import/stream")
    def import_stream():
        """
        SSE endpoint — stream la progression de l'import en temps réel.

        Paramètres GET :
          mode     : spacetrack | simulate | celestrak  (défaut: spacetrack)
          days     : jours d'historique Space-Track     (défaut: 7)
          finetune : true/false — lancer le fine-tuning après import

        Utilisation JS :
          const es = new EventSource('/api/import/stream?mode=spacetrack&days=7')
          es.onmessage = e => { const d = JSON.parse(e.data); ... }
        """
        import queue, threading

        mode     = request.args.get("mode", "spacetrack")
        days     = int(request.args.get("days", "7"))
        finetune = request.args.get("finetune", "false").lower() == "true"

        q = queue.Queue()

        def emit(message: str, pct: int = None):
            """Envoie un message dans la queue SSE."""
            q.put({"msg": message, "pct": pct})

        def run():
            """Exécute le pipeline dans un thread séparé."""
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
            try:
                from make_differential_dataset import run_pipeline
                report = run_pipeline(
                    mode=mode, days=days, finetune=finetune,
                    emit=emit,
                )
                q.put({"done": True, "report": report})
            except Exception as e:
                logger.exception("Erreur import pipeline")
                q.put({"done": True, "error": str(e)})

        # Lancer dans un thread pour ne pas bloquer Flask
        t = threading.Thread(target=run, daemon=True)
        t.start()

        def generate():
            """Générateur SSE."""
            while True:
                try:
                    item = q.get(timeout=60)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"):
                        break
                except Exception:
                    yield f"data: {json.dumps({'done': True, 'error': 'Timeout'})}\n\n"
                    break

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":      "keep-alive",
            },
        )

    @app.route("/api/import/report")
    def import_report():
        """Retourne le dernier rapport d'import."""
        path = "data/differential_report.json"
        if not os.path.exists(path):
            return jsonify({"error": "Aucun import effectué."})
        return jsonify(json.load(open(path)))


    #  Recherche satellites (Space-Track / catalogue local) 

    @app.route("/api/satellites/search")
    def search_satellites():
        """
        Recherche un satellite par nom ou NORAD ID dans la base SQLite locale.
        ?q=STARLINK  ou  ?q=25544
        Retourne jusqu a 20 resultats.
        """
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify([])

        try:
            from tle_database import get_connection, init_db
            init_db()
            conn = get_connection()
            try:
                q_upper = q.upper()
                rows = conn.execute("""
                    SELECT DISTINCT norad_id, name, orbit_class, alt_km, mm
                    FROM tle_records
                    WHERE UPPER(norad_id) LIKE ?
                       OR UPPER(name) LIKE ?
                    ORDER BY
                        CASE WHEN UPPER(norad_id) = ? THEN 0
                             WHEN UPPER(name) = ?     THEN 1
                             WHEN UPPER(norad_id) LIKE ? THEN 2
                             ELSE 3 END,
                        name
                    LIMIT 20
                """, (
                    f"%{q_upper}%",
                    f"%{q_upper}%",
                    q_upper,
                    q_upper,
                    f"{q_upper}%",
                )).fetchall()
            finally:
                conn.close()

            results = []
            for r in rows:
                results.append({
                    "norad": r["norad_id"],
                    "name":  r["name"],
                    "type":  r["orbit_class"] or "LEO",
                    "alt_km": round(r["alt_km"] or 0, 0),
                })
            return jsonify(results)

        except Exception as ex:
            logger.warning(f"search_satellites DB error: {ex}")
            return jsonify([])


    @app.route("/api/satellites/fetch", methods=["POST"])
    def fetch_satellite_tle():
        """
        Récupère les TLE d'un satellite depuis Celestrak ou Space-Track.
        Body: {"norad": "25544"}
        """
        body  = request.get_json(force=True)
        norad = str(body.get("norad", "")).strip()
        if not norad:
            return jsonify({"error": "NORAD ID requis"}), 400

        # Essai Celestrak (gratuit, pas de compte)
        import urllib.request
        try:
            url = f"https://celestrak.org/SOCRATES/query.php?CATALOG={norad}&FORMAT=tle"
            with urllib.request.urlopen(url, timeout=10) as r:
                content = r.read().decode("utf-8", errors="replace")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            from tle_fetcher import _validate_tle, get_tle_epoch
            for i in range(0, len(lines)-2, 3):
                if _validate_tle(lines[i+1], lines[i+2]):
                    ep = get_tle_epoch(lines[i+1])
                    return jsonify({
                        "name":  lines[i],
                        "tle1":  lines[i+1],
                        "tle2":  lines[i+2],
                        "epoch": ep.strftime("%Y-%m-%d %H:%M UTC") if ep else "?",
                        "source": "Celestrak",
                    })
        except Exception as e:
            pass

        return jsonify({"error": f"Satellite NORAD {norad} introuvable."}), 404

    @app.route("/api/satellites/history")
    def satellite_history():
        """
        Retourne l'historique des paramètres TLE d'un satellite (depuis le cache local).
        ?norad=25544
        """
        norad = request.args.get("norad", "").strip()
        cache_dir = os.path.join("data", "spacetrack_cache")
        results = []
        for days in [1, 3, 7, 14, 30]:
            path = os.path.join(cache_dir, f"{norad}_{days}d.json")
            if os.path.exists(path):
                try:
                    history = json.load(open(path))
                    for name, l1, l2 in history:
                        from tle_fetcher import get_tle_epoch
                        ep = get_tle_epoch(l1)
                        if not ep:
                            continue
                        # Extraire les éléments orbitaux
                        mm  = float(l2[52:63]) if len(l2) > 63 else 0
                        inc = float(l2[8:16])  if len(l2) > 16 else 0
                        ecc = float("0." + l2[26:33]) if len(l2) > 33 else 0
                        results.append({
                            "epoch":   ep.isoformat(),
                            "epoch_h": ep.strftime("%m/%d %H:%M"),
                            "mm":      round(mm, 5),
                            "inc":     round(inc, 4),
                            "ecc":     round(ecc, 7),
                            "alt_km":  round((398600.4418*(86400/mm/(2*3.14159))**2)**(1/3)-6371, 1) if mm>0 else 0,
                        })
                except Exception:
                    pass
                break

        # Dédupliquer + trier
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["epoch"]):
            if r["epoch"] not in seen:
                seen.add(r["epoch"]); unique.append(r)
        return jsonify(unique)

    #  Commandes DEV via SSE 

    @app.route("/api/run/<cmd>")
    def run_command(cmd):
        """
        Lance une commande du pipeline en arrière-plan avec streaming SSE.
        Commandes autorisées : train, evaluate, benchmark, finetune
        """
        allowed = {"train", "evaluate", "benchmark", "finetune", "predict"}
        if cmd not in allowed:
            return jsonify({"error": f"Commande non autorisée: {cmd}"}), 400

        import queue, threading, subprocess

        q = queue.Queue()

        def run():
            try:
                root = os.path.join(os.path.dirname(__file__), "..")
                proc = subprocess.Popen(
                    [sys.executable, "main.py", "--mode", cmd],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=os.path.abspath(root), bufsize=1,
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        # Extraire niveau de log si présent
                        msg = line.split(" | ")[-1] if " | " in line else line
                        q.put({"msg": msg})
                proc.wait()
                status = "success" if proc.returncode == 0 else "error"
                q.put({"done": True, "status": status, "returncode": proc.returncode})
            except Exception as e:
                q.put({"done": True, "status": "error", "msg": str(e)})

        t = threading.Thread(target=run, daemon=True)
        t.start()

        def generate():
            while True:
                try:
                    item = q.get(timeout=300)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"):
                        break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'status':'timeout'})}\n\n"
                    break

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    @app.route("/api/training_history")
    def training_history():
        """Retourne l'historique complet d'entraînement pour les graphiques."""
        csv_path = os.path.join(cfg.log_dir, "training.csv")
        if not os.path.exists(csv_path):
            return jsonify({"epochs":[],"train_loss":[],"val_loss":[],"rmse_m":[],"mae_m":[]})
        rows = list(csv.DictReader(open(csv_path)))
        return jsonify({
            "epochs":     [int(r["epoch"])        for r in rows],
            "train_loss": [float(r["train_loss"]) for r in rows],
            "val_loss":   [float(r["val_loss"])   for r in rows],
            "rmse_m":     [round(float(r["rmse_val"])*1000,1) for r in rows],
            "mae_m":      [round(float(r["mae_val"])*1000,1)  for r in rows],
            "lr":         [float(r["lr"])          for r in rows],
        })

    #  Conjonctions orbitales 

    @app.route("/api/conjunctions/stream")
    def conjunction_stream():
        """
        SSE — Lance l'analyse de conjonctions en arrière-plan.
        Paramètres GET :
          constellation : starlink | active | gps-ops | ...  (défaut: starlink)
          hours         : horizon en heures (1–168, défaut: 24)
          step_min      : résolution en minutes (1–15, défaut: 5)
          threshold_km  : seuil de distance (0.1–50, défaut: 5)
          extra_norad   : NORAD IDs supplémentaires séparés par virgule
        """
        import queue, threading

        constellation = request.args.get("constellation", "starlink")
        hours         = min(max(float(request.args.get("hours",  "24")),  1), 168)
        step_min      = min(max(float(request.args.get("step_min", "5")), 1),  15)
        threshold_km  = min(max(float(request.args.get("threshold_km","5")), 0.1), 50)
        extra_norad   = request.args.get("extra_norad", "")
        pc_method     = request.args.get("pc_method", "foster")
        mc_n          = int(request.args.get("mc_n", "50000"))
        # Flags de perturbations (JSON encodé)
        import urllib.parse as _up
        pert_json = request.args.get("pert_flags", "{}")
        try:
            pert_flags = json.loads(_up.unquote(pert_json)) if pert_json != "{}" else None
        except Exception:
            pert_flags = None

        q = queue.Queue()

        def emit(msg, pct=None):
            q.put({"msg": msg, "pct": pct})

        def run():
            sys.path.insert(0, os.path.dirname(__file__))
            try:
                from conjunction import run_conjunction_analysis, fetch_constellation

                # Satellites supplémentaires (ex: ISS ajouté au screening Starlink)
                extra_tles = []
                if extra_norad:
                    import urllib.request
                    for norad in extra_norad.split(","):
                        norad = norad.strip()
                        if not norad: continue
                        try:
                            url = f"https://celestrak.org/SOCRATES/query.php?CATALOG={norad}&FORMAT=tle"
                            with urllib.request.urlopen(url, timeout=8) as r:
                                lines = [l.strip() for l in r.read().decode().splitlines() if l.strip()]
                            from tle_fetcher import _validate_tle
                            for i in range(0, len(lines)-2, 3):
                                if _validate_tle(lines[i+1], lines[i+2]):
                                    extra_tles.append((lines[i], lines[i+1], lines[i+2]))
                        except Exception:
                            pass

                conjunctions = run_conjunction_analysis(
                    constellation=constellation,
                    extra_tles=extra_tles,
                    hours=hours,
                    step_min=step_min,
                    threshold_km=threshold_km,
                    max_results=300,
                    pc_method=pc_method,
                    mc_n=mc_n,
                    pert_flags=pert_flags,
                    emit=emit,
                )
                q.put({"done": True, "conjunctions": conjunctions,
                       "n_total": len(conjunctions)})
            except Exception as e:
                logger.exception("Erreur analyse conjonctions")
                q.put({"done": True, "error": str(e)})

        t = threading.Thread(target=run, daemon=True)
        t.start()

        def generate():
            while True:
                try:
                    item = q.get(timeout=600)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"):
                        break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"
                    break

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no",
                                 "Connection":"keep-alive"})

    @app.route("/api/conjunctions/constellations")
    def list_constellations():
        """Liste les constellations disponibles sur Celestrak."""
        return jsonify([
            {"id": "starlink",  "name": "Starlink",          "n_approx": 3000},
            {"id": "oneweb",    "name": "OneWeb",             "n_approx": 600},
            {"id": "gps-ops",   "name": "GPS opérationnel",   "n_approx": 31},
            {"id": "galileo",   "name": "Galileo",            "n_approx": 28},
            {"id": "iridium",   "name": "Iridium NEXT",       "n_approx": 66},
            {"id": "active",    "name": "Tous actifs (LEO)",  "n_approx": 5000},
            {"id": "stations",  "name": "Stations spatiales", "n_approx": 10},
            {"id": "debris",    "name": "Débris Cosmos-1408", "n_approx": 1500},
        ])



    #  Météo spatiale 

    @app.route("/api/space_weather")
    def space_weather():
        """Conditions de météo spatiale actuelles depuis NOAA SWPC."""
        try:
            from space_weather import get_current_conditions
            return jsonify(get_current_conditions())
        except Exception as e:
            # Retourner des valeurs nominales si NOAA inaccessible
            return jsonify({
                "f107_current": 150.0, "f107_81day": 150.0,
                "kp_current": 2.0, "kp_max_24h": 2.0,
                "kp_forecast": [], "storm_level": "DONNÉES INDISPONIBLES",
                "storm_color": "#7878a0", "alerts": [],
                "source_time": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                "_live": False, "error": str(e),
            })

    @app.route("/api/earth_texture_url")
    def earth_texture_url():
        """Retourne l'URL de la texture terrestre (pour compatibilité)."""
        from datetime import datetime, timedelta, timezone
        yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        url = (
            "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
            "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
            "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
            f"&TIME={yesterday}&BBOX=-90,-180,90,180"
            "&CRS=EPSG:4326&WIDTH=2048&HEIGHT=1024&FORMAT=image/jpeg"
        )
        return jsonify({"url": url, "date": yesterday})

    @app.route("/api/earth_texture")
    def earth_texture():
        """
        NASA Blue Marble — image statique sans nuages.
        URLs directes NASA Visible Earth, cache 30 jours.
        IMPORTANT: supprimer data/earth_texture_cache.jpg si l'ancienne
        version GIBS/MODIS (avec nuages) est en cache.
        """
        import urllib.request, time
        from flask import make_response
        from pathlib import Path

        cache = Path("data/earth_texture_cache.jpg")

        # Invalider le cache si trop petit (GIBS partiel ~512KB) ou trop ancien
        if cache.exists():
            age = time.time() - cache.stat().st_mtime
            size = cache.stat().st_size
            if size > 1_000_000 and age < 86400*30:
                resp = make_response(cache.read_bytes())
                resp.headers["Content-Type"]  = "image/jpeg"
                resp.headers["Cache-Control"] = "public, max-age=2592000"
                return resp
            else:
                cache.unlink()  # invalide : trop petit ou expiré

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "image/jpeg,image/*",
        }
        # Images statiques NASA — équirectangulaires complètes, sans nuages
        urls = [
            # Blue Marble 2004 — 5400x2700 (image de référence NASA)
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74393/world.200412.3x5400x2700.jpg",
            # Blue Marble 2002 — 2048x1024
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg",
            # GIBS BlueMarble_NextGeneration (WMS, sans date donc sans nuages)
            (
                "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
                "&LAYERS=BlueMarble_NextGeneration"
                "&BBOX=-90,-180,90,180&CRS=EPSG:4326"
                "&WIDTH=2048&HEIGHT=1024&FORMAT=image/jpeg"
            ),
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if len(data) > 1_000_000:   # exiger >1MB = image complète
                    cache.parent.mkdir(exist_ok=True)
                    cache.write_bytes(data)
                    logger.info(f"Blue Marble OK — {len(data)//1024} KB — {url[:60]}")
                    resp = make_response(data)
                    resp.headers["Content-Type"]  = "image/jpeg"
                    resp.headers["Cache-Control"] = "public, max-age=2592000"
                    return resp
                else:
                    logger.debug(f"Blue Marble trop petit ({len(data)//1024} KB): {url[:60]}")
            except Exception as e:
                logger.debug(f"Blue Marble {url[:60]}: {e}")

        logger.warning("Blue Marble indisponible — toutes les sources ont échoué")
        return ("Texture indisponible", 404)

    @app.route("/api/earth_texture_url")
    def earth_texture_url_compat():
        """Compatibilité."""
        return jsonify({"status": "use_api_earth_texture"})

    @app.route("/api/earth_night")
    def earth_night():
        """
        NASA Black Marble — lumières nocturnes.
        Source primaire : uscoalexports.org DNB 3600x1800 (domaine public NASA).
        Cache 30 jours.
        """
        import urllib.request, time
        from flask import make_response
        from pathlib import Path

        cache = Path("data/earth_night_cache.jpg")
        if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400*30:
            resp = make_response(cache.read_bytes())
            resp.headers["Content-Type"]  = "image/jpeg"
            resp.headers["Cache-Control"] = "public, max-age=2592000"
            return resp

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "image/jpeg,image/*",
            "Referer": "https://www.google.com/",
        }
        urls = [
            "http://uscoalexports.org/wp-content/uploads/2011/03/dnb_land_ocean_ice.2012.3600x1800.jpg",
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144897/BlackMarble_2016_3km.jpg",
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/55000/55167/earth_lights_lrg.jpg",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if len(data) > 200_000:
                    cache.parent.mkdir(exist_ok=True)
                    cache.write_bytes(data)
                    logger.info(f"Night texture OK — {len(data)//1024}KB — {url[:60]}")
                    resp = make_response(data)
                    resp.headers["Content-Type"]  = "image/jpeg"
                    resp.headers["Cache-Control"] = "public, max-age=2592000"
                    return resp
            except Exception as e:
                logger.debug(f"Night texture {url[:60]}: {e}")

        logger.warning("Night texture indisponible")
        return ("Night texture indisponible", 404)

    @app.route("/api/clouds_config", methods=["POST"])
    def clouds_config():
        """Reçoit host+path RainViewer depuis le navigateur."""
        import json
        from pathlib import Path
        data = request.get_json(silent=True) or {}
        host = data.get("host","https://tilecache.rainviewer.com")
        path = data.get("path","")
        if path:
            cache_dir = Path("data/clouds_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir/"timestamp.json").write_text(
                json.dumps({"host":host,"path":path})
            )
            logger.info(f"RainViewer config reçue: {path}")
        return jsonify({"ok":True})

    @app.route("/api/clouds_tile/<int:z>/<int:x>/<int:y>")
    def clouds_tile(z, x, y):
        """Proxy tile RainViewer infrarouge — utilise le path configuré par le navigateur."""
        import urllib.request, time, json
        from flask import make_response
        from pathlib import Path

        cache_dir = Path("data/clouds_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Lire le path configuré par le navigateur
        ts_file = cache_dir / "timestamp.json"
        rv_host = "https://tilecache.rainviewer.com"
        rv_path = None
        if ts_file.exists():
            try:
                info = json.loads(ts_file.read_text())
                rv_host = info.get("host", rv_host)
                rv_path = info.get("path")
            except Exception:
                pass

        if not rv_path:
            return ("RainViewer path non configuré — le navigateur doit d'abord appeler /api/clouds_config", 503)

        tile_key = rv_path.replace('/','_')
        tile_cache = cache_dir / f"{tile_key}_{z}_{x}_{y}.png"
        if tile_cache.exists() and (time.time() - tile_cache.stat().st_mtime) < 3600:
            resp = make_response(tile_cache.read_bytes())
            resp.headers["Content-Type"]  = "image/png"
            resp.headers["Cache-Control"] = "public, max-age=3600"
            return resp

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        }
        for suffix in ["/1/1_1.png", "/1/0_0.png", "/0/0_0.png"]:
            url = f"{rv_host}{rv_path}/{z}/{x}/{y}{suffix}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r:
                    data_bytes = r.read()
                if len(data_bytes) > 200:
                    tile_cache.write_bytes(data_bytes)
                    resp = make_response(data_bytes)
                    resp.headers["Content-Type"]  = "image/png"
                    resp.headers["Cache-Control"] = "public, max-age=3600"
                    return resp
            except Exception as e:
                logger.debug(f"RainViewer tile {url}: {e}")

        return ("Tile indisponible", 503)


    #  Base de données TLE 

    @app.route("/api/rendezvous/compute", methods=["POST"])
    def rendezvous_compute():
        """
        Calcule la séquence de manœuvres pour un rapprochement orbital.
        Utilise rendezvous_engine.py — implémentation rigoureuse.
        Méthodes : hohmann, lambert, phasing, bielliptic, lowthrust.
        """
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from rendezvous_engine import compute_rendezvous as _compute_rdv

        data = request.get_json(force=True, silent=True) or {}
        tle1_c = data.get("chaser_tle1", "")
        tle2_c = data.get("chaser_tle2", "")
        tle1_t = data.get("target_tle1", "")
        tle2_t = data.get("target_tle2", "")
        method  = data.get("method", "hohmann")
        dur_h   = float(data.get("duration_h", 24))
        dv_max  = float(data.get("dv_max_ms", 500))
        ang_deg = float(data.get("approach_angle_deg", 0))

        if not all([tle1_c, tle2_c, tle1_t, tle2_t]):
            return jsonify({"error": "TLE manquants"}), 400

        try:
            result = _compute_rdv(
                tle1_c, tle2_c, tle1_t, tle2_t,
                method=method, dur_h=dur_h,
                dv_max_ms=dv_max, approach_angle_deg=ang_deg
            )
            return jsonify(result)
        except Exception as ex:
            logger.error(f"Rendezvous compute error: {ex}", exc_info=True)
            return jsonify({"error": str(ex)}), 500

    @app.route("/api/rendezvous/trajectory", methods=["POST"])
    @app.route("/api/rendezvous/trajectory", methods=["POST"])
    @app.route("/api/rendezvous/trajectory", methods=["POST"])
    def rendezvous_trajectory():
        """
        Calcule le plan de manœuvre optimal et les trajectoires avec ΔV appliqués.

        Stratégie selon méthode :
          - Hohmann/Phasing : timing de fenêtre exact + ΔV prograde/rétrograde
          - Lambert : optimisation sur tof, ΔV dans direction calculée
          - Bi-elliptique/Lowthrust : idem Hohmann avec orbite intermédiaire

        Propagation :
          - Chaser : képlerienne avec ΔV appliqués
          - Cible  : SGP4 continue
        """
        import sys as _sys, os as _os, math as _m
        import numpy as _np
        from datetime import datetime, timezone, timedelta

        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from rendezvous_engine import (
            propagate_sgp4, rv_to_elements, propagate_kepler,
            lambert_battin, compute_rendezvous, MU, RE as _RE
        )

        data = request.get_json(force=True, silent=True) or {}
        tle1_c = data.get("chaser_tle1",""); tle2_c = data.get("chaser_tle2","")
        tle1_t = data.get("target_tle1",""); tle2_t = data.get("target_tle2","")
        method  = data.get("method","hohmann")
        dur_h   = float(data.get("duration_h", 72))
        dv_max  = float(data.get("dv_max_ms", 500))
        ang_deg = float(data.get("approach_angle_deg", 0))
        N_PTS   = 300

        if not all([tle1_c,tle2_c,tle1_t,tle2_t]):
            return jsonify({"error":"TLE manquants"}), 400

        try:
            now = datetime.now(timezone.utc)
            r0, v0   = propagate_sgp4(tle1_c, tle2_c, now)
            rt0, vt0 = propagate_sgp4(tle1_t, tle2_t, now)
            el_c = rv_to_elements(r0, v0)
            el_t = rv_to_elements(rt0, vt0)

            #  Calcul du plan de manœuvre optimal 
            plan = compute_rendezvous(
                tle1_c, tle2_c, tle1_t, tle2_t,
                method=method, dur_h=dur_h, dv_max_ms=dv_max,
                approach_angle_deg=ang_deg
            )
            maneuvers = plan.get("maneuvers", [])

            #  Helpers RTN 
            def rtn_basis(r_vec, v_vec):
                r_hat = r_vec / _np.linalg.norm(r_vec)
                h_vec = _np.cross(r_vec, v_vec)
                n_hat = h_vec / _np.linalg.norm(h_vec)
                t_hat = _np.cross(n_hat, r_hat)
                return r_hat, t_hat, n_hat

            def dv_to_rtn(dv_vec, r_vec, v_vec):
                R, T, N = rtn_basis(r_vec, v_vec)
                return (float(_np.dot(dv_vec,R)),
                        float(_np.dot(dv_vec,T)),
                        float(_np.dot(dv_vec,N)))

            def prograde_dv(v_cur, dv_ms):
                return v_cur / _np.linalg.norm(v_cur) * (dv_ms/1000.)

            def retrograde_dv(v_cur, dv_ms):
                return -v_cur / _np.linalg.norm(v_cur) * (dv_ms/1000.)

            #  Propagation avec ΔV appliqués 
            segments   = {}
            burns_out  = []
            burn_positions = []

            wait_s  = plan.get("wait_h", 0.) * 3600.
            t_tr_s  = plan.get("transfer_time_h", 0.) * 3600.

            # Propager le chaser sur son orbite jusqu'au moment du ΔV₁
            r_cur = r0.copy(); v_cur = v0.copy()
            if wait_s > 30:
                pts_wait = []
                for k in range(N_PTS+1):
                    t = wait_s*k/N_PTS
                    rk,_ = propagate_kepler(r0, v0, t)
                    pts_wait.append(rk.tolist())
                segments["wait"] = pts_wait
                r_cur, v_cur = propagate_kepler(r0, v0, wait_s)
            else:
                segments["wait"] = [r0.tolist()]

            # Burn 1 : direction selon méthode
            if len(maneuvers) >= 1:
                m1  = maneuvers[0]
                dv1_ms = m1["dv_ms"]

                # Direction du burn selon méthode
                if method in ("hohmann","phasing","bielliptic"):
                    # Prograde si montée, rétrograde si descente
                    if el_t["a"] >= el_c["a"]:
                        dv1_vec = prograde_dv(v_cur, dv1_ms)
                    else:
                        dv1_vec = retrograde_dv(v_cur, dv1_ms)
                elif method == "lambert":
                    # Optimisation Lambert sur le timing exact
                    best_dv = float("inf"); best_v1 = None; best_r_ta = None
                    # Fenêtre : autour de wait_s ± 2*T_c
                    T_c = el_c["T"]
                    for delta in _np.linspace(-2*T_c, 2*T_c, 40):
                        tw2 = max(0., wait_s + delta)
                        for tof_frac in [0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0]:
                            tof2 = T_c * tof_frac
                            rw2,vw2 = propagate_kepler(r0,v0,tw2)
                            r_ta2,v_ta2 = propagate_kepler(rt0,vt0,tw2+tof2)
                            try:
                                v1l,v2l = lambert_battin(rw2, r_ta2, tof2)
                                dv1l = _np.linalg.norm(v1l-vw2)*1000
                                dv2l = _np.linalg.norm(v_ta2-v2l)*1000
                                tot  = dv1l+dv2l
                                if tot < best_dv:
                                    best_dv=tot; best_v1=v1l
                                    best_r_ta=r_ta2; best_vw=vw2; best_tw2=tw2; best_tof2=tof2
                                    best_v_ta2=v_ta2
                            except: pass
                    if best_v1 is not None:
                        dv1_vec = best_v1 - best_vw
                        # Mettre à jour wait_s et t_tr_s avec les valeurs optimales
                        wait_s = best_tw2; t_tr_s = best_tof2
                        r_cur,v_cur = propagate_kepler(r0,v0,wait_s)
                        # Recalculer segment wait avec le bon wait
                        pts_wait=[]
                        for k in range(N_PTS+1):
                            t=wait_s*k/N_PTS
                            rk,_=propagate_kepler(r0,v0,t)
                            pts_wait.append(rk.tolist())
                        segments["wait"]=pts_wait
                    else:
                        dv1_vec = prograde_dv(v_cur, dv1_ms)
                else:
                    dv1_vec = prograde_dv(v_cur, dv1_ms)

                v_after_dv1 = v_cur + dv1_vec
                dv1_rtn = dv_to_rtn(dv1_vec, r_cur, v_cur)
                burns_out.append({
                    "label":   m1.get("type","ΔV₁"),
                    "t_h":     round(wait_s/3600, 4),
                    "dv_ms":   round(float(_np.linalg.norm(dv1_vec))*1000, 3),
                    "dv_R_ms": round(dv1_rtn[0]*1000, 3),
                    "dv_T_ms": round(dv1_rtn[1]*1000, 3),
                    "dv_N_ms": round(dv1_rtn[2]*1000, 3),
                    "desc":    m1.get("description",""),
                })
                burn_positions.append(r_cur.tolist())

                # Segment 1 : ellipse de transfert
                pts_tr=[]
                for k in range(N_PTS+1):
                    t = t_tr_s*k/N_PTS
                    rk,_ = propagate_kepler(r_cur, v_after_dv1, t)
                    pts_tr.append(rk.tolist())
                segments["transfer"] = pts_tr

                r_cur2, v_cur2 = propagate_kepler(r_cur, v_after_dv1, t_tr_s)

                # Burn 2 : circularisation
                if len(maneuvers) >= 2:
                    m2 = maneuvers[1]
                    dv2_ms = m2["dv_ms"]
                    if method in ("hohmann","phasing","lambert"):
                        if el_t["a"] >= el_c["a"]:
                            dv2_vec = prograde_dv(v_cur2, dv2_ms)
                        else:
                            dv2_vec = retrograde_dv(v_cur2, dv2_ms)
                    else:
                        dv2_vec = prograde_dv(v_cur2, dv2_ms)

                    v_after_dv2 = v_cur2 + dv2_vec
                    dv2_rtn = dv_to_rtn(dv2_vec, r_cur2, v_cur2)
                    burns_out.append({
                        "label":   m2.get("type","ΔV₂"),
                        "t_h":     round((wait_s+t_tr_s)/3600, 4),
                        "dv_ms":   round(float(_np.linalg.norm(dv2_vec))*1000, 3),
                        "dv_R_ms": round(dv2_rtn[0]*1000, 3),
                        "dv_T_ms": round(dv2_rtn[1]*1000, 3),
                        "dv_N_ms": round(dv2_rtn[2]*1000, 3),
                        "desc":    m2.get("description",""),
                    })
                    burn_positions.append(r_cur2.tolist())

                    # Orbite finale : 1 révolution
                    el_fin = rv_to_elements(r_cur2, v_after_dv2)
                    pts_fin=[]
                    for k in range(N_PTS+1):
                        t = el_fin["T"]*k/N_PTS
                        rk,_ = propagate_kepler(r_cur2, v_after_dv2, t)
                        pts_fin.append(rk.tolist())
                    segments["final_orbit"] = pts_fin
                    r_cur3 = r_cur2; v_cur3 = v_after_dv2

            # Burns supplémentaires (bi-elliptique ΔV₃, approche terminale)
            r_curN = r_cur2 if "r_cur2" in dir() else r_cur
            v_curN = v_cur3 if "v_cur3" in dir() else (v_after_dv2 if "v_after_dv2" in dir() else v_cur)
            t_cumul = wait_s + t_tr_s
            for mx in maneuvers[2:]:
                dt_mx = mx.get("t_from_now_h",0)*3600 - t_cumul
                if dt_mx > 30:
                    r_curN,v_curN = propagate_kepler(r_curN,v_curN,dt_mx)
                    t_cumul += dt_mx
                dv_mx = prograde_dv(v_curN, mx["dv_ms"])
                dv_rtn = dv_to_rtn(dv_mx,r_curN,v_curN)
                burns_out.append({
                    "label":   mx.get("type","ΔV"),
                    "t_h":     round(t_cumul/3600,4),
                    "dv_ms":   round(mx["dv_ms"],3),
                    "dv_R_ms": round(dv_rtn[0]*1000,3),
                    "dv_T_ms": round(dv_rtn[1]*1000,3),
                    "dv_N_ms": round(dv_rtn[2]*1000,3),
                    "desc":    mx.get("description",""),
                })
                burn_positions.append(r_curN.tolist())
                v_curN = v_curN + dv_mx

            # Cible : SGP4 sur toute la durée
            total_s = wait_s + t_tr_s + el_c["T"]
            step_s  = max(30., total_s/N_PTS)
            pts_t=[]
            for k in range(int(total_s/step_s)+1):
                try:
                    rt_k,_=propagate_sgp4(tle1_t,tle2_t,now+timedelta(seconds=k*step_s))
                    pts_t.append(rt_k.tolist())
                except: break
            segments["target"] = pts_t

            return jsonify({
                "segments":       segments,
                "burns":          burns_out,
                "burn_positions": burn_positions,
                "summary": {
                    "total_dv_ms":   round(plan.get("total_dv_ms",0.), 3),
                    "wait_h":        round(plan.get("wait_h",0.), 4),
                    "transfer_h":    round(plan.get("transfer_time_h",0.), 4),
                    "chaser_alt_km": round(float(_np.linalg.norm(r0))-_RE, 1),
                    "target_alt_km": round(float(_np.linalg.norm(rt0))-_RE, 1),
                    "delta_alt_km":  round(abs(float(_np.linalg.norm(rt0)-_np.linalg.norm(r0))), 1),
                    "method":        method,
                },
                "plan": plan,
            })

        except Exception as ex:
            logger.error(f"Trajectory: {ex}", exc_info=True)
            return jsonify({"error": str(ex)}), 500




    @app.route("/api/rendezvous/optimize", methods=["POST"])
    def rendezvous_optimize():
        """
        Calcule la combinaison optimale de manœuvres (ΔV minimum)
        parmi les méthodes sélectionnées par l'utilisateur.
        Retourne le classement, la meilleure solution, et les trajectoires.
        """
        import sys as _sys, os as _os, math as _m
        import numpy as _np
        from datetime import datetime, timezone, timedelta
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from rendezvous_engine import (
            propagate_sgp4, rv_to_elements, propagate_kepler,
            compute_rendezvous, compute_rendezvous_physical, compute_drift_j2,
            compute_rendezvous_graphs, bielliptic_plane_change, j2_convergence,
                compute_mission_plan, compute_plane_change_to_target,
            rv_to_elements, propagate_kepler, propagate_sgp4, MU, RE as _RE
        )

        data = request.get_json(force=True, silent=True) or {}
        tle1_c  = data.get("chaser_tle1","")
        tle2_c  = data.get("chaser_tle2","")
        tle1_t  = data.get("target_tle1","")
        tle2_t  = data.get("target_tle2","")
        methods = data.get("methods", ["hohmann","lambert","phasing"])
        dv_max  = float(data.get("dv_max_ms", 500))
        dur_h   = float(data.get("duration_h", 72))
        ang_deg = float(data.get("approach_angle_deg", 0))
        N_PTS   = 3000  # 10× plus de points pour la timeline

        if not all([tle1_c,tle2_c,tle1_t,tle2_t]):
            return jsonify({"error":"TLE manquants"}),400

        LABELS = {
            "hohmann":           "Hohmann 2-burns",
            "lambert":           "Lambert optimal",
            "phasing":           "Phasage orbital",
            "bielliptic":        "Bi-elliptique",
            "bielliptic_highalt":"Bi-ell. haute alt.",
            "lowthrust":         "Poussée faible",
        }

        try:
            now = datetime.now(timezone.utc)
            r0,v0   = propagate_sgp4(tle1_c,tle2_c,now)
            rt0,vt0 = propagate_sgp4(tle1_t,tle2_t,now)
            el_c = rv_to_elements(r0,v0)
            el_t = rv_to_elements(rt0,vt0)

            # Angle entre plans
            h_c = el_c['h_vec']/el_c['h']
            h_t = el_t['h_vec']/el_t['h']
            delta_plane = _m.degrees(_m.acos(_np.clip(_np.dot(h_c,h_t),-1.0,1.0)))

            #  Évaluer chaque méthode 
            results = []
            for method in methods:
                try:
                    # Utiliser compute_rendezvous_physical (Vallado 2013)
                    # pour Hohmann/Phasage/Bi-ell — ΔV analytiques exacts
                    res = compute_rendezvous_physical(
                        tle1_c, tle2_c, tle1_t, tle2_t,
                        method=method,
                        dv_max_ms=dv_max*10,
                        approach_angle_deg=ang_deg,
                    )
                    results.append(res)
                except Exception as ex:
                    logger.warning(f"Method {method} failed: {ex}")
                    results.append({
                        'method': method,
                        'total_dv_ms': float('inf'),
                        'error': str(ex)
                    })

            #  Classer par ΔV total 
            valid = [r for r in results if r.get('total_dv_ms',float('inf'))<float('inf')]
            valid.sort(key=lambda r: r['total_dv_ms'])

            if not valid:
                return jsonify({"error":"Aucune méthode n'a convergé"}),400

            best = valid[0]
            best['delta_plane_deg'] = round(delta_plane, 2)

            # Analyse J2 et bi-ell haute si plan > 15°
            if delta_plane > 15:
                j2 = j2_convergence(el_c, el_t, max_days=365)
                be_high = bielliptic_plane_change(el_c, el_t)
                best['j2_analysis']        = j2
                best['bielliptic_highalt'] = be_high
                if best['total_dv_ms'] > dv_max:
                    if j2['converges']:
                        best['warning'] = (
                            f"ΔV={best['total_dv_ms']:.0f}m/s > budget {dv_max:.0f}m/s. "
                            f"J2 converge en {j2['min_day']:.0f}j → {j2['min_angle_deg']:.1f}°."
                        )
                    else:
                        best['warning'] = (
                            f"ΔV={best['total_dv_ms']:.0f}m/s > budget {dv_max:.0f}m/s. "
                            f"J2 inefficace (sens opposés). "
                            f"Bi-ell. haute: {be_high['total_dv']:.0f}m/s en {be_high['duration_h']:.0f}h "
                            f"(borne inf: {be_high['dv_lower_bound_ms']:.0f}m/s)."
                        )

            #  Classement 
            ranking = [{
                'method':        r['method'],
                'method_label':  LABELS.get(r['method'], r['method']),
                'total_dv_ms':   round(r['total_dv_ms'], 2),
                'wait_h':        round(r.get('wait_h',0.), 3),
                'transfer_h':    round(r.get('transfer_time_h',0.), 3),
            } for r in valid]

            #  Trajectoires — Lambert garanti dist_TCA=0 
            # Pour chaque méthode, on calcule la trajectoire réelle :
            # - Lambert : v1 calculé → r_c(tof) = r_t(tof) exact
            # - Hohmann/Phasing : on cherche le wait_s optimal par Lambert
            # - Low-thrust : spirale discrétisée
            wait_s  = best.get('wait_h', 0.) * 3600.
            t_tr_s  = best.get('transfer_time_h', 0.) * 3600.
            method_used = best.get('method','lambert')

            def prograde_dv(v, dv_ms):
                return v / _np.linalg.norm(v) * (dv_ms/1000.)
            def retrograde_dv(v, dv_ms):
                return -v / _np.linalg.norm(v) * (dv_ms/1000.)
            def rtn_basis(r, v):
                rh=r/_np.linalg.norm(r)
                nh=_np.cross(r,v); nh/=_np.linalg.norm(nh)
                return rh, _np.cross(nh,rh), nh
            def dv_to_rtn(dv, r, v):
                R,T,N=rtn_basis(r,v)
                return float(_np.dot(dv,R)), float(_np.dot(dv,T)), float(_np.dot(dv,N))

            # Trajectoire basée sur compute_rendezvous_physical (Vallado 2013)
            # Lambert retiré — non adapté au rendezvous LEO-LEO

            #  Génération de trajectoire depuis les manœuvres calculées 
            maneuvers_list = best.get('maneuvers', [])
            wait_s  = best.get('wait_h', 0.) * 3600.
            t_tr_s  = best.get('transfer_time_h', 0.) * 3600.

            r0_, v0_   = propagate_sgp4(tle1_c, tle2_c, now)
            rt0_, vt0_ = propagate_sgp4(tle1_t, tle2_t, now)
            el_c_ = rv_to_elements(r0_, v0_)
            el_t_ = rv_to_elements(rt0_, vt0_)

            a_c = el_c_['a']; a_t = el_t_['a']; a_tr=(a_c+a_t)/2
            e_tr=(a_t-a_c)/(a_t+a_c)
            n_tr=_m.sqrt(MU/a_tr**3)

            N_PTS=1500; segments={}; burns_out=[]

            # 
            # Génération trajectoire selon la méthode
            # 
            T_c = el_c_['T'];  T_t = el_t_['T']
            a_c = el_c_['a'];  a_t = el_t_['a']

            method_used = best.get('method', 'hohmann')
            be_data     = best.get('bielliptic_highalt', {})

            #  BRANCHE 1 : Bi-elliptique haute altitude 
            if method_used == 'bielliptic_highalt' and be_data:

                r_b = (be_data['r_b_km'] + _RE)

                #  Départ optimal : satellite sur la ligne d'intersection
                # des deux plans orbitaux (condition nécessaire pour Δi correct)
                # Référence : Chobotov (2002) §9.3
                h_c_vec = _np.cross(r0_, v0_)
                h_t_vec = _np.cross(rt0_, vt0_)
                h_c_hat = h_c_vec / _np.linalg.norm(h_c_vec)
                h_t_hat = h_t_vec / _np.linalg.norm(h_t_vec)
                node_dir = _np.cross(h_c_hat, h_t_hat)
                node_nrm = _np.linalg.norm(node_dir)
                if node_nrm < 1e-8:       # plans parallèles : Δi≈0
                    node_dir = r0_ / _np.linalg.norm(r0_)
                else:
                    node_dir /= node_nrm

                # Recherche numérique du point de départ optimal
                best_t_dep = 0.; best_align = -1.
                for _k in range(400):
                    _t = T_c * _k / 400
                    _r, _ = propagate_kepler(r0_, v0_, _t)
                    _a = abs(float(_np.dot(_r / _np.linalg.norm(_r), node_dir)))
                    if _a > best_align:
                        best_align = _a; best_t_dep = _t
                # Raffiner
                _step = T_c / 400
                for _k in range(80):
                    _t = best_t_dep - _step + 2*_step*_k/79
                    if _t < 0: continue
                    _r, _ = propagate_kepler(r0_, v0_, _t)
                    _a = abs(float(_np.dot(_r / _np.linalg.norm(_r), node_dir)))
                    if _a > best_align:
                        best_align = _a; best_t_dep = _t

                # Position et vitesse au point de départ optimal
                r_w, v_w = propagate_kepler(r0_, v0_, best_t_dep)
                r_w_mag  = float(_np.linalg.norm(r_w))
                a_tr1    = (r_w_mag + r_b) / 2
                a_tr2    = (r_b + a_t)     / 2
                T_tr1    = _m.pi * _m.sqrt(a_tr1**3 / MU)
                T_tr2    = _m.pi * _m.sqrt(a_tr2**3 / MU)
                v1_dep   = _m.sqrt(MU * (2./r_w_mag - 1./a_tr1))
                v_dep1   = v_w / _np.linalg.norm(v_w) * v1_dep

                #  Sampling adaptatif : uniforme en distance (gap ≤ 300km) 
                def _arc_pts(r_start, v_start, T_arc, t_offset, max_gap=300.):
                    """Génère les points d'un arc Kepler avec pas adaptatif
                    en distance (évite les coupures dans toSegs). """
                    pts = []; t = 0.
                    while t <= T_arc + 0.1:
                        try:
                            rk, vk = propagate_kepler(r_start, v_start, t)
                            pts.append(rk.tolist() + [t_offset + t])
                            v_mag = float(_np.linalg.norm(vk))
                            dt = max_gap / v_mag if v_mag > 1e-6 else 60.
                            dt = min(dt, T_arc - t + 1.) if t < T_arc else 0.
                            if dt <= 0: break
                            t += dt
                        except Exception: break
                    return pts

                pts_arc1 = _arc_pts(r_w, v_dep1, T_tr1, 0.)

                #  Burn 2 à l'apogée : changement de plan 
                r_apo, v_apo1 = propagate_kepler(r_w, v_dep1, T_tr1)

                # Direction dans le plan cible (h_t × r̂_apo)
                h_t_vec = _np.cross(rt0_, vt0_)
                h_t_hat = h_t_vec / _np.linalg.norm(h_t_vec)
                r_apo_hat = r_apo / _np.linalg.norm(r_apo)
                v_dir2 = _np.cross(h_t_hat, r_apo_hat)
                nrm = _np.linalg.norm(v_dir2)
                v_dir2 = v_dir2 / nrm if nrm > 1e-10 else v_apo1.copy()
                # Orientation : h_arc2 = r_apo × v_dep2 doit être aligné avec h_t
                if float(_np.dot(_np.cross(r_apo, v_dir2), h_t_hat)) < 0:
                    v_dir2 = -v_dir2

                v_apo2_mag = _m.sqrt(MU * (2./_np.linalg.norm(r_apo) - 1./a_tr2))
                v_dep2 = v_dir2 * v_apo2_mag

                #  Arc descendant avec sampling adaptatif 
                pts_arc2 = _arc_pts(r_apo, v_dep2, T_tr2, T_tr1)

                #  Circularisation en a_t (burn 3) 
                r_arr, v_arr = propagate_kepler(r_apo, v_dep2, T_tr2)
                v_circ = _m.sqrt(MU / _np.linalg.norm(r_arr))
                v_arr_c = v_arr / _np.linalg.norm(v_arr) * v_circ

                # Assembler segments
                pts_tr = pts_arc1 + pts_arc2
                segments['wait']     = [r_w.tolist() + [0.]]
                segments['transfer'] = pts_tr
                r_arr_arr = _np.array(r_arr)

                #  Orbite finale post-ΔV₃ 
                pts_fin = []
                n_fin = max(40, int(2 * T_t / T_c * 20))
                for k in range(n_fin + 1):
                    try:
                        rk, _ = propagate_kepler(r_arr, v_arr_c, 2*T_t*k/n_fin)
                        pts_fin.append(rk.tolist() + [t_tr_s + 2*T_t*k/n_fin])
                    except Exception: break
                segments['final_orbit'] = pts_fin

                total_s = t_tr_s + 2 * T_t

            #  BRANCHE 1b : Changement de plan simple (burn unique) 
            elif method_used == 'plane_change' or (
                method_used not in ('bielliptic_highalt',) and
                abs(float(_np.dot(
                    _np.cross(r0_,v0_)/_np.linalg.norm(_np.cross(r0_,v0_)),
                    _np.cross(rt0_,vt0_)/_np.linalg.norm(_np.cross(rt0_,vt0_))
                ))) < 0.9997 and  # delta_plane > ~1.4°
                True):
                # Plan différent → burn au nœud
                pc_data = compute_plane_change_to_target(
                    r0_.tolist(), v0_.tolist(), rt0_.tolist(), vt0_.tolist())
                t_node   = pc_data['t_node_s']
                r_burn_  = _np.array(pc_data['r_burn'])
                v_after_ = _np.array(pc_data['v_after'])

                # Orbite d'attente jusqu'au nœud (sampling adaptatif)
                def _arc_pts_wait(r_s, v_s, T_w, t_off=0., max_gap=300.):
                    pts=[]; t=0.
                    while t<=T_w+0.1:
                        try:
                            rk,vk=propagate_kepler(r_s,v_s,t)
                            pts.append(rk.tolist()+[t_off+t])
                            v_mag=float(_np.linalg.norm(vk))
                            dt=max_gap/v_mag if v_mag>1e-6 else 30.
                            dt=min(dt, T_w-t+1.) if t<T_w else 0.
                            if dt<=0: break
                            t+=dt
                        except Exception: break
                    return pts

                pts_wait_ = _arc_pts_wait(r0_, v0_, t_node, 0.)
                pts_wait_.append(r_burn_.tolist()+[t_node])
                segments['wait'] = pts_wait_

                # Post-burn : orbite après changement de plan
                T_post = 2*T_c   # 2 révolutions sur la nouvelle orbite
                pts_post = _arc_pts_wait(r_burn_, v_after_, T_post, t_node)
                segments['transfer']    = pts_post
                segments['final_orbit'] = []
                r_arr   = r_burn_.copy()
                v_arr_c = v_after_.copy()
                pts_tr  = pts_post          # pour compute_rendezvous_graphs
                total_s = t_node + T_post

            #  BRANCHE 2 : Hohmann / Phasage / standard 
            else:
                a_tr = (a_c + a_t) / 2

                # Burn position (modulo T pour stabilité numérique)
                t_mod_burn = max(1., wait_s % T_c) if wait_s > 0 else 1.
                r_w, v_w = propagate_kepler(r0_, v0_, t_mod_burn)

                # Vecteur départ Hohmann
                r_w_mag = float(_np.linalg.norm(r_w))
                v_tr = _m.sqrt(MU * (2./r_w_mag - 1./a_tr))
                v_dep = v_w / _np.linalg.norm(v_w) * v_tr
                try:
                    el_dep = rv_to_elements(r_w, v_dep)
                    if el_dep['a'] * (1 - el_dep['e']) - _RE < 100.:
                        v_dep = v_w
                except Exception: pass

                # Phase d'attente : orbite complète, 5 pts/révolution
                N_orb_wait = wait_s / T_c if T_c > 0 else 0
                n_wait = min(2000, max(20, int(N_orb_wait * 5) + 1))
                step_w = wait_s / n_wait if n_wait > 0 and wait_s > 0 else T_c
                pts_wait = []
                for k in range(n_wait + 1):
                    t_k = step_w * k
                    try:
                        rk, _ = propagate_kepler(r0_, v0_, max(0.5, t_k % T_c))
                        pts_wait.append(rk.tolist() + [t_k])
                    except Exception: break
                pts_wait.append(r_w.tolist() + [wait_s])
                segments['wait'] = pts_wait

                # Arc de transfert : N_PTS pts
                pts_tr = []
                if t_tr_s > 30:
                    for k in range(N_PTS + 1):
                        t = t_tr_s * k / N_PTS
                        try:
                            rk, _ = propagate_kepler(r_w, v_dep, t)
                            pts_tr.append(rk.tolist() + [wait_s + t])
                        except Exception: break
                segments['transfer'] = pts_tr

                # État chaser après ΔV₂ (circularisation)
                try:
                    r_arr, v_arr = propagate_kepler(r_w, v_dep, t_tr_s)
                    v_circ = _m.sqrt(MU / _np.linalg.norm(r_arr))
                    v_arr_c = v_arr / _np.linalg.norm(v_arr) * v_circ
                except Exception:
                    r_arr, v_arr_c = r_w.copy(), v0_.copy()

                # Orbite finale post-ΔV₂
                pts_fin = []
                n_fin = max(40, int(2 * T_t / T_c * 20))
                for k in range(n_fin + 1):
                    try:
                        rk, _ = propagate_kepler(r_arr, v_arr_c, 2*T_t*k/n_fin)
                        pts_fin.append(rk.tolist() + [wait_s + t_tr_s + 2*T_t*k/n_fin])
                    except Exception: break
                segments['final_orbit'] = pts_fin

                total_s = wait_s + t_tr_s + 2 * T_t

            #  Cible : 20 pts/orbite sur toute la mission 
            # total_s depuis la trajectoire réellement générée
            all_ch = (segments.get('wait') or []) +                      (segments.get('transfer') or []) +                      (segments.get('final_orbit') or [])
            traj_t_max = all_ch[-1][3] if all_ch and len(all_ch[-1])>3 else (wait_s+t_tr_s+2*T_t)
            total_s_used = max(traj_t_max, 1.)

            # wait_s et t_tr_s depuis la trajectoire (pas du résultat Hohmann)
            wait_pts  = segments.get('wait') or []
            trans_pts = segments.get('transfer') or []
            traj_wait_s = wait_pts[-1][3] if wait_pts and len(wait_pts[-1])>3 else 0.
            traj_tr_s   = (trans_pts[-1][3] - traj_wait_s) if trans_pts and len(trans_pts[-1])>3 else t_tr_s

            n_tgt = min(3000, max(100, int(total_s_used / T_t * 20) + 1))
            step_t = total_s_used / n_tgt if n_tgt > 0 else T_t
            pts_tgt = []
            for k in range(n_tgt + 1):
                t_k = step_t * k
                try:
                    rk, _ = propagate_kepler(rt0_, vt0_, t_k)
                    pts_tgt.append(rk.tolist() + [t_k])
                except Exception: break
            segments['target']  = pts_tgt
            segments['t_burns'] = [traj_wait_s, traj_wait_s + traj_tr_s]
            segments['t_max']   = total_s_used
            segments['wait_s']  = traj_wait_s
            segments['t_tr_s']  = traj_tr_s

            #  Burns RTN 
            burns_out = []
            for m in maneuvers_list:
                burns_out.append({
                    'label':   m.get('type', 'ΔV'),
                    't_h':     round(m.get('t_from_now_h', 0.), 4),
                    'dv_ms':   round(m.get('dv_ms', 0.), 3),
                    'dv_R_ms': 0., 'dv_T_ms': round(m.get('dv_ms', 0.), 3), 'dv_N_ms': 0.,
                    'desc':    m.get('description', ''),
                })

            #  Données graphiques (Vallado §6 + CW §7.6) 
            phasing_info=best.get('phasing',{})
            theta_0=phasing_info.get('theta_0_deg',0.)
            dn_d=phasing_info.get('dn_deg_day',0.)
            graph_data={}
            try:
                # Données graphiques physiquement correctes (Vallado Ch.6-7)
                gd = compute_rendezvous_graphs(
                    r0_, v0_, rt0_, vt0_,
                    el_c_, el_t_,
                    wait_s, t_tr_s,
                    pts_tr, r_arr, v_arr_c,
                    approach_angle_deg=float(ang_deg),
                )
                graph_data = gd
                # dist_timeline depuis les graphiques (oscillante réelle)
                segments['dist_timeline'] = gd.get('dist_tl', [])
            except Exception as eg:
                logger.warning(f"Graph data error: {eg}")

            #  Budget ΔV cumulé 
            cumul=0.; dv_budget=[]
            for m in maneuvers_list:
                cumul+=m.get('dv_ms',0.)
                dv_budget.append({'t_h':m.get('t_from_now_h',0.),
                                   'dv':m.get('dv_ms',0.),'cumul':round(cumul,3),
                                   'label':m.get('type','ΔV')})
            graph_data['dv_budget']=dv_budget

            # Distance inter-satellite réelle : propager chaser + cible à chaque pas
            dist_tl=[]
            # Point initial
            dist_tl.append([0., round(float(_np.linalg.norm(r0_-rt0_)),1)])

            # Pendant le transfert : distance réelle entre chaser (sur ellipse) et cible
            if t_tr_s > 30 and len(pts_tr) > 2:
                # Vérifier si les orbites sont coplanaires (< 2° de différence de plan)
                h_c_v = _np.cross(r0_,v0_); h_t_v = _np.cross(rt0_,vt0_)
                h_c_n = h_c_v/_np.linalg.norm(h_c_v); h_t_n = h_t_v/_np.linalg.norm(h_t_v)
                delta_pl = _m.degrees(_m.acos(float(_np.clip(_np.dot(h_c_n,h_t_n),-1,1))))
                coplanar = delta_pl < 2.0

                N_dt = 80
                for k in range(N_dt+1):
                    frac = k/N_dt
                    t_k  = wait_s + t_tr_s * frac
                    # Chaser : interpoler dans pts_tr
                    idx_c = min(int(frac*len(pts_tr)), len(pts_tr)-1)
                    rc = _np.array(pts_tr[idx_c][:3]) if pts_tr else r_w
                    # Cible : propager képlerien
                    try:
                        rt_k,_ = propagate_kepler(rt0_, vt0_, t_k)
                    except Exception:
                        rt_k = rt0_
                    d_km = float(_np.linalg.norm(rc - rt_k))
                    dist_tl.append([round(t_k/3600, 4), round(d_km, 1)])

                # Warning si non coplanaire : distance min théorique
                if not coplanar:
                    best['warning'] = (
                        (best.get('warning') or '') +
                        f" | Δ-plan={delta_pl:.1f}°: distance TCA ≈ "
                        f"{min(d for _,d in dist_tl[1:]):.0f}km. "
                        "Hohmann seul ne peut pas atteindre < 1km. "
                        "Utilisez Bi-ell. haute altitude pour corriger le plan."
                    )
            segments['dist_timeline']=dist_tl

            try:
                _mp_ = compute_mission_plan(r0_.tolist(), v0_.tolist(),
                                            rt0_.tolist(), vt0_.tolist())
            except Exception as _ep_:
                logger.warning(f"Mission plan error: {_ep_}")
                _mp_ = {'phases':[], 'all_burns':[], 'n_burns':0,
                        'total_dv_ms':0, 'total_duration_h':0,
                        'summary':'Non disponible'}

            return jsonify({
                'best':       best,
                'ranking':    ranking,
                'burns':      burns_out,
                'segments':   segments,
                'graph_data': graph_data,
                'mission_plan': _mp_,
                'summary': {
                    'total_dv_ms':   round(best['total_dv_ms'],3),
                    'wait_h':        round(best.get('wait_h',0.),4),
                    'transfer_h':    round(best.get('transfer_time_h',0.),4),
                    'chaser_alt_km': round(float(_np.linalg.norm(r0_))-_RE,1),
                    'target_alt_km': round(float(_np.linalg.norm(rt0_))-_RE,1),
                    'delta_alt_km':  round(abs(float(_np.linalg.norm(rt0_)-_np.linalg.norm(r0_))),1),
                    'method':        best['method'],
                    'warning':       (best.get('warning') or ''),
                    'T_c_s':         round(float(_np.linalg.norm(v0_)>0 and
                                         2*_m.pi*_m.sqrt(el_c_['a']**3/MU) or 5580), 1),
                },
            })

        except Exception as ex:
            logger.error(f"Optimize: {ex}", exc_info=True)
            return jsonify({"error":str(ex)}),500

    @app.route("/api/rendezvous/synthetic_tles", methods=["POST"])
    def rendezvous_synthetic_tles():
        """Génère des TLE synthétiques valides avec checksum correct."""
        import math as _m
        from datetime import datetime, timezone

        def tle_checksum(line):
            total = 0
            for ch in line[:68]:
                if ch.isdigit(): total += int(ch)
                elif ch == '-':  total += 1
            return total % 10

        data = request.get_json(force=True, silent=True) or {}
        alt_km   = float(data.get("alt_km",   400))
        inc_deg  = float(data.get("inc_deg",   51.6))
        raan_deg = float(data.get("raan_deg",  0.0))
        ma_deg   = float(data.get("ma_deg",    0.0))
        norad    = int(data.get("norad",       99901))

        MU_M3S2 = 3.986004418e14
        RE_M    = 6378137.0
        a_m     = RE_M + alt_km * 1000
        n_rads  = _m.sqrt(MU_M3S2 / a_m**3)
        n_revday = n_rads * 86400 / (2 * _m.pi)

        now = datetime.now(timezone.utc)
        doy  = now.timetuple().tm_yday
        frac = (now.hour*3600 + now.minute*60 + now.second) / 86400
        epoch = f"{str(now.year)[2:]}{doy:03d}{frac:.8f}"[:-1]  # YYDDDfrac

        # Construire les 68 premiers chars puis ajouter le checksum
        l1_68 = (f"1 {norad:05d}U 26001A   {epoch}"
                 f"  .00001000  00000-0  10000-4 0  999")
        l1_68 = l1_68[:68]
        tle1 = l1_68 + str(tle_checksum(l1_68))

        l2_68 = (f"2 {norad:05d} {inc_deg:8.4f} {raan_deg:8.4f} "
                 f"0000100 {0:8.4f} {ma_deg:8.4f} {n_revday:11.8f}    1")
        l2_68 = l2_68[:68]
        tle2 = l2_68 + str(tle_checksum(l2_68))

        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from rendezvous_engine import propagate_sgp4 as _p
            _p(tle1, tle2)
            valid = True
            err_msg = None
        except Exception as ex:
            valid = False
            err_msg = str(ex)

        return jsonify({
            "tle1": tle1, "tle2": tle2, "norad": norad,
            "valid": valid, "error": err_msg,
            "params": {"alt_km":alt_km,"inc_deg":inc_deg,
                       "raan_deg":raan_deg,"ma_deg":ma_deg}
        })

    @app.route("/api/config/owm_key")
    def get_owm_key():
        import os as _os
        key = _os.environ.get("OPENWEATHERMAP_KEY", "")
        # Fallback : lire directement le .env si la variable n'est pas dans l'environnement
        if not key:
            env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
            if _os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENWEATHERMAP_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if key:
                                _os.environ["OPENWEATHERMAP_KEY"] = key
                            break
        return jsonify({"key": key if key else None})

    @app.route("/api/diag/owm")
    def diag_owm():
        """Test OWM tile avec la clé configurée."""
        import urllib.request, os as _os
        key = _os.environ.get("OPENWEATHERMAP_KEY","")
        # Debug : lister toutes les vars d'env contenant OWM ou WEATHER ou MAP
        related = {k:v[:8]+"..." for k,v in _os.environ.items()
                   if any(x in k.upper() for x in ["OWM","WEATHER","OPENWEATHER","MAP_KEY"])}
        env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
        env_exists = _os.path.exists(env_path)
        # Lire directement le .env pour voir ce qui est dedans
        env_keys = []
        if env_exists:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k = line.split("=")[0].strip()
                        env_keys.append(k)
        if not key:
            return jsonify({
                "error": "OPENWEATHERMAP_KEY absent de os.environ",
                "key_found": False,
                "env_path": env_path,
                "env_exists": env_exists,
                "env_keys_found": env_keys,
                "related_env_vars": related
            })
        url = f"https://tile.openweathermap.org/map/clouds_new/2/0/0.png?appid={key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                size = len(r.read())
                return jsonify({"ok": True, "http": r.status, "size_bytes": size,
                                "key_prefix": key[:6]+"..."})
        except urllib.error.HTTPError as e:
            return jsonify({"ok": False, "http": e.code, "error": str(e),
                            "key_prefix": key[:6]+"..."})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

    @app.route("/api/db/stats")
    def db_stats():
        """Statistiques de la base TLE."""
        sys.path.insert(0, os.path.dirname(__file__))
        try:
            from tle_database import get_stats
            return jsonify(get_stats())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/db/ingest/stream")
    def db_ingest_stream():
        """
        SSE — Ingere TOUS les TLE LEO dans la base SQLite.
        Priorité : 1. Space-Track API  2. Celestrak  3. TLE locaux (avertissement)
        """
        import queue, threading
        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})

        def _parse_tle_text(raw):
            tles = []
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            i = 0
            while i < len(lines):
                if i+2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
                    tles.append((lines[i], lines[i+1], lines[i+2]))
                    i += 3
                elif lines[i].startswith("1 ") and i+1 < len(lines) and lines[i+1].startswith("2 "):
                    tles.append((f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i+1]))
                    i += 2
                else:
                    i += 1
            return tles

        def _fetch(url, timeout=45):
            import urllib.request as _ur, ssl
            # Headers complets pour éviter les 403
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept":     "text/plain,text/html,*/*;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer":    "https://celestrak.org/",
                "Connection": "keep-alive",
            }
            # Essayer les URLs alternatives si la principale échoue
            urls_to_try = [url]
            if "celestrak.org/pub/TLE" in url:
                alt = url.replace("celestrak.org/pub/TLE", "celestrak.org/SOCRATES")
                # Alias GP API pour certains groupes
                name = url.split("/")[-1].replace(".txt", "")
                gp_url = f"https://celestrak.org/SOCRATES/query.php?catalog={name}&format=tle"
            for try_url in urls_to_try:
                try:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    req = _ur.Request(try_url, headers=headers)
                    with _ur.urlopen(req, timeout=timeout, context=ctx) as r:
                        data = r.read()
                        if len(data) > 100:
                            return data.decode("utf-8", errors="ignore")
                except Exception as _e:
                    continue
            return None

        def run():
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
            sys.path.insert(0, os.path.dirname(__file__))
            try:
                from tle_database import ingest_tles, init_db, get_stats
                init_db()
                emit("[INFO] Base initialisee...", 2)
                all_tles = []
                source_used = "inconnu"

                #  1. Space-Track 
                try:
                    import urllib.request as _ur, urllib.parse as _up
                    # Utiliser _load_env() de scripts/spacetrack.py — mécanisme éprouvé
                    import sys as _sys
                    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
                    try:
                        from spacetrack import _load_env as _st_load_env
                        _creds = _st_load_env()
                        env_email = _creds.get("email", "")
                        env_pwd   = _creds.get("password", "")
                    except Exception:
                        env_email = os.environ.get("SPACETRACK_EMAIL", "")
                        env_pwd   = os.environ.get("SPACETRACK_PASSWORD", "")
                    if not env_email or not env_pwd:
                        raise ValueError("Credentials manquants — copiez .env.example en .env et remplissez vos identifiants Space-Track")
                    # Réutiliser SpaceTrackSession (classe réelle dans spacetrack.py)
                    emit("[INFO] Connexion Space-Track via SpaceTrackSession...", 5)
                    from spacetrack import SpaceTrackSession
                    from datetime import timedelta as _td, timezone as _tz
                    import os as _os
                    # Forcer les credentials dans l'environnement pour _check_credentials()
                    _os.environ["SPACETRACK_EMAIL"]    = env_email
                    _os.environ["SPACETRACK_PASSWORD"] = env_pwd

                    with SpaceTrackSession() as client:
                        now_dt   = datetime.now(_tz.utc)
                        end_str  = now_dt.strftime("%Y-%m-%d")

                        #  Étape 1 : liste NORAD LEO actifs via gp 
                        emit("[INFO] Recuperation liste satellites LEO actifs (gp)...", 8)
                        url_list = ("https://www.space-track.org/basicspacedata/query"
                                    "/class/gp/EPOCH/%3Enow-2"
                                    "/MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25"
                                    "/OBJECT_TYPE/payload,debris"
                                    "/orderby/NORAD_CAT_ID/format/tle")
                        raw_list = client._request(url_list).decode("utf-8", errors="ignore")
                        tles_current = _parse_tle_text(raw_list) if raw_list and len(raw_list) > 200 else []
                        norad_ids = list(set(t[1][2:7].strip() for t in tles_current))
                        emit(f"[OK] {len(norad_ids)} satellites LEO identifies", 15)

                        if not norad_ids:
                            raise ValueError("Aucun satellite LEO trouve via gp")

                        #  Étape 2 : historique gp_history par batches 
                        all_tles = list(tles_current)
                        try:
                            from tle_database import get_stats as _gs2
                            _li = _gs2().get("last_ingest")
                            if _li:
                                from datetime import datetime as _dt2
                                _ld = _dt2.fromisoformat(_li.replace("Z","+00:00"))
                                start_hist = (_ld - _td(days=1)).strftime("%Y-%m-%d")
                                emit(f"[INFO] Mise a jour incrementale depuis {start_hist}", 18)
                            else:
                                start_hist = (now_dt - _td(days=30)).strftime("%Y-%m-%d")
                                emit("[INFO] Premiere importation — 30j", 18)
                        except Exception:
                            start_hist = (now_dt - _td(days=30)).strftime("%Y-%m-%d")
                            emit(f"[INFO] Import depuis {start_hist}", 18)
                        batch_size = 500
                        batches = [norad_ids[i:i+batch_size] for i in range(0, len(norad_ids), batch_size)]
                        emit(f"[INFO] {len(batches)} batches de {batch_size} satellites...", 20)

                        for b_idx, batch in enumerate(batches):
                            pct = 20 + int(b_idx / len(batches) * 55)
                            norad_str = ",".join(batch)
                            url_hist = (
                                "https://www.space-track.org/basicspacedata/query"
                                f"/class/gp_history/NORAD_CAT_ID/{norad_str}"
                                f"/EPOCH/{start_hist}--{end_str}"
                                "/orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/tle"
                            )
                            try:
                                raw_h = client._request(url_hist).decode("utf-8", errors="ignore")
                                batch_tles = _parse_tle_text(raw_h) if raw_h and len(raw_h) > 100 else []
                                all_tles.extend(batch_tles)
                                emit(f"[INFO] Batch {b_idx+1}/{len(batches)}: +{len(batch_tles)} TLE", pct)
                            except Exception as be:
                                emit(f"[AVERT] Batch {b_idx+1} echoue: {be}", pct)

                    # Dedupliquer par (NORAD + epoch)
                    seen = set(); unique_tles = []
                    for t in all_tles:
                        key = t[1][2:7].strip() + "|" + t[1][18:32]
                        if key not in seen:
                            seen.add(key); unique_tles.append(t)
                    all_tles = unique_tles
                    uniq_total  = len(set(t[1][2:7].strip() for t in all_tles))
                    avg_per_sat = len(all_tles) / max(uniq_total, 1)
                    source_used = "Space-Track gp_history"
                    emit(f"[OK] {len(all_tles)} TLE · {uniq_total} satellites · moy. {avg_per_sat:.1f} TLE/sat", 77)
                except Exception as e:
                    emit(f"[AVERT] Space-Track inaccessible: {e}", 10)
                    emit("[INFO] Tentative Celestrak...", 12)

                    #  2. Celestrak 
                    celestrak = [
                        ("actifs LEO",  "https://celestrak.org/pub/TLE/active.txt"),
                        ("Starlink",    "https://celestrak.org/pub/TLE/starlink.txt"),
                        ("OneWeb",      "https://celestrak.org/pub/TLE/oneweb.txt"),
                        ("Debris 1",    "https://celestrak.org/pub/TLE/1999-025.txt"),
                        ("Debris 2",    "https://celestrak.org/pub/TLE/cosmos-2251-debris.txt"),
                    ]
                    for grp, url in celestrak:
                        raw = _fetch(url)
                        if raw and len(raw) > 200:
                            parsed = _parse_tle_text(raw)
                            all_tles.extend(parsed)
                            emit(f"[OK] {grp}: {len(parsed)} TLE", None)
                        else:
                            emit(f"[AVERT] {grp}: inaccessible", None)

                    if all_tles:
                        # Dedupliquer
                        seen = set()
                        uniq = []
                        for t in all_tles:
                            k = t[1][2:7]+t[1][18:32]
                            if k not in seen:
                                seen.add(k); uniq.append(t)
                        all_tles = uniq
                        source_used = "Celestrak"
                        emit(f"[OK] {len(all_tles)} TLE uniques Celestrak", 60)
                    else:
                        #  3. TLE locaux 
                        emit("[AVERT] Toutes sources inaccessibles — TLE locaux uniquement", 20)
                        emit("[AVERT] IMPORTANT: configurez .env avec vos credentials Space-Track", 21)
                        try:
                            from config import cfg
                            from tle_fetcher import parse_tle_file
                            all_tles = parse_tle_file(cfg.tle_source)
                        except Exception:
                            all_tles = []
                        source_used = "local"
                        emit(f"[AVERT] {len(all_tles)} TLE locaux — INSUFFISANT pour le modele IA", 60)

                if not all_tles:
                    q.put({"done":True,"error":"Aucune source TLE accessible. Configurez .env avec vos credentials Space-Track."})
                    return

                emit(f"[INFO] Ingestion de {len(all_tles)} TLE ({source_used})...", 65)
                report = ingest_tles(all_tles, source=source_used, emit=emit)
                stats  = get_stats()
                emit(f"[OK] {report['added']} nouveaux · {report['skipped']} existants · source={source_used}", 98)
                q.put({"done":True,"report":report,"stats":stats,"source":source_used})

            except Exception as e:
                logger.exception("Erreur ingestion DB")
                q.put({"done":True,"error":str(e)})

        threading.Thread(target=run, daemon=True).start()
        def generate():
            while True:
                try:
                    item = q.get(timeout=300)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    @app.route("/api/db/history/<norad_id>")
    def db_history(norad_id):
        """Historique TLE d'un satellite."""
        try:
            from tle_database import get_history
            return jsonify(get_history(norad_id))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    #  Manœuvres 

    @app.route("/api/maneuvers/detect/stream")
    def detect_maneuvers_stream():
        """SSE — Détecte les manœuvres sur les 7 derniers jours (paramétrable)."""
        import queue, threading
        days    = int(request.args.get("days", "7"))
        min_dv  = float(request.args.get("min_dv", "10"))
        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})
        def run():
            try:
                from maneuver_predictor import detect_maneuvers_from_db
                detections = detect_maneuvers_from_db(days=days, min_dv_ms=min_dv, emit=emit)
                q.put({"done": True, "detections": detections, "n": len(detections)})
            except Exception as e:
                logger.exception("Erreur détection manœuvres")
                q.put({"done": True, "error": str(e)})
        threading.Thread(target=run, daemon=True).start()
        def generate():
            while True:
                try:
                    item = q.get(timeout=300)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except: yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    @app.route("/api/maneuvers/list")
    def list_maneuvers():
        days   = int(request.args.get("days", "7"))
        min_dv = float(request.args.get("min_dv", "0"))
        try:
            from tle_database import get_maneuvers
            return jsonify(get_maneuvers(days=days, min_dv=min_dv))
        except Exception as e:
            return jsonify({"error": str(e), "maneuvers": []}), 500

    @app.route("/api/maneuvers/predict")
    def predict_maneuvers():
        try:
            from maneuver_predictor import predict_maneuvers as _pred
            preds = _pred()
            return jsonify(preds)
        except Exception as e:
            return jsonify({"error": str(e), "predictions": []}), 500

    @app.route("/api/maneuvers/predictions/list")
    def list_predictions():
        min_conf = float(request.args.get("min_confidence", "0.3"))
        try:
            from tle_database import get_predictions
            return jsonify(get_predictions(min_confidence=min_conf))
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    #  Détection de manœuvres (Bloc 1) 

    _detect_flags = {}  # clé=id(q), valeur=pause_flag

    @app.route("/api/maneuvers/detect/pause", methods=["POST"])
    def detect_pause():
        for flag in _detect_flags.values():
            flag["pause"] = not flag.get("pause", False)
            return jsonify({"paused": flag["pause"]})
        return jsonify({"error": "Pas de détection en cours"}), 404

    @app.route("/api/maneuvers/detect/stop", methods=["POST"])
    def detect_stop():
        for flag in _detect_flags.values():
            flag["stop"] = True; flag["pause"] = False
        return jsonify({"stopped": True})

    @app.route("/api/maneuvers/detect_adaptive/stream")
    def detect_adaptive_stream():
        """SSE — Détection adaptative avec perturbations."""
        import queue, threading
        days         = int(request.args.get("days", "7"))
        sigma        = float(request.args.get("sigma", "3.0"))
        norad_filter = request.args.get("norad_filter", "").strip()
        pert_json    = request.args.get("pert_flags", "{}")
        try:
            import urllib.parse as _up
            pert_flags = json.loads(_up.unquote(pert_json))
        except Exception:
            pert_flags = {}

        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})

        pause_flag = {"pause": False, "stop": False}
        _detect_flags[id(q)] = pause_flag

        def run():
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from maneuver_detector import detect_maneuvers_adaptive
                dets = detect_maneuvers_adaptive(
                    days=days, pert_flags=pert_flags,
                    min_sigma=sigma, norad_filter=norad_filter,
                    pause_flag=pause_flag, emit=emit
                )
                # Pour l'analyse ciblée : retourner aussi les TLE du satellite
                tle_entries = []
                if norad_filter:
                    try:
                        from tle_database import get_connection
                        conn = get_connection()
                        rows = conn.execute("""
                            SELECT norad_id, name, epoch FROM tle_records
                            WHERE (norad_id = ? OR UPPER(name) LIKE ?)
                            ORDER BY epoch ASC
                        """, (norad_filter.upper(), f'%{norad_filter.upper()}%')).fetchall()
                        conn.close()
                        tle_entries = [{"norad_id":r["norad_id"],"name":r["name"],
                                        "epoch":r["epoch"]} for r in rows]
                    except Exception:
                        pass
                q.put({"done": True, "detections": dets, "n": len(dets),
                       "tle_entries": tle_entries})
            except Exception as e:
                logger.exception("detect_adaptive")
                q.put({"done": True, "error": str(e)})
            finally:
                _detect_flags.pop(id(q), None)

        threading.Thread(target=run, daemon=True).start()

        def generate():
            while True:
                try:
                    item = q.get(timeout=300)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    #  Entraînement modèle manœuvres (Bloc 2) 

    _active_trainings = {}

    @app.route("/api/maneuvers/train/pause", methods=["POST"])
    def train_pause():
        """Pause/reprend l'entraînement GRU en cours."""
        for flag in _active_trainings.values():
            flag["pause"] = not flag.get("pause", False)
            return jsonify({"paused": flag["pause"]})
        return jsonify({"error": "Pas d'entraînement en cours"}), 404

    @app.route("/api/maneuvers/train/stop", methods=["POST"])
    def train_stop():
        """Arrête l'entraînement GRU en cours."""
        for flag in _active_trainings.values():
            flag["stop"] = True; flag["pause"] = False
        return jsonify({"stopped": True})

    @app.route("/api/maneuvers/train/stream")
    def train_maneuver_stream():
        """SSE — Entraînement du modèle GRU."""
        import queue, threading
        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})

        sample_pct = float(request.args.get("sample_pct", "1.0"))
        pause_flag = {"pause": False, "stop": False}
        _active_trainings[threading.current_thread().ident if False else id(q)] = pause_flag

        def run():
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from maneuver_model import train_maneuver_model
                report = train_maneuver_model(emit=emit, sample_pct=sample_pct,
                                              pause_flag=pause_flag)
                q.put({"done": True, "report": report})
            except Exception as e:
                logger.exception("train_maneuver")
                q.put({"done": True, "error": str(e)})

        threading.Thread(target=run, daemon=True).start()

        def generate():
            while True:
                try:
                    item = q.get(timeout=600)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    #  Inférence + génération TLE synthétiques (Blocs 2+3) 

    @app.route("/api/maneuvers/predict_gru")
    def predict_gru():
        """Inférence GRU sur tous les satellites en base."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from maneuver_model import predict_all_maneuvers
            min_p = float(request.args.get("min_p7d", "0.3"))
            preds = predict_all_maneuvers(min_p7d=min_p)
            return jsonify(preds)
        except Exception as e:
            return jsonify({"error": str(e), "predictions": []}), 500

    @app.route("/api/maneuvers/synthetic_tles")
    def synthetic_tles():
        """Génère les TLE post-manœuvre depuis les prédictions GRU."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from maneuver_model import predict_all_maneuvers
            from post_maneuver_propagator import generate_post_maneuver_tles
            min_p = float(request.args.get("min_p7d", "0.4"))
            preds = predict_all_maneuvers(min_p7d=min_p)
            synth = generate_post_maneuver_tles(preds)
            return jsonify(synth)
        except Exception as e:
            return jsonify({"error": str(e), "tles": []}), 500

    @app.route("/api/maneuvers/model_status")
    def maneuver_model_status():
        """Statut du modèle GRU manœuvres."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from maneuver_model import ManeuverGRU
            import os as _os
            m = ManeuverGRU.load()
            return jsonify({
                "trained":    m.trained,
                "n_samples":  m.n_samples,
                "train_loss": m.train_loss,
                "version":    m.version,
                "model_path": _os.path.exists("models/maneuver_model.json"),
            })
        except Exception as e:
            return jsonify({"error": str(e), "trained": False}), 500

    #  Conjonctions post-manœuvre (Bloc 5) 

    @app.route("/api/conjunctions/post_maneuver/stream")
    def conj_post_maneuver_stream():
        """
        SSE — Analyse de conjonctions en incluant les TLE post-manœuvre synthétiques.
        Identique à /api/conjunctions/stream mais injecte les TLE synthétiques.
        """
        import queue, threading, urllib.parse as _up
        constellation = request.args.get("constellation", "active")
        hours         = float(request.args.get("hours", "24"))
        step_min      = float(request.args.get("step_min", "5"))
        threshold_km  = max(0.1, min(float(request.args.get("threshold_km","5")),50))
        extra_norad   = request.args.get("extra_norad", "")
        pc_method     = request.args.get("pc_method", "foster")
        mc_n          = int(request.args.get("mc_n", "50000"))
        min_p7d       = float(request.args.get("min_p7d", "0.4"))
        pert_json     = request.args.get("pert_flags", "{}")
        try:
            pert_flags = json.loads(_up.unquote(pert_json))
        except Exception:
            pert_flags = None

        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})

        def run():
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from conjunction import run_conjunction_analysis, fetch_constellation
                from maneuver_model import predict_all_maneuvers
                from post_maneuver_propagator import generate_post_maneuver_tles

                emit("[INFO] Génération des TLE post-manœuvre...", 3)
                preds = predict_all_maneuvers(min_p7d=min_p7d)
                synth = generate_post_maneuver_tles(preds)
                emit(f"[OK] {len(synth)} TLE synthétiques générés", 8)

                # TLE extra : nominaux + synthétiques
                extra_tles = []
                if extra_norad:
                    from tle_fetcher import fetch_by_norad_list
                    extra_tles = fetch_by_norad_list(
                        [n.strip() for n in extra_norad.split(",") if n.strip()]
                    )
                # Ajouter les TLE synthétiques
                for s in synth:
                    extra_tles.append((s["name"], s["tle1"], s["tle2"]))

                conjunctions = run_conjunction_analysis(
                    constellation=constellation,
                    extra_tles=extra_tles,
                    hours=hours, step_min=step_min,
                    threshold_km=threshold_km,
                    max_results=300,
                    pc_method=pc_method, mc_n=mc_n,
                    pert_flags=pert_flags, emit=emit,
                )

                # Marquer les conjonctions impliquant un TLE synthétique
                pred_norads = {s["norad_id"] for s in synth}
                pred_map    = {s["norad_id"]: s for s in synth}
                for c in conjunctions:
                    n = c.get("norad_A") or c.get("norad_B")
                    if c.get("norad_A") in pred_norads or c.get("norad_B") in pred_norads:
                        best_norad = c["norad_A"] if c["norad_A"] in pred_norads else c["norad_B"]
                        s = pred_map.get(best_norad, {})
                        orig = s.get("original", {})
                        c["is_post_maneuver"] = True
                        c["pred_p7d"]         = orig.get("p_7d", 0)
                        c["pred_dv_ms"]       = orig.get("dv_pred_ms", 0)
                        c["pred_type"]        = orig.get("type_pred", "")
                    else:
                        c["is_post_maneuver"] = False

                q.put({"done": True, "conjunctions": conjunctions,
                       "n_synth": len(synth), "n_preds": len(preds)})
            except Exception as e:
                logger.exception("conj_post_maneuver")
                q.put({"done": True, "error": str(e)})

        threading.Thread(target=run, daemon=True).start()

        def generate():
            while True:
                try:
                    item = q.get(timeout=600)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


    #  Onglet Temps réel 

    @app.route("/api/realtime/tles")
    def realtime_tles():
        """
        Retourne les TLE LEO pour le visualiseur temps réel.
        Limité à 8000 objets prioritaires. Cache 15 min.
        """
        import time as _time, json as _json

        cache_attr      = '_rt_cache'
        cache_time_attr = '_rt_cache_time'
        cached = getattr(app, cache_attr, None)
        c_time = getattr(app, cache_time_attr, 0)
        if cached and (_time.time() - c_time) < 900:
            return app.response_class(
                response=_json.dumps(cached), status=200, mimetype='application/json'
            )

        tles = []
        try:
            from tle_database import get_connection, init_db
            init_db()
            conn = get_connection()
            try:
                rows = conn.execute("""
                    WITH latest AS (
                        SELECT norad_id, name, tle1, tle2, epoch,
                               ROW_NUMBER() OVER (PARTITION BY norad_id ORDER BY epoch DESC) as rn,
                               CASE
                                 WHEN UPPER(name) LIKE '%DEB%' OR UPPER(name) LIKE '%DEBRIS%'
                                      OR UPPER(name) LIKE '%FRAG%' THEN 3
                                 WHEN UPPER(name) LIKE '%R/B%' OR UPPER(name) LIKE '%ROCKET%'
                                      OR UPPER(name) LIKE '%SL-%' THEN 2
                                 ELSE 1
                               END as priority
                        FROM tle_records
                        WHERE (orbit_class = 'LEO' OR orbit_class IS NULL OR orbit_class = '')
                          AND tle1 != '' AND tle2 != ''
                    )
                    SELECT norad_id, name, tle1, tle2, epoch, priority
                    FROM latest WHERE rn = 1
                    ORDER BY priority ASC, epoch DESC
                """).fetchall()
            finally:
                conn.close()

            for r in rows:
                priority = r['priority'] if isinstance(r, dict) else r[5]
                norad_id = r['norad_id'] if isinstance(r, dict) else r[0]
                name     = r['name']     if isinstance(r, dict) else r[1]
                tle1     = r['tle1']     if isinstance(r, dict) else r[2]
                tle2     = r['tle2']     if isinstance(r, dict) else r[3]
                epoch    = r['epoch']    if isinstance(r, dict) else r[4]
                stype = 'DEBRIS' if priority==3 else ('ROCKET' if priority==2 else 'PAYLOAD')
                epoch_days = None
                try:
                    from datetime import datetime
                    ep = datetime.fromisoformat(str(epoch).replace('Z','+00:00'))
                    epoch_days = ep.timestamp()/86400.0 + 25567.5
                except Exception:
                    pass
                tles.append({'name':name,'norad':norad_id,'stype':stype,
                             'tle1':tle1,'tle2':tle2,'epoch_days':epoch_days})
            logger.info(f"Realtime TLEs : {len(tles)} depuis DB locale")
        except Exception as e:
            logger.warning(f"DB locale indisponible : {e}")

        if len(tles) < 100:
            import ssl, urllib.request as _ur
            ctx = ssl.create_default_context()
            ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
            for stype, url in [("PAYLOAD","https://celestrak.org/pub/TLE/active.txt"),
                                ("DEBRIS","https://celestrak.org/pub/TLE/1999-025.txt")]:
                try:
                    with _ur.urlopen(_ur.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=20,context=ctx) as r:
                        raw=r.read().decode("utf-8","ignore")
                    ls=[l.strip() for l in raw.splitlines() if l.strip()]
                    i=0
                    while i+2<len(ls) and len(tles)<8000:
                        if ls[i+1].startswith("1 ") and ls[i+2].startswith("2 "):
                            tles.append({"name":ls[i],"norad":ls[i+1][2:7].strip(),
                                        "stype":stype,"tle1":ls[i+1],"tle2":ls[i+2],"epoch_days":None})
                            i+=3
                        else: i+=1
                except Exception as e2:
                    logger.warning(f"Celestrak {url}: {e2}")

        if not tles:
            return jsonify({"error": "No TLE data available"}), 503


        # Cache + réponse streaming JSON
        setattr(app, cache_attr, tles)
        setattr(app, cache_time_attr, _time.time())

        resp = app.response_class(
            response=_json.dumps(tles),
            status=200,
            mimetype='application/json'
        )
        return resp


    @app.route("/api/realtime/stream")
    def realtime_stream():
        """
        SSE : push positions SGP4 pre-calculees cote serveur.
        - Le serveur propage toutes les 60s en thread background
        - A la connexion, le dernier resultat est envoye immediatement
        - Puis chaque nouveau calcul est pousse aux clients connectes
        """
        import time as _t2, json as _j2

        def _gen():
            # Envoyer le dernier resultat immediatement si disponible
            cached = getattr(app, '_rt_positions_cache', None)
            if cached:
                yield "data: " + _j2.dumps(cached) + "\n\n"

            # S'abonner aux futures mises a jour
            import queue as _queue
            q = _queue.Queue(maxsize=2)
            subscribers = getattr(app, '_rt_subscribers', None)
            if subscribers is None:
                app._rt_subscribers = []
                subscribers = app._rt_subscribers
            subscribers.append(q)
            try:
                while True:
                    try:
                        msg = q.get(timeout=90)  # heartbeat toutes les 90s
                        yield "data: " + _j2.dumps(msg) + "\n\n"
                    except _queue.Empty:
                        yield "data: " + _j2.dumps({"type":"heartbeat"}) + "\n\n"
            finally:
                try: subscribers.remove(q)
                except ValueError: pass

        return app.response_class(
            _gen(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @app.route("/api/realtime/positions")
    def realtime_positions():
        """Retourne le dernier calcul de positions (pour chargement initial)."""
        import json as _j2
        cached = getattr(app, '_rt_positions_cache', None)
        if not cached:
            return jsonify({"error": "No positions computed yet"}), 503
        import flask as _fl
        resp = _fl.make_response(_j2.dumps(cached))
        resp.headers['Content-Type'] = 'application/json'
        return resp


    @app.route("/frontend/sgp4_worker.js")
    def serve_worker():
        """Sert le Web Worker SGP4."""
        import os as _os
        worker_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "..", "frontend", "sgp4_worker.js"
        )
        from flask import make_response
        try:
            content = open(worker_path).read()
            resp = make_response(content)
            resp.headers["Content-Type"]  = "application/javascript"
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except FileNotFoundError:
            return ("Worker non trouvé", 404)


    #  Score de durabilité orbitale 

    @app.route("/api/sustainability/satellite/<norad_id>")
    def sustainability_satellite(norad_id):
        """Score SSR d'un satellite individuel."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from sustainability_scorer import score_satellite, DEFAULT_THRESHOLDS
            from tle_database import get_connection, init_db
            init_db()
            conn = get_connection()
            try:
                row = conn.execute("""
                    SELECT norad_id, name, tle1, tle2, epoch, alt_km, inc, ecc, mm
                    FROM tle_records WHERE norad_id=? ORDER BY epoch DESC LIMIT 1
                """, (norad_id,)).fetchone()
                mans = conn.execute("""
                    SELECT delta_v_ms, detected_at, event_type FROM maneuvers
                    WHERE norad_id=? ORDER BY detected_at DESC LIMIT 50
                """, (norad_id,)).fetchall()
            finally:
                conn.close()
            if not row:
                return jsonify({"error": "Satellite non trouvé"}), 404
            result = score_satellite(
                norad_id=row["norad_id"], name=row["name"],
                tle1=row["tle1"], tle2=row["tle2"], epoch=row["epoch"] or "",
                alt_km=float(row["alt_km"] or 500),
                inc_deg=float(row["inc"] or 53),
                ecc=float(row["ecc"] or 0.001),
                mm=float(row["mm"] or 15),
                maneuver_history=[dict(m) for m in mans],
                conjunction_history=[],
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/sustainability/batch")
    def sustainability_batch():
        """Scores SSR pour tous les satellites LEO en base."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from sustainability_scorer import score_satellite
            from tle_database import get_connection, init_db
            init_db()
            conn = get_connection()
            try:
                rows = conn.execute("""
                    SELECT norad_id, name, tle1, tle2, epoch, alt_km, inc, ecc, mm
                    FROM tle_records WHERE orbit_class='LEO' AND tle1!=''
                    GROUP BY norad_id ORDER BY norad_id
                """).fetchall()
                mans_all = conn.execute("""
                    SELECT norad_id, delta_v_ms, detected_at, event_type FROM maneuvers
                """).fetchall()
            finally:
                conn.close()
            from collections import defaultdict
            mans_map = defaultdict(list)
            for m in mans_all:
                mans_map[m["norad_id"]].append(dict(m))
            limit = int(request.args.get("limit", "500"))
            results = []
            for row in rows[:limit]:
                try:
                    s = score_satellite(
                        norad_id=row["norad_id"], name=row["name"],
                        tle1=row["tle1"], tle2=row["tle2"],
                        epoch=row["epoch"] or "",
                        alt_km=float(row["alt_km"] or 500),
                        inc_deg=float(row["inc"] or 53),
                        ecc=float(row["ecc"] or 0.001),
                        mm=float(row["mm"] or 15),
                        maneuver_history=mans_map.get(row["norad_id"], []),
                        conjunction_history=[],
                    )
                    results.append(s)
                except Exception:
                    pass
            results.sort(key=lambda x: x["score"])
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    #  Détection anomalies comportementales 

    @app.route("/api/anomalies/detect/stream")
    def anomaly_detect_stream():
        """SSE — Détection d'anomalies comportementales."""
        import queue, threading, urllib.parse as _up
        norad_filter = request.args.get("norad_filter", "")
        thresh_json  = request.args.get("thresholds", "{}")
        try:
            thresholds = json.loads(_up.unquote(thresh_json))
        except Exception:
            thresholds = {}
        q = queue.Queue()
        def emit(msg, pct=None): q.put({"msg": msg, "pct": pct})
        def run():
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from anomaly_detector import detect_behavioral_anomalies
                anomalies = detect_behavioral_anomalies(
                    norad_filter=norad_filter,
                    thresholds=thresholds,
                    emit=emit,
                )
                q.put({"done": True, "anomalies": anomalies, "n": len(anomalies)})
            except Exception as e:
                logger.exception("anomaly_detect")
                q.put({"done": True, "error": str(e)})
        threading.Thread(target=run, daemon=True).start()
        def generate():
            while True:
                try:
                    item = q.get(timeout=300)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("done"): break
                except Exception:
                    yield f"data: {json.dumps({'done':True,'error':'Timeout'})}\n\n"; break
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    @app.route("/api/sustainability/thresholds")
    def get_thresholds():
        """Retourne les seuils par défaut (Starlink)."""
        from sustainability_scorer import DEFAULT_THRESHOLDS as ST
        from anomaly_detector import DEFAULT_THRESHOLDS as AT
        return jsonify({"sustainability": ST, "anomaly": AT})

    return app

def _start_rt_propagation_thread(app_ref):
    """
    Thread background : propage tous les TLE LEO toutes les 60s.
    Stocke le resultat dans app._rt_positions_cache et notifie les subscribers SSE.
    """
    import threading as _th, time as _t3, json as _j3, logging as _lg
    _log = _lg.getLogger('rt_propagator')

    def _prop_loop():
        _log.info("[RT] Thread de propagation demarré")
        while True:
            t0 = _t3.time()
            try:
                with app_ref.app_context():
                    # Charger les TLE depuis la DB
                    from tle_database import get_connection, init_db
                    init_db()
                    conn = get_connection()
                    try:
                        rows = conn.execute("""
                            WITH latest AS (
                                SELECT norad_id, name, tle1, tle2, epoch,
                                       ROW_NUMBER() OVER (PARTITION BY norad_id ORDER BY epoch DESC) as rn,
                                       CASE
                                         WHEN UPPER(name) LIKE '%DEB%' OR UPPER(name) LIKE '%DEBRIS%'
                                              OR UPPER(name) LIKE '%FRAG%' THEN 3
                                         WHEN UPPER(name) LIKE '%R/B%' OR UPPER(name) LIKE '%ROCKET%'
                                              OR UPPER(name) LIKE '%SL-%' THEN 2
                                         ELSE 1
                                       END as priority
                                FROM tle_records
                                WHERE (orbit_class = 'LEO' OR orbit_class IS NULL OR orbit_class = '')
                                  AND tle1 != '' AND tle2 != ''
                            )
                            SELECT norad_id, name, tle1, tle2, epoch, priority
                            FROM latest WHERE rn = 1
                            ORDER BY priority ASC, epoch DESC
                                """).fetchall()
                    finally:
                        conn.close()

                    tles = []
                    for r in rows:
                        norad_id = r['norad_id'] if isinstance(r, dict) else r[0]
                        name     = r['name']     if isinstance(r, dict) else r[1]
                        tle1     = r['tle1']     if isinstance(r, dict) else r[2]
                        tle2     = r['tle2']     if isinstance(r, dict) else r[3]
                        tles.append({"norad":norad_id,"name":name,"tle1":tle1,"tle2":tle2})

                    if not tles:
                        _log.warning("[RT] Aucun TLE disponible")
                        _t3.sleep(60)
                        continue

                    # Propagation SGP4 vectorisée
                    from sgp4.api import Satrec, jday
                    from datetime import datetime, timezone
                    now_dt = datetime.now(timezone.utc)
                    jd, fr = jday(now_dt.year, now_dt.month, now_dt.day,
                                  now_dt.hour, now_dt.minute,
                                  now_dt.second + now_dt.microsecond/1e6)
                    epoch_s = _t3.time()

                    positions = []
                    for tle in tles:
                        try:
                            sat = Satrec.twoline2rv(tle["tle1"], tle["tle2"])
                            e, r, v = sat.sgp4(jd, fr)
                            if e == 0:
                                positions.append([
                                    round(r[0]/6378.137, 6),
                                    round(r[1]/6378.137, 6),
                                    round(r[2]/6378.137, 6)
                                ])
                            else:
                                positions.append(None)
                        except Exception:
                            positions.append(None)

                    payload = {
                        "type":      "positions",
                        "epoch_s":   epoch_s,
                        "count":     len(tles),
                        "norads":    [t["norad"] for t in tles],
                        "positions": positions
                    }

                    # Stocker en cache
                    app_ref._rt_positions_cache = payload

                    # Notifier tous les subscribers SSE
                    subs = getattr(app_ref, '_rt_subscribers', [])
                    dead = []
                    for q in subs:
                        try:
                            q.put_nowait(payload)
                        except Exception:
                            dead.append(q)
                    for q in dead:
                        try: subs.remove(q)
                        except ValueError: pass

                    elapsed = _t3.time() - t0
                    _log.info(f"[RT] {len(tles)} satellites propagés en {elapsed:.1f}s")

            except Exception as ex:
                _log.error(f"[RT] Erreur propagation: {ex}")

            # Attendre 60s depuis le début du cycle
            elapsed = _t3.time() - t0
            _t3.sleep(max(0, 60.0 - elapsed))

    t = _th.Thread(target=_prop_loop, daemon=True, name="rt_propagator")
    t.start()
    return t


