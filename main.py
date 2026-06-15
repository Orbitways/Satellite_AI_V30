"""
main.py — Point d'entrée principal du pipeline satellite_ai_v2.

Modes :
    train      Entraînement complet + évaluation automatique
    evaluate   Évaluation + rapport HTML
    export     Export ONNX + quantification INT8
    finetune   Fine-tuning incrémental (EWC + replay buffer)
    predict    Démo d'inférence (PyTorch ou ONNX Runtime)
    info       Affiche la config et les stats du dataset
    benchmark  Mesure la latence d'inférence (CPU)
"""

import os
import sys
import logging
import argparse
import pickle
import numpy as np
from datetime import datetime, timezone

# Ajouter src/ au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import cfg
from tle_fetcher import parse_tle_file
from sgp4_utils import generate_trajectory
from dataset import build_dataset, save_scalers, load_scalers
from model import build_model, count_parameters
from train import train
from evaluate import evaluate
from export import export_onnx, quantize_int8, benchmark_onnx
from predict import OrbitalPredictor
from continual import EWC, ReplayBuffer, finetune
from torch.utils.data import TensorDataset, DataLoader
import torch


# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs(cfg.log_dir,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(cfg.log_dir, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _build_model() -> torch.nn.Module:
    """Instancie le modèle selon cfg.model_type avec les bons kwargs."""
    common = dict(input_size=cfg.input_size, residual_size=cfg.residual_size,
                  dropout=cfg.dropout)
    if cfg.model_type == "tcn":
        return build_model("tcn", **common,
                           channels=cfg.tcn_channels,
                           kernel_size=cfg.tcn_kernel_size)
    return build_model("gru", **common,
                       hidden_size=cfg.gru_hidden,
                       num_layers=cfg.gru_layers)


def load_satellite_data(tle_path: str):
    tles = parse_tle_file(tle_path)
    satellite_data = []
    for sat_id, (name, tle1, tle2) in enumerate(tles):
        logger.info(f"Génération trajectoire : {name} (sat_id={sat_id})")
        try:
            states, residuals = generate_trajectory(
                tle1, tle2, sat_id=sat_id,
                n_points=cfg.points_per_sat,
                step_minutes=cfg.step_minutes,
            )
            satellite_data.append((states, residuals))
        except Exception as e:
            logger.warning(f"Satellite {name} ignoré : {e}")
    return satellite_data, tles


# ─── Mode : info ──────────────────────────────────────────────────────────────

def mode_info():
    print(cfg.summary())

    tles = parse_tle_file(cfg.tle_source)
    print(f"\n  Satellites chargés ({len(tles)}) :")
    for i, (name, l1, _) in enumerate(tles):
        from tle_fetcher import get_tle_epoch
        epoch = get_tle_epoch(l1)
        epoch_str = epoch.strftime("%Y-%m-%d %H:%M UTC") if epoch else "?"
        print(f"    [{i}] {name:<20}  époque TLE : {epoch_str}")

    ckpt = cfg.checkpoint_path
    if os.path.exists(ckpt):
        sz = os.path.getsize(ckpt) / 1024
        print(f"\n  Checkpoint : {ckpt}  ({sz:.1f} KB)")
        model = _build_model()
        print(f"  Paramètres : {count_parameters(model):,}")
    else:
        print(f"\n  Aucun checkpoint trouvé — lancer --mode train")

    csv_path = os.path.join(cfg.log_dir, "training.csv")
    if os.path.exists(csv_path):
        import csv
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            best = min(rows, key=lambda r: float(r["val_loss"]))
            print(f"  Meilleur entraînement : epoch={best['epoch']}  val_loss={float(best['val_loss']):.5f}  rmse={float(best['rmse_val'])*1000:.1f} m")


# ─── Mode : entraînement ──────────────────────────────────────────────────────

def mode_train():
    logger.info("=" * 60)
    logger.info("MODE : ENTRAÎNEMENT")
    logger.info("=" * 60)
    logger.info("\n" + cfg.summary())

    satellite_data, tles = load_satellite_data(cfg.tle_source)
    if not satellite_data:
        raise RuntimeError("Aucune donnée satellite disponible.")

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test), scalers = build_dataset(
        satellite_data,
        window=cfg.window_size,
        horizon=cfg.horizon,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    save_scalers(scalers, cfg.scaler_path)

    model = _build_model()
    logger.info(f"Modèle : {cfg.model_type.upper()} | {count_parameters(model):,} paramètres")

    history = train(
        model,
        train_data=(X_tr, y_tr),
        val_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        lr_patience=cfg.lr_patience,
        grad_clip=cfg.grad_clip,
        loss_type=cfg.loss,
        huber_delta=cfg.huber_delta,
        checkpoint_path=cfg.checkpoint_path,
        log_path=os.path.join(cfg.log_dir, "training.csv"),
    )

    # Sauvegarder le test set
    np.save(os.path.join(cfg.model_dir, "X_test.npy"), X_test)
    np.save(os.path.join(cfg.model_dir, "y_test.npy"), y_test)
    logger.info(f"Entraînement terminé — best val_loss={history['best_val_loss']:.4f}")

    # Évaluation automatique post-entraînement
    logger.info("─" * 60)
    logger.info("Évaluation automatique post-entraînement...")
    report_path = os.path.join(cfg.model_dir, "report.html")
    evaluate(model, (X_test, y_test), scaler_residual=scalers,
             sat_id=0, report_path=report_path)

    return model, scalers, X_test, y_test


# ─── Mode : évaluation ────────────────────────────────────────────────────────

def mode_evaluate():
    logger.info("=" * 60)
    logger.info("MODE : ÉVALUATION")
    logger.info("=" * 60)

    model = _build_model()
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
    scalers = load_scalers(cfg.scaler_path)

    X_test_path = os.path.join(cfg.model_dir, "X_test.npy")
    if not os.path.exists(X_test_path):
        logger.error("Test set introuvable — lancer --mode train d'abord.")
        return

    X_test = np.load(X_test_path)
    y_test = np.load(os.path.join(cfg.model_dir, "y_test.npy"))

    return evaluate(
        model, (X_test, y_test), scaler_residual=scalers,
        sat_id=0, report_path=os.path.join(cfg.model_dir, "report.html"),
    )


# ─── Mode : benchmark ─────────────────────────────────────────────────────────

def mode_benchmark():
    """Mesure la latence d'inférence sur CPU (100 runs, médiane + P99)."""
    import time
    logger.info("=" * 60)
    logger.info("MODE : BENCHMARK LATENCE CPU")
    logger.info("=" * 60)

    model = _build_model()
    if os.path.exists(cfg.checkpoint_path):
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, cfg.window_size, cfg.input_size, dtype=torch.float32)
    N = 200

    # Warmup
    with torch.no_grad():
        for _ in range(20): model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(N):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    logger.info(f"Modèle       : {cfg.model_type.upper()}  {count_parameters(model):,} params")
    logger.info(f"Latence CPU  : médiane={np.median(lat):.3f} ms | P99={np.percentile(lat,99):.3f} ms | max={lat.max():.3f} ms")
    logger.info(f"Débit        : {1000/np.median(lat):.0f} inférences/s")

    # Contexte embarqué
    freq_hz = 1000 / np.median(lat)
    logger.info(f"Fréquence IA : {freq_hz:.0f} Hz  (1 correction toutes {np.median(lat):.1f} ms)")
    if freq_hz > 10:
        logger.info(": Compatible temps-réel LEO (10 Hz requis)")
    else:
        logger.warning(": Trop lent pour temps-réel — envisager quantification INT8")

    # Benchmark ONNX si disponible
    if os.path.exists(cfg.onnx_path):
        benchmark_onnx(cfg.onnx_path, window_size=cfg.window_size, input_size=cfg.input_size)


# ─── Mode : export ────────────────────────────────────────────────────────────

def mode_export():
    logger.info("=" * 60)
    logger.info("MODE : EXPORT EMBARQUÉ (ONNX + INT8)")
    logger.info("=" * 60)

    model = _build_model()
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))

    export_onnx(model, cfg.onnx_path,
                window_size=cfg.window_size, input_size=cfg.input_size)
    quantize_int8(model, cfg.onnx_int8_path,
                  window_size=cfg.window_size, input_size=cfg.input_size)
    benchmark_onnx(cfg.onnx_path,
                   window_size=cfg.window_size, input_size=cfg.input_size)


# ─── Mode : fine-tuning ───────────────────────────────────────────────────────

def mode_finetune(new_tle_path: str):
    logger.info("=" * 60)
    logger.info(f"MODE : FINE-TUNING depuis {new_tle_path}")
    logger.info("=" * 60)

    model = _build_model()
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
    scalers = load_scalers(cfg.scaler_path)

    satellite_data, _ = load_satellite_data(new_tle_path)
    (X_new, y_new), (X_val, y_val), _, _ = build_dataset(
        satellite_data, window=cfg.window_size, train_ratio=0.70, val_ratio=0.15
    )

    # EWC depuis le test set historique
    ewc = None
    X_old_path = os.path.join(cfg.model_dir, "X_test.npy")
    if os.path.exists(X_old_path):
        X_old = np.load(X_old_path)
        y_old = np.load(os.path.join(cfg.model_dir, "y_test.npy"))
        n = min(500, len(X_old))
        old_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_old[:n], dtype=torch.float32),
                torch.tensor(y_old[:n], dtype=torch.float32),
            ),
            batch_size=32,
        )
        ewc = EWC(model, old_loader)
        logger.info(f"EWC activé — FIM calculée sur {n} échantillons historiques")

    replay = ReplayBuffer(max_ratio=cfg.replay_ratio)

    finetune(
        model,
        new_train_data=(X_new, y_new),
        new_val_data=(X_val, y_val),
        ewc=ewc,
        replay_buffer=replay,
        ewc_lambda=cfg.ewc_lambda,
        epochs=cfg.finetune_epochs,
        lr=cfg.finetune_lr,
        checkpoint_path=cfg.checkpoint_path,
    )

    # Évaluation post fine-tuning
    logger.info("Évaluation post fine-tuning...")
    X_test = np.load(X_old_path)
    y_test = np.load(os.path.join(cfg.model_dir, "y_test.npy"))
    evaluate(model, (X_test, y_test), scaler_residual=scalers,
             sat_id=0, report_path=os.path.join(cfg.model_dir, "report_finetune.html"))


# ─── Mode : prédiction ────────────────────────────────────────────────────────

def mode_predict(use_onnx: bool = False):
    logger.info("=" * 60)
    logger.info(f"MODE : PRÉDICTION ({'ONNX Runtime' if use_onnx else 'PyTorch'})")
    logger.info("=" * 60)

    if use_onnx:
        predictor = OrbitalPredictor.from_onnx(cfg.onnx_path, cfg.scaler_path, sat_id=0)
    else:
        model = _build_model()
        predictor = OrbitalPredictor.from_pytorch(model, cfg.checkpoint_path, cfg.scaler_path)

    from datetime import timedelta
    from sgp4_utils import tle_to_state

    tles = parse_tle_file(cfg.tle_source)
    name, tle1, tle2 = tles[0]

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_states = []
    for i in range(cfg.window_size):
        dt = now + timedelta(minutes=i * cfg.step_minutes)
        state = tle_to_state(tle1, tle2, dt)
        if state is not None:
            window_states.append(np.append(state, [i * cfg.step_minutes * 60, 0.0]))

    if len(window_states) < cfg.window_size:
        logger.error("Fenêtre insuffisante pour la prédiction.")
        return

    window = np.array(window_states[-cfg.window_size:], dtype=np.float64)
    t_next = now + timedelta(minutes=cfg.window_size * cfg.step_minutes)
    sgp4_pos = tle_to_state(tle1, tle2, t_next)[:3]

    delta     = predictor.predict_residual(window)
    corrected = sgp4_pos + delta

    logger.info(f"Satellite      : {name}")
    logger.info(f"Horizon        : +{cfg.window_size * cfg.step_minutes} min")
    logger.info(f"SGP4 position  : {sgp4_pos.round(3)} km (ECI)")
    logger.info(f"Correction IA  : Δ = {(delta*1000).round(2)} m")
    logger.info(f"Position finale: {corrected.round(3)} km (ECI)")
    logger.info(f"Norme Δ        : {np.linalg.norm(delta)*1000:.1f} m")


# ─── Mode : ingest ────────────────────────────────────────────────────────────

def mode_ingest(tle_path: str, do_finetune: bool = False):
    from ingest import ingest_new_tles, ManeuverDetector

    logger.info("=" * 60)
    logger.info(f"MODE : INGESTION  {tle_path}")
    logger.info("=" * 60)

    detector = ManeuverDetector(
        threshold_maneuver_m=500.0,
        threshold_anomaly_m=200.0,
        threshold_inoperable_m=5000.0,
        min_consecutive=3,
    )

    report = ingest_new_tles(
        tle_path=tle_path,
        model_dir=cfg.model_dir,
        data_dir=os.path.dirname(cfg.tle_source) or "data",
        do_finetune=do_finetune,
        detector=detector,
    )

    logger.info(f"Rapport sauvegardé : {cfg.model_dir}/ingest_report.json")
    return report


# ─── Mode : serve (frontend + API) ───────────────────────────────────────────

def mode_serve(port: int = 5000):
    import socket

    # ── Charger le .env avant tout ───────────────────────────────────────────
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        _loaded = []
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
                    _loaded.append(_k)
        logger.info(f".env chargé depuis {_env_path} — clés: {_loaded}")
    else:
        logger.warning(f".env introuvable à {_env_path} — créez ce fichier avec vos credentials")

    logger.info("=" * 60)
    logger.info(f"MODE : SERVEUR WEB")
    logger.info("=" * 60)

    # Trouver un port libre si le port demandé est occupé
    def is_port_free(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) != 0

    if not is_port_free(port):
        logger.warning(f"Port {port} occupé (AirPlay sur macOS utilise 5000).")
        for candidate in range(port + 1, port + 20):
            if is_port_free(candidate):
                port = candidate
                break
        else:
            logger.error("Aucun port libre trouvé entre 5000 et 5019.")
            return

    logger.info(f"Serveur : http://localhost:{port}  (ou http://127.0.0.1:{port})")
    logger.info("Safari/Firefox/Chrome : utiliser http://localhost:{} (pas 0.0.0.0)".format(port))
    logger.info("Arrêter avec Ctrl+C")

    try:
        from api import create_app, _start_rt_propagation_thread
        app = create_app()
        _start_rt_propagation_thread(app)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except ImportError:
        logger.error("pip install flask flask-cors  (requis pour le mode serve)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Satellite AI v2 — Correction de résidus SGP4 par TCN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --mode info
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode benchmark
  python main.py --mode export
  python main.py --mode predict
  python main.py --mode finetune --tle data/new_tle.txt
  python main.py --mode predict --onnx
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train","evaluate","export","finetune","predict","info","benchmark","ingest","serve"],
        default="info",
        help="Mode d'exécution (défaut: info)",
    )
    parser.add_argument("--tle",  default=cfg.tle_source,
                        help="Chemin vers le fichier TLE (pour train/finetune)")
    parser.add_argument("--finetune", action="store_true",
                        help="Déclencher le fine-tuning après ingestion (mode ingest)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port pour le serveur web (mode serve)")
    parser.add_argument("--onnx", action="store_true",
                        help="Utiliser ONNX Runtime pour l'inférence (mode predict)")
    parser.add_argument("--model", choices=["tcn", "gru"], default=None,
                        help="Forcer le type de modèle (override config)")
    args = parser.parse_args()

    # Override optionnels
    if args.model:
        cfg.model_type = args.model
    if args.tle != cfg.tle_source:
        cfg.tle_source = args.tle

    if   args.mode == "info":      mode_info()
    elif args.mode == "train":     mode_train()
    elif args.mode == "evaluate":  mode_evaluate()
    elif args.mode == "benchmark": mode_benchmark()
    elif args.mode == "export":    mode_export()
    elif args.mode == "finetune":  mode_finetune(args.tle)
    elif args.mode == "predict":   mode_predict(use_onnx=args.onnx)
    elif args.mode == "ingest":    mode_ingest(args.tle, do_finetune=args.finetune)
    elif args.mode == "serve":     mode_serve(port=args.port)
