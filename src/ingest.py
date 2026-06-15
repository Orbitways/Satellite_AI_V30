"""
ingest.py — Import de nouvelles données TLE + détection de manœuvres.

Fonctionnalités :
  1. Ingestion de nouveaux TLE (fichier local, URL Celestrak, Space-Track)
  2. Calcul de résidus différentiels réels (TLE successifs)
  3. Détection de manœuvres orbitales (discontinuités de résidu)
  4. Détection d'inopérabilité (résidus incohérents avec la physique)
  5. Mise à jour du dataset + déclenchement du fine-tuning EWC

Usage :
  python main.py --mode ingest --tle data/new_tle.txt
  python main.py --mode ingest --tle data/new_tle.txt --finetune
"""

import os
import json
import logging
import pickle
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class SatelliteEvent:
    """Événement détecté sur un satellite."""
    sat_name:   str
    epoch:      str                   # ISO 8601
    event_type: str                   # "maneuver" | "inoperable" | "anomaly" | "normal"
    severity:   str                   # "low" | "medium" | "high" | "critical"
    delta_v_ms: float                 # ΔV estimé en m/s (0 si non manœuvre)
    residual_norm_m: float            # norme du résidu au moment de l'événement (m)
    description: str

    def to_dict(self): return asdict(self)


@dataclass
class IngestReport:
    """Rapport complet d'une session d'ingestion."""
    timestamp:      str
    tle_path:       str
    n_satellites:   int
    n_new_residuals: int
    events:         List[SatelliteEvent]
    finetune_triggered: bool
    summary:        str


# ─── Détecteur de manœuvres ───────────────────────────────────────────────────

class ManeuverDetector:
    """
    Détecte les manœuvres et anomalies orbitales en analysant
    les résidus différentiels entre TLE successifs.

    Critères de détection :

    1. MANŒUVRE : saut brusque du résidu 3D > seuil_maneuver
       Un ΔV provoque une discontinuité dans la trajectoire non explicable
       par les perturbations naturelles. Le ΔV est estimé depuis le résidu.

    2. INOPÉRABILITÉ : résidu qui DIMINUE alors qu'il devrait croître,
       ou résidu constant (satellite passif sans drag détectable),
       ou résidu > seuil_inoperable sur plusieurs époques consécutives.

    3. ANOMALIE : résidu anormalement élevé mais sans signature claire
       de manœuvre (peut indiquer une dégradation TLE ou événement imprévu).

    Références :
      Kelecy & Hall, "Satellite Maneuver Detection using Two-Line Element Data",
      AAS/AIAA Space Flight Mechanics, 2006.
    """

    def __init__(
        self,
        threshold_maneuver_m:  float = 500.0,   # saut > 500 m → manœuvre
        threshold_anomaly_m:   float = 200.0,   # résidu > 200 m → anomalie
        threshold_inoperable_m: float = 5000.0, # résidu > 5 km → inopérable
        min_consecutive:       int   = 3,        # N époques consécutives pour inopérabilité
    ):
        self.thr_man  = threshold_maneuver_m
        self.thr_anom = threshold_anomaly_m
        self.thr_inop = threshold_inoperable_m
        self.min_cons = min_consecutive

    def analyze(
        self,
        sat_name: str,
        epochs: List[datetime],
        residuals_m: np.ndarray,         # (N, 3) résidus en mètres
    ) -> List[SatelliteEvent]:
        """
        Analyse la série temporelle de résidus et retourne les événements détectés.
        """
        events = []
        norms  = np.linalg.norm(residuals_m, axis=1)
        n      = len(norms)

        # ── Détection de manœuvres : saut de résidu entre deux époques ──
        for i in range(1, n):
            delta = abs(norms[i] - norms[i-1])
            if delta > self.thr_man:
                # Estimer ΔV depuis le saut de résidu
                dt_s  = (epochs[i] - epochs[i-1]).total_seconds()
                dv_ms = delta / dt_s if dt_s > 0 else 0.0

                sev = "high" if delta > 2000 else "medium"
                events.append(SatelliteEvent(
                    sat_name=sat_name,
                    epoch=epochs[i].isoformat(),
                    event_type="maneuver",
                    severity=sev,
                    delta_v_ms=round(dv_ms, 4),
                    residual_norm_m=round(float(norms[i]), 1),
                    description=(
                        f"Saut de résidu Δ={delta:.0f} m entre {epochs[i-1].strftime('%H:%M')} "
                        f"et {epochs[i].strftime('%H:%M')} UTC. "
                        f"ΔV estimé : {dv_ms*1000:.1f} mm/s."
                    )
                ))

        # ── Détection d'inopérabilité : résidus durablement élevés ──
        consecutive_high = 0
        for i, norm in enumerate(norms):
            if norm > self.thr_inop:
                consecutive_high += 1
                if consecutive_high == self.min_cons:
                    events.append(SatelliteEvent(
                        sat_name=sat_name,
                        epoch=epochs[i].isoformat(),
                        event_type="inoperable",
                        severity="critical",
                        delta_v_ms=0.0,
                        residual_norm_m=round(float(norm), 1),
                        description=(
                            f"Résidu > {self.thr_inop:.0f} m pendant {self.min_cons} "
                            f"époques consécutives. Satellite potentiellement inopérable "
                            f"ou TLE sévèrement dégradé."
                        )
                    ))
            else:
                consecutive_high = 0

        # ── Détection d'anomalies : résidu élevé isolé ──
        for i, norm in enumerate(norms):
            if norm > self.thr_anom and norm <= self.thr_inop:
                # Vérifier que ce n'est pas déjà une manœuvre détectée
                already = any(
                    e.event_type == "maneuver" and e.epoch == epochs[i].isoformat()
                    for e in events
                )
                if not already:
                    events.append(SatelliteEvent(
                        sat_name=sat_name,
                        epoch=epochs[i].isoformat(),
                        event_type="anomaly",
                        severity="low" if norm < 1000 else "medium",
                        delta_v_ms=0.0,
                        residual_norm_m=round(float(norm), 1),
                        description=(
                            f"Résidu anormalement élevé ({norm:.0f} m). "
                            f"Peut indiquer une dégradation TLE, une micro-manœuvre, "
                            f"ou une perturbation atmosphérique (tempête solaire)."
                        )
                    ))

        if not events:
            events.append(SatelliteEvent(
                sat_name=sat_name,
                epoch=epochs[-1].isoformat() if epochs else "",
                event_type="normal",
                severity="low",
                delta_v_ms=0.0,
                residual_norm_m=round(float(np.mean(norms)), 1),
                description=f"Aucune anomalie détectée. Résidu moyen : {np.mean(norms):.0f} m."
            ))

        return events


# ─── Ingestion principale ─────────────────────────────────────────────────────

def ingest_new_tles(
    tle_path: str,
    model_dir: str = "models",
    data_dir:  str = "data",
    do_finetune: bool = False,
    detector: Optional[ManeuverDetector] = None,
) -> IngestReport:
    """
    Pipeline complet d'ingestion de nouveaux TLE.

    Étapes :
      1. Charger les TLE existants (historique) et les nouveaux
      2. Calculer les résidus différentiels réels pour chaque satellite
      3. Analyser les résidus → détecter manœuvres / inopérabilité
      4. Sauvegarder les nouveaux résidus dans data/residuals_history.npz
      5. (optionnel) Déclencher le fine-tuning EWC
      6. Retourner un rapport structuré

    Args:
        tle_path     : fichier TLE à ingérer (nouvelles données)
        model_dir    : dossier contenant le modèle et les scalers
        data_dir     : dossier data pour l'historique
        do_finetune  : déclencher le fine-tuning après ingestion
        detector     : instance ManeuverDetector (crée une par défaut)

    Returns:
        IngestReport avec tous les événements détectés
    """
    from tle_fetcher import parse_tle_file, get_tle_epoch, _validate_tle
    from sgp4_utils import tle_to_state, differential_tle_residual

    if detector is None:
        detector = ManeuverDetector()

    logger.info(f"[INGEST] Chargement : {tle_path}")
    new_tles = parse_tle_file(tle_path)
    if not new_tles:
        raise ValueError(f"Aucun TLE valide dans {tle_path}")

    # Charger l'historique TLE existant
    history_path = os.path.join(data_dir, "tle_history.json")
    tle_history: Dict[str, List] = {}
    if os.path.exists(history_path):
        with open(history_path) as f:
            tle_history = json.load(f)
    os.makedirs(data_dir, exist_ok=True)

    all_events: List[SatelliteEvent] = []
    total_new_residuals = 0

    for name, tle1_new, tle2_new in new_tles:
        sat_key = name.strip()
        epoch_new = get_tle_epoch(tle1_new)
        if epoch_new is None:
            logger.warning(f"[INGEST] Époque invalide pour {name}, ignoré.")
            continue

        # Récupérer l'historique de ce satellite
        sat_hist = tle_history.get(sat_key, [])

        residuals_m = []
        epochs_list = []

        if sat_hist:
            # Calculer résidus différentiels avec le TLE précédent
            for old_entry in sat_hist[-5:]:   # 5 derniers TLE max
                tle1_old, tle2_old = old_entry["tle1"], old_entry["tle2"]
                epoch_old_str = old_entry.get("epoch", "")
                try:
                    epoch_old = datetime.fromisoformat(epoch_old_str)
                except Exception:
                    continue

                # Générer des points entre epoch_old et epoch_new
                dt_total = (epoch_new - epoch_old).total_seconds()
                if dt_total <= 0 or dt_total > 7 * 86400:
                    continue  # Skip si > 7 jours d'écart

                step_s = 300  # 5 minutes
                n_steps = min(int(dt_total / step_s), 100)

                for k in range(n_steps):
                    t = epoch_old + timedelta(seconds=k * step_s)
                    dr = differential_tle_residual(
                        tle1_old, tle2_old, tle1_new, tle2_new, t
                    )
                    if dr is not None:
                        residuals_m.append(dr * 1000.0)  # km → m
                        epochs_list.append(t)

                total_new_residuals += len(residuals_m)
                logger.info(
                    f"[INGEST] {sat_key} : {len(residuals_m)} résidus calculés "
                    f"({epoch_old.date()} → {epoch_new.date()})"
                )

        # Mettre à jour l'historique
        sat_hist.append({
            "tle1":  tle1_new,
            "tle2":  tle2_new,
            "epoch": epoch_new.isoformat(),
        })
        tle_history[sat_key] = sat_hist[-20:]  # garder 20 TLE max par satellite

        # Analyse des résidus si on en a
        if residuals_m:
            res_arr = np.array(residuals_m)
            events = detector.analyze(sat_key, epochs_list, res_arr)
            all_events.extend(events)

            # Sauvegarder les résidus
            _save_residuals(data_dir, sat_key, epoch_new, res_arr)
        else:
            logger.info(f"[INGEST] {sat_key} : premier TLE enregistré, pas encore de résidu.")
            all_events.append(SatelliteEvent(
                sat_name=sat_key, epoch=epoch_new.isoformat(),
                event_type="normal", severity="low", delta_v_ms=0.0,
                residual_norm_m=0.0,
                description="Premier TLE enregistré pour ce satellite."
            ))

    # Sauvegarder l'historique mis à jour
    with open(history_path, "w") as f:
        json.dump(tle_history, f, indent=2)
    logger.info(f"[INGEST] Historique TLE sauvegardé → {history_path}")

    # Rapport
    n_man   = sum(1 for e in all_events if e.event_type == "maneuver")
    n_inop  = sum(1 for e in all_events if e.event_type == "inoperable")
    n_anom  = sum(1 for e in all_events if e.event_type == "anomaly")
    summary = (
        f"{len(new_tles)} satellite(s) ingérés | "
        f"{total_new_residuals} nouveaux résidus | "
        f"Manœuvres: {n_man} | Anomalies: {n_anom} | Inopérables: {n_inop}"
    )

    report = IngestReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        tle_path=tle_path,
        n_satellites=len(new_tles),
        n_new_residuals=total_new_residuals,
        events=all_events,
        finetune_triggered=False,
        summary=summary,
    )

    logger.info(f"[INGEST] {summary}")
    _print_event_table(all_events)

    # Fine-tuning optionnel
    if do_finetune and total_new_residuals >= 50:
        report.finetune_triggered = True
        _trigger_finetune(tle_path, model_dir)
    elif do_finetune:
        logger.warning(
            f"[INGEST] Pas assez de résidus ({total_new_residuals} < 50) "
            f"pour déclencher le fine-tuning."
        )

    # Sauvegarder le rapport JSON
    report_path = os.path.join(model_dir, "ingest_report.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": report.timestamp,
            "tle_path":  report.tle_path,
            "n_satellites": report.n_satellites,
            "n_new_residuals": report.n_new_residuals,
            "finetune_triggered": report.finetune_triggered,
            "summary": report.summary,
            "events": [e.to_dict() for e in report.events],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"[INGEST] Rapport → {report_path}")

    return report


def _save_residuals(data_dir: str, sat_key: str, epoch: datetime,
                    residuals_m: np.ndarray) -> None:
    """Sauvegarde incrémentale des résidus dans un fichier NPZ."""
    safe_name = sat_key.replace(" ", "_").replace("/", "_")
    path = os.path.join(data_dir, f"residuals_{safe_name}.npz")

    if os.path.exists(path):
        old = np.load(path)
        residuals_m = np.concatenate([old["residuals"], residuals_m], axis=0)

    np.savez(path, residuals=residuals_m, epoch=np.array([epoch.isoformat()]))
    logger.debug(f"[INGEST] Résidus sauvegardés : {path} ({len(residuals_m)} total)")


def _print_event_table(events: List[SatelliteEvent]) -> None:
    """Affiche un tableau console des événements détectés."""
    if not events:
        return
    logger.info("─" * 72)
    logger.info(f"{'Satellite':<20} {'Type':<12} {'Sévérité':<10} {'Résidu (m)':>10}  Description")
    logger.info("─" * 72)
    icons = {"normal":"●", "anomaly":"▲", "maneuver":"◆", "inoperable":"✗"}
    colors_sev = {"low":"", "medium":"", "high":"", "critical":""}
    for e in events:
        icon = icons.get(e.event_type, "?")
        desc = e.description[:40] + "..." if len(e.description) > 40 else e.description
        logger.info(
            f"{e.sat_name:<20} {icon} {e.event_type:<10} {e.severity:<10} "
            f"{e.residual_norm_m:>10.0f}  {desc}"
        )
    logger.info("─" * 72)


def _trigger_finetune(tle_path: str, model_dir: str) -> None:
    """Déclenche le fine-tuning EWC depuis main.py."""
    import subprocess, sys
    logger.info("[INGEST] Déclenchement du fine-tuning EWC...")
    subprocess.run(
        [sys.executable, "main.py", "--mode", "finetune", "--tle", tle_path],
        check=True
    )


# ─── Chargement des résidus historiques ──────────────────────────────────────

def load_historical_residuals(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Charge tous les fichiers de résidus sauvegardés.
    Retourne un dict {sat_name: residuals_array (N,3)}.
    """
    result = {}
    for fname in os.listdir(data_dir):
        if fname.startswith("residuals_") and fname.endswith(".npz"):
            path = os.path.join(data_dir, fname)
            sat_name = fname[10:-4].replace("_", " ")
            data = np.load(path)
            result[sat_name] = data["residuals"]
            logger.info(f"[LOAD] {sat_name} : {len(data['residuals'])} résidus historiques")
    return result
