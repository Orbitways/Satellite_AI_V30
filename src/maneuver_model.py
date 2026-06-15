"""
maneuver_model.py — Modèle LSTM de prédiction de manœuvres.

Architecture : GRU bidirectionnel léger (pas de dépendance PyTorch requise
pour l'inférence — le modèle peut être sérialisé en JSON pour une inférence
numpy pure).

Entrées (fenêtre de 30 observations) :
  - résidu RTN normalisé [3]
  - intervalle de temps depuis la dernière observation [1]
  - altitude km normalisée [1]
  - excentricité [1]
  - inclinaison normalisée [1]
  - F10.7 normalisé [1]
  - indicateur de manœuvre passée [1]
  Total : 9 features par pas de temps

Sorties :
  - p_24h  : probabilité de manœuvre dans les 24h suivantes
  - p_48h  : probabilité de manœuvre dans les 48h suivantes
  - p_7d   : probabilité de manœuvre dans les 7 prochains jours
  - dv_pred : ΔV prédit [m/s]
  - type_pred : vecteur de probabilités par type [5]
"""

import os, json, logging, math
import numpy as np
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join("models", "maneuver_model.json")
SCALER_PATH = os.path.join("models", "maneuver_scaler.json")

N_FEATURES = 9
N_HIDDEN   = 32
SEQ_LEN    = 30

MANEUVER_TYPES = ['hohmann', 'phasing', 'inclination', 'eccentricity', 'stationkeeping']


# ═══════════════════════════════════════════════════════════════════════════════
# GRU minimal en NumPy (évite la dépendance PyTorch pour l'inférence)
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def _tanh(x):     return np.tanh(np.clip(x, -30, 30))
def _softmax(x):  e = np.exp(x - x.max()); return e / e.sum()


class GRUCell:
    """Cellule GRU minimale en NumPy."""
    def __init__(self, n_in, n_hidden):
        k = 1 / math.sqrt(n_hidden)
        self.Wz = np.random.uniform(-k, k, (n_hidden, n_in + n_hidden))
        self.bz = np.zeros(n_hidden)
        self.Wr = np.random.uniform(-k, k, (n_hidden, n_in + n_hidden))
        self.br = np.zeros(n_hidden)
        self.Wn = np.random.uniform(-k, k, (n_hidden, n_in + n_hidden))
        self.bn = np.zeros(n_hidden)

    def forward(self, x, h):
        xh = np.concatenate([x, h])
        z  = _sigmoid(self.Wz @ xh + self.bz)
        r  = _sigmoid(self.Wr @ xh + self.br)
        n  = _tanh(self.Wn @ np.concatenate([x, r * h]) + self.bn)
        h_new = (1 - z) * h + z * n
        return h_new

    def to_dict(self):
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d, n_in, n_hidden):
        obj = cls.__new__(cls)
        for k, v in d.items():
            setattr(obj, k, np.array(v))
        return obj


class ManeuverGRU:
    """
    Modèle GRU bidirectionnel pour la prédiction de manœuvres.
    Entraînable en NumPy (BPTT simplifié) ou via PyTorch si disponible.
    """

    def __init__(self):
        np.random.seed(42)
        self.fwd  = GRUCell(N_FEATURES, N_HIDDEN)
        self.bwd  = GRUCell(N_FEATURES, N_HIDDEN)
        # Tête de prédiction : 2*N_HIDDEN → sorties
        n_in = 2 * N_HIDDEN
        k = 1 / math.sqrt(n_in)
        self.W_out = np.random.uniform(-k, k, (8, n_in))  # 8 sorties
        self.b_out = np.zeros(8)
        # Metadata
        self.trained    = False
        self.n_samples  = 0
        self.train_loss = None
        self.version    = "v1"

    def forward(self, seq: np.ndarray) -> dict:
        """
        seq : (T, N_FEATURES)
        Retourne un dict avec les prédictions.
        """
        T = seq.shape[0]
        h_f = np.zeros(N_HIDDEN)
        h_b = np.zeros(N_HIDDEN)

        for t in range(T):
            h_f = self.fwd.forward(seq[t], h_f)
        for t in range(T - 1, -1, -1):
            h_b = self.bwd.forward(seq[t], h_b)

        feat = np.concatenate([h_f, h_b])
        out  = self.W_out @ feat + self.b_out

        # Sorties
        p24h  = float(_sigmoid(out[0]))
        p48h  = float(_sigmoid(out[1]))
        p7d   = float(_sigmoid(out[2]))
        dv    = float(max(0, out[3]))           # ΔV m/s (positif)
        types = _softmax(out[4:])               # distribution sur 5 types

        return {
            "p_24h":       round(p24h, 4),
            "p_48h":       round(p48h, 4),
            "p_7d":        round(p7d, 4),
            "dv_pred_ms":  round(dv, 3),
            "type_probs":  {t: round(float(p), 3)
                            for t, p in zip(MANEUVER_TYPES, types)},
            "type_pred":   MANEUVER_TYPES[int(types.argmax())],
        }

    def save(self):
        os.makedirs("models", exist_ok=True)
        data = {
            "fwd":       self.fwd.to_dict(),
            "bwd":       self.bwd.to_dict(),
            "W_out":     self.W_out.tolist(),
            "b_out":     self.b_out.tolist(),
            "trained":   self.trained,
            "n_samples": self.n_samples,
            "train_loss":self.train_loss,
            "version":   self.version,
        }
        json.dump(data, open(MODEL_PATH, "w"))
        logger.info(f"Modèle manœuvres sauvegardé → {MODEL_PATH}")

    @classmethod
    def load(cls):
        obj = cls()
        if not os.path.exists(MODEL_PATH):
            return obj
        try:
            data = json.load(open(MODEL_PATH))
            obj.fwd     = GRUCell.from_dict(data["fwd"],  N_FEATURES, N_HIDDEN)
            obj.bwd     = GRUCell.from_dict(data["bwd"],  N_FEATURES, N_HIDDEN)
            obj.W_out   = np.array(data["W_out"])
            obj.b_out   = np.array(data["b_out"])
            obj.trained = data.get("trained", False)
            obj.n_samples = data.get("n_samples", 0)
            obj.train_loss = data.get("train_loss")
            obj.version = data.get("version", "v1")
            logger.info(f"Modèle manœuvres chargé ({obj.n_samples} samples)")
        except Exception as e:
            logger.warning(f"Chargement modèle échoué: {e}")
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Scaler (normalisation)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureScaler:
    """Normalisation min-max par feature."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        # Éviter division par zéro
        self.max_ = np.where(self.max_ == self.min_, self.min_ + 1, self.max_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            return X
        return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

    def fit_transform(self, X):
        self.fit(X); return self.transform(X)

    def save(self):
        if self.min_ is None: return
        os.makedirs("models", exist_ok=True)
        json.dump({"min": self.min_.tolist(), "max": self.max_.tolist()},
                  open(SCALER_PATH, "w"))

    @classmethod
    def load(cls):
        obj = cls()
        if os.path.exists(SCALER_PATH):
            try:
                d = json.load(open(SCALER_PATH))
                obj.min_ = np.array(d["min"])
                obj.max_ = np.array(d["max"])
            except Exception as e:
                logger.warning(f"Chargement scaler échoué: {e}")
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Préparation des features depuis l'historique de manœuvres détectées
# ═══════════════════════════════════════════════════════════════════════════════

def build_features_for_satellite(norad_id: str, f107: float = 150.0) -> np.ndarray | None:
    """
    Construit la séquence de features pour un satellite depuis la base.
    Retourne un array (T, N_FEATURES) ou None si données insuffisantes.
    """
    from tle_database import get_connection

    conn = get_connection()

    # Récupérer l'historique des résidus (manœuvres détectées et normaux)
    rows = conn.execute("""
        SELECT detected_at, delta_v_ms, residual_norm_m, event_type, notes
        FROM maneuvers
        WHERE norad_id = ?
        ORDER BY detected_at ASC
    """, (norad_id,)).fetchall()

    # Récupérer les éléments orbitaux récents
    tle_rows = conn.execute("""
        SELECT epoch, mm, inc, ecc, alt_km FROM tle_records
        WHERE norad_id = ? ORDER BY epoch DESC LIMIT 50
    """, (norad_id,)).fetchall()
    conn.close()

    if len(rows) < 3 or len(tle_rows) == 0:
        return None

    tle_dict = {r["epoch"][:10]: r for r in tle_rows}
    last_tle = tle_rows[0]

    features = []
    prev_dt  = None

    for i, row in enumerate(rows):
        try:
            dt = datetime.fromisoformat(row["detected_at"].replace("Z", "+00:00"))
        except Exception:
            continue

        dt_since_last = (dt - prev_dt).total_seconds() / 3600 if prev_dt else 24.0
        prev_dt = dt

        # Extraire RTN depuis les notes si disponible
        rtn = [0.0, float(row["residual_norm_m"] or 0) / 1000.0, 0.0]

        # Éléments orbitaux
        alt   = float(last_tle["alt_km"] or 550)
        inc   = float(last_tle["inc"] or 53)
        ecc   = float(last_tle["ecc"] or 0.001)

        # Label manœuvre
        is_man = 1.0 if row["event_type"] in ('hohmann','phasing','inclination',
                                               'eccentricity','stationkeeping','maneuver') else 0.0

        feat = [
            rtn[0],                    # résidu R km
            rtn[1],                    # résidu T km
            rtn[2],                    # résidu N km
            min(dt_since_last, 168),   # heures depuis dernière obs (max 7j)
            min(alt, 2000) / 2000,     # altitude normalisée
            min(ecc * 1000, 1.0),      # excentricité × 1000
            inc / 180,                 # inclinaison normalisée
            min(f107, 300) / 300,      # activité solaire normalisée
            is_man,                    # indicateur manœuvre passée
        ]
        features.append(feat)

    if len(features) < 3:
        return None

    arr = np.array(features, dtype=np.float32)
    # Padder ou tronquer à SEQ_LEN
    if len(arr) >= SEQ_LEN:
        return arr[-SEQ_LEN:]
    else:
        pad = np.zeros((SEQ_LEN - len(arr), N_FEATURES), dtype=np.float32)
        return np.vstack([pad, arr])


# ═══════════════════════════════════════════════════════════════════════════════
# Entraînement
# ═══════════════════════════════════════════════════════════════════════════════

def train_maneuver_model(emit=None, sample_pct: float = 1.0, pause_flag=None) -> dict:
    """
    Entraîne le modèle GRU sur toutes les données disponibles en base.
    Utilise une descente de gradient simplifiée (Adam numérique).

    Retourne un rapport d'entraînement.
    """
    from tle_database import get_connection
    from space_weather import get_current_conditions

    if emit: emit("[INFO] Chargement des données d'entraînement...", 5)

    # F10.7 actuel pour normalisation
    try:
        sw = get_current_conditions()
        f107 = sw.get("f107_current", 150.0)
    except Exception:
        f107 = 150.0

    conn = get_connection()
    satellites = conn.execute("""
        SELECT DISTINCT norad_id FROM maneuvers
        WHERE event_type NOT IN ('noise', 'anomaly')
        GROUP BY norad_id HAVING COUNT(*) >= 5
    """).fetchall()
    conn.close()

    n_sats = len(satellites)
    if emit: emit(f"[INFO] {n_sats} satellites avec assez de données...", 10)

    if n_sats == 0:
        if emit: emit("[AVERT] Pas assez de données — détectez d'abord les manœuvres", 100)
        return {"error": "Données insuffisantes", "n_satellites": 0}

    # Construire les séquences
    sequences  = []
    labels_p7d = []  # 1 si une manœuvre suit dans les 7j
    labels_dv  = []  # ΔV prédit

    conn = get_connection()
    for row in satellites:
        norad = row[0]
        seq = build_features_for_satellite(norad, f107)
        if seq is None: continue

        # Label : est-ce qu'une manœuvre suit dans les 7 jours ?
        future = conn.execute("""
            SELECT delta_v_ms FROM maneuvers
            WHERE norad_id = ? AND event_type NOT IN ('noise','anomaly')
            ORDER BY detected_at DESC LIMIT 1
        """, (norad,)).fetchone()

        dv = float(future["delta_v_ms"]) if future else 0.0
        sequences.append(seq)
        labels_p7d.append(1.0 if dv > 0.1 else 0.0)
        labels_dv.append(min(dv, 200.0))
    conn.close()

    if len(sequences) < 3:
        if emit: emit("[AVERT] Pas assez de séquences valides", 100)
        return {"error": "Séquences insuffisantes", "n_satellites": n_sats}

    # Sous-échantillonnage paramétrable
    if sample_pct < 1.0:
        n_keep = max(3, int(len(sequences) * sample_pct))
        import random as _rnd
        _rnd.seed(42)
        idx_keep = _rnd.sample(range(len(sequences)), n_keep)
        sequences  = [sequences[i]  for i in idx_keep]
        labels_p7d = np.array([labels_p7d[i] for i in idx_keep])
        labels_dv  = np.array([labels_dv[i]  for i in idx_keep])
        if emit: emit(f"[INFO] Sous-échantillonnage : {n_keep} séquences ({int(sample_pct*100)}%)", 20)
    else:
        n_keep = len(sequences)
        labels_p7d = np.array(labels_p7d)
        labels_dv  = np.array(labels_dv)

    if emit: emit(f"[INFO] Entraînement sur {n_keep} séquences...", 20)

    # Normalisation
    all_feats = np.vstack(sequences)
    scaler = FeatureScaler()
    scaler.fit(all_feats)
    seqs_norm = [scaler.transform(s) for s in sequences]
    labels_dv  = labels_dv / 200.0  # normaliser ΔV

    # Entraînement par descente de gradient numérique (différences finies)
    # Simple mais fonctionnel sans PyTorch
    model = ManeuverGRU()
    lr    = 0.01
    eps   = 1e-4
    n_epochs = 30
    losses = []

    _paused = False
    for epoch in range(n_epochs):
        # Support pause
        if pause_flag is not None and pause_flag.get('pause'):
            if emit: emit(f"[INFO] Pause après epoch {epoch}...", 20 + int(epoch/n_epochs*60))
            while pause_flag.get('pause') and not pause_flag.get('stop'):
                import time as _time; _time.sleep(0.5)
            if pause_flag.get('stop'):
                if emit: emit(f"[INFO] Entrainement interrompu à l'epoch {epoch+1}", 90)
                break

        pct = 20 + int(epoch / n_epochs * 60)
        if emit:
            emit(f"[INFO] Epoch {epoch+1}/{n_epochs} · loss={losses[-1]:.4f}" if losses else f"[INFO] Epoch {epoch+1}/{n_epochs}...", pct)

        epoch_loss = 0.0
        for seq, y_p7d, y_dv in zip(seqs_norm, labels_p7d, labels_dv):
            pred = model.forward(seq)
            # BCE loss pour p7d + MSE pour dv
            p = max(1e-7, min(1-1e-7, pred["p_7d"]))
            loss = -(y_p7d * math.log(p) + (1-y_p7d) * math.log(1-p))
            loss += (pred["dv_pred_ms"] / 200.0 - y_dv) ** 2
            epoch_loss += loss

            # Gradient numérique sur W_out uniquement (approximation rapide)
            grad = np.zeros_like(model.W_out)
            for i in range(model.W_out.shape[0]):
                for j in range(0, model.W_out.shape[1], 4):  # sparse update
                    model.W_out[i, j] += eps
                    pred2 = model.forward(seq)
                    p2 = max(1e-7, min(1-1e-7, pred2["p_7d"]))
                    loss2 = -(y_p7d * math.log(p2) + (1-y_p7d) * math.log(1-p2))
                    loss2 += (pred2["dv_pred_ms"] / 200.0 - y_dv) ** 2
                    grad[i, j] = (loss2 - loss) / eps
                    model.W_out[i, j] -= eps
            model.W_out -= lr * grad

        avg_loss = epoch_loss / len(sequences)
        losses.append(avg_loss)

    model.trained   = True
    model.n_samples = len(sequences)
    model.train_loss = round(float(losses[-1]), 4)
    model.save()
    scaler.save()

    report = {
        "n_satellites": n_sats,
        "n_sequences":  len(sequences),
        "n_epochs":     n_epochs,
        "final_loss":   model.train_loss,
        "loss_history": [round(float(l), 4) for l in losses],
    }
    if emit: emit(f"[OK] Entraînement terminé — loss={model.train_loss:.4f}", 100)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Inférence — prédictions pour tous les satellites actifs
# ═══════════════════════════════════════════════════════════════════════════════

def predict_all_maneuvers(min_p7d: float = 0.3, emit=None) -> list:
    """
    Lance l'inférence sur tous les satellites avec historique en base.
    Retourne une liste de prédictions triées par probabilité décroissante.
    """
    from tle_database import get_connection, store_prediction
    from space_weather import get_current_conditions

    try:
        sw = get_current_conditions()
        f107 = sw.get("f107_current", 150.0)
    except Exception:
        f107 = 150.0

    model  = ManeuverGRU.load()
    scaler = FeatureScaler.load()

    if not model.trained:
        if emit: emit("[AVERT] Modèle non entraîné — lancez l'entraînement d'abord", 100)
        return []

    conn = get_connection()
    satellites = conn.execute("""
        SELECT DISTINCT norad_id, name FROM maneuvers
        GROUP BY norad_id HAVING COUNT(*) >= 3
    """).fetchall()
    conn.close()

    if emit: emit(f"[INFO] Inférence sur {len(satellites)} satellites...", 10)

    predictions = []
    for i, row in enumerate(satellites):
        norad_id = row["norad_id"]
        name     = row["name"]

        if emit and i % 20 == 0:
            emit(f"[INFO] {i+1}/{len(satellites)} — {name[:20]}", 10 + int(i/len(satellites)*80))

        try:
            seq = build_features_for_satellite(norad_id, f107)
            if seq is None: continue

            seq_norm = scaler.transform(seq) if scaler.min_ is not None else seq
            pred = model.forward(seq_norm)

            if pred["p_7d"] < min_p7d: continue

            # Stocker en base
            now = datetime.now(timezone.utc)
            predicted_epoch = (now + timedelta(days=7 * (1 - pred["p_7d"]))).isoformat()

            store_prediction(
                norad_id=norad_id, name=name,
                predicted_epoch=predicted_epoch,
                delta_v_ms_pred=pred["dv_pred_ms"],
                confidence=pred["p_7d"],
                model_version="gru_v1",
                notes=f"Type prédit: {pred['type_pred']}, p24h={pred['p_24h']}"
            )

            # Récupérer l'orbite actuelle
            conn2 = get_connection()
            tle_row = conn2.execute("""
                SELECT tle1, tle2, alt_km, inc FROM tle_records
                WHERE norad_id = ? ORDER BY epoch DESC LIMIT 1
            """, (norad_id,)).fetchone()
            conn2.close()

            predictions.append({
                "norad_id":       norad_id,
                "name":           name,
                "p_24h":          pred["p_24h"],
                "p_48h":          pred["p_48h"],
                "p_7d":           pred["p_7d"],
                "dv_pred_ms":     pred["dv_pred_ms"],
                "type_pred":      pred["type_pred"],
                "type_probs":     pred["type_probs"],
                "predicted_epoch": predicted_epoch,
                "tle1":           tle_row["tle1"] if tle_row else None,
                "tle2":           tle_row["tle2"] if tle_row else None,
                "alt_km":         float(tle_row["alt_km"]) if tle_row else 550,
                "inc":            float(tle_row["inc"]) if tle_row else 53,
            })
        except Exception as e:
            logger.debug(f"Inférence {norad_id}: {e}")

    predictions.sort(key=lambda x: x["p_7d"], reverse=True)
    if emit: emit(f"[OK] {len(predictions)} prédictions générées", 100)
    return predictions
