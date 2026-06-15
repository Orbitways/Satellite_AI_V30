"""
spacetrack.py — Client Space-Track.org pour récupérer l'historique TLE.

Utilise les credentials du fichier .env à la racine du projet.
Respecte le rate-limiting de l'API (max 30 req/min, pause entre requêtes).

Documentation API : https://www.space-track.org/documentation
"""

import os, json, time, logging, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constantes API ────────────────────────────────────────────────────────────
BASE_URL    = "https://www.space-track.org"
LOGIN_URL   = f"{BASE_URL}/ajaxauth/login"
QUERY_URL   = f"{BASE_URL}/basicspacedata/query"
LOGOUT_URL  = f"{BASE_URL}/ajaxauth/logout"

# Satellites d'intérêt : NORAD ID → nom usuel
SATELLITES = {
    "25544": "ISS (ZARYA)",
    "44713": "STARLINK-1007",
    "28654": "NOAA 18",
    "25994": "TERRA",
}

# Cache local pour éviter de re-télécharger si les données sont récentes
CACHE_DIR = Path("data/spacetrack_cache")


def _load_env() -> dict:
    """
    Charge les credentials depuis .env (racine du projet) ou variables d'environnement.
    Ordre de priorité : variables système > fichier .env
    """
    creds = {
        "email":    os.environ.get("SPACETRACK_EMAIL", ""),
        "password": os.environ.get("SPACETRACK_PASSWORD", ""),
    }

    # Chercher .env depuis la racine du projet
    root = Path(__file__).parent.parent
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "SPACETRACK_EMAIL"    and not creds["email"]:
                    creds["email"] = v
                if k == "SPACETRACK_PASSWORD" and not creds["password"]:
                    creds["password"] = v

    return creds


def _check_credentials():
    """Vérifie que les credentials sont disponibles. Lève ValueError sinon."""
    creds = _load_env()
    email    = creds["email"]
    password = creds["password"]

    if not email or not password:
        raise ValueError(
            "Credentials Space-Track manquants.\n"
            "Créez un fichier .env à la racine du projet avec :\n"
            "  SPACETRACK_EMAIL=votre@email.com\n"
            "  SPACETRACK_PASSWORD=votremotdepasse\n"
            "Compte gratuit sur : https://www.space-track.org/auth/createAccount"
        )
    return email, password


class SpaceTrackSession:
    """
    Session authentifiée Space-Track.

    Gère :
    - L'authentification par cookie de session
    - Le rate-limiting (max 30 req/min imposé par l'API)
    - Le cache local (évite de re-télécharger des données récentes)
    - La déconnexion propre

    Usage :
        with SpaceTrackSession() as st:
            history = st.get_tle_history("25544", days=7)
    """

    def __init__(self):
        self._cookie: Optional[str] = None
        self._last_request: float = 0.0
        self._request_count: int = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self._login()
        return self

    def __exit__(self, *_):
        self._logout()

    def _rate_limit(self):
        """Respecte le rate-limit Space-Track : 30 req/min max."""
        min_interval = 2.1  # secondes entre requêtes (≈ 28 req/min, marge de sécurité)
        elapsed = time.time() - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()
        self._request_count += 1

    def _request(self, url: str, data: Optional[bytes] = None) -> bytes:
        """Effectue une requête HTTP avec gestion des cookies."""
        self._rate_limit()
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if self._cookie:
            req.add_header("Cookie", self._cookie)
        req.add_header("User-Agent", "satellite-ai-v2/1.0")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Capturer le cookie de session à la connexion
                if not self._cookie:
                    set_cookie = resp.getheader("Set-Cookie", "")
                    if set_cookie:
                        # Garder seulement le nom=valeur sans les attributs
                        self._cookie = set_cookie.split(";")[0]
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ConnectionError(f"HTTP {e.code} sur {url}: {body[:200]}")

    def _login(self):
        """S'authentifie auprès de Space-Track."""
        email, password = _check_credentials()
        logger.info("Connexion à Space-Track...")
        data = urllib.parse.urlencode({
            "identity": email,
            "password": password,
        }).encode("utf-8")
        resp = self._request(LOGIN_URL, data=data)
        resp_text = resp.decode("utf-8", errors="replace")
        if "Failed" in resp_text or "Invalid" in resp_text:
            raise ConnectionError(
                "Authentification Space-Track échouée. "
                "Vérifiez SPACETRACK_EMAIL et SPACETRACK_PASSWORD dans .env"
            )
        logger.info("✓ Connecté à Space-Track")

    def _logout(self):
        """Déconnexion propre."""
        try:
            self._request(LOGOUT_URL)
            logger.debug("Déconnecté de Space-Track")
        except Exception:
            pass

    def get_tle_history(
        self,
        norad_id: str,
        days: int = 7,
        emit: Optional[callable] = None,
    ):
        """
        Récupère l'historique TLE d'un satellite sur les `days` derniers jours.

        Args:
            norad_id : identifiant NORAD (ex: "25544" pour l'ISS)
            days     : nombre de jours d'historique (1-30, max API Space-Track)
            emit     : callback(message) pour le suivi de progression

        Returns:
            Liste de (name, line1, line2) ordonnée chronologiquement (du plus ancien au plus récent)
        """
        # Vérifier le cache local
        cache_file = CACHE_DIR / f"{norad_id}_{days}d.json"
        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_h < 2.0:  # Cache valide 2h
                logger.info(f"[Cache] NORAD {norad_id} — données fraîches ({age_h:.1f}h)")
                if emit: emit(f"Cache local utilisé (données vieilles de {age_h:.1f}h)")
                return json.loads(cache_file.read_text())

        # Dates de requête
        now  = datetime.now(timezone.utc)
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")

        # URL API gp_history (General Perturbations history)
        url = (
            f"{QUERY_URL}/class/gp_history"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/{start}--{end}"
            f"/orderby/EPOCH%20asc"
            f"/format/tle"
        )

        sat_name = SATELLITES.get(norad_id, f"NORAD-{norad_id}")
        msg = f"Téléchargement TLE historique : {sat_name} ({days}j)"
        logger.info(msg)
        if emit: emit(msg)

        try:
            content = self._request(url).decode("utf-8", errors="replace")
        except ConnectionError as e:
            raise ConnectionError(f"Erreur récupération NORAD {norad_id}: {e}")

        # Parser les TLE (format 3-lignes)
        from tle_fetcher import _validate_tle
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        history = []
        i = 0
        while i + 2 < len(lines):
            name, l1, l2 = lines[i], lines[i+1], lines[i+2]
            if _validate_tle(l1, l2):
                history.append((name, l1, l2))
            i += 3

        if not history:
            # Fallback : essayer le format JSON si TLE vide
            logger.warning(f"Format TLE vide pour NORAD {norad_id}, tentative JSON...")
            url_json = url.replace("/format/tle", "/format/json")
            try:
                content_json = self._request(url_json).decode("utf-8")
                records = json.loads(content_json)
                history = _parse_gp_json(records)
            except Exception:
                pass

        # Mettre en cache
        cache_file.write_text(json.dumps(history))

        msg = f"✓ {sat_name} : {len(history)} TLE récupérés sur {days} jours"
        logger.info(msg)
        if emit: emit(msg)

        return history


def _parse_gp_json(records: list):
    """Parse les enregistrements JSON de Space-Track en tuples (name, tle1, tle2)."""
    from tle_fetcher import _validate_tle
    history = []
    for r in records:
        name = r.get("OBJECT_NAME", "UNKNOWN")
        l1   = r.get("TLE_LINE1", "")
        l2   = r.get("TLE_LINE2", "")
        if l1 and l2 and _validate_tle(l1, l2):
            history.append((name, l1, l2))
    return history


def fetch_all_satellites(
    days: int = 7,
    emit: Optional[callable] = None,
):
    """
    Télécharge l'historique TLE de tous les satellites configurés.

    Args:
        days : jours d'historique (recommandé : 7 pour l'entraînement initial,
               1-2 pour les mises à jour régulières)
        emit : callback pour la progression

    Returns:
        dict { norad_id: [(name, tle1, tle2), ...] }
    """
    results = {}
    with SpaceTrackSession() as st:
        for norad_id, name in SATELLITES.items():
            msg = f"[{list(SATELLITES.keys()).index(norad_id)+1}/{len(SATELLITES)}] {name}"
            if emit: emit(msg)
            try:
                history = st.get_tle_history(norad_id, days=days, emit=emit)
                results[norad_id] = history
            except Exception as e:
                logger.error(f"Erreur {name} (NORAD {norad_id}): {e}")
                if emit: emit(f"⚠ Erreur {name}: {e}")
                results[norad_id] = []

    total = sum(len(v) for v in results.values())
    msg = f"✓ Total : {total} TLE téléchargés pour {len(results)} satellites"
    logger.info(msg)
    if emit: emit(msg)
    return results


def check_credentials_available() -> dict:
    """
    Vérifie si les credentials sont configurés (sans se connecter).
    Retourne un dict avec le statut.
    """
    try:
        creds = _load_env()
        email    = creds["email"]
        password = creds["password"]
        if email and password:
            masked = email[:3] + "***@" + email.split("@")[-1] if "@" in email else "***"
            return {"ok": True, "email": masked}
        return {"ok": False, "reason": "Credentials manquants dans .env"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}
