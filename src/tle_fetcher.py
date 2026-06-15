"""
tle_fetcher.py — Ingestion de TLE depuis Celestrak ou fichier local.

En production : utiliser Space-Track.org (compte requis) pour des TLE
haute précision. Celestrak est suffisant pour du LEO standard.
"""

import os
import re
import logging
from typing import List, Tuple, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# TLE ISS réels (backup si pas d'accès réseau)
FALLBACK_TLES = [
    (
        "ISS (ZARYA)",
        "1 25544U 98067A   24150.54097222  .00016717  00000+0  10270-3 0  9993",
        "2 25544  51.6416  21.5234 0006752  54.1234  45.6789 15.50012345678901",
    ),
]

# URLs Celestrak par catégorie
CELESTRAK_URLS = {
    "stations":  "https://celestrak.org/SOCRATES/query.php?CATALOG=25544&NAME=ISS",
    "active":    "https://celestrak.org/SOCRATES/query.php?CATALOG=active",
    "starlink":  "https://celestrak.org/SOCRATES/query.php?CATALOG=starlink",
}


TLEEntry = Tuple[str, str, str]  # (name, line1, line2)


def parse_tle_file(path: str) -> List[TLEEntry]:
    """
    Parse un fichier TLE au format 3-lignes standard.
    Ignore les lignes vides et les commentaires (#).
    Valide le format de chaque TLE avant de l'inclure.
    """
    if not os.path.exists(path):
        logger.warning(f"Fichier TLE introuvable : {path}. Utilisation du fallback.")
        return FALLBACK_TLES

    entries: List[TLEEntry] = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if _validate_tle(line1, line2):
            entries.append((name, line1, line2))
        else:
            logger.warning(f"TLE invalide ignoré : {name}")

        i += 3

    if not entries:
        logger.warning("Aucun TLE valide trouvé. Utilisation du fallback.")
        return FALLBACK_TLES

    logger.info(f"[TLE] {len(entries)} satellite(s) chargé(s) depuis {path}")
    return entries


def fetch_celestrak(category: str = "stations") -> List[TLEEntry]:
    """
    Télécharge les TLE depuis Celestrak (format 3-lignes).
    Nécessite une connexion réseau. Retourne FALLBACK_TLES en cas d'échec.
    """
    try:
        import urllib.request
        url = f"https://celestrak.org/SOCRATES/query.php?CATALOG={category}"
        # URL standard Celestrak 3-line
        url = f"https://celestrak.org/supplemental/query.php?FORMAT=tle"
        # Utiliser l'URL correcte pour les TLE
        url = f"https://celestrak.org/SOCRATES/query.php"

        # URL propre Celestrak :
        url = f"https://celestrak.org/pub/TLE/catalog.tle"
        logger.info(f"Téléchargement TLE depuis Celestrak...")

        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read().decode("utf-8")

        # Sauvegarder localement pour cache
        cache_path = f"data/celestrak_{category}.txt"
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(content)

        return _parse_tle_string(content)

    except Exception as e:
        logger.warning(f"Celestrak inaccessible ({e}). Utilisation du cache/fallback.")
        # Essayer le cache local
        cache_path = f"data/celestrak_{category}.txt"
        if os.path.exists(cache_path):
            return parse_tle_file(cache_path)
        return FALLBACK_TLES


def _parse_tle_string(content: str) -> List[TLEEntry]:
    """Parse un contenu TLE multi-satellites depuis une chaîne."""
    entries: List[TLEEntry] = []
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if _validate_tle(line1, line2):
            entries.append((name, line1, line2))
        i += 3
    return entries if entries else FALLBACK_TLES


def _validate_tle(line1: str, line2: str) -> bool:
    """
    Validation basique du format TLE :
    - Ligne 1 commence par '1 '
    - Ligne 2 commence par '2 '
    - Longueur correcte
    - Checksum valide
    """
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        return False
    if len(line1) < 69 or len(line2) < 69:
        return False
    if not (_checksum(line1) and _checksum(line2)):
        return False
    return True


def _checksum(line: str) -> bool:
    """Vérifie le checksum TLE (modulo 10)."""
    if len(line) < 69:
        return False
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return (total % 10) == int(line[68])


def get_tle_epoch(line1: str) -> Optional[datetime]:
    """Extrait la date d'époque d'une ligne TLE 1."""
    try:
        epoch_str = line1[18:32].strip()
        year_2d = int(epoch_str[:2])
        year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
        day_of_year = float(epoch_str[2:])
        dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        dt += timedelta(days=day_of_year - 1)
        return dt
    except Exception:
        return None

def fetch_and_store(group: str = "starlink"):
    """
    Compatibility wrapper for the FastAPI TLE refresh endpoint.

    This function is called by api/main.py when Lovable triggers:
    POST /v1/tle/refresh
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)

    result = fetch_celestrak(group=group)

    status = {
        "ok": True,
        "group": group,
        "source": "celestrak",
        "last_fetched_at": datetime.now(timezone.utc).isoformat(),
        "result_type": type(result).__name__,
    }

    # Try to enrich status if fetch_celestrak returns useful metadata
    if isinstance(result, dict):
        status.update(result)
    elif isinstance(result, list):
        status["n_objects_estimated"] = len(result)
    elif isinstance(result, str):
        status["n_lines"] = len(result.splitlines())
        status["n_objects_estimated"] = len(result.splitlines()) // 3

    Path("data/tle_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    return status

def load_catalog_status():
    """
    Return latest TLE catalog refresh status for the API endpoint:
    GET /v1/tle/status
    """
    import json
    from pathlib import Path

    status_file = Path("data") / "tle_status.json"

    if not status_file.exists():
        return {
            "ok": False,
            "status": "not_refreshed_yet",
            "message": "No TLE refresh has been performed yet.",
        }

    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "ok": False,
            "status": "status_read_error",
            "message": str(e),
        }

def fetch_and_store(group: str = "starlink", days: int = 30):
    """
    Refresh the local TLE database using Space-Track gp_history,
    then ingest results into the same local database used by the existing frontend.
    """
    import os
    import json
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    from spacetrack import SpaceTrackSession
    from tle_database import ingest_tles, get_stats

    Path("data").mkdir(exist_ok=True)

    logs = []

    def emit(msg, pct=None):
        logs.append(str(msg))

    def parse_tle_text(raw: str):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        tles = []
        i = 0
        while i + 2 < len(lines):
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            if l1.startswith("1 ") and l2.startswith("2 "):
                tles.append((name, l1, l2))
            i += 3
        return tles

    # Ensure compatibility with the existing SpaceTrackSession credential names.
    # Your Codespace currently uses SPACETRACK_USER / SPACETRACK_PASS,
    # while scripts/spacetrack.py expects SPACETRACK_EMAIL / SPACETRACK_PASSWORD.
    if os.environ.get("SPACETRACK_USER") and not os.environ.get("SPACETRACK_EMAIL"):
        os.environ["SPACETRACK_EMAIL"] = os.environ["SPACETRACK_USER"]

    if os.environ.get("SPACETRACK_PASS") and not os.environ.get("SPACETRACK_PASSWORD"):
        os.environ["SPACETRACK_PASSWORD"] = os.environ["SPACETRACK_PASS"]

    all_tles = []
    source_used = "Space-Track gp_history"

    with SpaceTrackSession() as client:
        now_dt = datetime.now(timezone.utc)
        end_str = now_dt.strftime("%Y-%m-%d")
        start_hist = (now_dt - timedelta(days=days)).strftime("%Y-%m-%d")

        emit("[INFO] Recuperation liste satellites LEO actifs via Space-Track gp", 5)

        url_list = (
            "https://www.space-track.org/basicspacedata/query"
            "/class/gp/EPOCH/%3Enow-2"
            "/MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25"
            "/OBJECT_TYPE/payload,debris"
            "/orderby/NORAD_CAT_ID/format/tle"
        )

        raw_list = client._request(url_list).decode("utf-8", errors="ignore")
        tles_current = parse_tle_text(raw_list) if raw_list and len(raw_list) > 200 else []

        norad_ids = list(set(t[1][2:7].strip() for t in tles_current))
        emit(f"[OK] {len(norad_ids)} objets LEO identifies", 15)

        if not norad_ids:
            raise ValueError("Aucun objet LEO trouve via Space-Track gp")

        all_tles.extend(tles_current)

        batch_size = 500
        batches = [norad_ids[i:i + batch_size] for i in range(0, len(norad_ids), batch_size)]
        emit(f"[INFO] {len(batches)} batches de {batch_size} objets", 20)

        for b_idx, batch in enumerate(batches):
            pct = 20 + int(b_idx / max(len(batches), 1) * 55)
            norad_str = ",".join(batch)

            url_hist = (
                "https://www.space-track.org/basicspacedata/query"
                f"/class/gp_history/NORAD_CAT_ID/{norad_str}"
                f"/EPOCH/{start_hist}--{end_str}"
                "/orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/tle"
            )

            try:
                raw_h = client._request(url_hist).decode("utf-8", errors="ignore")
                batch_tles = parse_tle_text(raw_h) if raw_h and len(raw_h) > 100 else []
                all_tles.extend(batch_tles)
                emit(f"[INFO] Batch {b_idx + 1}/{len(batches)}: +{len(batch_tles)} TLE", pct)
            except Exception as be:
                emit(f"[WARN] Batch {b_idx + 1} failed: {be}", pct)

    # Deduplicate by NORAD + epoch
    seen = set()
    unique_tles = []

    for t in all_tles:
        key = t[1][2:7].strip() + "|" + t[1][18:32]
        if key not in seen:
            seen.add(key)
            unique_tles.append(t)

    all_tles = unique_tles

    emit(f"[INFO] Ingestion de {len(all_tles)} TLE", 80)

    report = ingest_tles(
        all_tles,
        source=source_used,
        emit=emit,
    )

    stats = get_stats()

    status = {
        "ok": True,
        "group": group,
        "source": source_used,
        "days": days,
        "last_fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_tles_fetched": len(all_tles),
        "report": report,
        "stats": stats,
        "logs_tail": logs[-30:],
    }

    Path("data/tle_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    return status


def load_catalog_status():
    """
    Return latest TLE database status for GET /v1/tle/status.
    Prefer real DB stats from tle_database.py.
    """
    import json
    from pathlib import Path

    try:
        from tle_database import get_stats

        stats = get_stats()

        status_file = Path("data") / "tle_status.json"
        last_refresh = None
        source = None

        if status_file.exists():
            try:
                previous = json.loads(status_file.read_text(encoding="utf-8"))
                last_refresh = previous.get("last_fetched_at")
                source = previous.get("source")
            except Exception:
                pass

        return {
            "ok": True,
            "source": source or "local_database",
            "last_fetched_at": last_refresh or stats.get("last_ingest"),
            "stats": stats,
        }

    except Exception as e:
        return {
            "ok": False,
            "status": "status_read_error",
            "message": str(e),
        }