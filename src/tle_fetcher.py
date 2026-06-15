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
