"""
earth_capture.py — Capture de l'image satellite temps réel depuis zoom.earth
via Playwright (navigateur headless).

zoom.earth affiche des tuiles satellite Mapbox/ESRI actualisées toutes les 10 min.
On capture un screenshot de la vue monde, on le recadre et on le compresse.

Nécessite : pip install playwright && python -m playwright install chromium
"""

import os, time, logging, base64
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_PATH  = Path("data/earth_texture_cache.jpg")
CACHE_MAX_AGE_S = 600   # 10 minutes — correspond à la fréquence de mise à jour zoom.earth


def capture_zoom_earth(force: bool = False) -> bytes | None:
    """
    Capture une image satellite de zoom.earth.
    
    Retourne les bytes JPEG de l'image (2048×1024 environ),
    ou None si la capture échoue.
    
    Le résultat est mis en cache 10 minutes pour éviter de relancer
    un navigateur à chaque requête.
    """
    # Vérifier le cache
    if not force and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_S:
            logger.info(f"Cache texture valide ({int(age)}s < {CACHE_MAX_AGE_S}s)")
            return CACHE_PATH.read_bytes()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright non installé — pip install playwright")
        return None

    logger.info("Capture zoom.earth via Playwright...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ]
            )
            page = browser.new_page(
                viewport={"width": 1920, "height": 960},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )

            # Naviguer vers zoom.earth en vue satellite globale
            page.goto(
                "https://zoom.earth/maps/satellite/#view=0,0,3z",
                wait_until="networkidle",
                timeout=30000
            )

            # Attendre que les tuiles satellite soient chargées
            page.wait_for_timeout(4000)

            # Masquer l'UI (barre de navigation, labels, etc.)
            page.evaluate("""
                () => {
                    // Masquer tous les éléments UI
                    const selectors = [
                        '.leaflet-control-container',
                        '.top-bar', '.bottom-bar',
                        '.attribution', 'header', 'footer',
                        '.ui-overlay', '.controls',
                        '[class*="control"]', '[class*="toolbar"]',
                        '[class*="legend"]', '[class*="watermark"]'
                    ];
                    selectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => {
                            el.style.display = 'none';
                        });
                    });
                }
            """)
            page.wait_for_timeout(500)

            # Screenshot de la zone de la carte uniquement
            # Chercher le canvas ou div de la carte
            map_el = page.query_selector('.leaflet-container') or \
                     page.query_selector('#map') or \
                     page.query_selector('canvas')

            if map_el:
                img_bytes = map_el.screenshot(type="jpeg", quality=85)
            else:
                # Fallback : screenshot pleine page recadré
                img_bytes = page.screenshot(
                    type="jpeg", quality=85,
                    clip={"x": 0, "y": 0, "width": 1920, "height": 960}
                )

            browser.close()

            if len(img_bytes) < 50000:
                logger.warning(f"Image trop petite ({len(img_bytes)} bytes) — capture probablement échouée")
                return None

            # Redimensionner à 2048×1024 via PIL si disponible
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((2048, 1024), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=82, optimize=True)
                img_bytes = buf.getvalue()
                logger.info(f"Image redimensionnée → {len(img_bytes)//1024} KB")
            except ImportError:
                logger.info(f"PIL non disponible, image brute {len(img_bytes)//1024} KB")

            # Sauvegarder le cache
            os.makedirs("data", exist_ok=True)
            CACHE_PATH.write_bytes(img_bytes)
            logger.info(f"Cache sauvegardé : {CACHE_PATH}")
            return img_bytes

    except Exception as e:
        logger.warning(f"Capture zoom.earth échouée : {e}")
        return None


def get_earth_texture_bytes() -> bytes | None:
    """
    Point d'entrée principal : retourne l'image satellite en JPEG.
    Essaie d'abord zoom.earth, fallback sur NASA GIBS.
    """
    # 1. Essayer le cache
    if CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_S:
            return CACHE_PATH.read_bytes()

    # 2. Capture zoom.earth
    result = capture_zoom_earth()
    if result:
        return result

    # 3. Fallback NASA GIBS
    try:
        import urllib.request
        from datetime import datetime, timedelta, timezone
        for delta in [2, 3, 4]:
            date_str = (datetime.now(timezone.utc) - timedelta(days=delta)).strftime("%Y-%m-%d")
            url = (
                "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
                "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
                f"&TIME={date_str}&BBOX=-90,-180,90,180"
                "&CRS=EPSG:4326&WIDTH=2048&HEIGHT=1024&FORMAT=image/jpeg"
            )
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            })
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
                if len(data) > 50000:
                    os.makedirs("data", exist_ok=True)
                    CACHE_PATH.write_bytes(data)
                    return data
    except Exception as e:
        logger.warning(f"NASA GIBS fallback échoué : {e}")

    return None
