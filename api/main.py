# api/main.py
import os
import sys
import time
import math
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make src/ and scripts/ importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from conjunction import run_conjunction_analysis, propagate_all, RE_KM  # noqa: E402
from tle_fetcher import fetch_and_store, load_catalog_status, load_refresh_progress
from tle_database import lookup_tle, get_latest_tles

app = FastAPI(
    title="Orbitways Insurer API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for prototype. Tighten later.
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

API_KEY = os.environ.get("ORBITWAYS_API_KEY")


def _check_auth(authorization: Optional[str]):
    """
    Shared bearer-token check.

    If ORBITWAYS_API_KEY is not set, auth is disabled.
    This is convenient for local dev, but set ORBITWAYS_API_KEY for demos.
    """
    if not API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1]

    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


class AssessmentRequest(BaseModel):
    constellation: str = Field("starlink", max_length=64)
    hours: float = Field(24, ge=1, le=168)
    step_min: float = Field(5, ge=1, le=60)
    threshold_km: float = Field(5, ge=0.1, le=50)
    max_results: int = Field(50, ge=1, le=500)
    pc_method: Literal["foster", "patera", "montecarlo"] = "foster"


@app.get("/")
def root():
    return {
        "service": "Orbitways Insurer API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    spacetrack_configured = bool(
        os.environ.get("SPACETRACK_EMAIL")
        or os.environ.get("SPACETRACK_USER")
    )

    return {
        "status": "ok",
        "spacetrack": spacetrack_configured,
        "auth_enabled": bool(API_KEY),
    }


@app.post("/v1/assessments")
def assessments(
    req: AssessmentRequest,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    t0 = time.time()

    try:
        conjunctions = run_conjunction_analysis(
            constellation=req.constellation,
            hours=req.hours,
            step_min=req.step_min,
            threshold_km=req.threshold_km,
            max_results=req.max_results,
            pc_method=req.pc_method,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")

    max_pc = max((c.get("Pc", 0.0) for c in conjunctions), default=0.0)

    source = (
        "spacetrack"
        if os.environ.get("SPACETRACK_EMAIL") or os.environ.get("SPACETRACK_USER")
        else "celestrak"
    )

    sat_a = {c.get("sat_A") for c in conjunctions if c.get("sat_A")}
    sat_b = {c.get("sat_B") for c in conjunctions if c.get("sat_B")}

    return {
        "ok": True,
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_satellites": len(sat_a | sat_b),
        "n_conjunctions": len(conjunctions),
        "max_pc": max_pc,
        "elapsed_s": round(time.time() - t0, 2),
        "conjunctions": conjunctions,
    }


@app.post("/v1/tle/refresh")
def refresh_tle(
    bg: BackgroundTasks,
    group: str = "starlink",
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    bg.add_task(fetch_and_store, group=group)

    return {
        "ok": True,
        "group": group,
        "status": "refresh_started",
    }


@app.get("/v1/tle/status")
def tle_status():
    return load_catalog_status()

@app.get("/v1/tle/refresh/progress")
def tle_refresh_progress():
    return load_refresh_progress()    

@app.get("/v1/tle/lookup")
def tle_lookup(
    q: str,
    limit: int = 10,
):
    """
    Lookup latest TLE records by NORAD ID or satellite name.

    Used by the orbit visualization frontend.
    """
    q = q.strip()

    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter q")

    try:
        results = lookup_tle(q=q, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TLE lookup failed: {e}")

    if not results:
        raise HTTPException(status_code=404, detail=f"No TLE found for query: {q}")

    return {
        "ok": True,
        "query": q,
        "count": len(results),
        "results": results,
    }

def _vec3(values):
    return [float(values[0]), float(values[1]), float(values[2])]


def _norm_km(values):
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _float_or_none(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _period_min_from_mean_motion(mm):
    mm = _float_or_none(mm)

    if not mm or mm <= 0:
        return None

    return round(1440.0 / mm, 2)


def _scene_key(row):
    return f"{row.get('norad_id')}::{row.get('name', '')}"


def _guess_object_type(name: str | None):
    """
    Best-effort classification.

    The current DB does not store Space-Track OBJECT_TYPE directly, so this
    is only a display heuristic. If OBJECT_TYPE is added later to the DB,
    replace this function with the real catalog field.
    """
    n = (name or "").upper()

    if "DEB" in n or "DEBRIS" in n:
        return "debris"

    if "R/B" in n or "ROCKET BODY" in n or "ROCKET" in n:
        return "rocket_body"

    return "payload_active"

@app.get("/v1/leo/scene")
def leo_scene(
    selected_norad: Optional[str] = None,
    max_objects: int = 30000,
):
    """
    Build the real-time 3D LEO scene for the frontend.

    This endpoint returns:
    - one current propagated position per LEO object
    - optional selected satellite metadata
    - optional selected satellite orbit points

    The frontend renders the 3D scene; the backend provides the physics data.
    """
    max_objects = max(1, min(int(max_objects), 30000))
    selected_norad = (selected_norad or "").strip()

    try:
        records = get_latest_tles(limit=max_objects, orbit_class="LEO")

        selected_record = None
        selected_error = None

        if selected_norad:
            selected_matches = lookup_tle(q=selected_norad, limit=1)
            selected_record = selected_matches[0] if selected_matches else None

            if selected_record:
                existing_norads = {row["norad_id"] for row in records}

                if selected_record["norad_id"] not in existing_norads:
                    records.append(selected_record)
            else:
                selected_error = f"No TLE found for NORAD ID {selected_norad}"

        if not records:
            raise HTTPException(
                status_code=404,
                detail="No LEO TLE records available. Refresh the TLE database first.",
            )

        keyed_records = []
        seen_keys = set()

        for row in records:
            key = _scene_key(row)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            keyed_records.append((key, row))

        # Current cloud position: use one SGP4 sample only.
        # Very short horizon + 1-minute step gives one current propagated point.
        cloud_tles = [
            (key, row["tle1"], row["tle2"])
            for key, row in keyed_records
        ]

        cloud_states = propagate_all(
            cloud_tles,
            hours=0.001,
            step_min=1.0,
            pert_flags=None,
            emit=None,
        )

        objects = []

        for key, row in keyed_records:
            state = cloud_states.get(key)

            if not state:
                continue

            pos0 = state["pos_km"][0]
            alt_now = round(_norm_km(pos0) - RE_KM, 1)

            objects.append(
                {
                    "norad_id": str(row["norad_id"]),
                    "name": row["name"],
                    "object_type": _guess_object_type(row.get("name")),
                    "orbit_class": row.get("orbit_class") or state.get("orbit_class"),
                    "alt_km": alt_now,
                    "position_km": _vec3(pos0),
                }
            )

        selected_payload = None

        if selected_record:
            selected_key = _scene_key(selected_record)

            selected_states = propagate_all(
                [(selected_key, selected_record["tle1"], selected_record["tle2"])],
                hours=2.0,
                step_min=2.0,
                pert_flags=None,
                emit=None,
            )

            selected_state = selected_states.get(selected_key)

            if selected_state:
                pos0 = selected_state["pos_km"][0]
                vel0 = selected_state["vel_km_s"][0]

                selected_payload = {
                    "norad_id": str(selected_record["norad_id"]),
                    "name": selected_record["name"],
                    "tle1": selected_record["tle1"],
                    "tle2": selected_record["tle2"],
                    "epoch": selected_record.get("epoch"),
                    "orbit_class": selected_record.get("orbit_class") or selected_state.get("orbit_class"),
                    "alt_km": round(_norm_km(pos0) - RE_KM, 1),
                    "catalog_alt_km": _float_or_none(selected_record.get("alt_km")),
                    "inc": _float_or_none(selected_record.get("inc")),
                    "ecc": _float_or_none(selected_record.get("ecc")),
                    "mm": _float_or_none(selected_record.get("mm")),
                    "period_min": _period_min_from_mean_motion(selected_record.get("mm")),
                    "current_position_km": _vec3(pos0),
                    "current_velocity_km_s": _vec3(vel0),
                    "orbit_points_km": [
                        _vec3(point)
                        for point in selected_state["pos_km"]
                    ],
                }
            else:
                selected_error = f"Propagation failed for NORAD ID {selected_norad}"

        return {
            "ok": True,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "frame": "ECI",
            "earth_radius_km": RE_KM,
            "total_objects": len(records),
            "rendered_objects": len(objects),
            "objects": objects,
            "selected": selected_payload,
            "selected_error": selected_error,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LEO scene generation failed: {e}",
        )