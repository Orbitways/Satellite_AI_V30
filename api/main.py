# api/main.py
import os
import sys
import time
import math
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from underwriting import run_target_risk_analysis, select_orbital_environment_catalog
from historical_underwriting import run_historical_target_risk
from cdm_database import (
    cdm_status,
    ingest_cdm_records,
    parse_cdm_csv,
    run_historical_cdm_analysis,
)

app = FastAPI(title="Orbitways Insurer API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

API_KEY = os.environ.get("ORBITWAYS_API_KEY")


def _check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


def _parse_catalog_orbit_class(value: Optional[str]):
    raw_class = (value or "").strip().upper()
    if raw_class in ("", "ALL", "ANY", "NONE", "NULL"):
        return None
    if raw_class in ("LEO", "MEO", "GEO", "HEO"):
        return raw_class
    raise HTTPException(status_code=400, detail="catalog_orbit_class must be one of LEO, MEO, GEO, HEO or ALL")


class AssessmentRequest(BaseModel):
    constellation: str = Field("starlink", max_length=64)
    hours: float = Field(24, ge=1, le=168)
    step_min: float = Field(5, ge=1, le=60)
    threshold_km: float = Field(5, ge=0.1, le=50)
    max_results: int = Field(50, ge=1, le=500)
    pc_method: Literal["foster", "patera", "montecarlo"] = "foster"


class TargetRiskRequest(BaseModel):
    target_norad: str = Field(..., min_length=1, max_length=10)
    horizon_days: float = Field(7, ge=0.25, le=30)
    step_min: float = Field(30, ge=1, le=180)
    screening_miss_distance_threshold_km: float = Field(10, ge=0.1, le=100)
    cdm_pc_threshold: float = Field(1e-7, gt=0, le=1)
    cdm_miss_distance_threshold_km: float = Field(5, ge=0.1, le=100)
    maneuver_pc_threshold: float = Field(1e-4, gt=0, le=1)
    maneuver_miss_distance_threshold_km: float = Field(1, ge=0.01, le=100)
    catalog_orbit_class: Optional[str] = Field("LEO", max_length=8)

    altitude_band_km: Optional[float] = Field(300, ge=10, le=5000)
    inclination_band_deg: Optional[float] = Field(20, ge=0, le=180)
    include_debris: bool = True
    include_inactive_satellites: bool = True
    include_active_satellites: bool = True
    include_crossing_orbits: bool = True

    max_catalog_objects: int = Field(30000, ge=100, le=50000)
    max_events: int = Field(50, ge=1, le=500)
    pc_method: Literal["foster", "patera", "montecarlo"] = "foster"


class HistoricalTargetRiskRequest(BaseModel):
    target_norad: str = Field(..., min_length=1, max_length=10)
    lookback_days: float = Field(90, ge=1, le=365)
    bucket_days: float = Field(7, ge=1, le=30)
    step_min: float = Field(60, ge=5, le=360)
    screening_miss_distance_threshold_km: float = Field(10, ge=0.1, le=100)
    cdm_pc_threshold: float = Field(1e-7, gt=0, le=1)
    cdm_miss_distance_threshold_km: float = Field(5, ge=0.1, le=100)
    maneuver_pc_threshold: float = Field(1e-4, gt=0, le=1)
    maneuver_miss_distance_threshold_km: float = Field(1, ge=0.01, le=100)
    catalog_orbit_class: Optional[str] = Field("LEO", max_length=8)
    max_catalog_objects: int = Field(8000, ge=100, le=50000)
    max_tle_age_days: float = Field(14, ge=1, le=90)
    altitude_band_km: Optional[float] = Field(300, ge=10, le=2000)
    max_events_per_bucket: int = Field(20, ge=1, le=200)
    pc_method: Literal["foster", "patera", "montecarlo"] = "foster"


class CdmImportRequest(BaseModel):
    source: str = Field("manual", max_length=128)
    records: Optional[list[dict[str, Any]]] = None
    csv_text: Optional[str] = None


class HistoricalCdmRequest(BaseModel):
    target_norad: str = Field(..., min_length=1, max_length=10)
    lookback_days: float = Field(365, ge=1, le=3650)
    bucket_days: float = Field(30, ge=1, le=365)
    time_axis: Literal["tca", "creation_date"] = "tca"
    cdm_pc_threshold: float = Field(1e-7, gt=0, le=1)
    cdm_miss_distance_threshold_km: float = Field(5, ge=0.001, le=1000)
    maneuver_pc_threshold: float = Field(1e-4, gt=0, le=1)
    maneuver_miss_distance_threshold_km: float = Field(1, ge=0.001, le=1000)
    max_events: int = Field(200, ge=1, le=2000)


@app.get("/")
def root():
    return {"service": "Orbitways Insurer API", "version": "0.6.0", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    spacetrack_configured = bool(os.environ.get("SPACETRACK_EMAIL") or os.environ.get("SPACETRACK_USER"))
    return {"status": "ok", "spacetrack": spacetrack_configured, "auth_enabled": bool(API_KEY)}


@app.post("/v1/assessments")
def assessments(req: AssessmentRequest, authorization: Optional[str] = Header(None)):
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
    source = "spacetrack" if os.environ.get("SPACETRACK_EMAIL") or os.environ.get("SPACETRACK_USER") else "celestrak"
    sat_a = {c.get("sat_A") for c in conjunctions if c.get("sat_A")}
    sat_b = {c.get("sat_B") for c in conjunctions if c.get("sat_B")}
    return {"ok": True, "source": source, "fetched_at": datetime.now(timezone.utc).isoformat(), "n_satellites": len(sat_a | sat_b), "n_conjunctions": len(conjunctions), "max_pc": max_pc, "elapsed_s": round(time.time() - t0, 2), "conjunctions": conjunctions}


@app.post("/v1/underwriting/target-risk")
def underwriting_target_risk(req: TargetRiskRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = time.time()
    target_norad = (req.target_norad or "").strip()
    if not target_norad.isdigit():
        raise HTTPException(status_code=400, detail="target_norad must be numeric")

    catalog_orbit_class = _parse_catalog_orbit_class(req.catalog_orbit_class)

    try:
        target_matches = lookup_tle(q=target_norad, limit=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"target TLE lookup failed: {e}")
    if not target_matches:
        raise HTTPException(status_code=404, detail=f"No TLE found for target NORAD ID {target_norad}. Refresh the TLE database first.")

    try:
        raw_catalog_records = get_latest_tles(limit=req.max_catalog_objects, orbit_class=catalog_orbit_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"catalog TLE lookup failed: {e}")
    if not raw_catalog_records:
        raise HTTPException(status_code=404, detail="No catalog TLE records available. Refresh the TLE database first.")

    try:
        selected_catalog_records, environment_report = select_orbital_environment_catalog(
            target_record=target_matches[0],
            catalog_records=raw_catalog_records,
            altitude_band_km=req.altitude_band_km,
            inclination_band_deg=req.inclination_band_deg,
            include_debris=req.include_debris,
            include_inactive_satellites=req.include_inactive_satellites,
            include_active_satellites=req.include_active_satellites,
            include_crossing_orbits=req.include_crossing_orbits,
            max_selected_objects=req.max_catalog_objects,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"orbital environment selection failed: {e}")

    if not selected_catalog_records:
        raise HTTPException(status_code=404, detail="No candidate objects found in the selected orbital environment. Widen altitude/inclination bands or include more object classes.")

    try:
        result = run_target_risk_analysis(
            target_record=target_matches[0],
            catalog_records=selected_catalog_records,
            horizon_days=req.horizon_days,
            step_min=req.step_min,
            screening_miss_distance_threshold_km=req.screening_miss_distance_threshold_km,
            cdm_pc_threshold=req.cdm_pc_threshold,
            cdm_miss_distance_threshold_km=req.cdm_miss_distance_threshold_km,
            maneuver_pc_threshold=req.maneuver_pc_threshold,
            maneuver_miss_distance_threshold_km=req.maneuver_miss_distance_threshold_km,
            pc_method=req.pc_method,
            max_events=req.max_events,
            environment_selection=environment_report,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"target risk analysis failed: {e}")

    result["catalog"]["orbit_class_filter"] = catalog_orbit_class or "ALL"
    result["catalog"]["catalog_compute_cap"] = req.max_catalog_objects
    result["catalog"]["note"] = "Catalog selection is based on the target orbital environment. catalog_compute_cap is an internal performance cap, not a risk-model input."

    return {"ok": True, "source": "local_tle_database", "fetched_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": round(time.time() - t0, 2), **result}


@app.post("/v1/underwriting/historical-target-risk")
def underwriting_historical_target_risk(req: HistoricalTargetRiskRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = time.time()
    target_norad = (req.target_norad or "").strip()
    if not target_norad.isdigit():
        raise HTTPException(status_code=400, detail="target_norad must be numeric")

    catalog_orbit_class = _parse_catalog_orbit_class(req.catalog_orbit_class)
    try:
        result = run_historical_target_risk(
            target_norad=target_norad,
            lookback_days=req.lookback_days,
            bucket_days=req.bucket_days,
            step_min=req.step_min,
            screening_miss_distance_threshold_km=req.screening_miss_distance_threshold_km,
            cdm_pc_threshold=req.cdm_pc_threshold,
            cdm_miss_distance_threshold_km=req.cdm_miss_distance_threshold_km,
            maneuver_pc_threshold=req.maneuver_pc_threshold,
            maneuver_miss_distance_threshold_km=req.maneuver_miss_distance_threshold_km,
            catalog_orbit_class=catalog_orbit_class,
            max_catalog_objects=req.max_catalog_objects,
            max_tle_age_days=req.max_tle_age_days,
            altitude_band_km=req.altitude_band_km,
            pc_method=req.pc_method,
            max_events_per_bucket=req.max_events_per_bucket,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"historical target risk replay failed: {e}")
    return {"ok": True, "source": "local_tle_history", "fetched_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": round(time.time() - t0, 2), **result}


@app.post("/v1/cdm/import")
def cdm_import(req: CdmImportRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    records: list[dict[str, Any]] = []
    if req.records:
        records.extend(req.records)
    if req.csv_text:
        records.extend(parse_cdm_csv(req.csv_text))
    if not records:
        raise HTTPException(status_code=400, detail="Provide either records[] or csv_text containing CDM records")
    try:
        report = ingest_cdm_records(records, source=req.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDM import failed: {e}")
    return report


@app.get("/v1/cdm/status")
def cdm_database_status(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    try:
        return cdm_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDM status failed: {e}")


@app.post("/v1/underwriting/historical-cdms")
def underwriting_historical_cdms(req: HistoricalCdmRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = time.time()
    target_norad = (req.target_norad or "").strip()
    if not target_norad.isdigit():
        raise HTTPException(status_code=400, detail="target_norad must be numeric")
    try:
        result = run_historical_cdm_analysis(
            target_norad=target_norad,
            lookback_days=req.lookback_days,
            bucket_days=req.bucket_days,
            time_axis=req.time_axis,
            cdm_pc_threshold=req.cdm_pc_threshold,
            cdm_miss_distance_threshold_km=req.cdm_miss_distance_threshold_km,
            maneuver_pc_threshold=req.maneuver_pc_threshold,
            maneuver_miss_distance_threshold_km=req.maneuver_miss_distance_threshold_km,
            max_events=req.max_events,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"historical CDM analysis failed: {e}")
    return {"ok": True, "source": "imported_cdm_records", "fetched_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": round(time.time() - t0, 2), **result}


@app.post("/v1/tle/refresh")
def refresh_tle(bg: BackgroundTasks, group: str = "starlink", authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    bg.add_task(fetch_and_store, group=group)
    return {"ok": True, "group": group, "status": "refresh_started"}


@app.get("/v1/tle/status")
def tle_status():
    return load_catalog_status()


@app.get("/v1/tle/refresh/progress")
def tle_refresh_progress():
    return load_refresh_progress()


@app.get("/v1/tle/lookup")
def tle_lookup(q: str, limit: int = 10):
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter q")
    try:
        results = lookup_tle(q=q, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TLE lookup failed: {e}")
    if not results:
        raise HTTPException(status_code=404, detail=f"No TLE found for query: {q}")
    return {"ok": True, "query": q, "count": len(results), "results": results}


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


def _clean(value):
    return str(value or "").strip()


def _scene_object_type(row: dict) -> tuple[str, str]:
    """
    Return a UI-oriented scene object type and classification source.

    Values intentionally match the frontend legend:
    - payload_active
    - payload_inactive
    - debris
    - rocket_body
    """
    object_type = _clean(row.get("object_type")).upper()
    decay_date = _clean(row.get("decay_date"))
    ops_status = _clean(row.get("ops_status_code")).upper()

    if object_type:
        if "DEBRIS" in object_type:
            return "debris", "spacetrack_satcat_object_type"
        if "ROCKET" in object_type or "R/B" in object_type:
            return "rocket_body", "spacetrack_satcat_object_type"
        if "PAYLOAD" in object_type:
            if decay_date or ops_status in {"-", "D"}:
                return "payload_inactive", "spacetrack_satcat_payload_status"
            return "payload_active", "spacetrack_satcat_payload_status"

    name = _clean(row.get("meta_object_name") or row.get("name") or row.get("object_name")).upper()
    if "DEB" in name or "DEBRIS" in name:
        return "debris", "name_heuristic"
    if "R/B" in name or "ROCKET BODY" in name or "ROCKET" in name:
        return "rocket_body", "name_heuristic"
    if "OBJECT" in name:
        return "payload_inactive", "name_heuristic"
    return "payload_active", "name_heuristic"


def _scene_metadata(row: dict):
    return {
        "object_type": row.get("object_type"),
        "rcs_size": row.get("rcs_size"),
        "country": row.get("country"),
        "launch_date": row.get("launch_date"),
        "decay_date": row.get("decay_date"),
        "ops_status_code": row.get("ops_status_code"),
        "metadata_source": row.get("metadata_source"),
        "metadata_updated_at": row.get("metadata_updated_at"),
    }


def _empty_scene_type_counts():
    return {
        "payload_active": 0,
        "payload_inactive": 0,
        "debris": 0,
        "rocket_body": 0,
    }


@app.get("/v1/leo/scene")
def leo_scene(selected_norad: Optional[str] = None, max_objects: int = 30000):
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
            raise HTTPException(status_code=404, detail="No LEO TLE records available. Refresh the TLE database first.")

        keyed_records = []
        seen_keys = set()
        for row in records:
            key = _scene_key(row)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            keyed_records.append((key, row))

        cloud_tles = [(key, row["tle1"], row["tle2"]) for key, row in keyed_records]
        state_epoch_unix = time.time()
        cloud_states = propagate_all(cloud_tles, hours=0.001, step_min=1.0, pert_flags=None, emit=None)

        objects = []
        object_type_counts = _empty_scene_type_counts()
        classification_method_counts = {}
        metadata_available = 0

        for key, row in keyed_records:
            state = cloud_states.get(key)
            if not state:
                continue
            pos0 = state["pos_km"][0]
            vel0 = state["vel_km_s"][0]
            alt_now = round(_norm_km(pos0) - RE_KM, 1)
            object_type, object_type_source = _scene_object_type(row)
            object_type_counts[object_type] = object_type_counts.get(object_type, 0) + 1
            classification_method_counts[object_type_source] = classification_method_counts.get(object_type_source, 0) + 1
            if row.get("object_type") or row.get("metadata_source"):
                metadata_available += 1

            objects.append({
                "norad_id": str(row["norad_id"]),
                "name": row.get("meta_object_name") or row["name"],
                "object_type": object_type,
                "object_type_source": object_type_source,
                "metadata": _scene_metadata(row),
                "orbit_class": row.get("orbit_class") or state.get("orbit_class"),
                "alt_km": alt_now,
                "position_km": _vec3(pos0),
                "velocity_km_s": _vec3(vel0),
            })

        selected_payload = None
        if selected_record:
            selected_key = _scene_key(selected_record)
            selected_states = propagate_all([(selected_key, selected_record["tle1"], selected_record["tle2"])], hours=2.0, step_min=2.0, pert_flags=None, emit=None)
            selected_state = selected_states.get(selected_key)
            if selected_state:
                pos0 = selected_state["pos_km"][0]
                vel0 = selected_state["vel_km_s"][0]
                object_type, object_type_source = _scene_object_type(selected_record)
                selected_payload = {
                    "norad_id": str(selected_record["norad_id"]),
                    "name": selected_record.get("meta_object_name") or selected_record["name"],
                    "object_type": object_type,
                    "object_type_source": object_type_source,
                    "metadata": _scene_metadata(selected_record),
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
                    "state_epoch_unix": state_epoch_unix,
                    "orbit_points_km": [_vec3(point) for point in selected_state["pos_km"]],
                }
            else:
                selected_error = f"Propagation failed for NORAD ID {selected_norad}"

        return {
            "ok": True,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "state_epoch_unix": state_epoch_unix,
            "frame": "ECI",
            "earth_radius_km": RE_KM,
            "total_objects": len(records),
            "rendered_objects": len(objects),
            "metadata_available_objects": metadata_available,
            "object_type_counts": object_type_counts,
            "classification_method_counts": classification_method_counts,
            "classification_method": "Space-Track/SATCAT metadata when available; name heuristic fallback for objects without metadata",
            "objects": objects,
            "selected": selected_payload,
            "selected_error": selected_error,
            "motion": {
                "model": "linear_velocity_interpolation_between_backend_sgp4_snapshots",
                "velocity_units": "km/s",
                "position_units": "km",
                "recommended_refresh_s": 10,
                "max_client_interpolation_s": 30,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LEO scene generation failed: {e}")
