# api/main.py
import os
import sys
import time
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

from conjunction import run_conjunction_analysis  # noqa: E402
from tle_fetcher import fetch_and_store, load_catalog_status, load_refresh_progress

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
