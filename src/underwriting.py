"""
underwriting.py — Target-centric underwriting KPIs for one satellite.

This module answers the first Orbway demo need: given one target satellite
(NORAD ID resolved by the API layer), estimate annualized conjunction/CDM/
avoidance-maneuver indicators from a target-vs-catalog propagation.

Scope intentionally kept narrow:
- baseline orbital risk only;
- no ARGO/ICAR benefit yet;
- TLE-only, SGP4 propagation;
- closest approach per secondary object over the requested horizon.
"""

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from conjunction import (
    HARD_BODY,
    SIGMA_TLE,
    compute_Pc_Foster,
    compute_Pc_MonteCarlo,
    compute_Pc_Patera,
    propagate_all,
    _eci_to_geo,
)


MAX_PROPAGATION_STEPS = 500  # mirrors conjunction.propagate_all safety cap


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _record_key(row: dict) -> str:
    return f"{row.get('name', 'UNKNOWN')} [{row.get('norad_id', 'NA')}]"


def _hard_body_radius_km(name_a: str, name_b: str) -> float:
    pair = f"{name_a} {name_b}".lower()
    if "starlink" in pair:
        return HARD_BODY["starlink"]
    if any(x in pair for x in ["iss", "zarya"]):
        return HARD_BODY["iss"]
    if any(x in pair for x in ["debris", " deb", "r/b", "rocket body"]):
        return HARD_BODY["debris"]
    return HARD_BODY["default"]


def _combined_sigma_km(orbit_class: Optional[str]) -> float:
    sg = SIGMA_TLE.get(orbit_class or "LEO", SIGMA_TLE["LEO"])
    return math.sqrt(sg["r"] ** 2 + sg["t"] ** 2 + sg["n"] ** 2) / math.sqrt(3)


def _compute_pc(method: str, pos_a, vel_a, pos_b, vel_b, sigma: float, r_hard: float) -> float:
    if method == "patera":
        return compute_Pc_Patera(pos_a, vel_a, pos_b, vel_b, sigma, sigma, r_hard)
    if method == "montecarlo":
        return compute_Pc_MonteCarlo(pos_a, vel_a, pos_b, vel_b, sigma, sigma, r_hard)
    return compute_Pc_Foster(pos_a, vel_a, pos_b, vel_b, sigma, sigma, r_hard)


def _decision_level(
    pc: float,
    miss_km: float,
    cdm_pc_threshold: float,
    cdm_miss_distance_threshold_km: float,
    maneuver_pc_threshold: float,
    maneuver_miss_distance_threshold_km: float,
) -> str:
    if pc >= maneuver_pc_threshold or miss_km <= maneuver_miss_distance_threshold_km:
        return "maneuver"
    if pc >= cdm_pc_threshold or miss_km <= cdm_miss_distance_threshold_km:
        return "cdm"
    return "conjunction"


def _annualized_rate(count: int, effective_days: float) -> Optional[float]:
    if effective_days <= 0:
        return None
    return round(float(count) * 365.25 / effective_days, 2)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _confidence(
    target_record: dict,
    catalog_count: int,
    effective_days: float,
    requested_days: float,
    pc_method: str,
) -> dict:
    score = 45
    drivers = []

    target_epoch = _parse_datetime(target_record.get("epoch"))
    if target_epoch:
        age_days = abs((datetime.now(timezone.utc) - target_epoch).total_seconds()) / 86400
        if age_days <= 3:
            score += 15
            drivers.append("fresh target TLE epoch <= 3 days")
        elif age_days <= 14:
            score += 8
            drivers.append("target TLE epoch <= 14 days")
        else:
            score -= 10
            drivers.append("stale target TLE epoch > 14 days")
    else:
        score -= 8
        drivers.append("target TLE epoch unavailable")

    if catalog_count >= 20000:
        score += 18
        drivers.append("large screened catalog >= 20k objects")
    elif catalog_count >= 5000:
        score += 10
        drivers.append("medium screened catalog >= 5k objects")
    elif catalog_count >= 1000:
        score += 5
        drivers.append("limited screened catalog >= 1k objects")
    else:
        score -= 10
        drivers.append("small screened catalog < 1k objects")

    if effective_days >= 14:
        score += 10
        drivers.append("projection horizon >= 14 effective days")
    elif effective_days >= 7:
        score += 6
        drivers.append("projection horizon >= 7 effective days")
    else:
        score -= 5
        drivers.append("short projection horizon < 7 effective days")

    if effective_days + 1e-9 < requested_days:
        score -= 10
        drivers.append("requested horizon truncated by propagation step cap")

    if pc_method == "foster":
        score -= 5
        drivers.append("Pc uses isotropic covariance proxy, not operator covariance")
    elif pc_method == "patera":
        drivers.append("Pc computed with Patera numerical encounter-plane integration")
    elif pc_method == "montecarlo":
        score += 3
        drivers.append("Pc computed with Monte-Carlo sampling")

    score = max(0, min(100, score))
    return {"score": score, "drivers": drivers}


def run_target_risk_analysis(
    target_record: dict,
    catalog_records: list,
    horizon_days: float = 7.0,
    step_min: float = 30.0,
    screening_miss_distance_threshold_km: float = 10.0,
    cdm_pc_threshold: float = 1e-7,
    cdm_miss_distance_threshold_km: float = 5.0,
    maneuver_pc_threshold: float = 1e-4,
    maneuver_miss_distance_threshold_km: float = 1.0,
    pc_method: str = "foster",
    max_events: int = 50,
) -> dict:
    """
    Estimate underwriting KPIs for one target satellite.

    The function propagates the target and all candidate secondary objects over
    the requested horizon, keeps the closest approach per secondary object, then
    derives baseline annualized indicators.
    """
    target_norad = str(target_record.get("norad_id", "")).strip()
    if not target_norad:
        raise ValueError("target_record must include norad_id")

    requested_hours = float(horizon_days) * 24.0
    dt_s = float(step_min) * 60.0
    requested_steps = int(requested_hours * 3600.0 / max(dt_s, 1.0)) + 1
    expected_steps = min(requested_steps, MAX_PROPAGATION_STEPS)
    effective_days = max(0.0, ((expected_steps - 1) * float(step_min)) / (60.0 * 24.0))

    target_key = _record_key(target_record)
    records_by_key = {target_key: target_record}
    tles = [(target_key, target_record["tle1"], target_record["tle2"])]

    for row in catalog_records:
        norad = str(row.get("norad_id", "")).strip()
        if not norad or norad == target_norad:
            continue
        if not row.get("tle1") or not row.get("tle2"):
            continue
        key = _record_key(row)
        if key in records_by_key:
            continue
        records_by_key[key] = row
        tles.append((key, row["tle1"], row["tle2"]))

    if len(tles) < 2:
        raise ValueError("not enough catalog objects to run target risk analysis")

    states = propagate_all(
        tles,
        hours=requested_hours,
        step_min=step_min,
        pert_flags=None,
        emit=None,
    )

    target_state = states.get(target_key)
    if not target_state:
        raise ValueError(f"target propagation failed for NORAD {target_norad}")

    target_pos = target_state["pos_km"]
    target_vel = target_state["vel_km_s"]
    times = target_state["times"]
    actual_steps = len(times)
    actual_effective_days = max(0.0, ((actual_steps - 1) * float(step_min)) / (60.0 * 24.0))

    events = []

    for key, state in states.items():
        if key == target_key:
            continue
        pos = state["pos_km"]
        vel = state["vel_km_s"]
        n = min(len(pos), len(target_pos), len(times))
        if n <= 0:
            continue

        deltas = pos[:n] - target_pos[:n]
        distances = np.linalg.norm(deltas, axis=1)
        t_idx = int(np.argmin(distances))
        miss_km = float(distances[t_idx])

        if miss_km > screening_miss_distance_threshold_km:
            continue

        secondary_record = records_by_key.get(key, {})
        orbit_class = target_state.get("orbit_class") or state.get("orbit_class") or target_record.get("orbit_class")
        sigma = _combined_sigma_km(orbit_class)
        r_hard = _hard_body_radius_km(target_key, key)
        pc = _compute_pc(
            pc_method,
            target_pos[t_idx],
            target_vel[t_idx],
            pos[t_idx],
            vel[t_idx],
            sigma,
            r_hard,
        )
        v_rel = float(np.linalg.norm(target_vel[t_idx] - vel[t_idx]))
        t_ca = times[t_idx]
        pos_ca = (target_pos[t_idx] + pos[t_idx]) / 2.0
        lat, lon, alt = _eci_to_geo(pos_ca, t_ca)
        level = _decision_level(
            pc,
            miss_km,
            cdm_pc_threshold,
            cdm_miss_distance_threshold_km,
            maneuver_pc_threshold,
            maneuver_miss_distance_threshold_km,
        )

        events.append(
            {
                "id": f"{target_norad}_{state.get('norad', secondary_record.get('norad_id', 'NA'))}_{t_idx:04d}",
                "target_norad": target_norad,
                "target_name": target_record.get("name"),
                "secondary_norad": str(state.get("norad", secondary_record.get("norad_id", ""))),
                "secondary_name": secondary_record.get("name", key),
                "t_ca": t_ca.isoformat(),
                "t_ca_h": round((t_ca - times[0]).total_seconds() / 3600.0, 2),
                "miss_dist_km": round(miss_km, 4),
                "v_rel_km_s": round(v_rel, 3),
                "Pc": pc,
                "Pc_str": f"{pc:.2e}",
                "pc_method": pc_method,
                "decision_level": level,
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "alt_km": round(alt, 1),
            }
        )

    risk_order = {"maneuver": 0, "cdm": 1, "conjunction": 2}
    events.sort(
        key=lambda e: (
            risk_order.get(e["decision_level"], 9),
            -float(e.get("Pc", 0.0)),
            float(e.get("miss_dist_km", 999999.0)),
        )
    )

    conjunction_count = len(events)
    cdm_count = sum(1 for e in events if e["decision_level"] in ("cdm", "maneuver"))
    maneuver_count = sum(1 for e in events if e["decision_level"] == "maneuver")
    max_pc = max((float(e["Pc"]) for e in events), default=0.0)
    min_miss = min((float(e["miss_dist_km"]) for e in events), default=None)

    return {
        "target": {
            "norad_id": target_norad,
            "name": target_record.get("name"),
            "epoch": target_record.get("epoch"),
            "orbit_class": target_record.get("orbit_class"),
            "alt_km": _safe_float(target_record.get("alt_km")),
            "inc": _safe_float(target_record.get("inc")),
            "source": target_record.get("source"),
        },
        "analysis_window": {
            "requested_days": float(horizon_days),
            "effective_days": round(actual_effective_days, 3),
            "step_min": float(step_min),
            "requested_steps": requested_steps,
            "propagated_steps": actual_steps,
            "truncated_by_step_cap": actual_effective_days + 1e-9 < float(horizon_days),
        },
        "catalog": {
            "candidate_objects": len(tles) - 1,
            "propagated_objects": max(0, len(states) - 1),
            "excluded_target_norad": target_norad,
        },
        "thresholds": {
            "screening_miss_distance_threshold_km": screening_miss_distance_threshold_km,
            "cdm_pc_threshold": cdm_pc_threshold,
            "cdm_miss_distance_threshold_km": cdm_miss_distance_threshold_km,
            "maneuver_pc_threshold": maneuver_pc_threshold,
            "maneuver_miss_distance_threshold_km": maneuver_miss_distance_threshold_km,
        },
        "kpis": {
            "conjunction_events": {
                "period_count": conjunction_count,
                "annualized_rate": _annualized_rate(conjunction_count, actual_effective_days),
                "definition": "unique secondary objects whose closest approach to the target is within the screening miss-distance threshold",
            },
            "cdm_equivalent_alerts": {
                "period_count": cdm_count,
                "annualized_rate": _annualized_rate(cdm_count, actual_effective_days),
                "definition": "screened events exceeding the CDM Pc threshold or CDM miss-distance threshold",
            },
            "avoidance_maneuvers": {
                "period_count": maneuver_count,
                "annualized_rate": _annualized_rate(maneuver_count, actual_effective_days),
                "definition": "screened events exceeding the maneuver Pc threshold or maneuver miss-distance threshold",
            },
            "max_pc": max_pc,
            "max_pc_str": f"{max_pc:.2e}",
            "min_miss_distance_km": min_miss,
        },
        "top_events": events[:max_events],
        "confidence": _confidence(
            target_record=target_record,
            catalog_count=max(0, len(states) - 1),
            effective_days=actual_effective_days,
            requested_days=float(horizon_days),
            pc_method=pc_method,
        ),
        "methodology": {
            "model": "target-vs-catalog SGP4 propagation from local TLE database",
            "event_grouping": "one closest-approach event per secondary object over the analysis window",
            "annualization": "period_count * 365.25 / effective_days",
            "baseline_only": True,
        },
        "limitations": [
            "TLE-only propagation; no operator ephemerides or covariance messages ingested yet.",
            "Pc uses simplified isotropic covariance proxies derived from orbit class.",
            "CDM and maneuver counts are threshold-derived CDM-equivalent indicators, not official 18 SDS CDM messages.",
            "No ARGO risk-reduction effect is applied in this baseline endpoint.",
        ],
    }
