from datetime import datetime, timedelta, timezone
import math, sqlite3, os
import numpy as np
from sgp4.api import Satrec, jday

from tle_database import DB_PATH, init_db
from conjunction import SIGMA_TLE, HARD_BODY, compute_Pc_Foster, compute_Pc_Patera, compute_Pc_MonteCarlo

MAX_STEPS_PER_BUCKET = 500


def _conn():
    init_db()
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _as_dt(s):
    if not s:
        return None
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _obj_class(name):
    n = (name or "").upper()
    if "DEB" in n or "DEBRIS" in n:
        return "debris"
    if "R/B" in n or "ROCKET" in n or "OBJECT" in n:
        return "inactive_satellite"
    return "active_satellite"


def _sigma(orbit_class):
    s = SIGMA_TLE.get(orbit_class or "LEO", SIGMA_TLE["LEO"])
    return math.sqrt(s["r"] ** 2 + s["t"] ** 2 + s["n"] ** 2) / math.sqrt(3)


def _rhard(a, b):
    p = f"{a or ''} {b or ''}".lower()
    if "starlink" in p:
        return HARD_BODY["starlink"]
    if "iss" in p or "zarya" in p:
        return HARD_BODY["iss"]
    if "deb" in p or "r/b" in p or "rocket" in p:
        return HARD_BODY["debris"]
    return HARD_BODY["default"]


def _pc(method, pa, va, pb, vb, sig, rh):
    if method == "patera":
        return compute_Pc_Patera(pa, va, pb, vb, sig, sig, rh)
    if method == "montecarlo":
        return compute_Pc_MonteCarlo(pa, va, pb, vb, sig, sig, rh)
    return compute_Pc_Foster(pa, va, pb, vb, sig, sig, rh)


def _level(pc, miss, cdm_pc, cdm_miss, man_pc, man_miss):
    if pc >= man_pc or miss <= man_miss:
        return "maneuver"
    if pc >= cdm_pc or miss <= cdm_miss:
        return "cdm"
    return "conjunction"


def _series(tle1, tle2, times):
    sat = Satrec.twoline2rv(tle1, tle2)
    pos, vel = [], []
    for t in times:
        u = t.astimezone(timezone.utc).replace(tzinfo=None)
        jd, fr = jday(u.year, u.month, u.day, u.hour, u.minute, u.second + u.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e:
            return None, None
        pos.append(r); vel.append(v)
    return np.array(pos), np.array(vel)


def _target_before(c, norad, snap):
    r = c.execute("""
        SELECT norad_id,name,tle1,tle2,epoch,alt_km,orbit_class,source
        FROM tle_records WHERE norad_id=? AND epoch<=?
        ORDER BY epoch DESC LIMIT 1
    """, (str(norad), snap.isoformat())).fetchone()
    return dict(r) if r else None


def _catalog_before(c, snap, target_norad, orbit_class, limit, max_age_days, target_alt, altitude_band):
    cutoff = (snap - timedelta(days=float(max_age_days))).isoformat()
    params = [snap.isoformat(), cutoff]
    where = "WHERE epoch<=? AND epoch>=?"
    if orbit_class:
        where += " AND orbit_class=?"
        params.append(orbit_class)
    rows = c.execute(f"""
        SELECT r.norad_id,r.name,r.tle1,r.tle2,r.epoch,r.alt_km,r.orbit_class,r.source
        FROM tle_records r JOIN (
          SELECT norad_id, MAX(epoch) latest_epoch FROM tle_records
          {where} GROUP BY norad_id ORDER BY norad_id LIMIT ?
        ) x ON r.norad_id=x.norad_id AND r.epoch=x.latest_epoch
        WHERE r.norad_id!=? ORDER BY r.norad_id
    """, (*params, int(limit), str(target_norad))).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        if target_alt is not None and altitude_band is not None and d.get("alt_km") is not None:
            if abs(float(d["alt_km"]) - float(target_alt)) > float(altitude_band):
                continue
        out.append(d)
    return out


def _times(start, days, step_min):
    n = min(int(float(days) * 1440 / float(step_min)) + 1, MAX_STEPS_PER_BUCKET)
    return [start + timedelta(minutes=i * float(step_min)) for i in range(n)]


def _annualized(n, days):
    return None if days <= 0 else round(float(n) * 365.25 / float(days), 2)


def run_historical_target_risk(target_norad, lookback_days=90, bucket_days=7, step_min=60,
    screening_miss_distance_threshold_km=10, cdm_pc_threshold=1e-7,
    cdm_miss_distance_threshold_km=5, maneuver_pc_threshold=1e-4,
    maneuver_miss_distance_threshold_km=1, catalog_orbit_class="LEO",
    max_catalog_objects=8000, max_tle_age_days=14, altitude_band_km=300,
    pc_method="foster", max_events_per_bucket=20):

    if not os.path.exists(DB_PATH):
        raise ValueError("TLE database missing. Refresh historical TLEs first.")
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=float(lookback_days))
    target_norad = str(target_norad).strip()
    c = _conn()
    try:
        db_min, db_max = c.execute("SELECT MIN(epoch), MAX(epoch) FROM tle_records").fetchone()
        target_records = c.execute("SELECT COUNT(*) FROM tle_records WHERE norad_id=?", (target_norad,)).fetchone()[0]
        if target_records == 0:
            raise ValueError(f"No historical TLE records for target NORAD {target_norad}.")
        buckets, t = [], start
        while t < now:
            b0, b1 = t, min(t + timedelta(days=float(bucket_days)), now)
            target = _target_before(c, target_norad, b0)
            if not target:
                buckets.append({"bucket_start": b0.isoformat(), "bucket_end": b1.isoformat(), "ok": False, "error": "No target TLE before bucket."})
                t = b1; continue
            times = _times(b0, (b1 - b0).total_seconds() / 86400, step_min)
            eff_days = max(0, (len(times)-1) * float(step_min) / 1440)
            pt, vt = _series(target["tle1"], target["tle2"], times)
            if pt is None:
                buckets.append({"bucket_start": b0.isoformat(), "bucket_end": b1.isoformat(), "ok": False, "error": "Target propagation failed."})
                t = b1; continue
            catalog = _catalog_before(c, b0, target_norad, catalog_orbit_class, max_catalog_objects, max_tle_age_days, target.get("alt_km"), altitude_band_km)
            by_class = {"debris": {"conjunction": 0, "cdm": 0, "maneuver": 0}, "inactive_satellite": {"conjunction": 0, "cdm": 0, "maneuver": 0}, "active_satellite": {"conjunction": 0, "cdm": 0, "maneuver": 0}}
            events = []
            for sec in catalog:
                ps, vs = _series(sec["tle1"], sec["tle2"], times)
                if ps is None:
                    continue
                dist = np.linalg.norm(ps - pt, axis=1)
                i = int(np.argmin(dist)); miss = float(dist[i])
                if miss > float(screening_miss_distance_threshold_km):
                    continue
                pcv = _pc(pc_method, pt[i], vt[i], ps[i], vs[i], _sigma(target.get("orbit_class") or sec.get("orbit_class")), _rhard(target.get("name"), sec.get("name")))
                lvl = _level(pcv, miss, cdm_pc_threshold, cdm_miss_distance_threshold_km, maneuver_pc_threshold, maneuver_miss_distance_threshold_km)
                oc = _obj_class(sec.get("name")); by_class.setdefault(oc, {"conjunction": 0, "cdm": 0, "maneuver": 0})
                by_class[oc]["conjunction"] += 1
                if lvl in ("cdm", "maneuver"): by_class[oc]["cdm"] += 1
                if lvl == "maneuver": by_class[oc]["maneuver"] += 1
                events.append({"secondary_norad": sec.get("norad_id"), "secondary_name": sec.get("name"), "object_class": oc, "t_ca": times[i].isoformat(), "miss_dist_km": round(miss, 4), "Pc": pcv, "Pc_str": f"{pcv:.2e}", "decision_level": lvl})
            events.sort(key=lambda e: (-float(e.get("Pc", 0)), float(e.get("miss_dist_km", 1e9))))
            conj = len(events); cdm = sum(1 for e in events if e["decision_level"] in ("cdm", "maneuver")); man = sum(1 for e in events if e["decision_level"] == "maneuver")
            buckets.append({"bucket_start": b0.isoformat(), "bucket_end": b1.isoformat(), "ok": True, "target_epoch": target.get("epoch"), "catalog_objects": len(catalog), "effective_days": round(eff_days, 3), "conjunction_events": conj, "cdm_equivalent_alerts": cdm, "avoidance_maneuvers": man, "by_object_class": by_class, "max_pc": max((float(e["Pc"]) for e in events), default=0), "min_miss_distance_km": min((float(e["miss_dist_km"]) for e in events), default=None), "top_events": events[:int(max_events_per_bucket)]})
            t = b1
        replayed_days = sum(float(b.get("effective_days") or 0) for b in buckets if b.get("ok"))
        total_conj = sum(int(b.get("conjunction_events") or 0) for b in buckets if b.get("ok"))
        total_cdm = sum(int(b.get("cdm_equivalent_alerts") or 0) for b in buckets if b.get("ok"))
        total_man = sum(int(b.get("avoidance_maneuvers") or 0) for b in buckets if b.get("ok"))
        return {"ok": True, "mode": "historical_tle_replay", "target_norad": target_norad, "lookback_days": lookback_days, "bucket_days": bucket_days, "step_min": step_min, "thresholds": {"screening_miss_distance_threshold_km": screening_miss_distance_threshold_km, "cdm_pc_threshold": cdm_pc_threshold, "cdm_pc_threshold_str": f"{cdm_pc_threshold:.2e}", "cdm_miss_distance_threshold_km": cdm_miss_distance_threshold_km, "maneuver_pc_threshold": maneuver_pc_threshold, "maneuver_pc_threshold_str": f"{maneuver_pc_threshold:.2e}", "maneuver_miss_distance_threshold_km": maneuver_miss_distance_threshold_km}, "history_coverage": {"db_earliest_epoch": db_min, "db_latest_epoch": db_max, "target_tle_records": target_records, "effective_replayed_days": round(replayed_days, 3)}, "summary": {"conjunction_events": total_conj, "cdm_equivalent_alerts": total_cdm, "avoidance_maneuvers": total_man, "annualized_conjunction_events": _annualized(total_conj, replayed_days), "annualized_cdm_equivalent_alerts": _annualized(total_cdm, replayed_days), "annualized_avoidance_maneuvers": _annualized(total_man, replayed_days)}, "time_series": buckets, "object_class_method": "heuristic_from_object_name", "limitations": ["Historical replay uses stored TLE records, not official CDM messages.", "Active/inactive/debris split is heuristic because the database does not yet store authoritative operational status.", "No local TCA refinement is applied yet."]}
    finally:
        c.close()
