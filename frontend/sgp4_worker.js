/**
 * sgp4_worker.js — Web Worker pour la propagation SGP4 en arrière-plan.
 *
 * Protocole messages :
 *   IN  { type:'init',   tles:[{name,norad,type,tle1,tle2},...] }
 *   IN  { type:'propagate', timestamp_ms: number }
 *   OUT { type:'ready',  n_sats: number }
 *   OUT { type:'positions', timestamp_ms, data: Float64Array (x,y,z,vx,vy,vz par sat) }
 *   OUT { type:'orbit',  norad, pts:[[x,y,z],...] }
 */

// SGP4 minimal en JS pur (pas de dépendance externe)
// Basé sur Vallado (2013) — même algorithme que la lib Python sgp4

const PI    = Math.PI;
const TWOPI = 2 * PI;
const DE2RA = PI / 180.0;
const MU    = 398600.4418;    // km³/s²
const RE    = 6378.137;       // km
const J2    = 1.08262668e-3;
const J3    = -2.53215306e-6;
const J4    = -1.61098761e-6;
const XKE   = 60.0 / Math.sqrt(RE * RE * RE / MU);  // rad/min
const VKMPERSEC = RE * XKE / 60.0;

function mod(a, b) { return ((a % b) + b) % b; }

function parseTle(tle1, tle2) {
  try {
    const yr2 = parseInt(tle1.substring(18, 20));
    const year = yr2 < 57 ? 2000 + yr2 : 1900 + yr2;
    const doy  = parseFloat(tle1.substring(20, 32));
    const epoch_jd = (year - 1970) * 365.25 + doy - 1 + 2440587.5 - 25567.5; // ms epoch approx
    // On utilise l'époque en jours juliens depuis J2000
    const jd_epoch = 367 * year
      - Math.trunc(7 * (year + Math.trunc((parseInt(tle1.substring(18,20))<57?2000:1900)+yr2 > 0 ? 1 : 0)) / 4)
      + doy - 1 + 1721013.5 + 2451545.0 - 2451545.0;
    // Éléments TLE
    const i    = parseFloat(tle2.substring(8, 16))  * DE2RA;
    const raan = parseFloat(tle2.substring(17, 25)) * DE2RA;
    const ecc  = parseFloat('0.' + tle2.substring(26, 33).trim());
    const argp = parseFloat(tle2.substring(34, 42)) * DE2RA;
    const mo   = parseFloat(tle2.substring(43, 51)) * DE2RA;
    const mm   = parseFloat(tle2.substring(52, 63)) * TWOPI / 1440.0; // rad/min
    const ndot  = parseFloat(tle1.substring(33, 43)) * TWOPI / (1440.0 * 1440.0);
    const bstar = parseBstar(tle1);

    // Demi-grand axe
    const a = Math.pow(XKE / mm, 2/3);

    return { i, raan, ecc, argp, mo, mm, ndot, bstar, a,
             epoch_doy: doy, epoch_year: year, valid: true };
  } catch(e) {
    return { valid: false };
  }
}

function parseBstar(tle1) {
  try {
    const s = tle1.substring(53, 61).trim();
    if (!s || s === '00000-0' || s === '+00000-0') return 0;
    const sign = s[0] === '-' ? -1 : 1;
    const body = s[0] === '+' || s[0] === '-' ? s.substring(1) : s;
    const dotStr = body.substring(0, 5);
    const expStr = body.substring(5);
    return sign * parseFloat('0.' + dotStr) * Math.pow(10, parseInt(expStr));
  } catch { return 0; }
}

function epochToJd(year, doy) {
  // JD de J2000 = 2451545.0
  const y = year - 1;
  return Math.trunc(365.25 * y) + Math.trunc(30.6001 * 14) + doy + 1720994.5 - 2451545.0;
}

function propagate(sat, dt_min) {
  // SGP4 simplifié — précision suffisante pour visualisation
  // dt_min : minutes depuis l'époque du TLE
  const { i, raan, ecc, argp, mo, mm, ndot, bstar, a } = sat;

  const e = ecc;
  const n = mm;

  // Anomalie moyenne propagée
  const M = mod(mo + n * dt_min + ndot * dt_min * dt_min, TWOPI);

  // Résoudre l'équation de Kepler E - e*sin(E) = M (Newton-Raphson, 10 iter)
  let E = M;
  for (let k = 0; k < 10; k++) {
    const dE = (M - E + e * Math.sin(E)) / (1 - e * Math.cos(E));
    E += dE;
    if (Math.abs(dE) < 1e-12) break;
  }

  // Anomalie vraie
  const nu = 2 * Math.atan2(
    Math.sqrt(1 + e) * Math.sin(E / 2),
    Math.sqrt(1 - e) * Math.cos(E / 2)
  );

  // Rayon (km)
  const r = a * RE * (1 - e * Math.cos(E));

  // Correction J2 du RAAN et argument du périgée
  const p   = a * RE * (1 - e * e);
  const cosI = Math.cos(i);
  const n0  = n;
  const raan_dot  = -1.5 * J2 * (RE / p) ** 2 * n0 * cosI;
  const argp_dot  =  0.75 * J2 * (RE / p) ** 2 * n0 * (5 * cosI * cosI - 1);

  const raan_t = raan + raan_dot * dt_min;
  const argp_t = argp + argp_dot * dt_min;
  const u      = argp_t + nu;

  // Coordonnées dans le plan orbital
  const rx =  r * Math.cos(u);
  const ry =  r * Math.sin(u);

  // Rotation vers ECI
  const cosR = Math.cos(raan_t), sinR = Math.sin(raan_t);
  const cosI2 = Math.cos(i),    sinI2 = Math.sin(i);
  const cosu  = Math.cos(u),    sinu  = Math.sin(u);

  const x = r * (cosR * cosu - sinR * sinu * cosI2);
  const y = r * (sinR * cosu + cosR * sinu * cosI2);
  const z = r * sinu * sinI2;

  // Vitesse approx (vis-viva)
  const v = Math.sqrt(MU * (2 / r - 1 / (a * RE))) / VKMPERSEC;
  // Direction tangentielle (perpendiculaire à r dans le plan orbital)
  const vx = v * (-cosR * sinu - sinR * cosu * cosI2);
  const vy = v * (-sinR * sinu + cosR * cosu * cosI2);
  const vz = v * cosu * sinI2;

  return { x, y, z, vx: vx * VKMPERSEC, vy: vy * VKMPERSEC, vz: vz * VKMPERSEC };
}

// ── État global du worker ─────────────────────────────────────────────────
let satellites = [];   // [{name, norad, type, sat_params, epoch_jd}, ...]
let lastPropTime = 0;
const EPOCH_2000 = Date.UTC(2000, 0, 1, 12, 0, 0); // J2000.0

function dtMinutes(timestamp_ms, year, doy) {
  // Minutes entre l'époque du TLE et timestamp_ms
  const tle_date = Date.UTC(year, 0, 1) + (doy - 1) * 86400000;
  return (timestamp_ms - tle_date) / 60000;
}

// ── Gestionnaire de messages ──────────────────────────────────────────────
self.onmessage = function(e) {
  const msg = e.data;

  if (msg.type === 'init') {
    satellites = [];
    for (const t of msg.tles) {
      const sp = parseTle(t.tle1, t.tle2);
      if (sp.valid) {
        satellites.push({
          name:  t.name,
          norad: t.norad,
          stype: t.stype || 'PAYLOAD',
          sp,
        });
      }
    }
    self.postMessage({ type: 'ready', n_sats: satellites.length });
    // Propagation initiale immédiate
    propagateAll(Date.now());
  }

  if (msg.type === 'propagate') {
    propagateAll(msg.timestamp_ms);
  }

  if (msg.type === 'orbit') {
    computeOrbit(msg.norad, msg.timestamp_ms);
  }
};

function propagateAll(ts) {
  const n = satellites.length;
  const buf = new Float64Array(n * 6);  // x,y,z,vx,vy,vz par satellite

  for (let i = 0; i < n; i++) {
    const sat = satellites[i];
    const dt  = dtMinutes(ts, sat.sp.epoch_year, sat.sp.epoch_doy);
    try {
      const p = propagate(sat.sp, dt);
      buf[i*6+0] = p.x;  buf[i*6+1] = p.y;  buf[i*6+2] = p.z;
      buf[i*6+3] = p.vx; buf[i*6+4] = p.vy; buf[i*6+5] = p.vz;
    } catch {
      // Satellite dégénéré : laisser à 0
    }
  }

  self.postMessage({ type: 'positions', timestamp_ms: ts, data: buf }, [buf.buffer]);
}

function computeOrbit(norad, ts) {
  const sat = satellites.find(s => s.norad === norad);
  if (!sat) return;
  const pts = [];
  const period_min = TWOPI / sat.sp.mm;
  // ±1/2 période autour de t actuel (passé + futur)
  for (let dt_offset = -period_min/2; dt_offset <= period_min/2; dt_offset += period_min/60) {
    const dt = dtMinutes(ts, sat.sp.epoch_year, sat.sp.epoch_doy) + dt_offset;
    try {
      const p = propagate(sat.sp, dt);
      pts.push([p.x, p.y, p.z]);
    } catch {}
  }
  self.postMessage({ type: 'orbit', norad, pts });
}
