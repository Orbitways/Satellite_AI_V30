"""
perturbations.py — Perturbations orbitales non modélisées par SGP4.

Toutes les fonctions reçoivent r et v en MÈTRES et retournent m/s².
Formules dérivées du gradient du potentiel gravitationnel (Montenbruck & Gill,
"Satellite Orbits", Springer 2000).

J2 est EXCLU : SGP4 l'intègre déjà analytiquement.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

# ── Constantes physiques ──────────────────────────────────────────────────────
MU      = 3.986004418e14   # m³/s²
RE      = 6378136.3        # m
J3      = -2.53265648e-6
J4      = -1.61962159e-6
J5      = -2.27296082e-7
J6      =  5.40681239e-7
C_LIGHT = 299792458.0      # m/s
AU      = 1.495978707e11   # m
P_SRP   = 4.56e-6          # N/m²
GM_MOON = 4.9048695e12     # m³/s²
GM_SUN  = 1.32712440018e20 # m³/s²


@dataclass
class PerturbationConfig:
    j3: bool = True
    j4: bool = False
    j5: bool = False
    drag_residual: bool = True
    solar_pressure: bool = True
    moon_gravity: bool = False
    sun_gravity: bool = False
    albedo: bool = False
    relativity: bool = False
    Cd: float = 2.2
    A_m: float = 0.01
    Cr: float = 1.3
    A_srp: float = 0.01

    def to_dict(self): return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k): setattr(c, k, v)
        return c

    def active_names(self):
        return [k for k, v in self.__dict__.items() if isinstance(v, bool) and v]


def _grad_zonal(r, Jn, n):
    """
    Gradient du potentiel zonal J_n. Retourne m/s².
    Formule : -∇U_Jn  (Montenbruck & Gill eq. 3.28-3.32)

    U_Jn = -(μ/r)·Jn·(RE/r)^n·Pn(sin φ)  où sin φ = z/r

    Polynômes de Legendre P_n(t) :
      P3(t) = (5t³ - 3t)/2
      P4(t) = (35t⁴ - 30t² + 3)/8
      P5(t) = (63t⁵ - 70t³ + 15t)/8
      P6(t) = (231t⁶ - 315t⁴ + 105t² - 5)/16
    """
    x, y, z = r
    rn = np.linalg.norm(r)
    zr = z / rn
    r_hat = r / rn
    e_z = np.array([0., 0., 1.])
    C = MU * Jn * RE**n / rn**n

    if n == 3:
        P  = zr * (5*zr**2 - 3) / 2
        dP = (15*zr**2 - 3) / 2
    elif n == 4:
        P  = (35*zr**4 - 30*zr**2 + 3) / 8
        dP = (140*zr**3 - 60*zr) / 8
    elif n == 5:
        P  = zr * (63*zr**4 - 70*zr**2 + 15) / 8
        dP = (315*zr**4 - 210*zr**2 + 15) / 8
    elif n == 6:
        P  = (231*zr**6 - 315*zr**4 + 105*zr**2 - 5) / 16
        dP = (1386*zr**5 - 1260*zr**3 + 210*zr) / 16
    else:
        return np.zeros(3)

    # -∇U = (C/r²) · [(n+1)·P·r̂ - dP·(ê_z - zr·r̂)]
    return -(C / rn**2) * ((n + 1) * P * r_hat - dP * (e_z - zr * r_hat))


def accel_j3(r): return _grad_zonal(r, J3, 3)
def accel_j4(r): return _grad_zonal(r, J4, 4)
def accel_j5(r): return _grad_zonal(r, J5, 5)
def accel_j6(r): return _grad_zonal(r, J6, 6)
def accel_jn(r, Jn, n): return _grad_zonal(r, Jn, n)


def atmospheric_density(altitude_m):
    """Modèle CIRA-72 par couches exponentielles. kg/m³."""
    alt_km = altitude_m / 1000.0
    layers = [
        (100, 5.297e-7, 5.877), (150, 2.076e-9, 26.27),
        (200, 2.541e-10, 37.11), (300, 1.916e-11, 53.26),
        (400, 2.803e-12, 58.52), (500, 5.215e-13, 65.52),
        (600, 1.137e-13, 73.13), (700, 3.070e-14, 88.67),
        (800, 1.136e-14, 124.6), (900, 5.759e-15, 181.0),
        (1000, 3.561e-15, 268.0),
    ]
    for i in range(len(layers) - 1, -1, -1):
        h0, rho0, H = layers[i]
        if alt_km >= h0:
            return rho0 * np.exp(-(alt_km - h0) / H)
    return layers[0][1]


def accel_drag(r, v, Cd, A_m):
    """a = -½·Cd·(A/m)·ρ(h)·|v_rel|·v_rel  [m/s²]"""
    omega_E = np.array([0., 0., 7.2921150e-5])
    v_rel = v - np.cross(omega_E, r)
    v_mag = np.linalg.norm(v_rel)
    if v_mag < 1e-10: return np.zeros(3)
    alt = np.linalg.norm(r) - RE
    rho = atmospheric_density(max(alt, 80e3))
    return -0.5 * Cd * A_m * rho * v_mag * v_rel


def sun_position_eci(t_jd):
    """Position Soleil en ECI (m). Précision ~0.01°."""
    T = (t_jd - 2451545.0) / 36525.0
    M = np.radians(357.52911 + 35999.05029 * T)
    e = 0.016708634 - 0.000042037 * T
    C = np.radians((1.914602 - 0.004817*T)*np.sin(M) + 0.019993*np.sin(2*M))
    lon = np.radians(280.46646 + 36000.76983*T) + C
    nu  = M + C
    r_au = 1.000001018*(1 - e**2)/(1 + e*np.cos(nu))
    eps  = np.radians(23.439291 - 0.013004*T)
    return AU * r_au * np.array([np.cos(lon), np.cos(eps)*np.sin(lon), np.sin(eps)*np.sin(lon)])


def moon_position_eci(t_jd):
    """Position Lune en ECI (m). Précision ~1°."""
    T  = (t_jd - 2451545.0) / 36525.0
    Mp = np.radians(134.9633964 + 477198.8675055*T)
    F  = np.radians(93.2720950  + 483202.0175233*T)
    lon = np.radians(218.3164477 + 481267.88123421*T) + np.radians(6.289*np.sin(Mp))
    lat = np.radians(5.128*np.sin(F))
    r_km = 385000.56 + 20905.355*np.cos(Mp)
    eps  = np.radians(23.439291 - 0.013004*T)
    return 1e3 * r_km * np.array([
        np.cos(lat)*np.cos(lon),
        np.cos(eps)*np.cos(lat)*np.sin(lon) - np.sin(eps)*np.sin(lat),
        np.sin(eps)*np.cos(lat)*np.sin(lon) + np.cos(eps)*np.sin(lat),
    ])


def _cylindrical_shadow(r_sat, r_sun):
    """0 = ombre, 1 = lumière."""
    r_sun_hat = r_sun / np.linalg.norm(r_sun)
    proj = np.dot(r_sat, r_sun_hat)
    if proj > 0: return 1.0
    perp = np.linalg.norm(r_sat - proj * r_sun_hat)
    return 0.0 if perp < RE else 1.0


def accel_srp(r, t_jd, Cr, A_srp):
    """a = -P☉·Cr·(A/m)·(AU/d)²·ê_sat→sun  [m/s²]"""
    r_sun = sun_position_eci(t_jd)
    d = r - r_sun
    dist = np.linalg.norm(d)
    if dist < 1e6: return np.zeros(3)
    shadow = _cylindrical_shadow(r, r_sun)
    return shadow * (-P_SRP * Cr * A_srp * (AU/dist)**2 * d/dist)


def accel_third_body(r_sat, r_body, GM_body):
    """a = GM·[d/|d|³ - r_body/|r_body|³]  [m/s²]"""
    d = r_body - r_sat
    return GM_body * (d/np.linalg.norm(d)**3 - r_body/np.linalg.norm(r_body)**3)


def accel_albedo(r, t_jd, Cr, A_srp):
    """Albédo terrestre (α=0.30). [m/s²]"""
    r_hat = r / np.linalg.norm(r)
    r_sun = sun_position_eci(t_jd)
    cos_theta = max(0, np.dot(r_hat, r_sun/np.linalg.norm(r_sun)))
    alt = np.linalg.norm(r) - RE
    flux = P_SRP * 0.30 * cos_theta * (RE/(RE+alt))**2
    return flux * Cr * A_srp * r_hat


def accel_relativity(r, v):
    """Correction de Schwarzschild. [m/s²]"""
    rn = np.linalg.norm(r); vn = np.linalg.norm(v); rv = np.dot(r, v)
    return MU/(C_LIGHT**2 * rn**3) * ((4*MU/rn - vn**2)*r + 4*rv*v)


def total_perturbation(r, v, t_jd, cfg):
    """Somme de toutes les accélérations activées. [m/s²]"""
    a = np.zeros(3)
    if cfg.j3:            a += accel_j3(r)
    if cfg.j4:            a += accel_j4(r)
    if cfg.j5:            a += accel_j5(r)
    if cfg.drag_residual: a += accel_drag(r, v, cfg.Cd, cfg.A_m) * 0.30
    if cfg.solar_pressure:a += accel_srp(r, t_jd, cfg.Cr, cfg.A_srp)
    if cfg.moon_gravity:  a += accel_third_body(r, moon_position_eci(t_jd), GM_MOON)
    if cfg.sun_gravity:   a += accel_third_body(r, sun_position_eci(t_jd), GM_SUN)
    if cfg.albedo:        a += accel_albedo(r, t_jd, cfg.Cr, cfg.A_srp)
    if cfg.relativity:    a += accel_relativity(r, v)
    return a


def compute_residual_from_perturbations(r_km, v_km_s, t_jd, dt_s, cfg):
    """Résidu Δpos = ½·a·dt² en km."""
    a = total_perturbation(r_km*1e3, v_km_s*1e3, t_jd, cfg)
    return 0.5 * a * dt_s**2 / 1000.0
