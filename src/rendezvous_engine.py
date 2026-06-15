"""
rendezvous_engine.py — Moteur de rendezvous orbital rigoureux.

Méthodes implémentées :
  - Hohmann      : 2 impulsions, orbites quasi-circulaires coplanaires
  - Lambert      : 2 impulsions, temps fixé, algorithme Battin (1984)
  - Phasage      : N révolutions, orbite de phasage optimisée
  - Bi-elliptique: 3 impulsions, orbite intermédiaire haute
  - Poussée faible: spirale logarithmique, formule Edelbaum (1961)

Approche terminale :
  - Manœuvre CW (Clohessy-Wiltshire) pour l'angle d'approche finale

Références :
  Vallado (2013) ; Battin (1984) ; Curtis (2014) ; Edelbaum (1961)
"""

import math
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

MU = 398600.4418   # km³/s²
RE = 6378.137      # km

#  Mécanique orbitale de base 

def propagate_sgp4(tle1, tle2, dt=None):
    """SGP4 → (r km, v km/s) ECI à l'instant dt (datetime UTC, défaut: maintenant)."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Nettoyage défensif des TLE
    tle1 = tle1.strip().replace('\r','').replace('\t',' ')
    tle2 = tle2.strip().replace('\r','').replace('\t',' ')

    # Corriger l'epoch si mal formatée (ex: "26155.0.9891550" → "26156.00761574")
    # Format correct : cols 18-31 = YYDDD.DDDDDDDD (pas de double point)
    if len(tle1) >= 32:
        epoch_field = tle1[18:32].strip()
        parts = epoch_field.split('.')
        if len(parts) > 2:
            # Double point : reconstruire avec les bons morceaux
            year_doy = parts[0]
            frac = parts[1] + (parts[2] if len(parts)>2 else '')
            fixed_epoch = f"{year_doy}.{frac}"
            tle1 = tle1[:18] + fixed_epoch.ljust(14) + tle1[32:]

    # Validation basique du format TLE
    if not tle1.startswith('1 ') or not tle2.startswith('2 '):
        # Tentative de correction : peut-être les lignes sont inversées
        if tle1.startswith('2 ') and tle2.startswith('1 '):
            tle1, tle2 = tle2, tle1
        else:
            raise ValueError(
                f"Format TLE invalide — TLE1 doit commencer par '1 ', TLE2 par '2 '. "
                f"Reçu: '{tle1[:10]}' / '{tle2[:10]}'"
            )

    if len(tle1) < 69 or len(tle2) < 69:
        # Compléter avec des espaces si trop court (certaines sources tronquent)
        tle1 = tle1.ljust(69)
        tle2 = tle2.ljust(69)

    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
    err, r, v = sat.sgp4(jd, fr)

    ERROR_CODES = {
        1: "éléments moyens hors limites (e≥1 ou a<0.95 ER) — TLE corrompu ou orbite dégénérée",
        2: "mean motion < 0",
        3: "pert elements > 1 (eccentricity issue at secular update)",
        4: "semi-latus rectum < 0",
        5: "epoch elements are sub-orbital",
        6: "satellite has decayed",
    }
    if err != 0:
        desc = ERROR_CODES.get(err, f"code inconnu {err}")
        raise ValueError(f"SGP4 erreur {err}: {desc}")

    return np.array(r), np.array(v)


def rv_to_elements(r_vec, v_vec):
    """Vecteurs ECI (r km, v km/s) → dict éléments képlériens.
    Lève ValueError si l'orbite est hyperbolique (a < 0)."""
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross([0., 0., 1.], h_vec)
    n = np.linalg.norm(n_vec)
    e_vec = ((v**2 - MU/r)*r_vec - np.dot(r_vec, v_vec)*v_vec) / MU
    e = np.linalg.norm(e_vec)
    energy = v**2/2 - MU/r
    a = -MU / (2*energy)
    if a <= 0:
        raise ValueError(f"Orbite hyperbolique (a={a:.1f}km, energy={energy:.4f})")
    i = math.acos(np.clip(h_vec[2]/h, -1, 1))
    RAAN = math.atan2(n_vec[1], n_vec[0]) if n > 1e-10 else 0.
    if RAAN < 0: RAAN += 2*math.pi
    argp = math.acos(np.clip(np.dot(n_vec/(n+1e-20), e_vec/(e+1e-20)), -1, 1)) \
           if n > 1e-10 and e > 1e-10 else 0.
    if e > 1e-10 and e_vec[2] < 0: argp = 2*math.pi - argp
    nu = math.acos(np.clip(np.dot(e_vec/(e+1e-20), r_vec/r), -1, 1)) if e > 1e-10 \
         else math.acos(np.clip(np.dot(n_vec/(n+1e-20), r_vec/r), -1, 1))
    if np.dot(r_vec, v_vec) < 0: nu = 2*math.pi - nu
    T = 2*math.pi*math.sqrt(a**3/MU)
    return dict(a=a, e=e, i=i, RAAN=RAAN, argp=argp, nu=nu,
                T=T, h=h, h_vec=h_vec, e_vec=e_vec, n_vec=n_vec)


def vis_viva(r, a):
    return math.sqrt(MU*(2/r - 1/a))


def propagate_kepler(r0, v0, dt_s):
    """
    Propagation képlerienne via variables universelles (Bate et al. 1971).
    Retourne (r, v) après dt_s secondes.
    Robuste aux orbites quasi-circulaires, elliptiques et légèrement hyperboliques.
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0_mag
    alpha = 2/r0_mag - v0_mag**2/MU   # 1/a (>0 ellipse, <0 hyperbole)

    # Vérification : orbite trop hyperbolique → propagation impossible
    if alpha < -1.0/(RE*100):  # a > 100*RE → fuite du système
        raise ValueError(
            f"Orbite hyperbolique divergente (alpha={alpha:.2e}). "
            f"ΔV trop grand ou direction incorrecte."
        )

    # Variable universelle chi par Newton
    chi = math.sqrt(MU)*abs(dt_s)*abs(alpha) if abs(alpha) > 1e-10 else math.sqrt(r0_mag)
    for _ in range(50):
        psi = chi**2 * alpha
        if psi > 1e-6:
            sp = math.sqrt(psi)
            C = (1 - math.cos(sp))/psi
            S = (sp - math.sin(sp))/psi**1.5
        elif psi < -1e-6:
            sp = math.sqrt(-psi)
            # Protéger contre l'overflow de cosh/sinh pour |sp| > 500
            if sp > 500:
                raise ValueError(f"Orbite trop hyperbolique (sp={sp:.1f})")
            C = (1 - math.cosh(sp))/psi
            S = (math.sinh(sp) - sp)/(-psi)**1.5
        else:
            C, S = 0.5, 1/6
        r_new = chi**2*C + vr0/math.sqrt(MU)*chi*(1 - psi*S) + r0_mag*(1 - psi*C)
        F = r0_mag*vr0/math.sqrt(MU)*chi**2*C + (1 - r0_mag*alpha)*chi**3*S + r0_mag*chi - math.sqrt(MU)*dt_s
        dFdchi = r_new
        dchi = -F/dFdchi if abs(dFdchi) > 1e-12 else 0
        chi += dchi
        if abs(dchi) < 1e-10: break
    psi = chi**2*alpha
    if psi > 1e-6:
        sp=math.sqrt(psi); C=(1-math.cos(sp))/psi; S=(sp-math.sin(sp))/psi**1.5
    elif psi < -1e-6:
        sp=math.sqrt(-psi); C=(1-math.cosh(sp))/psi; S=(math.sinh(sp)-sp)/(-psi)**1.5
    else:
        C,S=0.5,1/6
    r_mag = chi**2*C + vr0/math.sqrt(MU)*chi*(1-psi*S) + r0_mag*(1-psi*C)
    f = 1 - chi**2*C/r0_mag
    g = dt_s - chi**3*S/math.sqrt(MU)
    f_ = math.sqrt(MU)*chi*(psi*S-1)/(r_mag*r0_mag)
    g_ = 1 - chi**2*C/r_mag
    return f*r0 + g*v0, f_*r0 + g_*v0


#  Lambert (Battin 1984) 

def _stumpff(z):
    if z > 1e-6:
        sq=math.sqrt(z); return (1-math.cos(sq))/z, (sq-math.sin(sq))/z**1.5
    elif z < -1e-6:
        sq=math.sqrt(-z); return (1-math.cosh(sq))/z, (math.sinh(sq)-sq)/(-z)**1.5
    return 0.5 - z/24, 1/6 - z/120


def lambert_battin(r1_vec, r2_vec, tof, prograde=True):
    """
    Résout le problème de Lambert par la méthode de Battin (1984).
    Convergence robuste via Newton sur variable z universelle.
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    cos_nu = np.clip(np.dot(r1_vec, r2_vec)/(r1*r2), -1, 1)
    nu = math.acos(cos_nu)
    cross_z = np.cross(r1_vec, r2_vec)[2]
    if prograde and cross_z < 0:    nu = 2*math.pi - nu
    if not prograde and cross_z >= 0: nu = 2*math.pi - nu

    A = math.sin(nu)*math.sqrt(r1*r2/(1-cos_nu))
    if abs(A) < 1e-10:
        raise ValueError("Lambert: trajectoire dégénérée (nu≈0 ou 180°)")

    z = 0.0
    for _ in range(300):
        C, S = _stumpff(z)
        if C < 1e-14: z += 0.1; continue
        y = r1 + r2 + A*(z*S - 1)/math.sqrt(C)
        if A > 0 and y < 0: z += 0.1; continue

        sqrt_y = math.sqrt(max(y, 0))
        t_z = (y/C)**1.5 * S/math.sqrt(MU) + A*sqrt_y/math.sqrt(MU)

        # Dérivée numérique dt/dz
        dz = max(abs(z)*1e-4, 1e-6)
        C2, S2 = _stumpff(z+dz)
        y2 = r1 + r2 + A*((z+dz)*S2-1)/math.sqrt(max(C2,1e-14))
        if y2 > 0:
            t_z2 = (y2/C2)**1.5*S2/math.sqrt(MU) + A*math.sqrt(y2)/math.sqrt(MU)
            dtdz = (t_z2 - t_z)/dz
        else:
            dtdz = 1e-3

        res = t_z - tof
        if abs(res) < 1e-8: break
        step = res/dtdz if abs(dtdz) > 1e-14 else 0.01*(1 if res<0 else -1)
        z -= min(max(step, -2.0), 2.0)

    C, S = _stumpff(z)
    y = r1 + r2 + A*(z*S-1)/math.sqrt(max(C, 1e-14))
    f  = 1 - y/r1
    g  = A*math.sqrt(max(y,0)/MU)
    if abs(g) < 1e-12: raise ValueError("Lambert: g≈0")
    g_ = 1 - y/r2
    v1 = (r2_vec - f*r1_vec)/g
    v2 = (g_*r2_vec - r1_vec)/g
    return v1, v2


#  Méthodes de transfert 

def _hohmann(el_c, el_t, r_c, v_c, r_t, v_t):
    ac, at = el_c['a'], el_t['a']
    if el_c['e'] > 0.15 or el_t['e'] > 0.15:
        raise ValueError(f"Hohmann: excentricités trop élevées "
                         f"(e_c={el_c['e']:.4f}, e_t={el_t['e']:.4f}). "
                         f"Utilisez Lambert.")
    vc = math.sqrt(MU/ac)
    vt = math.sqrt(MU/at)
    a_tr = (ac+at)/2
    dv1 = abs(vc*(math.sqrt(2*at/(ac+at)) - 1))
    dv2 = abs(vt*(1 - math.sqrt(2*ac/(ac+at))))
    t_tr_s = math.pi*math.sqrt(a_tr**3/MU)
    t_tr_h = t_tr_s/3600

    # Déphasage angulaire réel dans le plan orbital
    nc = math.sqrt(MU/ac**3)
    nt = math.sqrt(MU/at**3)
    r_c_hat = r_c/np.linalg.norm(r_c)
    r_t_hat = r_t/np.linalg.norm(r_t)
    h_c_hat = el_c['h_vec']/el_c['h']
    along = np.cross(h_c_hat, r_c_hat)
    phi = math.atan2(np.dot(r_t_hat, along), np.dot(r_t_hat, r_c_hat))
    if phi < 0: phi += 2*math.pi

    phi_req = math.pi - nt*t_tr_s   # angle requis au départ
    phi_req = phi_req % (2*math.pi)

    dphi = (phi_req - phi) % (2*math.pi)
    T_syn = abs(2*math.pi/(nc - nt)) if abs(nc-nt) > 1e-12 else float('inf')
    wait_s = dphi/abs(nc-nt) if abs(nc-nt) > 1e-12 else 0.
    wait_h = wait_s/3600

    return {
        'total_dv_ms':     round((dv1+dv2)*1000, 4),
        'transfer_time_h': round(t_tr_h, 4),
        'wait_h':          round(wait_h, 4),
        'description': (
            f"Hohmann {ac-RE:.0f}→{at-RE:.0f}km | Δalt={abs(at-ac):.1f}km | "
            f"Attente phase {wait_h:.2f}h ({dphi*180/math.pi:.1f}°→{phi_req*180/math.pi:.1f}°) | "
            f"Transfert {t_tr_h:.3f}h"
        ),
        'maneuvers': [
            {'type':'Hohmann ΔV₁', 't_from_now_h':round(wait_h,4),
             'dv_ms':round(dv1*1000,4),
             'description':f"Injection ellipse de transfert (a={a_tr:.1f}km). Prograde."},
            {'type':'Hohmann ΔV₂', 't_from_now_h':round(wait_h+t_tr_h,4),
             'dv_ms':round(dv2*1000,4),
             'description':f"Circularisation orbite cible (alt={at-RE:.0f}km). Prograde."},
        ],
    }


def _lambert(r_c, v_c, r_t, v_t, dur_h):
    el_c = rv_to_elements(r_c, v_c)
    el_t = rv_to_elements(r_t, v_t)
    nc = math.sqrt(MU/el_c['a']**3)
    nt = math.sqrt(MU/el_t['a']**3)
    T_short = min(el_c['T'], el_t['T'])

    best = None; best_dv = float('inf')
    # 30 échantillons entre T/4 et dur_h
    n_s = 30
    t_min = T_short/4; t_max = dur_h*3600
    for k in range(n_s):
        tof = t_min + (t_max-t_min)*k/(n_s-1)
        # Propager cible (képlerien)
        try:
            r_t2, v_t2 = propagate_kepler(r_t, v_t, tof)
        except Exception:
            continue
        for prog in [True, False]:
            try:
                v1, v2 = lambert_battin(r_c, r_t2, tof, prograde=prog)
                dv1 = np.linalg.norm(v1 - v_c)
                dv2 = np.linalg.norm(v_t2 - v2)
                total = (dv1+dv2)*1000
                if total < best_dv:
                    best_dv = total
                    best = dict(tof=tof, dv1=dv1*1000, dv2=dv2*1000,
                                v1=v1, v2=v2, r_t2=r_t2)
            except Exception:
                continue

    if best is None:
        raise ValueError("Lambert: aucune solution convergée sur la fenêtre demandée")

    tof_h = best['tof']/3600
    cos_ang = np.clip(np.dot(r_c, best['r_t2'])/(np.linalg.norm(r_c)*np.linalg.norm(best['r_t2'])),-1,1)
    ang = math.degrees(math.acos(cos_ang))
    return {
        'total_dv_ms':     round(best_dv, 4),
        'transfer_time_h': round(tof_h, 4),
        'wait_h': 0.,
        'description': (
            f"Lambert optimal sur {n_s} durées [T/4..{dur_h:.0f}h] | "
            f"tof={tof_h:.3f}h | angle={ang:.1f}° | "
            f"ΔV₁={best['dv1']:.3f}m/s ΔV₂={best['dv2']:.3f}m/s"
        ),
        'maneuvers': [
            {'type':'Lambert ΔV₁', 't_from_now_h':0.,
             'dv_ms':round(best['dv1'],4),
             'description':'Impulsion initiale Lambert (direction optimale).'},
            {'type':'Lambert ΔV₂', 't_from_now_h':round(tof_h,4),
             'dv_ms':round(best['dv2'],4),
             'description':'Impulsion de rencontre — annulation vitesse relative.'},
        ],
    }


def _phasing(el_c, el_t, r_c, v_c, r_t, v_t, dur_h):
    ac, at = el_c['a'], el_t['a']
    nc = math.sqrt(MU/ac**3)
    nt = math.sqrt(MU/at**3)
    Tc = 2*math.pi/nc

    r_c_hat = r_c/np.linalg.norm(r_c)
    r_t_hat = r_t/np.linalg.norm(r_t)
    h_c_hat = el_c['h_vec']/el_c['h']
    along = np.cross(h_c_hat, r_c_hat)
    phi = math.atan2(np.dot(r_t_hat, along), np.dot(r_t_hat, r_c_hat))
    if phi < 0: phi += 2*math.pi

    best = None; best_dv = float('inf')
    for N in range(1, 16):
        # Le chaser doit parcourir N tours + rattraper phi
        t_phase = (N*2*math.pi + phi) / nt  # cible parcourt N*2π + phi
        if t_phase > dur_h*3600 and N > 1: continue
        if t_phase <= 0: continue
        T_ph = t_phase / N
        a_ph = (MU*(T_ph/(2*math.pi))**2)**(1/3)
        if a_ph < RE+150: continue
        vc_circ = math.sqrt(MU/ac)
        dv = abs(vc_circ - vis_viva(ac, a_ph))
        total = dv*2*1000
        if total < best_dv:
            best_dv = total
            best = dict(N=N, T_ph=T_ph, a_ph=a_ph, t_h=t_phase/3600, dv=dv)

    if best is None:
        # Fallback : phasage 1 tour, orbite légèrement modifiée
        T_ph = Tc*(1 + phi/(2*math.pi))
        a_ph = (MU*(T_ph/(2*math.pi))**2)**(1/3)
        vc_circ = math.sqrt(MU/ac)
        dv = abs(vc_circ - vis_viva(ac, a_ph))
        best = dict(N=1, T_ph=T_ph, a_ph=a_ph, t_h=T_ph/3600, dv=dv)
        best_dv = dv*2*1000

    return {
        'total_dv_ms':     round(best_dv, 4),
        'transfer_time_h': round(best['t_h'], 4),
        'wait_h': 0.,
        'description': (
            f"Phasage N={best['N']} révolutions sur {best['t_h']:.2f}h | "
            f"Orbite de phasage: alt={best['a_ph']-RE:.0f}km "
            f"({'au-dessus' if best['a_ph']>ac else 'en-dessous'}) | "
            f"Déphasage initial: {phi*180/math.pi:.1f}°"
        ),
        'maneuvers': [
            {'type':'Phasage ΔV₁', 't_from_now_h':0.,
             'dv_ms':round(best['dv']*1000,4),
             'description':(f"Entrée orbite phasage (a={best['a_ph']:.1f}km, "
                            f"T={best['T_ph']/60:.2f}min). "
                            f"{'Prograde' if best['a_ph']>ac else 'Rétrograde'}.")},
            {'type':'Phasage ΔV₂', 't_from_now_h':round(best['t_h'],4),
             'dv_ms':round(best['dv']*1000,4),
             'description':f"Retour orbite nominale après {best['N']} révolutions."},
        ],
    }


def _bielliptic(el_c, el_t):
    ac, at = el_c['a'], el_t['a']
    vc = math.sqrt(MU/ac); vt = math.sqrt(MU/at)
    ratio = max(ac,at)/min(ac,at)
    r_hi = max(ac,at)

    # Optimisation sur r_b
    best = None; best_dv = float('inf')
    dv_hoh = (abs(vc*(math.sqrt(2*at/(ac+at))-1)) + abs(vt*(1-math.sqrt(2*ac/(ac+at)))))*1000
    for f in [1.5,2,2.5,3,4,5,7,10,15]:
        rb = r_hi*f
        dv1 = abs(vis_viva(ac,(ac+rb)/2) - vc)
        dv2 = abs(vis_viva(rb,(rb+at)/2) - vis_viva(rb,(ac+rb)/2))
        dv3 = abs(vt - vis_viva(at,(rb+at)/2))
        tot = (dv1+dv2+dv3)*1000
        t1 = math.pi*math.sqrt(((ac+rb)/2)**3/MU)/3600
        t2 = math.pi*math.sqrt(((rb+at)/2)**3/MU)/3600
        if tot < best_dv:
            best_dv=tot; best=dict(rb=rb,f=f,dv1=dv1,dv2=dv2,dv3=dv3,t1=t1,t2=t2)

    adv = dv_hoh - best_dv
    return {
        'total_dv_ms':     round(best_dv,4),
        'transfer_time_h': round(best['t1']+best['t2'],4),
        'wait_h': 0.,
        'description': (
            f"Bi-elliptique {ac-RE:.0f}→{best['rb']-RE:.0f}→{at-RE:.0f}km | "
            f"Ratio={ratio:.2f} | "
            f"{'Avantage vs Hohmann: '+str(round(adv,2))+' m/s' if adv>0 else 'Hohmann plus économique de '+str(round(-adv,2))+' m/s'}"
        ),
        'maneuvers': [
            {'type':'Bi-ell. ΔV₁','t_from_now_h':0.,
             'dv_ms':round(best['dv1']*1000,4),
             'description':f"Injection vers apogée intermédiaire ({best['rb']-RE:.0f}km)."},
            {'type':'Bi-ell. ΔV₂','t_from_now_h':round(best['t1'],4),
             'dv_ms':round(best['dv2']*1000,4),
             'description':f"Injection depuis apogée vers orbite cible."},
            {'type':'Bi-ell. ΔV₃','t_from_now_h':round(best['t1']+best['t2'],4),
             'dv_ms':round(best['dv3']*1000,4),
             'description':f"Circularisation orbite cible ({at-RE:.0f}km)."},
        ],
    }


def _lowthrust(el_c, el_t):
    ac, at = el_c['a'], el_t['a']
    di = abs(el_c['i'] - el_t['i'])
    vc = math.sqrt(MU/ac); vt = math.sqrt(MU/at)
    # Edelbaum (1961)
    dv = math.sqrt(max(0.0, vc**2 + vt**2 - 2*vc*vt*math.cos(math.pi/2*di)))*1000
    # T/m typique HET: 5e-5 m/s²
    t_h = dv/5e-5/3600
    isp = 1800.; g0 = 9.80665e-3
    prop_frac = 1 - math.exp(-dv/1000/(isp*g0/1000))
    return {
        'total_dv_ms':     round(dv,4),
        'transfer_time_h': round(t_h,4),
        'wait_h': 0.,
        'description': (
            f"Edelbaum {ac-RE:.0f}km/{math.degrees(el_c['i']):.1f}° → "
            f"{at-RE:.0f}km/{math.degrees(el_t['i']):.1f}° | "
            f"Δi={math.degrees(di):.2f}° | "
            f"Isp={isp:.0f}s, T/m=5×10⁻⁵m/s² | "
            f"Masse propulsif {prop_frac*100:.1f}% | Durée {t_h:.1f}h"
        ),
        'maneuvers': [
            {'type':'Low-thrust continu','t_from_now_h':0.,
             'dv_ms':round(dv,4),
             'description':f"Spirale continue {t_h:.1f}h. Poussée tangentielle."},
        ],
    }


def _approach_cw(el_t, approach_deg, t_base_h):
    """Manœuvre terminale Clohessy-Wiltshire — approche à angle imposé."""
    n = math.sqrt(MU/el_t['a']**3)  # rad/s
    ang = math.radians(approach_deg)
    # Direction d'approche dans LVLH (x=radial, y=along-track)
    # 0°→along-track +y, 90°→radial +x, 180°→along-track -y
    d_hat = np.array([math.sin(ang), math.cos(ang), 0.])

    # Résidu typique après transfert principal : 10km along-track
    r_rel = np.array([0., 10., 0.])  # km
    r_final = d_hat * 0.1            # 100m finale

    # Durée approche ≈ quart de période
    dt = math.pi/(2*n)
    nt = n*dt
    c, s = math.cos(nt), math.sin(nt)

    # Matrice CW : v_init → Δr_final - Δr_free
    Mcw = np.array([[s/n,        2*(1-c)/n  ],
                    [-2*(1-c)/n, (4*s-3*nt)/n]])

    # Position libre sous CW (v_rel = 0 au début)
    x_free = (4 - 3*c)*r_rel[0]
    y_free = 6*(s - nt)*r_rel[0] + r_rel[1]
    dr = r_final[:2] - np.array([x_free, y_free])

    try:
        dv_req = np.linalg.solve(Mcw, dr)   # km/s
    except np.linalg.LinAlgError:
        dv_req = np.zeros(2)

    dv_ms = np.linalg.norm(dv_req)*1000
    return {
        'type': f'Approche terminale {approach_deg}°',
        't_from_now_h': round(t_base_h + 0.05, 4),
        'dv_ms': round(dv_ms, 4),
        'description': (
            f"CW approche {approach_deg}° (0°=V-bar colinéaire, 90°=R-bar radial). "
            f"10km → 100m en {dt/60:.1f}min."
        ),
    }



#  Précession nodale J2 
J2 = 1.08262668e-3

def j2_raan_rate(a_km: float, i_rad: float) -> float:
    """dΩ/dt (rad/s) dû à J2 pour une orbite circulaire."""
    n = math.sqrt(MU / a_km**3)
    return -1.5 * n * J2 * (RE / a_km)**2 * math.cos(i_rad)


def plane_angle_from_elements(RAAN_c, i_c, RAAN_t, i_t) -> float:
    """Angle (rad) entre deux plans orbitaux."""
    hc = np.array([math.sin(i_c)*math.sin(RAAN_c),
                  -math.sin(i_c)*math.cos(RAAN_c),
                   math.cos(i_c)])
    ht = np.array([math.sin(i_t)*math.sin(RAAN_t),
                  -math.sin(i_t)*math.cos(RAAN_t),
                   math.cos(i_t)])
    return math.acos(np.clip(np.dot(hc, ht), -1.0, 1.0))


def j2_convergence(el_c: dict, el_t: dict,
                   max_days: float = 365) -> dict:
    """
    Détermine si la précession J2 peut réduire l'angle entre les deux plans.
    Retourne le minimum d'angle atteignable et le délai correspondant.

    Un angle entre plans converge via J2 uniquement si les deux orbites
    précèdent dans le même sens (cos(i_c) et cos(i_t) de même signe)
    ET à des vitesses différentes.
    """
    dOmega_c = j2_raan_rate(el_c['a'], el_c['i'])
    dOmega_t = j2_raan_rate(el_t['a'], el_t['i'])

    RAAN_c = el_c['RAAN']
    RAAN_t = el_t['RAAN']
    i_c    = el_c['i']
    i_t    = el_t['i']

    # Même sens de précession ?
    same_direction = (dOmega_c * dOmega_t > 0)

    min_angle = float('inf')
    min_day   = 0.0

    step = 0.1  # jours
    for k in range(int(max_days / step)):
        t_s = k * step * 86400
        Rc  = (RAAN_c + dOmega_c * t_s) % (2 * math.pi)
        Rt  = (RAAN_t + dOmega_t * t_s) % (2 * math.pi)
        ang = plane_angle_from_elements(Rc, i_c, Rt, i_t)
        if ang < min_angle:
            min_angle = ang
            min_day   = k * step

    converges = min_angle < math.radians(5)  # < 5° → convergence utile

    return {
        'converges':         bool(converges),
        'same_direction':    bool(same_direction),
        'min_angle_deg':     float(round(math.degrees(min_angle), 2)),
        'min_day':           float(round(min_day, 1)),
        'dOmega_c_deg_day':  float(round(math.degrees(dOmega_c) * 86400, 3)),
        'dOmega_t_deg_day':  float(round(math.degrees(dOmega_t) * 86400, 3)),
    }


def bielliptic_plane_change(el_c: dict, el_t: dict) -> dict:
    """
    Stratégie bi-elliptique haute pour changement de plan + altitude.
    Trois impulsions :
      ΔV₁ : injection sur l'ellipse montante (périgée = a_c, apogée = r_b)
      ΔV₂ : à l'apogée r_b : changement de plan + injection ellipse descendante (combiné)
      ΔV₃ : circularisation sur l'orbite cible (a_t)

    Optimise r_b ∈ [max(a_c,a_t)×1.1 .. 300 000km] pour minimiser ΔV_total.

    Référence : Vallado (2013), §6.3.4 — Combined maneuvers.
    """
    ac = el_c['a']
    at = el_t['a']
    vc = math.sqrt(MU / ac)
    vt = math.sqrt(MU / at)

    # Angle entre les deux plans orbitaux
    h_c = el_c['h_vec'] / el_c['h']
    h_t = el_t['h_vec'] / el_t['h']
    delta_i = math.acos(np.clip(np.dot(h_c, h_t), -1.0, 1.0))

    # Grille d'optimisation sur r_b
    r_b_candidates = []
    for exp in np.linspace(math.log10(max(ac, at) * 1.1),
                           math.log10(RE + 300000), 60):
        r_b_candidates.append(10**exp)

    best_dv  = float('inf')
    best     = None

    for r_b in r_b_candidates:
        # Vitesse à l'apogée de l'ellipse montante (a_c → r_b)
        v_apo_up   = math.sqrt(MU * (2 / r_b - 1 / ((ac + r_b) / 2)))
        # Vitesse à l'apogée de l'ellipse descendante (r_b → a_t)
        v_apo_down = math.sqrt(MU * (2 / r_b - 1 / ((r_b + at) / 2)))
        # ΔV₂ combiné : changement de plan + changement d'ellipse en un burn
        dv2 = math.sqrt(v_apo_up**2 + v_apo_down**2
                        - 2 * v_apo_up * v_apo_down * math.cos(delta_i))
        # ΔV₁ : prograde depuis ac
        v_peri1 = math.sqrt(MU * (2 / ac - 1 / ((ac + r_b) / 2)))
        dv1 = abs(v_peri1 - vc)
        # ΔV₃ : circularisation en at
        v_peri2 = math.sqrt(MU * (2 / at - 1 / ((r_b + at) / 2)))
        dv3 = abs(vt - v_peri2)

        total = (dv1 + dv2 + dv3) * 1000

        t1 = math.pi * math.sqrt(((ac + r_b) / 2)**3 / MU) / 3600
        t2 = math.pi * math.sqrt(((r_b + at) / 2)**3 / MU) / 3600

        if total < best_dv:
            best_dv = total
            best = {
                'r_b_km':    round(r_b - RE, 0),
                'dv1_ms':    round(dv1 * 1000, 2),
                'dv2_ms':    round(dv2 * 1000, 2),
                'dv3_ms':    round(dv3 * 1000, 2),
                'total_dv':  round(total, 2),
                'duration_h': round(t1 + t2, 2),
                'delta_i_deg': round(math.degrees(delta_i), 2),
            }

    # Borne inférieure théorique (r_b → ∞)
    dv_lower = (vc * (math.sqrt(2) - 1) + vt * (math.sqrt(2) - 1)) * 1000
    best['dv_lower_bound_ms'] = round(dv_lower, 0)

    best['maneuvers'] = [
        {
            'type':        'Bi-ell. haute ΔV₁',
            't_from_now_h': 0.0,
            'dv_ms':        best['dv1_ms'],
            'description':  f"Injection ellipse montante (apogée r_b={best['r_b_km']:.0f}km). Prograde.",
        },
        {
            'type':        'Bi-ell. haute ΔV₂',
            't_from_now_h': round(best['duration_h'] / 2, 4),
            'dv_ms':        best['dv2_ms'],
            'description':  (f"Changement de plan Δi={best['delta_i_deg']:.1f}° + injection "
                             f"ellipse descendante. Burn combiné à r_b={best['r_b_km']:.0f}km."),
        },
        {
            'type':        'Bi-ell. haute ΔV₃',
            't_from_now_h': round(best['duration_h'], 4),
            'dv_ms':        best['dv3_ms'],
            'description':  f"Circularisation orbite cible ({at - RE:.0f}km). Prograde/rétrograde.",
        },
    ]
    return best


#  Point d'entrée 

def compute_rendezvous(tle1_c, tle2_c, tle1_t, tle2_t,
                       method='hohmann', dur_h=24., dv_max_ms=500.,
                       approach_angle_deg=0.):
    now = datetime.now(timezone.utc)
    r_c, v_c = propagate_sgp4(tle1_c, tle2_c, now)
    r_t, v_t = propagate_sgp4(tle1_t, tle2_t, now)
    el_c = rv_to_elements(r_c, v_c)
    el_t = rv_to_elements(r_t, v_t)

    dispatch = {
        'hohmann':    lambda: _hohmann(el_c, el_t, r_c, v_c, r_t, v_t),
        'lambert':    lambda: _lambert(r_c, v_c, r_t, v_t, dur_h),
        'phasing':    lambda: _phasing(el_c, el_t, r_c, v_c, r_t, v_t, dur_h),
        'bielliptic': lambda: _bielliptic(el_c, el_t),
        'lowthrust':  lambda: _lowthrust(el_c, el_t),
    }
    if method not in dispatch:
        raise ValueError(f"Méthode inconnue: {method}")

    res = dispatch[method]()

    # Manœuvre terminale si angle != 0
    if approach_angle_deg != 0. and method != 'lowthrust':
        t_base = res.get('wait_h',0.) + res.get('transfer_time_h',0.)
        m_app = _approach_cw(el_t, approach_angle_deg, t_base)
        res['maneuvers'].append(m_app)
        res['total_dv_ms'] = round(res['total_dv_ms'] + m_app['dv_ms'], 4)
        res['approach_angle_deg'] = approach_angle_deg

    if res['total_dv_ms'] > dv_max_ms:
        res['warning'] = (f"ΔV={res['total_dv_ms']:.1f}m/s dépasse la limite "
                          f"{dv_max_ms:.0f}m/s.")

    #  Analyse du plan orbital 
    h_c = el_c['h_vec'] / el_c['h']
    h_t = el_t['h_vec'] / el_t['h']
    delta_plane = math.degrees(math.acos(np.clip(np.dot(h_c, h_t), -1.0, 1.0)))
    res['delta_plane_deg'] = round(delta_plane, 2)

    if res['total_dv_ms'] > dv_max_ms or delta_plane > 15:
        # Analyse J2
        j2 = j2_convergence(el_c, el_t, max_days=365)
        res['j2_analysis'] = j2
        # Bi-elliptique haute
        be = bielliptic_plane_change(el_c, el_t)
        res['bielliptic_highalt'] = be
        if res['total_dv_ms'] > dv_max_ms:
            if j2['converges']:
                res['warning'] = (
                    f"ΔV={res['total_dv_ms']:.0f}m/s > budget {dv_max_ms:.0f}m/s. "
                    f"J2 converge en {j2['min_day']:.0f}j → angle plan={j2['min_angle_deg']:.1f}°. "
                    f"Bi-elliptique haute: {be['total_dv']:.0f}m/s en {be['duration_h']:.0f}h."
                )
            else:
                res['warning'] = (
                    f"ΔV={res['total_dv_ms']:.0f}m/s > budget {dv_max_ms:.0f}m/s. "
                    f"J2 inefficace (plans en sens opposés: {j2['dOmega_c_deg_day']:.2f}°/j vs {j2['dOmega_t_deg_day']:.2f}°/j). "
                    f"Stratégie recommandée: bi-elliptique haute {be['r_b_km']:.0f}km → "
                    f"{be['total_dv']:.0f}m/s en {be['duration_h']:.0f}h "
                    f"(borne inf: {be['dv_lower_bound_ms']:.0f}m/s)."
                )

    res.update({
        'method':         method,
        'chaser_alt_km':  round(el_c['a']-RE, 2),
        'target_alt_km':  round(el_t['a']-RE, 2),
        'chaser_inc_deg': round(math.degrees(el_c['i']), 4),
        'target_inc_deg': round(math.degrees(el_t['i']), 4),
        'delta_inc_deg':  round(abs(math.degrees(el_c['i']-el_t['i'])), 4),
        'chaser_ecc':     round(el_c['e'], 6),
        'target_ecc':     round(el_t['e'], 6),
        'init_dist_km':   round(float(np.linalg.norm(r_c-r_t)), 2),
        'epoch_utc':      now.isoformat(),
    })
    return res


# 
# Rendezvous physique — Vallado (2013) Fundamentals of Astrodynamics, Ch. 6
# 

def _hohmann_dv(a1: float, a2: float) -> tuple:
    """
    Calcule ΔV₁ et ΔV₂ pour un transfert Hohmann entre a1 et a2 (km).
    Retourne (dv1_ms, dv2_ms, tof_s).
    Référence: Vallado (2013) Eq. 6-1.
    """
    a_tr = (a1 + a2) / 2
    v1  = math.sqrt(MU / a1)
    v2  = math.sqrt(MU / a2)
    vp  = math.sqrt(MU * (2/a1 - 1/a_tr))  # vitesse périgée ellipse transfert
    va  = math.sqrt(MU * (2/a2 - 1/a_tr))  # vitesse apogée
    dv1 = abs(vp - v1) * 1000               # m/s
    dv2 = abs(v2 - va) * 1000               # m/s
    tof = math.pi * math.sqrt(a_tr**3 / MU) # s (demi-période)
    return dv1, dv2, tof


def compute_mission_plan(r0, v0, rt0, vt0) -> list:
    """
    Plan de rendez-vous complet N manœuvres (Vallado 2013, Ch.6-7).
    Séquence : [Phase plane/alt] → [Phasage] → [Approche] → [Proximité CW]
    
    Retourne une liste ordonnée de phases, chacune avec :
      - type, burns (label, t_h, dv_ms, desc), duration_h
    """
    r0  = np.array(r0);  v0  = np.array(v0)
    rt0 = np.array(rt0); vt0 = np.array(vt0)

    el_c = rv_to_elements(r0,  v0 )
    el_t = rv_to_elements(rt0, vt0)
    a_c  = el_c['a'];  a_t = el_t['a']
    i_c  = el_c['i'];  i_t = el_t['i']  # rad
    delta_i = math.acos(np.clip(np.dot(
        el_c['h_vec']/el_c['h'], el_t['h_vec']/el_t['h']), -1., 1.))

    phases   = []
    t_elapsed = 0.            # s depuis t=0
    r_cur    = r0.copy()
    v_cur    = v0.copy()

    # 
    # PHASE 1 : Changement de plan + altitude (Bi-elliptique)
    # 
    delta_plane_rad = math.acos(np.clip(np.dot(el_c['h_vec']/el_c['h'], el_t['h_vec']/el_t['h']), -1., 1.))
    if delta_plane_rad > math.radians(0.1) or abs(a_c - a_t) > 10.:
        use_biell = (delta_plane_rad > math.radians(20.))  # bi-ell seulement si > 20°

        if use_biell:
            be = bielliptic_plane_change(el_c, el_t)
            r_b   = (be['r_b_km'] + RE)
            r_mag = float(np.linalg.norm(r_cur))
            a_tr1 = (r_mag + r_b) / 2
            a_tr2 = (r_b + a_t)   / 2
            T_tr1 = math.pi * math.sqrt(a_tr1**3 / MU)
            T_tr2 = math.pi * math.sqrt(a_tr2**3 / MU)

            # ΔV₁ : injection arc montant
            v_dep1_mag = math.sqrt(MU * (2./r_mag - 1./a_tr1))
            v_dep1     = v_cur / np.linalg.norm(v_cur) * v_dep1_mag
            dv1 = float(np.linalg.norm(v_dep1 - v_cur)) * 1000

            # Propagation arc montant
            r_apo, v_apo1 = propagate_kepler(r_cur, v_dep1, T_tr1)

            # ΔV₂ : changement de plan à l'apogée
            h_t_hat  = el_t['h_vec'] / el_t['h']
            r_apo_h  = r_apo / np.linalg.norm(r_apo)
            v_dir2   = np.cross(h_t_hat, r_apo_h)
            nrm = np.linalg.norm(v_dir2)
            if nrm < 1e-10: v_dir2 = v_apo1.copy()
            else:            v_dir2 /= nrm
            if np.dot(v_dir2, v_apo1) < 0: v_dir2 = -v_dir2

            v_apo2_mag = math.sqrt(MU * (2./np.linalg.norm(r_apo) - 1./a_tr2))
            v_dep2     = v_dir2 * v_apo2_mag
            dv2 = float(np.linalg.norm(v_dep2 - v_apo1)) * 1000

            # Propagation arc descendant
            r_peri, v_peri = propagate_kepler(r_apo, v_dep2, T_tr2)
            v_c3 = math.sqrt(MU / np.linalg.norm(r_peri))

            # ΔV₃ : circularisation
            v_circ_vec = v_peri / np.linalg.norm(v_peri) * v_c3
            dv3 = float(np.linalg.norm(v_circ_vec - v_peri)) * 1000

            dur1 = (T_tr1 + T_tr2) / 3600.
            phases.append({
                'phase': 1,
                'type':  'bielliptic_plane_change',
                'label': 'Changement de plan bi-elliptique',
                'duration_h': round(dur1, 3),
                'delta_i_deg': round(math.degrees(delta_i), 2),
                'r_b_km': be['r_b_km'],
                'burns': [
                    {'n': 1, 'label': 'ΔV₁ Injection arc montant',
                     't_h': round(t_elapsed/3600, 4),
                     'dv_ms': round(dv1, 2),
                     'desc': f'Prograde depuis {a_c-RE:.0f}km → apogée {be["r_b_km"]:.0f}km'},
                    {'n': 2, 'label': 'ΔV₂ Changement de plan (apogée)',
                     't_h': round((t_elapsed + T_tr1)/3600, 4),
                     'dv_ms': round(dv2, 2),
                     'desc': f'Rotation Δi={math.degrees(delta_i):.1f}° + injection arc descendant'},
                    {'n': 3, 'label': 'ΔV₃ Circularisation orbite cible',
                     't_h': round((t_elapsed + T_tr1 + T_tr2)/3600, 4),
                     'dv_ms': round(dv3, 2),
                     'desc': f'Circularisation à {a_t-RE:.0f}km'},
                ],
                'r_end': r_peri.tolist(), 'v_end': v_circ_vec.tolist(),
            })
            t_elapsed += T_tr1 + T_tr2
            r_cur = r_peri.copy(); v_cur = v_circ_vec.copy()

        elif delta_plane_rad > math.radians(0.1):
            # Changement de plan seul (Δplane < 20°) — burn unique au nœud
            pc = compute_plane_change_to_target(r0, v0, rt0, vt0)
            t_node = pc['t_node_s']
            r_burn = np.array(pc['r_burn']); v_after = np.array(pc['v_after'])
            n_burn = 1
            phases.append({
                'phase': 1, 'type': 'plane_change', 'label': 'Changement de plan orbital',
                'duration_h': round(t_node/3600, 3),
                'delta_plane_deg': pc['delta_plane'],
                'burns': [
                    {'n': n_burn,
                     'label': f'ΔV₁ Changement de plan (nœud)',
                     't_h': round((t_elapsed + t_node)/3600, 4),
                     'dv_ms': pc['dv_ms'],
                     'desc': (f'Δplane={pc["delta_plane"]:.2f}° → incidence nulle | '
                              f'Nœud à t+{t_node/60:.1f}min')},
                ],
                'r_end': r_burn.tolist(), 'v_end': v_after.tolist(),
            })
            t_elapsed += t_node
            r_cur = r_burn.copy(); v_cur = v_after.copy()

        elif abs(a_c - a_t) > 10.:
            # Hohmann simple (Δi ≤ 0.1°, changement d'altitude seul)
            v_c = float(np.linalg.norm(v_cur))
            a_tr = (a_c + a_t) / 2
            v1m = math.sqrt(MU * (2./float(np.linalg.norm(r_cur)) - 1./a_tr))
            v2m = math.sqrt(MU / a_t)
            T_h = math.pi * math.sqrt(a_tr**3 / MU)
            v_dep = v_cur / np.linalg.norm(v_cur) * v1m
            r_end, v_end = propagate_kepler(r_cur, v_dep, T_h)
            v_e2 = v_end / np.linalg.norm(v_end) * v2m
            phases.append({
                'phase': 1, 'type': 'hohmann', 'label': 'Hohmann altitude',
                'duration_h': round(T_h/3600, 3),
                'burns': [
                    {'n': 1, 'label': 'ΔV₁ Hohmann prograde',
                     't_h': round(t_elapsed/3600, 4),
                     'dv_ms': round(abs(v1m - v_c)*1000, 2),
                     'desc': f'{a_c-RE:.0f}km → {a_t-RE:.0f}km'},
                    {'n': 2, 'label': 'ΔV₂ Circularisation',
                     't_h': round((t_elapsed + T_h)/3600, 4),
                     'dv_ms': round(abs(v2m - float(np.linalg.norm(v_end)))*1000, 2),
                     'desc': f'Circularisation à {a_t-RE:.0f}km'},
                ],
                'r_end': r_end.tolist(), 'v_end': v_e2.tolist(),
            })
            t_elapsed += T_h
            r_cur = r_end.copy(); v_cur = v_e2.copy()

    # 
    # PHASE 2 : Phasage (orbite de phasage)
    # 
    el_c2 = rv_to_elements(r_cur, v_cur)
    rt_now, vt_now = propagate_kepler(rt0, vt0, t_elapsed)
    el_t2 = rv_to_elements(rt_now, vt_now)

    u_c2 = (el_c2['argp'] + el_c2['nu']) % (2*math.pi)
    u_t2 = (el_t2['argp'] + el_t2['nu']) % (2*math.pi)
    theta_0   = (u_t2 - u_c2) % (2*math.pi)   # target devant chaser
    a_now     = el_c2['a']
    T_t_now   = el_t2['T']

    # Choisir l'orbite de phasage : delta_a adaptatif
    # Objectif : durée < 200h si possible
    theta_req = 0.          # rad (on veut phase = 0)
    best_phase = None
    for dA_sign in [-1., +1.]:
        for dA in [5., 10., 20., 30., 50.]:
            a_ph = a_now + dA_sign * dA
            if a_ph < 6500 or a_ph > 50000: continue
            T_ph = 2*math.pi*math.sqrt(a_ph**3/MU)
            dn   = 2*math.pi/T_ph - 2*math.pi/T_t_now
            if abs(dn) < 1e-15: continue
            # Si dn > 0 (phase plus rapide) → angle cible diminue → delta = theta_0 - theta_req
            # Si dn < 0 (phase plus lente)  → angle cible augmente → delta = theta_req - theta_0
            if dn > 0:
                delta = (theta_0 - theta_req) % (2*math.pi)
            else:
                delta = (theta_req - theta_0) % (2*math.pi)
            t_ph = delta / abs(dn)
            if best_phase is None or t_ph < best_phase['t']:
                dv_in  = abs(math.sqrt(MU/a_ph) - math.sqrt(MU/a_now)) * 1000
                best_phase = {'t': t_ph, 'a_ph': a_ph, 'dA': dA*dA_sign,
                              'dv': dv_in, 'T_ph': T_ph}

    if best_phase is None:
        best_phase = {'t': 0., 'a_ph': a_now, 'dA': 0., 'dv': 0., 'T_ph': T_t_now}

    t_ph = best_phase['t']
    dv_ph = best_phase['dv']
    a_ph  = best_phase['a_ph']
    dA    = best_phase['dA']

    if t_ph > 60:    # > 1 min : phasage nécessaire
        desc_in  = ('Rétro' if dA < 0 else 'Prograde') + f' {abs(dA):.0f}km (T_syn={t_ph/3600:.1f}h)'
        phases.append({
            'phase': 2,
            'type':  'phasing_orbit',
            'label': 'Phasage orbital',
            'duration_h': round(t_ph/3600, 3),
            'phase_angle_deg': round(math.degrees(theta_0), 2),
            'delta_a_km': round(dA, 1),
            'burns': [
                {'n': len(sum([p['burns'] for p in phases],[]))+1,
                 'label': 'ΔV Injection orbite de phasage',
                 't_h': round(t_elapsed/3600, 4),
                 'dv_ms': round(dv_ph, 3),
                 'desc': desc_in},
                {'n': len(sum([p['burns'] for p in phases],[]))+2,
                 'label': 'ΔV Retour orbite cible',
                 't_h': round((t_elapsed + t_ph)/3600, 4),
                 'dv_ms': round(dv_ph, 3),
                 'desc': f'Retour à {a_now-RE:.0f}km'},
            ],
            'r_end': None, 'v_end': None,
        })
        t_elapsed += t_ph
        # Après phasage, chaser est sur orbite cible, aligné avec target

    # 
    # PHASE 3 : Approche terminale (Hohmann depuis standoff 5km)
    # 
    standoff_km = 5.
    a_approach  = a_now - standoff_km / 2.   # orbite légèrement plus basse
    T_appr      = math.pi * math.sqrt(((a_now + a_approach)/2)**3 / MU)
    dv_appr     = abs(math.sqrt(MU/a_approach) - math.sqrt(MU/a_now)) * 1000

    n_burn_base = len(sum([p['burns'] for p in phases], []))
    phases.append({
        'phase': 3,
        'type':  'approach',
        'label': 'Approche terminale (5km → 0km)',
        'duration_h': round(T_appr/3600, 3),
        'burns': [
            {'n': n_burn_base+1, 'label': 'ΔV Injection approche',
             't_h': round(t_elapsed/3600, 4),
             'dv_ms': round(dv_appr, 3),
             'desc': f'Rétrograde → standoff {standoff_km:.0f}km'},
            {'n': n_burn_base+2, 'label': 'ΔV Circularisation standoff',
             't_h': round((t_elapsed + T_appr)/3600, 4),
             'dv_ms': round(dv_appr, 3),
             'desc': 'Hold point K (5km)'},
        ],
    })
    t_elapsed += T_appr

    # 
    # PHASE 4 : Proximité CW (1km → dock)
    # 
    v_approach = 0.5   # m/s
    t_cw = 1000 / v_approach   # s (1km à 0.5 m/s)
    dv_cw = v_approach / 1000  # km/s (correction initiale)

    n_burn_base2 = len(sum([p['burns'] for p in phases], []))
    phases.append({
        'phase': 4,
        'type':  'cw_proximity',
        'label': f'Proximité CW (1km → dock)',
        'duration_h': round(t_cw/3600, 4),
        'approach_v_ms': v_approach,
        'burns': [
            {'n': n_burn_base2+1, 'label': 'ΔV V-bar approach',
             't_h': round(t_elapsed/3600, 4),
             'dv_ms': round(dv_cw*1000, 3),
             'desc': f'V-bar {v_approach:.1f} m/s'},
            {'n': n_burn_base2+2, 'label': 'ΔV Docking',
             't_h': round((t_elapsed+t_cw)/3600, 4),
             'dv_ms': round(dv_cw*1000, 3),
             'desc': 'Arrêt relatif + docking'},
        ],
    })

    #  Bilan total 
    all_burns = sum([p['burns'] for p in phases], [])
    total_dv  = sum(b['dv_ms'] for b in all_burns)
    total_h   = (t_elapsed + t_cw) / 3600

    return {
        'phases':     phases,
        'all_burns':  all_burns,
        'n_burns':    len(all_burns),
        'total_dv_ms': round(total_dv, 2),
        'total_duration_h': round(total_h, 2),
        'summary': (f"{len(phases)} phases, {len(all_burns)} manœuvres, "
                    f"ΔV={total_dv:.0f} m/s, durée={total_h/24:.1f} jours"),
    }


def compute_plane_change_to_target(r0, v0, rt0, vt0):
    """
    Changement de plan orbital optimal en 1 burn.
    Burn au nœud ascendant/descendant (intersection des deux plans).
    
    Applicable quand delta_plane <= 20° (satellite conventionnel).
    Référence : Vallado (2013) §6.2, Curtis (2013) §6.7
    
    Retourne :
      t_node_s    : temps d'attente jusqu'au nœud optimal (s)
      dv_ms       : module du ΔV (m/s)
      dv_vec      : vecteur ΔV en km/s (ECI)
      r_burn      : position au burn (km, ECI)
      v_before    : vitesse avant burn (km/s, ECI)
      v_after     : vitesse après burn (km/s, ECI)
      delta_plane : angle dièdre entre les deux plans (°)
      T_c         : période du chaser (s)
    """
    r0  = np.array(r0,  dtype=float)
    v0  = np.array(v0,  dtype=float)
    rt0 = np.array(rt0, dtype=float)
    vt0 = np.array(vt0, dtype=float)

    el_c = rv_to_elements(r0,  v0 )
    el_t = rv_to_elements(rt0, vt0)
    T_c  = el_c['T']

    # Plans orbitaux (vecteurs normaux unitaires)
    h_c = np.cross(r0,  v0 );  h_c_hat = h_c / np.linalg.norm(h_c)
    h_t = np.cross(rt0, vt0);  h_t_hat = h_t / np.linalg.norm(h_t)

    # Angle dièdre entre les plans
    cos_d = float(np.clip(np.dot(h_c_hat, h_t_hat), -1., 1.))
    delta_plane_deg = math.degrees(math.acos(cos_d))

    # Ligne d'intersection des deux plans = h_c × h_t (direction du nœud)
    node_line = np.cross(h_c_hat, h_t_hat)
    nrm = float(np.linalg.norm(node_line))
    if nrm < 1e-8:
        # Plans (quasi-)parallèles : pas de changement nécessaire
        return {
            't_node_s':     0.,
            'dv_ms':        0.,
            'dv_vec':       [0., 0., 0.],
            'r_burn':       r0.tolist(),
            'v_before':     v0.tolist(),
            'v_after':      v0.tolist(),
            'delta_plane':  delta_plane_deg,
            'T_c':          T_c,
        }
    node_hat = node_line / nrm

    # Chercher le passage au nœud le plus proche (dans [0, T_c])
    # Le nœud est là où la position est alignée avec node_hat
    best_t   = 0.
    best_dot = -1.
    N_search = 500
    for k in range(N_search):
        t_k = T_c * k / N_search
        r_k, _ = propagate_kepler(r0, v0, t_k)
        d = abs(float(np.dot(r_k / np.linalg.norm(r_k), node_hat)))
        if d > best_dot:
            best_dot = d
            best_t   = t_k

    # Raffiner
    step = T_c / N_search
    for k in range(100):
        t_k = best_t - step + 2*step*k/99
        if t_k < 0: continue
        r_k, _ = propagate_kepler(r0, v0, t_k)
        d = abs(float(np.dot(r_k / np.linalg.norm(r_k), node_hat)))
        if d > best_dot:
            best_dot = d
            best_t   = t_k

    # État au nœud
    r_burn, v_before = propagate_kepler(r0, v0, best_t)
    r_burn_mag = float(np.linalg.norm(r_burn))

    # Nouvelle vitesse : même magnitude, dans le plan cible
    # Direction : tangente à l'orbite cible passant par r_burn
    # = perpendiculaire à r_burn dans le plan cible
    v_dir_new = np.cross(h_t_hat, r_burn / r_burn_mag)
    nrm2 = float(np.linalg.norm(v_dir_new))
    if nrm2 < 1e-8:
        v_after = v_before.copy()
    else:
        v_dir_new /= nrm2
        # Sens prograde (cohérent avec le vecteur original)
        if float(np.dot(v_dir_new, v_before)) < 0:
            v_dir_new = -v_dir_new
        # Magnitude : vitesse circulaire au rayon r_burn (conservation énergie)
        v_mag = float(np.linalg.norm(v_before))
        v_after = v_dir_new * v_mag

    dv_vec = (v_after - v_before)
    dv_ms  = float(np.linalg.norm(dv_vec)) * 1000.

    return {
        't_node_s':     float(best_t),
        'dv_ms':        round(dv_ms, 3),
        'dv_vec':       dv_vec.tolist(),
        'r_burn':       r_burn.tolist(),
        'v_before':     v_before.tolist(),
        'v_after':      v_after.tolist(),
        'delta_plane':  round(delta_plane_deg, 4),
        'T_c':          T_c,
    }


def _phase_wait(el_c: dict, el_t: dict, tof_s: float,
               r0_c=None, v0_c=None, r0_t=None, v0_t=None) -> dict:
    """
    Calcule le temps d'attente de phasage optimal.
    Méthode numérique robuste si les vecteurs d'état sont fournis :
    minimise la distance réelle au TCA par propagation képlerienne exacte.
    Références : Vallado (2013) §6.4
    """
    n_c = math.sqrt(MU / el_c['a']**3)
    n_t = math.sqrt(MU / el_t['a']**3)
    u_c = el_c['argp'] + el_c['nu']
    u_t = el_t['argp'] + el_t['nu']
    theta_0   = (u_t - u_c) % (2*math.pi)
    theta_req = (math.pi - n_t * tof_s) % (2*math.pi)
    dn        = n_t - n_c
    T_syn     = 2*math.pi / abs(dn) if abs(dn) > 1e-12 else 1e9

    # Analytique (ordre 0)
    if dn < 0:
        delta_theta = (theta_0 - theta_req) % (2*math.pi)
    else:
        delta_theta = (theta_req - theta_0) % (2*math.pi)
    t_wait_analytic = delta_theta / abs(dn) if abs(dn) > 1e-12 else 0.

    # Recherche numérique si vecteurs disponibles (robuste à l'excentricité)
    t_wait = t_wait_analytic
    dist_tca = float('inf')

    if r0_c is not None and v0_c is not None and r0_t is not None and v0_t is not None:
        r0_c = np.array(r0_c); v0_c = np.array(v0_c)
        r0_t = np.array(r0_t); v0_t = np.array(v0_t)
        T_c = el_c['T']
        a_c = el_c['a']; a_t = el_t['a']
        a_tr = (a_c + a_t) / 2

        def _eval(tw):
            try:
                r_w, v_w = propagate_kepler(r0_c, v0_c, max(1., tw % T_c))
                v_tr = math.sqrt(MU * (2. / float(np.linalg.norm(r_w)) - 1. / a_tr))
                v_dep = v_w / np.linalg.norm(v_w) * v_tr
                r_arr, _ = propagate_kepler(r_w, v_dep, tof_s)
                r_tca, _ = propagate_kepler(r0_t, v0_t, tw + tof_s)
                return float(np.linalg.norm(r_arr - r_tca))
            except Exception:
                return float('inf')

        # Phase 1 : grille grossière sur T_syn (200 pts)
        N1 = 200
        for k in range(N1):
            tw = T_syn * k / N1
            d = _eval(tw)
            if d < dist_tca:
                dist_tca = d; t_wait = tw

        # Phase 2 : raffiner autour du minimum
        step = T_syn / N1
        for k in range(80):
            tw = max(0., t_wait - step + 2*step*k/79)
            d = _eval(tw)
            if d < dist_tca:
                dist_tca = d; t_wait = tw

    return {
        't_wait_s':      float(t_wait),
        'theta_0_deg':   float(math.degrees(theta_0)),
        'theta_req_deg': float(math.degrees(theta_req)),
        'T_syn_h':       float(T_syn / 3600),
        'dn_deg_day':    float(math.degrees(dn) * 86400),
        'dist_tca_km':   round(dist_tca, 2) if dist_tca < float('inf') else None,
    }


def compute_rendezvous_physical(
    tle1_c: str, tle2_c: str,
    tle1_t: str, tle2_t: str,
    method: str = 'hohmann',
    dv_max_ms: float = 500.,
    approach_angle_deg: float = 0.,
) -> dict:
    """
    Calcul de rendezvous selon Vallado (2013) Ch. 6.

    Pipeline selon la méthode :
      Hohmann / Phasage → calcul analytique angle de phase + ΔV Hohmann
      Bi-elliptique → 3 burns analytiques
      Bi-ell. haute alt. → changement de plan au nœud optimal
      Lowthrust → Edelbaum

    Retourne un dict avec maneuvers, segments, summary, warning.
    """
    dt = datetime.now(timezone.utc)
    r0,  v0  = propagate_sgp4(tle1_c, tle2_c, dt)
    rt0, vt0 = propagate_sgp4(tle1_t, tle2_t, dt)
    el_c = rv_to_elements(r0,  v0)
    el_t = rv_to_elements(rt0, vt0)

    #  Angle entre les plans orbitaux 
    h_c = el_c['h_vec'] / el_c['h']
    h_t = el_t['h_vec'] / el_t['h']
    delta_plane = math.acos(np.clip(np.dot(h_c, h_t), -1., 1.))
    delta_plane_deg = math.degrees(delta_plane)

    maneuvers  = []
    segments   = {}
    warning    = None
    N_PTS      = 3000

    #  Méthodes impulsionnelles 
    if method in ('hohmann', 'phasing', 'bielliptic', 'bielliptic_highalt'):

        # Phase 1 : correction de plan si nécessaire
        if delta_plane_deg > 5. and method != 'bielliptic_highalt':
            # Burn combiné au nœud d'intersection (Vallado §6.7)
            v_c = math.sqrt(MU / el_c['a'])
            v_t = math.sqrt(MU / el_t['a'])
            dv_plane = math.sqrt(v_c**2 + v_t**2
                                 - 2*v_c*v_t*math.cos(delta_plane)) * 1000
            warning = (
                f"Δ-plan={delta_plane_deg:.1f}° — correction de plan requise. "
                f"ΔV burn combiné (nœud) ≈ {dv_plane:.0f}m/s. "
                f"Utilisez la méthode Bi-ell. haute alt. pour Δi>{delta_plane_deg:.0f}°."
            )

        if method == 'bielliptic_highalt':
            be = bielliptic_plane_change(el_c, el_t)
            for m in be['maneuvers']:
                maneuvers.append(m)
            total_dv = be['total_dv']
            t_total_h = be['duration_h']
            description = (
                f"Bi-elliptique haute altitude r_b={be['r_b_km']:.0f}km — "
                f"Δ-plan={delta_plane_deg:.1f}° — borne inf={be['dv_lower_bound_ms']:.0f}m/s"
            )
            return {
                'method':          method,
                'total_dv_ms':     round(total_dv, 2),
                'wait_h':          0.,
                'transfer_time_h': round(t_total_h, 3),
                'maneuvers':       maneuvers,
                'description':     description,
                'delta_plane_deg': round(delta_plane_deg, 2),
                'chaser_alt_km':   round(el_c['a'] - RE, 1),
                'target_alt_km':   round(el_t['a'] - RE, 1),
                'delta_alt_km':    round(abs(el_t['a'] - el_c['a']), 1),
                'bielliptic_highalt': be,
            }

        # Phase 2 : Hohmann entre les deux altitudes
        a1 = el_c['a']; a2 = el_t['a']
        if method == 'bielliptic' and abs(a2/a1) > 1.5:
            # Bi-elliptique analytique (Vallado §6.3.3)
            r_b = a2 * 2.0  # orbite intermédiaire = 2× apogée cible
            a_tr1 = (a1 + r_b) / 2
            a_tr2 = (r_b + a2) / 2
            v1 = math.sqrt(MU/a1); v2 = math.sqrt(MU/a2)
            vp1 = math.sqrt(MU*(2/a1 - 1/a_tr1))
            va1 = math.sqrt(MU*(2/r_b - 1/a_tr1))
            va2 = math.sqrt(MU*(2/r_b - 1/a_tr2))
            vp2 = math.sqrt(MU*(2/a2 - 1/a_tr2))
            dv1_ms = abs(vp1 - v1)*1000
            dv2_ms = abs(va2 - va1)*1000
            dv3_ms = abs(v2 - vp2)*1000
            tof1 = math.pi*math.sqrt(a_tr1**3/MU)
            tof2 = math.pi*math.sqrt(a_tr2**3/MU)
            tof_s = tof1 + tof2
            total_dv = dv1_ms + dv2_ms + dv3_ms
        else:
            # Hohmann standard
            dv1_ms, dv2_ms, tof_s = _hohmann_dv(a1, a2)
            dv3_ms = 0.
            total_dv = dv1_ms + dv2_ms

        # Phase 3 : phasage numérique robuste (Kepler exact)
        phase = _phase_wait(el_c, el_t, tof_s,
                           r0_c=r0, v0_c=v0, r0_t=rt0, v0_t=vt0)
        t_wait_s = phase['t_wait_s']

        # Pour la méthode phasage : utiliser une orbite de phasage temporaire
        if method == 'phasing':
            # Orbite de phasage : ajuster a pour raccourcir l'attente
            # Nombre de révolutions souhaité N tel que N*T_phase = t_wait_s_original
            N_rev = max(1, int(t_wait_s / el_c['T']))
            T_phase = t_wait_s / N_rev if N_rev > 0 else el_c['T']
            a_phase = (MU * (T_phase/(2*math.pi))**2)**(1/3)
            dv_phase1 = abs(math.sqrt(MU/a_phase) - math.sqrt(MU/a1))*1000
            dv_phase2 = dv_phase1  # retour
            t_wait_s = N_rev * T_phase

            maneuvers.append({
                'type':           'ΔV₁ Phasage',
                't_from_now_h':   0.,
                'dv_ms':          round(dv_phase1, 3),
                'description':    (f"Orbite de phasage a={a_phase-RE:.0f}km, "
                                   f"N={N_rev} révolutions sur {t_wait_s/3600:.1f}h"),
            })
            maneuvers.append({
                'type':           'ΔV₂ Retour orbite initiale',
                't_from_now_h':   round(t_wait_s/3600, 4),
                'dv_ms':          round(dv_phase2, 3),
                'description':    'Retour sur orbite initiale avant Hohmann',
            })
            total_dv += dv_phase1 + dv_phase2

        # Burns Hohmann
        t_burn1_h = round(t_wait_s/3600, 4)
        t_burn2_h = round((t_wait_s + tof_s)/3600, 4)
        maneuvers.append({
            'type':           'ΔV Hohmann₁',
            't_from_now_h':   t_burn1_h,
            'dv_ms':          round(dv1_ms, 3),
            'description':    (f"Injection ellipse transfert, "
                               f"{'prograde' if a2 >= a1 else 'rétrograde'}, "
                               f"alt={a1-RE:.0f}→{(a1+a2)/2-RE:.0f}km"),
        })
        if method == 'bielliptic' and dv3_ms > 0.:
            t_mid_h = round((t_wait_s + tof1)/3600, 4)
            maneuvers.append({
                'type':           'ΔV Bi-ell₂',
                't_from_now_h':   t_mid_h,
                'dv_ms':          round(dv2_ms, 3),
                'description':    f"Burn intermédiaire à r_b={r_b-RE:.0f}km",
            })
        maneuvers.append({
            'type':           'ΔV Hohmann₂',
            't_from_now_h':   t_burn2_h,
            'dv_ms':          round(dv2_ms if method != 'bielliptic' else dv3_ms, 3),
            'description':    f"Circularisation orbite cible alt={a2-RE:.0f}km",
        })

        # Approche terminale CW si angle d'approche spécifié
        if approach_angle_deg != 0.:
            n_t = math.sqrt(MU / a2**3)
            v_approach = 1.0  # m/s d'approche V-bar
            ang = math.radians(approach_angle_deg)
            dv_cw = abs(v_approach * math.sin(ang))
            if dv_cw > 0.001:
                maneuvers.append({
                    'type':           'ΔV Approche CW',
                    't_from_now_h':   round(t_burn2_h + 0.5, 4),
                    'dv_ms':          round(dv_cw, 3),
                    'description':    (f"Correction CW angle={approach_angle_deg}°, "
                                       f"1km→<1km dans LVLH"),
                })
                total_dv += dv_cw

        if total_dv > dv_max_ms and not warning:
            warning = (
                f"ΔV total ({total_dv:.1f}m/s) > budget ({dv_max_ms:.0f}m/s). "
                f"Augmentez le budget ou utilisez la poussée faible."
            )

        description = (
            f"{'Hohmann' if method=='hohmann' else method.title()} — "
            f"Δalt={abs(a2-a1):.1f}km, Δ-plan={delta_plane_deg:.1f}°, "
            f"attente={t_wait_s/3600:.1f}h (T_syn={phase['T_syn_h']:.0f}h)"
        )

        return {
            'method':          method,
            'total_dv_ms':     round(total_dv, 3),
            'wait_h':          round(t_wait_s/3600, 4),
            'transfer_time_h': round(tof_s/3600, 4),
            'maneuvers':       maneuvers,
            'description':     description,
            'delta_plane_deg': round(delta_plane_deg, 2),
            'chaser_alt_km':   round(el_c['a'] - RE, 1),
            'target_alt_km':   round(el_t['a'] - RE, 1),
            'delta_alt_km':    round(abs(a2 - a1), 1),
            'phasing':         phase,
            'warning':         warning,
        }

    # Dérive J2 passive
    if method == 'drift_j2':
        return compute_drift_j2(el_c, el_t)
    # Poussée faible : Edelbaum
    return compute_rendezvous(tle1_c, tle2_c, tle1_t, tle2_t,
                              method=method, dv_max_ms=dv_max_ms)


def compute_rendezvous_graphs(
    r0_c, v0_c, r0_t, v0_t,
    el_c: dict, el_t: dict,
    wait_s: float, tof_s: float,
    pts_tr: list, r_arr, v_arr_c,
    approach_angle_deg: float = 0.,
) -> dict:
    """
    Génère les données des 4 graphiques rendezvous.
    Physiquement rigoureux — pas d'interpolation linéaire.

    Référence : Vallado (2013) Ch. 6-7, Curtis (2013) Ch. 7
    """
    r0_c  = np.array(r0_c);  v0_c  = np.array(v0_c)
    r0_t  = np.array(r0_t);  v0_t  = np.array(v0_t)
    r_arr = np.array(r_arr); v_arr_c = np.array(v_arr_c)

    T_c = el_c['T'];  T_t = el_t['T']
    a_t = el_t['a'];  n_t = math.sqrt(MU / a_t**3)
    total_s = wait_s + tof_s + 2 * T_t

    #  C — Angle de phase (u_t - u_c) vs temps 
    # Calculé à chaque orbite pendant le phasage (pas d'interpolation linéaire)
    N_ph = min(400, max(60, int(wait_s / T_c * 10)))  # 10 pts/orbite pendant attente
    step_ph = wait_s / N_ph if N_ph > 0 else T_c
    phase_tl = []
    for k in range(N_ph + 1):
        t_k = step_ph * k
        try:
            rc_, vc_ = propagate_kepler(r0_c, v0_c, t_k % T_c)
            rt_, vt_ = propagate_kepler(r0_t, v0_t, t_k)
            el_c_k = rv_to_elements(rc_, vc_)
            el_t_k = rv_to_elements(rt_, vt_)
            u_c = el_c_k['argp'] + el_c_k['nu']
            u_t = el_t_k['argp'] + el_t_k['nu']
            # Angle cible devant chaser (positif = cible en avance)
            delta_u = math.degrees((u_t - u_c) % (2 * math.pi))
            phase_tl.append([round(t_k / 3600, 4), round(delta_u, 2)])
        except Exception:
            continue
    # Ajouter le TCA (delta ≈ theta_req)
    if tof_s > 0:
        phase_tl.append([round((wait_s + tof_s) / 3600, 4), 0.])

    #  D — Altitude vs Temps 
    # Chaser + cible sur toute la mission
    N_alt = 300
    alt_tl = []
    for k in range(N_alt + 1):
        t_k = total_s * k / N_alt
        try:
            if t_k <= wait_s:
                rc_, _ = propagate_kepler(r0_c, v0_c, t_k % T_c)
                tag = 'wait'
            elif t_k <= wait_s + tof_s:
                frac = (t_k - wait_s) / tof_s
                idx_c = min(int(frac * len(pts_tr)), len(pts_tr) - 1) if pts_tr else 0
                rc_ = np.array(pts_tr[idx_c][:3]) if pts_tr else r_arr
                tag = 'transfer'
            else:
                t_after = t_k - (wait_s + tof_s)
                rc_, _ = propagate_kepler(r_arr, v_arr_c, t_after)
                tag = 'final'
            rt_, _ = propagate_kepler(r0_t, v0_t, t_k)
            alt_c = float(np.linalg.norm(rc_)) - RE
            alt_t = float(np.linalg.norm(rt_)) - RE
            alt_tl.append([round(t_k / 3600, 4), round(alt_c, 2), round(alt_t, 2), tag])
        except Exception:
            continue

    #  Distance inter-satellite réelle (oscillante) 
    # 20 pts/orbite → montre l'oscillation physique
    N_dist = min(2000, max(200, int(total_s / T_c * 20)))
    step_d = total_s / N_dist
    dist_tl = []
    for k in range(N_dist + 1):
        t_k = step_d * k
        try:
            if t_k <= wait_s:
                rc_, _ = propagate_kepler(r0_c, v0_c, t_k % T_c)
            elif t_k <= wait_s + tof_s and pts_tr:
                frac = (t_k - wait_s) / tof_s
                idx_c = min(int(frac * len(pts_tr)), len(pts_tr) - 1)
                rc_ = np.array(pts_tr[idx_c][:3])
            else:
                t_after = t_k - (wait_s + tof_s)
                rc_, _ = propagate_kepler(r_arr, v_arr_c, t_after)
            rt_, _ = propagate_kepler(r0_t, v0_t, t_k)
            d_km = float(np.linalg.norm(np.array(rc_) - rt_))
            dist_tl.append([round(t_k / 3600, 4), round(d_km, 1)])
        except Exception:
            continue

    #  B — CW V-bar/R-bar en LVLH (approche terminale ≤ 1 km) 
    # Repère LVLH centré sur la cible à l'arrivée
    # x = R-bar (radial, positif vers le haut)
    # y = V-bar (along-track, positif dans le sens du mouvement)
    # Équations CW : Vallado §7.6, Curtis §7.4
    #
    # Approche depuis K-point (1km) selon l'angle d'approche
    ang_rad = math.radians(approach_angle_deg)
    d_kp    = 1.0         # km — distance K-point
    x0_km   =  d_kp * math.sin(ang_rad)  # composante R-bar initiale
    y0_km   =  d_kp * math.cos(ang_rad)  # composante V-bar initiale
    # Vitesse d'approche : 0.5 m/s → vers l'origine
    v_approach_kms = 0.0005   # km/s = 0.5 m/s
    # Direction d'approche : vers (0,0)
    dist_kp = math.sqrt(x0_km**2 + y0_km**2)
    vx0 = -v_approach_kms * x0_km / dist_kp
    vy0 = -v_approach_kms * y0_km / dist_kp

    T_approach_s = dist_kp / v_approach_kms   # durée théorique sans drift CW
    N_cw = 300
    cw_data = []
    for k in range(N_cw + 1):
        t = T_approach_s * k / N_cw
        c = math.cos(n_t * t);  s = math.sin(n_t * t);  nt = n_t * t
        # CW equations (units: km)
        x = (4 - 3*c)*x0_km + s/n_t*vx0 + 2*(1-c)/n_t*vy0
        y = 6*(s - nt)*x0_km + y0_km - 2*(1-c)/n_t*vx0 + (4*s - 3*nt)/n_t*vy0
        # Convert to meters for proximity graph
        cw_data.append([round(y * 1000, 2), round(x * 1000, 2)])

    return dict(
        phase_tl  = phase_tl,
        alt_tl    = alt_tl,
        dist_tl   = dist_tl,
        cw_data   = cw_data,
        cw_params = dict(
            x0_m    = x0_km * 1000,
            y0_m    = y0_km * 1000,
            v_app   = v_approach_kms * 1000,
            T_app_s = T_approach_s,
            angle_deg = approach_angle_deg,
        ),
    )


def compute_drift_j2(el_c:dict, el_t:dict) -> dict:
    """
    Phasage passif via précession différentielle J2 (0 ΔV actif).
    Vallado §9.6.2 — applicable si les orbites précèdent dans le même sens
    et si l'écart de taux est suffisant.
    """
    j2=j2_convergence(el_c, el_t, max_days=365)
    if j2['converges']:
        return {
            'method':          'drift_j2',
            'total_dv_ms':     0.,
            'wait_h':          float(j2['min_day']*24),
            'transfer_time_h': 0.,
            'maneuvers': [{
                'type':          'Dérive J2 (passive)',
                't_from_now_h':  0.,
                'dv_ms':         0.,
                'description':   (
                    f"Phasage naturel par précession J2. "
                    f"dΩ_chaser={j2['dOmega_c_deg_day']:.3f}°/j, "
                    f"dΩ_cible={j2['dOmega_t_deg_day']:.3f}°/j. "
                    f"Convergence en {j2['min_day']:.0f} jours."
                ),
            }],
            'description':     (f"Phasage passif J2 — 0 ΔV actif. "
                                 f"Attente {j2['min_day']:.0f} jours "
                                 f"(angle min {j2['min_angle_deg']:.1f}°)"),
            'delta_plane_deg': float(j2['min_angle_deg']),
            'chaser_alt_km':   round(el_c['a']-RE, 1),
            'target_alt_km':   round(el_t['a']-RE, 1),
            'delta_alt_km':    round(abs(el_t['a']-el_c['a']), 1),
            'j2_analysis':     j2,
        }
    else:
        return {
            'method':          'drift_j2',
            'total_dv_ms':     float('inf'),
            'wait_h':          0.,
            'transfer_time_h': 0.,
            'maneuvers':       [],
            'description':     "Dérive J2 inefficace pour ce couple d'orbites.",
            'warning': (
                f"J2 ne converge pas — plans précèdent en sens opposés "
                f"({j2['dOmega_c_deg_day']:.2f}°/j vs {j2['dOmega_t_deg_day']:.2f}°/j). "
                "Utilisez un burn actif."
            ),
            'error': 'J2 diverges',
        }
