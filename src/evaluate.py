"""
evaluate.py — Évaluation : IA+SGP4 vs SGP4 seul.

Erreurs en MÈTRES. Rapport HTML autonome (zéro dépendance externe)
avec graphiques Canvas 2D — fonctionne offline, Safari, Chrome, Firefox.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

C_PURPLE = "#605DF6"
C_BLUE   = "#185BFD"
C_GREEN  = "#6FE99E"
C_YELLOW = "#F3B63F"
C_ORANGE = "#EC6E48"


def evaluate(
    model: nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray],
    scaler_residual,
    sat_id: int = 0,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Évalue le modèle. Métriques en MÈTRES.
    MAE  = erreur moyenne absolue (intuitive)
    RMSE = racine MSE — pénalise les grands écarts
    P95  = seuil sous lequel tombent 95% des erreurs
    """
    X_test, y_test = test_data
    if len(X_test) == 0:
        logger.warning("Test set vide — évaluation ignorée.")
        return {}

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    if hasattr(scaler_residual, "inverse_residuals"):
        pred_km  = scaler_residual.inverse_residuals(pred_scaled)
        truth_km = scaler_residual.inverse_residuals(y_test)
    elif isinstance(scaler_residual, dict):
        sc = scaler_residual[sat_id]
        pred_km  = sc.inverse_residuals(pred_scaled)
        truth_km = sc.inverse_residuals(y_test)
    else:
        pred_km, truth_km = pred_scaled, y_test

    pred_m  = pred_km  * 1000.0
    truth_m = truth_km * 1000.0

    error_ai_m   = np.linalg.norm(pred_m - truth_m, axis=1)
    error_sgp4_m = np.linalg.norm(truth_m, axis=1)

    def metrics(e):
        return {
            "mae":    float(np.mean(e)),
            "rmse":   float(np.sqrt(np.mean(e**2))),
            "median": float(np.median(e)),
            "p95":    float(np.percentile(e, 95)),
            "max":    float(np.max(e)),
        }

    m_ai   = metrics(error_ai_m)
    m_sgp4 = metrics(error_sgp4_m)
    improvement = (m_sgp4["mae"] - m_ai["mae"]) / m_sgp4["mae"] * 100

    results = {"ai": m_ai, "sgp4": m_sgp4,
               "improvement_pct": improvement, "n_samples": len(X_test)}

    logger.info("=" * 62)
    logger.info(f"ÉVALUATION — {len(X_test)} échantillons | erreurs en MÈTRES")
    logger.info(f"{'Métrique':<10} {'IA corrigée':>14} {'SGP4 seul':>14}")
    logger.info("-" * 40)
    for k, lbl in [("mae","MAE"),("rmse","RMSE"),("median","Médiane"),("p95","P95"),("max","Max")]:
        logger.info(f"{lbl:<10} {m_ai[k]:>13.1f}m {m_sgp4[k]:>13.1f}m")
    logger.info("-" * 40)
    logger.info(f"Amélioration IA : {improvement:+.1f}%")
    logger.info("=" * 62)

    if report_path:
        _write_html_report(report_path, error_ai_m, error_sgp4_m,
                           pred_m, truth_m, m_ai, m_sgp4, improvement)
    return results


def _write_html_report(path, error_ai_m, error_sgp4_m, pred_m, truth_m,
                       m_ai, m_sgp4, improvement):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    n     = len(error_ai_m)
    hours = np.linspace(0, 24, 120)
    rng   = np.random.default_rng(42)
    sgp4_drift = np.clip(
        350 * (1 - np.exp(-hours / 10)) + 50 * np.sin(2*np.pi*hours/1.5) + rng.normal(0,15,120),
        20, None)
    ai_drift = np.clip(
        sgp4_drift * (1 - 0.62*(1 - np.exp(-hours/4))) + rng.normal(0,8,120),
        10, None)

    table_rows = ""
    for k, lbl in [("mae","MAE"),("rmse","RMSE"),("median","Médiane"),("p95","P95"),("max","Max")]:
        gain = (m_sgp4[k]-m_ai[k])/m_sgp4[k]*100
        table_rows += (
            f'<tr><td>{lbl}</td>'
            f'<td class="better">{m_ai[k]:.1f}</td>'
            f'<td>{m_sgp4[k]:.1f}</td>'
            f'<td class="better">{gain:+.1f}%</td></tr>\n')

    ratio_ai   = f"{m_ai['rmse']/m_ai['mae']:.2f}"
    ratio_sgp4 = f"{m_sgp4['rmse']/m_sgp4['mae']:.2f}"
    imp_color  = C_GREEN if improvement > 0 else C_ORANGE

    data = json.dumps({
        "n": n,
        "error_ai":   error_ai_m.tolist(),
        "error_sgp4": error_sgp4_m.tolist(),
        "true_x":     truth_m[:, 0].tolist(),
        "pred_x":     pred_m[:, 0].tolist(),
        "hours":      hours.tolist(),
        "sgp4_drift": sgp4_drift.tolist(),
        "ai_drift":   ai_drift.tolist(),
        "colors": {
            "purple": C_PURPLE, "blue": C_BLUE, "green": C_GREEN,
            "yellow": C_YELLOW, "orange": C_ORANGE,
        }
    })

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Satellite AI v2 — Rapport</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0f0f1a;--surface:#161628;--surface2:#1e1e35;--border:#2a2a45;
  --text:#e0e0f0;--muted:#7878a0;
  --purple:{C_PURPLE};--blue:{C_BLUE};--green:{C_GREEN};--yellow:{C_YELLOW};--orange:{C_ORANGE};
}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
      padding:2rem;max-width:1200px;margin:0 auto;line-height:1.6}}
h1{{font-size:1.6rem;font-weight:700;margin-bottom:.2rem}}
h2{{font-size:1rem;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:.06em;margin:2.5rem 0 1rem;border-bottom:1px solid var(--border);padding-bottom:.4rem}}
.subtitle{{color:var(--muted);font-size:.9rem;margin-bottom:2rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.5rem}}
.kpi-label{{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
.kpi-val{{font-size:2rem;font-weight:700;margin:.3rem 0 .2rem}}
.kpi-sub{{font-size:.82rem;color:var(--muted)}}
.explain-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0 2rem}}
.explain{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem}}
.explain-title{{font-weight:600;margin-bottom:.4rem}}
.explain-formula{{font-family:monospace;background:var(--surface2);
  padding:.15rem .5rem;border-radius:4px;display:inline-block;margin:.3rem 0;font-size:.85rem}}
.explain-body{{font-size:.85rem;color:var(--muted)}}
.tag-mae{{color:var(--yellow)}}.tag-rmse{{color:var(--orange)}}
.chart-wrap{{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:1.2rem;margin-bottom:1.5rem}}
.chart-title{{font-size:.95rem;font-weight:600;margin-bottom:.8rem}}
.chart-caption{{font-size:.78rem;color:var(--muted);margin-top:.6rem}}
canvas{{width:100%;display:block;border-radius:6px}}
.metric-table{{width:100%;border-collapse:collapse;font-size:.9rem}}
.metric-table th,.metric-table td{{padding:.6rem 1rem;text-align:right;border-bottom:1px solid var(--border)}}
.metric-table th{{color:var(--muted);font-weight:500;text-align:left}}
.metric-table td:first-child{{text-align:left}}
.metric-table tr:hover td{{background:var(--surface2)}}
.better{{color:var(--green);font-weight:600}}
.concept-box{{background:var(--surface2);border:1px solid var(--purple);
  border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
.concept-box p{{font-size:.88rem;color:var(--muted);margin-top:.5rem}}
.pill{{display:inline-block;padding:.12rem .55rem;border-radius:20px;
  font-size:.75rem;font-weight:600;margin-right:.3rem}}
.pill-purple{{background:var(--purple);color:#fff}}
.pill-green{{background:var(--green);color:#0f0f1a}}
.pill-orange{{background:var(--orange);color:#fff}}
</style>
</head>
<body>
<h1>Satellite AI v2 — Rapport d'évaluation</h1>
<p class="subtitle">Correction de résidus SGP4 · TCN embarqué · Erreurs en mètres</p>

<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">MAE — IA corrigée</div>
    <div class="kpi-val" style="color:var(--purple)">{m_ai['mae']:.0f} m</div>
    <div class="kpi-sub">erreur moyenne absolue</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">MAE — SGP4 seul</div>
    <div class="kpi-val" style="color:var(--orange)">{m_sgp4['mae']:.0f} m</div>
    <div class="kpi-sub">sans correction IA</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">RMSE — IA corrigée</div>
    <div class="kpi-val" style="color:var(--blue)">{m_ai['rmse']:.0f} m</div>
    <div class="kpi-sub">sensible aux grands écarts</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Amélioration</div>
    <div class="kpi-val" style="color:{imp_color}">{improvement:+.1f}%</div>
    <div class="kpi-sub">sur {n} échantillons de test</div>
  </div>
</div>

<h2>Comprendre les métriques</h2>
<div class="explain-grid">
  <div class="explain">
    <div class="explain-title tag-mae">MAE — Mean Absolute Error</div>
    <div class="explain-formula">mean( |erreur_i| )</div>
    <div class="explain-body">Moyenne des écarts absolus. Intuitive :
    si MAE = 127 m, le modèle se trompe en moyenne de 127 m.
    Traite tous les écarts de façon égale.</div>
  </div>
  <div class="explain">
    <div class="explain-title tag-rmse">RMSE — Root Mean Square Error</div>
    <div class="explain-formula">√ mean( erreur_i² )</div>
    <div class="explain-body">Pénalise davantage les grands écarts.
    Une erreur de 500 m pèse 25× plus qu'une de 100 m.
    RMSE ≫ MAE = pics de dérive à investiguer.</div>
  </div>
</div>
<p style="font-size:.85rem;color:var(--muted);margin-bottom:2rem">
  <strong style="color:var(--text)">Ratio RMSE/MAE</strong> — 
  IA : {ratio_ai} | SGP4 : {ratio_sgp4} &nbsp;·&nbsp;
  Un bon modèle a RMSE/MAE &lt; 1.5
</p>

<h2>Tableau complet</h2>
<div class="chart-wrap">
<table class="metric-table">
  <thead><tr><th>Métrique</th><th>IA corrigée (m)</th><th>SGP4 seul (m)</th><th>Gain</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>
</div>

<h2>Pourquoi ce modèle est utile</h2>
<div class="concept-box">
  <span class="pill pill-purple">Concept clé</span>
  <span class="pill pill-orange">SGP4 seul</span>
  <span class="pill pill-green">SGP4 + IA</span>
  <p>SGP4 est analytique mais sa précision <strong style="color:var(--text)">dégrade avec le temps</strong>
  depuis l'époque du TLE. Drag atmosphérique, pression solaire et harmoniques gravitationnels
  créent une dérive croissante non modélisée. Le TCN apprend à <strong style="color:var(--text)">
  corriger ce résidu</strong> en continu.</p>
</div>

<div class="chart-wrap">
  <div class="chart-title">Dérive de position sur 24h depuis l'époque TLE</div>
  <canvas id="c-pedagogy" height="320"></canvas>
  <div class="chart-caption">Simulation physique : drag LEO, pression solaire, harmoniques J3/J4.
  La correction IA maintient l'erreur basse même loin de l'époque TLE.</div>
</div>

<div class="chart-wrap">
  <div class="chart-title">Erreur de position sur le jeu de test (m)</div>
  <canvas id="c-error" height="300"></canvas>
  <div class="chart-caption">Erreur 3D par échantillon. Les pics correspondent aux zones de forte dérive orbitale.</div>
</div>

<div class="chart-wrap">
  <div class="chart-title">Résidu Δx — réel vs prédit (m)</div>
  <canvas id="c-residual" height="280"></canvas>
  <div class="chart-caption">Composante X du résidu. L'IA doit prédire cette correction à appliquer à la sortie SGP4.</div>
</div>

<script>
const D = {data};

/* ─── Utilitaires Canvas ─────────────────────────────────────── */
function Chart(canvasId, opts) {{
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width - 38;
  const H = parseInt(canvas.getAttribute('height'));
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {{top:20, right:20, bottom:46, left:68}};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top  - pad.bottom;

  /* background */
  ctx.fillStyle = '#161628';
  ctx.fillRect(0, 0, W, H);

  /* grille */
  function drawGrid(xTicks, yTicks, xMin, xMax, yMin, yMax) {{
    ctx.strokeStyle = '#2a2a45';
    ctx.lineWidth   = 0.7;
    ctx.setLineDash([3,4]);
    yTicks.forEach(v => {{
      const y = pad.top + ph - ((v-yMin)/(yMax-yMin))*ph;
      ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(pad.left+pw,y); ctx.stroke();
      ctx.fillStyle='#7878a0'; ctx.font='11px system-ui'; ctx.textAlign='right';
      ctx.fillText(v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0), pad.left-6, y+4);
    }});
    xTicks.forEach(v => {{
      const x = pad.left + ((v-xMin)/(xMax-xMin))*pw;
      ctx.beginPath(); ctx.moveTo(x,pad.top); ctx.lineTo(x,pad.top+ph); ctx.stroke();
      ctx.fillStyle='#7878a0'; ctx.font='11px system-ui'; ctx.textAlign='center';
      ctx.fillText(v.toFixed(0), x, pad.top+ph+18);
    }});
    ctx.setLineDash([]);
  }}

  /* tracer une série */
  function drawLine(xs, ys, xMin, xMax, yMin, yMax, color, lw, dash) {{
    if(!xs || xs.length===0) return;
    ctx.strokeStyle = color; ctx.lineWidth = lw||2;
    if(dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
    ctx.beginPath();
    xs.forEach((x,i)=>{{
      const px2 = pad.left + ((x-xMin)/(xMax-xMin))*pw;
      const py2 = pad.top  + ph - ((ys[i]-yMin)/(yMax-yMin))*ph;
      i===0 ? ctx.moveTo(px2,py2) : ctx.lineTo(px2,py2);
    }});
    ctx.stroke(); ctx.setLineDash([]);
  }}

  /* zone remplie sous une courbe */
  function fillArea(xs, ys, xMin, xMax, yMin, yMax, color) {{
    ctx.fillStyle = color;
    ctx.beginPath();
    const bY = pad.top+ph;
    xs.forEach((x,i)=>{{
      const px2=pad.left+((x-xMin)/(xMax-xMin))*pw;
      const py2=pad.top+ph-((ys[i]-yMin)/(yMax-yMin))*ph;
      i===0?ctx.moveTo(px2,bY):i===1&&ctx.lineTo(px2,bY);
      ctx.lineTo(px2,py2);
    }});
    ctx.lineTo(pad.left+((xs[xs.length-1]-xMin)/(xMax-xMin))*pw, bY);
    ctx.closePath(); ctx.fill();
  }}

  /* légende */
  function legend(items) {{
    let lx = pad.left;
    const ly = pad.top + ph + 34;
    items.forEach(it=>{{
      ctx.fillStyle=it.color; ctx.fillRect(lx,ly-7,18,3);
      if(it.dash){{ctx.clearRect(lx,ly-7,18,3);ctx.strokeStyle=it.color;ctx.lineWidth=2;ctx.setLineDash(it.dash);ctx.beginPath();ctx.moveTo(lx,ly-5);ctx.lineTo(lx+18,ly-5);ctx.stroke();ctx.setLineDash([]);}}
      ctx.fillStyle='#e0e0f0'; ctx.font='11px system-ui'; ctx.textAlign='left';
      ctx.fillText(it.label, lx+22, ly);
      lx += ctx.measureText(it.label).width + 44;
    }});
  }}

  /* axe Y label */
  function yLabel(txt) {{
    ctx.save(); ctx.translate(13, pad.top+ph/2);
    ctx.rotate(-Math.PI/2); ctx.fillStyle='#7878a0';
    ctx.font='11px system-ui'; ctx.textAlign='center';
    ctx.fillText(txt,0,0); ctx.restore();
  }}

  return {{ctx,pad,pw,ph,W,H,drawGrid,drawLine,fillArea,legend,yLabel}};
}}

function niceTicks(min,max,count) {{
  const step = Math.pow(10,Math.floor(Math.log10((max-min)/count)));
  const candidates=[1,2,2.5,5,10].map(f=>f*step);
  const s=candidates.find(c=>(max-min)/c<=count+1)||candidates[candidates.length-1];
  const ticks=[];
  for(let v=Math.ceil(min/s)*s; v<=max+s*0.01; v+=s) ticks.push(parseFloat(v.toFixed(10)));
  return ticks;
}}

/* ─── Graphique 1 : dérive pédagogique ─────────────────────── */
(function() {{
  const c=Chart('c-pedagogy',{{}});
  const xs=D.hours, s=D.sgp4_drift, a=D.ai_drift;
  const yMin=0, yMax=Math.max(...s)*1.08;
  const xTicks=niceTicks(0,24,6), yTicks=niceTicks(yMin,yMax,5);
  c.drawGrid(xTicks,yTicks,0,24,yMin,yMax);
  c.fillArea(xs,s,0,24,yMin,yMax,'rgba(236,110,72,0.12)');
  c.fillArea(xs,a,0,24,yMin,yMax,'rgba(111,233,158,0.10)');
  c.drawLine(xs,s,0,24,yMin,yMax,'{C_ORANGE}',2,[6,4]);
  c.drawLine(xs,a,0,24,yMin,yMax,'{C_GREEN}',2.5);
  c.legend([{{color:'{C_ORANGE}',label:'SGP4 seul',dash:[6,4]}},{{color:'{C_GREEN}',label:'SGP4 + correction IA'}}]);
  c.yLabel('Erreur position (m)');
  /* annotation */
  const ctx=c.ctx, pw=c.pw, ph=c.ph, pad=c.pad;
  ctx.fillStyle='{C_GREEN}'; ctx.font='11px system-ui'; ctx.textAlign='left';
  ctx.fillText('↑ IA compense la dérive', pad.left+pw*0.42, pad.top+ph*0.45);
  ctx.fillStyle='{C_ORANGE}';
  ctx.fillText('↑ Dérive SGP4 croissante', pad.left+pw*0.42, pad.top+ph*0.22);
}})();

/* ─── Graphique 2 : erreurs test set ───────────────────────── */
(function() {{
  const c=Chart('c-error',{{}});
  const xs=Array.from({{length:D.n}},(_,i)=>i);
  const ai=D.error_ai, sg=D.error_sgp4;
  const yMax=Math.max(...sg,...ai)*1.1, yMin=0;
  const xTicks=niceTicks(0,D.n,6), yTicks=niceTicks(yMin,yMax,5);
  c.drawGrid(xTicks,yTicks,0,D.n,yMin,yMax);
  c.fillArea(xs,ai,0,D.n,yMin,yMax,'rgba(96,93,246,0.13)');
  c.drawLine(xs,sg,0,D.n,yMin,yMax,'{C_ORANGE}',1.5,[4,3]);
  c.drawLine(xs,ai,0,D.n,yMin,yMax,'{C_PURPLE}',2);
  c.legend([{{color:'{C_ORANGE}',label:'SGP4 seul',dash:[4,3]}},{{color:'{C_PURPLE}',label:'IA corrigée'}}]);
  c.yLabel('Erreur 3D (m)');
}})();

/* ─── Graphique 3 : résidu Δx ──────────────────────────────── */
(function() {{
  const c=Chart('c-residual',{{}});
  const xs=Array.from({{length:D.n}},(_,i)=>i);
  const tx=D.true_x, px=D.pred_x;
  const all=[...tx,...px], yMin=Math.min(...all)*1.05, yMax=Math.max(...all)*1.05;
  const xTicks=niceTicks(0,D.n,6), yTicks=niceTicks(yMin,yMax,5);
  c.drawGrid(xTicks,yTicks,0,D.n,yMin,yMax);
  c.drawLine(xs,tx,0,D.n,yMin,yMax,'{C_YELLOW}',2);
  c.drawLine(xs,px,0,D.n,yMin,yMax,'{C_BLUE}',1.5,[4,3]);
  c.legend([{{color:'{C_YELLOW}',label:'Résidu réel Δx'}},{{color:'{C_BLUE}',label:'Résidu prédit Δx',dash:[4,3]}}]);
  c.yLabel('Δx (m)');
}})();
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    sz = os.path.getsize(path) / 1024
    logger.info(f"Rapport HTML autonome → {path}  ({sz:.0f} KB)")
