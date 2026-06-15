# Satellite AI v2

Pipeline de prédiction orbitale par **apprentissage de résidus SGP4**.

## Concept

L'IA n'apprend **pas** à recalculer SGP4 (inutile — c'est analytique).
Elle apprend à corriger les **résidus** entre SGP4 et la réalité physique :

```
TLE → SGP4 (référence) → Δpos = Mesure - SGP4 → TCN léger → correction
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py --mode info        # Config + infos satellites + dernier run
python main.py --mode train       # Entraînement + évaluation automatique
python main.py --mode evaluate    # Rapport HTML interactif (models/report.html)
python main.py --mode benchmark   # Latence CPU (médiane + P99)
python main.py --mode export      # Export ONNX + quantification INT8
python main.py --mode predict     # Démo inférence PyTorch
python main.py --mode predict --onnx      # Démo inférence ONNX Runtime
python main.py --mode finetune --tle data/new_tle.txt   # Fine-tuning incrémental
python main.py --mode train --model gru   # Forcer le modèle GRU
```

## Résidus réels (production)

Deux approches pour remplacer les résidus simulés :

**A — TLE différentiels** (sans radar, immédiat) :
Celestrak publie de nouveaux TLE toutes ~2h pour l'ISS.
`differential_tle_residual(tle_old, tle_new, t)` = SGP4(nouveau) − SGP4(ancien)

**B — Ranging radar / données SP3** (précis, accès requis) :
Space-Track.org (compte gratuit), données IGS SP3.

## Déploiement embarqué

```
models/model.onnx      → ONNX Runtime (RPi 4, Jetson Nano)
models/model_int8_ts.pt → TorchScript INT8 (~4× plus léger)
```

```bash
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # Jetson AGX Orin
```

## Structure

```
satellite_ai_v2/
├── main.py                 Point d'entrée (7 modes)
├── src/
│   ├── config.py           Paramètres centralisés
│   ├── tle_fetcher.py      Ingestion TLE + validation checksum
│   ├── sgp4_utils.py       SGP4 + résidus simulés + résidus différentiels
│   ├── dataset.py          Split chronologique + scalers par satellite
│   ├── model.py            TCN causal dilataté + GRU léger
│   ├── train.py            Entraînement (early stop, Huber loss, scheduler)
│   ├── evaluate.py         Métriques en mètres + rapport HTML interactif
│   ├── export.py           Export ONNX + quantification INT8
│   ├── predict.py          Inférence PyTorch ou ONNX Runtime
│   └── continual.py        Fine-tuning EWC + replay buffer
├── data/sample_tle.txt     4 TLE exemples (checksums valides)
├── models/                 Checkpoints, scalers, test set, rapport
└── logs/                   training.csv + run.log
```

## Métriques

| Métrique | Définition | Usage |
|----------|-----------|-------|
| **MAE** | Erreur moyenne absolue | Référence principale |
| **RMSE** | Racine MSE — pénalise les gros écarts | Détection de dérives |
| **P95** | 95% des erreurs sous ce seuil | Garantie opérationnelle |

Si RMSE/MAE > 1.5 → pics d'erreur à investiguer.
