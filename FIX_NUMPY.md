# Fix NumPy 2.x incompatibilité

Si vous voyez l'erreur :
  "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"

## Solution rapide

```bash
# Option 1 : downgrade NumPy (recommandé)
pip install "numpy<2" --break-system-packages

# Option 2 : avec conda
conda install "numpy<2"

# Option 3 : créer un environnement isolé
conda create -n satellite python=3.11
conda activate satellite
pip install -r requirements.txt
```

## Modules concernés
- sgp4 (propagation orbitale)
- scipy (calculs scientifiques)

## Vérification
```bash
python -c "import numpy; print(numpy.__version__)"  # doit être < 2.0
python -c "from sgp4.api import Satrec; print('OK')"
```
