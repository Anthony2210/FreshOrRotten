# Code source

Ce dossier contient les scripts Python du projet.

Organisation :

- `model.py` : définition du baseline_model ;
- `train.py` : entraînement du CNN sur le `standard_split` ;
- `evaluate.py` : évaluation du meilleur modèle sur le `test_set`.

Scripts prévus plus tard :

- `data_loader.py` : chargement centralisé des images et des labels ;
- `split_strategy.py` : création des splits avancés ;
- `predict.py` : prédiction sur une image.

Le premier entraînement utilise un CNN simple.
Il n'utilise ni NoisyViT, ni transfer learning.

Commande d'entraînement :

```bash
python src/train.py
```

Commande d'évaluation :

```bash
python src/evaluate.py
```

Fichiers générés après entraînement :

- `models/baseline_model.keras` ;
- `reports/training_history.csv` ;
- `reports/standard_split.csv`.

Fichiers générés après évaluation :

- `reports/evaluation_metrics.csv` ;
- `reports/confusion_matrix.csv`.
