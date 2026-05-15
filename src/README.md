# Code source

Ce dossier contient les scripts Python du projet.

Organisation :

- `model.py` : définition du baseline_model ;
- `train.py` : entraînement du CNN sur le `standard_split` ;
- `evaluate.py` : évaluation du meilleur modèle sur le `test_set` ;
- `split_strategy.py` : création du `standard_split` et du `unseen_category_split`.

Scripts prévus plus tard :

- `data_loader.py` : chargement centralisé des images et des labels ;
- `predict.py` : prédiction sur une image.

Le premier entraînement utilise un CNN simple.
Il n'utilise ni NoisyViT, ni transfer learning.

Commande d'entraînement :

```bash
python src/train.py
```

Commande d'entraînement avec catégories non vues :

```bash
python src/train.py --split unseen --unseen-categories apple banana tomato
```

Commande d'évaluation classique :

```bash
python src/evaluate.py
```

Commande d'évaluation sur catégories non vues :

```bash
python src/evaluate.py --split unseen --unseen-categories apple banana tomato
```

Fichiers générés après entraînement :

- `models/baseline_model.keras` ;
- `reports/training_history.csv` ;
- `reports/standard_split.csv`.

Fichiers générés après entraînement sur catégories non vues :

- `models/baseline_model_unseen.keras` ;
- `reports/unseen_training_history.csv` ;
- `reports/unseen_category_split.csv`.

Fichiers générés après évaluation classique :

- `reports/evaluation_metrics.csv` ;
- `reports/confusion_matrix.csv`.

Fichiers générés après évaluation sur catégories non vues :

- `reports/unseen_category_split.csv` ;
- `reports/unseen_category_evaluation_metrics.csv` ;
- `reports/unseen_category_confusion_matrix.csv` ;
- `reports/unseen_category_metrics_by_product_type.csv` ;
- `reports/evaluation_comparison.csv`.
