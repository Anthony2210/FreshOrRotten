# Rapports et figures

Ce dossier contient les éléments générés pour analyser le projet.

Le dossier `figures/` sert à sauvegarder les graphiques produits par les notebooks.

Figures générées par `notebooks/01_data_exploration.ipynb` :

- `image_count_by_freshness.png` ;
- `image_count_by_product_type.png` ;
- `image_count_by_product_type_and_freshness.png`.

Après entraînement de la baseline CNN, les scripts peuvent générer :

- `training_history.csv` : historique `loss`, `accuracy`, `precision` et `recall` ;
- `standard_split.csv` : split utilisé pour train, validation et test ;
- `evaluation_metrics.csv` : métriques sur le `test_set` ;
- `confusion_matrix.csv` : matrice de confusion sur le `test_set`.

