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

Après évaluation sur catégories non vues, les scripts peuvent générer :

- `unseen_training_history.csv` : historique du modèle entraîné sans les catégories non vues ;
- `unseen_category_split.csv` : split avec catégories non vues dans le test ;
- `unseen_category_evaluation_metrics.csv` : métriques globales sur les catégories non vues ;
- `unseen_category_confusion_matrix.csv` : matrice de confusion sur les catégories non vues ;
- `unseen_category_metrics_by_product_type.csv` : métriques par type de produit ;
- `evaluation_comparison.csv` : comparaison entre `standard_split` et `unseen_category_split`.

Après analyse de l'incertitude calibrée, le script `src/uncertainty.py` peut générer :

- `uncertainty_metrics.csv` : comparaison entre baseline et prédictions acceptées ;
- `uncertainty_by_product_type.csv` : métriques d'incertitude par type de produit ;
- `uncertainty_calibration_grid.csv` : seuils testés sur validation.

Après analyse de l'incertitude par distance de features, le script `src/feature_distance_uncertainty.py` peut générer :

- `feature_distance_metrics.csv` : comparaison entre baseline et prédictions acceptées par distance ;
- `feature_distance_by_product_type.csv` : métriques par type de produit ;
- `feature_distance_calibration_grid.csv` : seuils de distance testés sur validation.

Après analyse de l'incertitude hybride, le script `src/hybrid_uncertainty.py` peut générer :

- `hybrid_uncertainty_metrics.csv` : comparaison entre baseline et prédictions acceptées par la règle hybride ;
- `hybrid_uncertainty_by_product_type.csv` : métriques hybrides par type de produit ;
- `hybrid_uncertainty_calibration_grid.csv` : combinaisons de seuils testées sur validation.

Après analyse du biais `product_type`, le script `src/feature_product_type_analysis.py` peut générer :

- `product_type_probe_metrics.csv` : métriques globales du classifieur de `product_type` ;
- `product_type_probe_by_product_type.csv` : métriques détaillées par type de produit ;
- `product_type_probe_confusion_matrix.csv` : matrice de confusion entre types de produits.

Le fichier `results.md` résume les résultats réels déjà obtenus.
