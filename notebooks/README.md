# Notebooks

Ce dossier contient les notebooks d'analyse.

Notebooks :

- `01_data_exploration.ipynb` : explorer le dataset Freshness44 et générer des figures exploratoires ;
- `02_generalization_test.ipynb` : tester la généralisation avec des catégories non vues ;
- `03_uncertainty_analysis.ipynb` : évaluer une règle d'incertitude calibrée sur validation ;
- `baseline_colab.ipynb` : exécuter l'entraînement et les évaluations sur Google Colab.

Le notebook `01_data_exploration.ipynb` sauvegarde les figures dans :

```text
reports/figures/
```

Figures générées :

- `image_count_by_freshness.png` ;
- `image_count_by_product_type.png` ;
- `image_count_by_product_type_and_freshness.png`.

Le notebook `02_generalization_test.ipynb` réentraîne la même baseline CNN avec certaines catégories absentes du train.
Le test est ensuite fait uniquement sur ces catégories.

Le notebook `03_uncertainty_analysis.ipynb` compare la baseline avec une règle `uncertain`.
Le seuil est calibré sur validation, puis testé sur le split classique et sur les catégories non vues.

Le notebook `baseline_colab.ipynb` prépare Colab, copie le dataset depuis Google Drive vers le disque local, lance le `standard_split`, puis le `unseen_category_split`.
