# Notebooks

Ce dossier contient les notebooks d'analyse.

Notebooks :

- `01_data_exploration.ipynb` : explorer le dataset Freshness44 et générer des figures exploratoires ;
- `02_generalization_test.ipynb` : tester la généralisation avec des catégories non vues ;
- `03_uncertainty_analysis.ipynb` : évaluer une règle d'incertitude calibrée sur validation ;
- `04_feature_distance_uncertainty.ipynb` : évaluer une incertitude basée sur les features internes du CNN ;
- `05_hybrid_uncertainty.ipynb` : combiner confiance sigmoid et distance de features ;
- `06_product_type_bias_analysis.ipynb` : tester si les features du CNN encodent le type du produit ;
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

Le notebook `04_feature_distance_uncertainty.ipynb` compare les images de test aux prototypes `fresh` et `rotten` calculés dans l'espace de features du CNN.
Cette analyse sert à vérifier si les catégories non vues sont plus éloignées des représentations apprises pendant l'entraînement.

Le notebook `05_hybrid_uncertainty.ipynb` teste une règle plus stricte.
Une image est acceptée seulement si le score sigmoid est assez confiant et si la distance de features reste faible.

Le notebook `06_product_type_bias_analysis.ipynb` entraîne un petit classifieur sur les features internes du CNN pour prédire `product_type`.
Cette analyse aide à mesurer si le modèle encode fortement les catégories vues pendant l'entraînement.

Le notebook `baseline_colab.ipynb` prépare Colab, copie le dataset depuis Google Drive vers le disque local, lance le `standard_split`, puis le `unseen_category_split`.
