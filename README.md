# FreshOrRotten

FreshOrRotten est un projet de data science et de deep learning.
L'objectif est de prédire si un fruit ou un légume est frais ou pourri à partir d'une image.

## Contexte

La classification d'images peut être utilisée pour reconnaître l'état visuel d'un produit alimentaire.
Dans ce projet, on s'intéresse à deux classes :

- `fresh` : produit frais ;
- `rotten` : produit pourri.

Le modèle prévu est un CNN simple créé avec TensorFlow / Keras.
Il servira de modèle de base pour étudier les performances et les limites d'une approche supervisée.

## Problématique

La question principale est la suivante :

Un CNN entraîné à reconnaître la fraîcheur sur certaines catégories de fruits et légumes peut-il généraliser à des catégories absentes de l'entraînement ?

Elle teste si le modèle apprend une notion générale de fraîcheur ou s'il dépend surtout des catégories vues pendant l'entraînement.

## Dataset

Le projet utilise le dataset Freshness44.
Ce dataset est issu de l'article :

**“Multi-Task NoisyViT for Enhanced Fruit and Vegetable Freshness Detection and Type Classification”**
par Siavash Esfandiari Fard, Tonmoy Ghosh et Edward Sazonov.

Le papier propose un modèle avancé multi-tâche.
FreshOrRotten ne cherche pas à reproduire ce modèle.
Le but est de créer un pipeline autour d'un modèle plus simple.

Le dataset complet doit rester local.
Les images brutes sont attendues dans :

```text
data/raw/
```

Ce dossier est ignoré par Git pour éviter de pousser les données sur GitHub.

## Exploration du dataset

Le notebook `notebooks/01_data_exploration.ipynb` sert à analyser le dataset avant l'entraînement.

Il permet de :

- compter les images par `product_type` et par `freshness` ;
- visualiser le nombre d'images `fresh` et `rotten` ;
- repérer les catégories avec peu d'images ;
- comparer, pour chaque produit, le nombre d'images `fresh` et `rotten` ;
- afficher quelques exemples d'images.

Les figures générées sont sauvegardées dans :

```text
reports/figures/
```

Ces figures viennent du dataset local.
Elles doivent être régénérées si le dataset ou son organisation change.

### Split classique

Le `standard_split` sépare les images en train, validation et test.
Les mêmes catégories de produits peuvent apparaître dans les trois ensembles.

Ce protocole mesure la performance classique du modèle.

### Split par catégories non vues

Le `unseen_category_split` retire certaines catégories du jeu d'entraînement.
Ces catégories sont utilisées uniquement pour le test.

Ce protocole permet de tester la généralisation à des produits jamais vus par le modèle pendant l'entraînement.

## Plateforme web prévue

Une plateforme web simple sera ajoutée dans le dossier `web/`.
Elle devra permettre à l'utilisateur de prendre une photo ou d'importer une image.

La plateforme affichera :

- la prédiction `fresh` ou `rotten` ;
- le score de confiance ;
- un message d'incertitude si le score est trop faible.

La classe `uncertain` ne sera pas une vraie classe entraînée.
Elle sera seulement une règle d'interprétation basée sur le score de confiance.

## Limites

Le modèle ne garantit pas une prédiction fiable sur tous les fruits et légumes du monde.

Les états intermédiaires entre `fresh` et `rotten` ne sont pas une vraie classe du dataset.
Ils seront seulement interprétés avec une règle d'incertitude.

La plateforme web sera une démonstration pédagogique.
Elle ne doit pas être utilisée comme outil médical, sanitaire ou professionnel.

Les images prises avec un smartphone peuvent varier selon la lumière, le cadrage, l'arrière-plan et la qualité de la caméra.

## Structure du projet

```text
FreshOrRotten/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
│   └── figures/
├── web/
└── docs/
```
