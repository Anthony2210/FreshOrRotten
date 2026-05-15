# Résultats

Ce fichier résume les premiers résultats réels du projet FreshOrRotten.
Les métriques ci-dessous viennent d'entraînements exécutés sur Google Colab avec GPU T4.

L'objectif n'est pas de battre l'état de l'art.
L'objectif est de comparer une baseline CNN simple sur deux protocoles :

- `standard_split` : les mêmes types de produits peuvent apparaître en train, validation et test ;
- `unseen_category_split` : certains produits sont absents du train et utilisés uniquement pour le test.

## Baseline CNN

Modèle utilisé :

- CNN simple ;
- classification binaire `fresh` / `rotten` ;
- sortie sigmoid ;
- loss `binary_crossentropy` ;
- optimizer Adam ;
- image size : 128 x 128 ;
- batch size Colab : 64 ;
- early stopping sur la `val_loss`.

Le modèle n'utilise ni NoisyViT, ni transfer learning.

## Fonctionnement du modèle

Le modèle prend en entrée une image RGB redimensionnée en `128 x 128`.
Chaque pixel est normalisé entre 0 et 1.
Cette normalisation garde le même ordre de grandeur pour toutes les images et aide le réseau à apprendre plus facilement.

Le CNN traite ensuite l'image en plusieurs étapes.

Les premières couches `Conv2D` cherchent des motifs simples, comme des bords, des couleurs ou des petites textures.
Les couches suivantes combinent ces motifs pour reconnaître des indices plus utiles, par exemple des zones sombres, des taches, une peau abîmée ou une texture moins régulière.

Après les deux premières convolutions, `MaxPooling2D` réduit la taille des représentations.
Cela diminue le coût de calcul et garde les informations les plus marquées.

La troisième convolution extrait des caractéristiques plus globales.
Ensuite, `GlobalAveragePooling2D` résume chaque carte de caractéristiques en une seule valeur.
Cela évite d'ajouter trop de paramètres et garde le modèle léger.

La couche `Dense` combine les caractéristiques extraites.
Le `Dropout` désactive une partie des neurones pendant l'entraînement.
Son rôle est de limiter l'overfitting.

La dernière couche utilise une activation sigmoid.
Elle produit un score entre 0 et 1 :

- score proche de 0 : prédiction `fresh` ;
- score proche de 1 : prédiction `rotten`.

Pour l'évaluation, le seuil utilisé est 0.5.
Si le score est supérieur ou égal à 0.5, l'image est prédite `rotten`.
Sinon, elle est prédite `fresh`.

## Résultats avec standard_split

Le modèle a d'abord été entraîné avec le `standard_split`.

Répartition :

| split | fresh | rotten | total |
|---|---:|---:|---:|
| train | 16872 | 15303 | 32175 |
| validation | 5624 | 5101 | 10725 |
| test | 5624 | 5101 | 10725 |

Le split est légèrement déséquilibré vers la classe `fresh`, mais les deux classes restent bien représentées.

Dernière epoch sauvegardée :

| métrique | train | validation |
|---|---:|---:|
| accuracy | 0.9026 | 0.8905 |
| loss | 0.2483 | 0.2701 |
| precision | 0.9035 | 0.8424 |
| recall | 0.8902 | 0.9471 |

Résultats sur le `test_set` :

| métrique | valeur |
|---|---:|
| loss | 0.2539 |
| accuracy | 0.8969 |
| precision | 0.9203 |
| recall | 0.8575 |
| F1-score | 0.8878 |

Dans ces métriques, la classe positive est `rotten`.

Matrice de confusion :

| | predicted_fresh | predicted_rotten |
|---|---:|---:|
| actual_fresh | 5245 | 379 |
| actual_rotten | 727 | 4374 |

Sur ce protocole, la baseline donne une performance solide.
Elle fait peu de fausses alertes `rotten` sur des produits frais.
Elle manque encore 727 images `rotten`, prédites comme `fresh`.

## Résultats avec unseen_category_split

Le deuxième protocole teste la généralisation à des catégories non vues.
Les produits retirés de l'entraînement sont :

- `apple` ;
- `banana` ;
- `tomato`.

Ces catégories sont utilisées uniquement dans le `test_set`.

Répartition :

| split | fresh | rotten | total |
|---|---:|---:|---:|
| train | 15387 | 12106 | 27493 |
| validation | 3847 | 3027 | 6874 |
| test | 8886 | 10372 | 19258 |

Répartition du `test_set` par produit :

| product_type | fresh | rotten | total |
|---|---:|---:|---:|
| apple | 3468 | 4263 | 7731 |
| banana | 3513 | 3605 | 7118 |
| tomato | 1905 | 2504 | 4409 |

Dernière epoch sauvegardée :

| métrique | train | validation |
|---|---:|---:|
| accuracy | 0.9055 | 0.9014 |
| loss | 0.2416 | 0.2509 |
| precision | 0.9028 | 0.8547 |
| recall | 0.8801 | 0.9349 |

Résultats sur le `test_set` unseen :

| métrique | valeur |
|---|---:|
| loss | 0.7925 |
| accuracy | 0.6706 |
| precision | 0.6405 |
| recall | 0.8853 |
| F1-score | 0.7433 |

Matrice de confusion :

| | predicted_fresh | predicted_rotten |
|---|---:|---:|
| actual_fresh | 3733 | 5153 |
| actual_rotten | 1190 | 9182 |

Résultats par `product_type` :

| product_type | images | accuracy | precision | recall | F1-score |
|---|---:|---:|---:|---:|---:|
| apple | 7731 | 0.7663 | 0.7548 | 0.8534 | 0.8011 |
| banana | 7118 | 0.6205 | 0.5772 | 0.9373 | 0.7145 |
| tomato | 4409 | 0.5838 | 0.5914 | 0.8646 | 0.7024 |

## Comparaison des protocoles

| protocole | accuracy | precision | recall | F1-score |
|---|---:|---:|---:|---:|
| standard_split | 0.8969 | 0.9203 | 0.8575 | 0.8878 |
| unseen_category_split | 0.6706 | 0.6405 | 0.8853 | 0.7433 |

La performance baisse nettement sur les catégories non vues.
Cette baisse n'est pas un échec du projet : elle répond directement à la problématique.

Le modèle semble mieux fonctionner quand les types de produits vus au test ressemblent à ceux vus pendant l'entraînement.
Quand `apple`, `banana` et `tomato` sont absents du train, le modèle garde un bon recall sur `rotten`, mais il prédit beaucoup trop d'images fraîches comme `rotten`.
Cela explique la précision plus faible sur le protocole unseen.

Cette observation suggère que la baseline apprend bien des indices visuels de fraîcheur, mais aussi des indices propres aux catégories vues.
Le `unseen_category_split` est donc utile pour mesurer une généralisation plus difficile et plus proche d'un usage réel.

## Limites à garder en tête

Ces résultats dépendent du dataset Freshness44 et du split choisi.
Ils ne prouvent pas que le modèle sera fiable sur tous les fruits et légumes possibles.

Les images prises avec un smartphone peuvent varier selon la lumière, le cadrage, l'arrière-plan et la qualité de la caméra.
La future plateforme web devra donc afficher un score de confiance et un message d'incertitude.

La classe `uncertain` ne doit pas être entraînée comme une vraie classe.
Elle doit rester une règle d'interprétation basée sur le score de confiance.
