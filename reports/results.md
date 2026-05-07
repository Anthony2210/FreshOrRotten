# Résultats

Ce fichier résume les premiers résultats réels du projet FreshOrRotten.
Les métriques ci-dessous viennent d'un entraînement exécuté sur Google Colab avec GPU T4.

## Baseline CNN

Modèle utilisé :

- CNN ;
- classification binaire `fresh` / `rotten` ;
- sortie sigmoid ;
- loss `binary_crossentropy` ;
- optimizer Adam ;
- image size : 128 x 128 ;
- batch size : 64 ;
- entraînement : 5 epochs.

Le modèle n'utilise ni NoisyViT, ni transfer learning.

## Fonctionnement du modèle

Le modèle prend en entrée une image RGB redimensionnée en `128 x 128`.
Chaque pixel est normalisé entre 0 et 1.
Cette normalisation aide le réseau à apprendre plus facilement.

Le CNN traite ensuite l'image en plusieurs étapes.

Les premières couches `Conv2D` cherchent des motifs simples, comme des bords, des couleurs ou des petites textures.
Les couches suivantes peuvent combiner ces motifs pour reconnaître des indices plus utiles, par exemple des zones sombres, des taches, une peau abîmée ou une texture moins régulière.

Après les deux premières convolutions, `MaxPooling2D` réduit la taille des représentations.
Cela diminue le coût de calcul et garde les informations les plus importantes.

La troisième convolution extrait des caractéristiques plus globales.
Ensuite, `GlobalAveragePooling2D` résume chaque carte de caractéristiques en une seule valeur.
Cela évite d'ajouter trop de paramètres et garde le modèle plus léger.

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

## Split utilisé

Le modèle a été entraîné avec le `standard_split`.

Répartition :

| split | fresh | rotten | total |
|---|---:|---:|---:|
| train | 16872 | 15303 | 32175 |
| validation | 5624 | 5101 | 10725 |
| test | 5624 | 5101 | 10725 |

Le split est légèrement déséquilibré vers la classe `fresh`, mais les deux classes restent bien représentées.

## Historique d'entraînement

Dernière epoch :

| métrique | train | validation |
|---|---:|---:|
| accuracy | 0.8512 | 0.8584 |
| loss | 0.3496 | 0.3302 |
| precision | 0.8427 | 0.9054 |
| recall | 0.8447 | 0.7842 |

La loss baisse sur le train et la validation.
À 5 epochs, le modèle apprend encore et ne montre pas de signe évident d'overfitting.

## Résultats sur le test_set

| métrique | valeur |
|---|---:|
| loss | 0.3253 |
| accuracy | 0.8622 |
| precision | 0.9097 |
| recall | 0.7885 |
| F1-score | 0.8448 |

Dans ces métriques, la classe positive est `rotten`.

## Matrice de confusion

| | predicted_fresh | predicted_rotten |
|---|---:|---:|
| actual_fresh | 5225 | 399 |
| actual_rotten | 1079 | 4022 |

Le modèle reconnaît bien une grande partie des images.
Il fait peu de fausses alertes `rotten` sur des produits frais.
En revanche, il manque encore 1079 images `rotten`, prédites comme `fresh`.

## Interprétation

Cette première baseline est correcte et donne une référence utile pour la suite du projet.

Le point à surveiller est le recall de la classe `rotten`.
Pour une application web, rater un produit pourri peut être plus problématique que signaler un doute sur un produit frais.

La prochaine étape importante est de comparer ces résultats avec le `unseen_category_split`.
Ce test permettra de savoir si le modèle apprend une notion générale de fraîcheur ou s'il dépend surtout des catégories vues pendant l'entraînement.
