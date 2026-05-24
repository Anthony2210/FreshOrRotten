# Pistes de recherche

Ce document garde des idées pour renforcer la contribution scientifique du projet.

## Point de départ

Les premiers résultats montrent une limite importante.
Le modèle obtient de bons résultats avec le `standard_split`, mais sa performance baisse sur le `unseen_category_split`.

Cela suggère que le CNN apprend une partie des indices de fraîcheur, mais aussi des indices liés aux catégories vues pendant l'entraînement.

## Contribution courte terme

La première contribution ajoutée au projet est l'incertitude calibrée.

Idée :

- calibrer un seuil de confiance sur le `validation_set` ;
- accepter seulement les prédictions assez confiantes ;
- classer les autres images comme `uncertain` ;
- comparer le comportement sur `standard_split` et `unseen_category_split`.

Cette méthode ne change pas le modèle.
Elle teste si le système peut éviter certaines prédictions risquées.
Le seuil doit être assez exigeant pour produire une vraie classe `uncertain`.
Un seuil trop faible peut accepter toutes les images et masquer les limites du modèle.

## Contribution intermédiaire : distance de features

Une deuxième contribution plus liée au modèle est l'incertitude par distance de features.

Idée :

- utiliser le CNN entraîné comme extracteur de représentations ;
- calculer un prototype `fresh` et un prototype `rotten` dans l'espace interne du modèle ;
- mesurer si une image de test est proche du prototype de sa classe prédite ;
- rejeter l'image comme `uncertain` si elle est trop éloignée.

Cette méthode répond mieux au biais identifié.
Elle ne regarde pas seulement le score sigmoid final.
Elle vérifie aussi si l'image ressemble, dans l'espace appris par le CNN, aux exemples vus pendant l'entraînement.

Pour le mémoire, cette méthode peut être présentée comme une proposition personnelle raisonnable :
elle part du constat que le modèle généralise mal à certains `product_type` non vus, puis propose une règle de rejet basée sur la représentation interne du modèle.

## Contribution suivante : règle hybride

Les premiers résultats montrent que la confiance sigmoid est plus efficace que la distance de features seule.
La distance de features reste intéressante, mais elle rejette parfois beaucoup d'images sans isoler assez bien les erreurs.

La suite logique est donc une règle hybride :

- accepter une image seulement si le score sigmoid est assez confiant ;
- vérifier en plus que l'image reste proche des prototypes appris dans l'espace de features ;
- calibrer les deux seuils sur le `validation_set` ;
- comparer le compromis accuracy / coverage sur `standard_split` et `unseen_category_split`.

Cette piste est plus défendable qu'un simple seuil fixe.
Elle part d'un biais observé, combine deux mesures complémentaires, puis évalue si cette combinaison améliore le rejet des prédictions risquées.

## Analyse du biais product_type

Une autre analyse importante consiste à tester ce que le CNN encode dans ses features internes.

Méthode :

- garder le CNN entraîné pour `fresh` / `rotten` ;
- extraire une couche interne du modèle ;
- entraîner un classifieur simple pour prédire `product_type` à partir de ces features ;
- comparer sa performance à une baseline qui prédit toujours le type majoritaire.

Si ce classifieur prédit bien `product_type`, cela montre que les représentations du CNN contiennent fortement l'information de catégorie.
Ce résultat soutient l'idée que le modèle peut dépendre des produits vus pendant l'entraînement.

Cette analyse fait le lien avec la baisse de performance sur le `unseen_category_split`.
Elle donne une preuve plus directe du biais, au lieu de seulement constater une baisse d'accuracy.

## Piste plus avancée

Une piste plus ambitieuse serait de construire un modèle moins dépendant du `product_type`.

Idée possible :

- un backbone CNN commun ;
- une tête pour prédire `fresh` / `rotten` ;
- une tête adversariale liée au `product_type`.

L'objectif serait d'apprendre des représentations utiles pour la fraîcheur, mais moins informatives sur le type du produit.

Cette idée inverse partiellement la logique du papier NoisyViT.
Le papier prédit à la fois la fraîcheur et le type.
Ici, l'objectif serait de limiter la dépendance au type pour mieux généraliser à des catégories non vues.

## Ce qu'il faudrait évaluer

Pour que cette piste soit défendable, il faudrait comparer :

- baseline CNN ;
- baseline CNN avec incertitude calibrée ;
- modèle plus robuste ou plus invariant au `product_type`.

Les métriques importantes seraient :

- accuracy, precision, recall, F1-score ;
- résultats par `product_type` ;
- performance sur catégories non vues ;
- taux d'images rejetées comme `uncertain` ;
- comparaison entre images acceptées et images rejetées.
