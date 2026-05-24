# Plateforme web

Ce dossier contient la première version statique de la plateforme FreshOrRotten.

Objectif :

- importer ou prendre une photo ;
- afficher un aperçu de l'image ;
- lancer une prédiction côté navigateur avec TensorFlow.js ;
- afficher la prédiction et le score de confiance ;
- afficher un message clair si le résultat est incertain.

La classe `uncertain` n'est pas entraînée.
Elle dépend seulement du score de confiance.

## Fichiers

- `index.html` : structure de la page ;
- `style.css` : mise en forme ;
- `app.js` : chargement du modèle, prétraitement de l'image et prédiction ;
- `model/` : emplacement du modèle TensorFlow.js.

## Tester en local

Il vaut mieux lancer un petit serveur local, car le navigateur charge le modèle avec `fetch`.

```bash
cd web
python -m http.server 8000
```

Puis ouvrir :

```text
http://localhost:8000
```

## Ajouter le modèle web

Après entraînement du modèle Keras, convertir le fichier `.keras` en TensorFlow.js.

Exemple :

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras_keras --output_format=tfjs_layers_model models/baseline_model.keras web/model
```

Après conversion, le dossier `web/model/` doit contenir :

- `model.json` ;
- un ou plusieurs fichiers `.bin`.

Pour ce projet, le fichier de poids généré pour la V1 web est :

```text
baseline-shard1of1.bin
```

Ces fichiers peuvent être trop lourds pour GitHub.
Ils sont donc ignorés par Git par défaut.
Ils peuvent être ajoutés localement avant un envoi FTP.

Si le modèle n'est pas présent, l'interface reste visible mais affiche un message indiquant que le modèle n'est pas chargé.

## Dépannage

Si l'interface affiche que le modèle est introuvable :

- ne pas ouvrir `index.html` directement en `file://` ;
- lancer un serveur local avec `python -m http.server 8000` depuis le dossier `web/` ;
- vérifier que `web/model/model.json` existe ;
- vérifier que le fichier `.bin` indiqué dans `model.json` est aussi présent ;
- avec FileZilla, envoyer aussi le dossier `web/model/`, car les fichiers du modèle sont ignorés par Git.

Si l'interface affiche que le modèle est trouvé mais illisible, ouvrir la console du navigateur.
Cela indique plutôt un problème de compatibilité TensorFlow.js ou de fichier `.bin` manquant.

## Règle d'incertitude

- confiance >= 80 % : afficher la prédiction ;
- confiance entre 60 % et 80 % : afficher `Résultat incertain` ;
- confiance < 60 % : demander une meilleure photo.

Cette plateforme est une démonstration pédagogique.
Elle ne doit pas être présentée comme un outil sanitaire ou professionnel.
