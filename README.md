# Urban Segmenter — Cityscapes Segmentation Lab

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-CV_lab-0F172A)

## Présentation du projet

Urban Segmenter est un laboratoire Computer Vision dédié à la segmentation sémantique de scènes urbaines à partir du dataset Cityscapes. Le dépôt regroupe des notebooks d’expérimentation, des architectures U-Net, une API FastAPI d’inférence et une interface Streamlit permettant de visualiser les prédictions sur quelques images de démonstration.

Le projet reste volontairement au niveau MVP portfolio : il montre une démarche complète et lisible, sans chercher à devenir une plateforme de production.

## Objectif technique

L’objectif est de prédire un masque de segmentation pour une image urbaine et de comparer visuellement le masque réel avec le masque prédit. Les classes Cityscapes sont regroupées en 8 familles principales pour simplifier l’analyse : arrière-plan, route, trottoir, bâtiment, construction, objet, végétation et véhicule.

## Architecture générale

```text
.
|-- app/                     # Interface Streamlit
|-- main.py                  # API FastAPI actuelle
|-- src/
|   |-- config.py            # Configuration des chemins et backends
|   |-- inference.py         # Inférence locale/API et fallback
|   |-- segmentation.py      # Mapping Cityscapes et classes
|   |-- visualization.py     # Conversion image et graphiques
|   |-- models/              # Architectures U-Net
|   |-- utils/               # Fonctions utilitaires
|   `-- metrics.py           # Métriques et losses
|-- data/processed/          # Images et masques de démonstration
|-- models/                  # Modèles Keras utilisés pour l’inférence locale
|-- notebooks/               # Notebooks d’entraînement / évaluation
|-- docs/                    # Documentation de synthèse du dépôt
|-- documents_soutenance_P8/ # Supports historiques conservés
|-- documents_soutenance_p9/ # Supports historiques conservés
|-- captures_soutenance_P8/  # Captures historiques conservées
|-- captures_soutenance_P9/  # Captures historiques conservées
|-- tests/                   # Tests unitaires légers
|-- requirements.txt         # Dépendances API FastAPI
|-- requirements_streamlit.txt
`-- README.md
```

`main.py` contient encore l’API FastAPI à la racine pour ne pas casser l’existant. Un déplacement vers `api/` pourra être envisagé plus tard si une refonte légère est décidée.

## Modèles testés

Le projet contient plusieurs architectures U-Net dans `src/models/`, notamment des variantes basées sur VGG16, MobileNetV2, ResNet50 et une version plus légère. Les modèles sauvegardés au format Keras sont placés dans `models/`.

Le modèle attendu par défaut pour la démo locale est :

```text
models/unet_effnetv2b0.keras
```

Une synthèse des rôles de chaque architecture est disponible dans `docs/model-comparison.md`.

## API FastAPI

L’API reçoit une image encodée en base64 et renvoie un masque prédit sous forme JSON. Elle est exposée par `main.py` avec une route principale :

```text
POST /predict
```

L’API réutilise désormais le module `src.inference` pour éviter de dupliquer la logique d’inférence entre FastAPI et Streamlit.

## Interface Streamlit

L’application Streamlit permet de sélectionner une image de démonstration, de lancer une prédiction, puis de comparer :

- l’image RGB d’origine ;
- le masque réel remappé vers 8 classes ;
- le masque prédit ;
- la répartition des classes en proportion de pixels.

La démo peut fonctionner en backend local, recommandé pour le portfolio, ou via une API distante en configurant les variables d’environnement.

La couche Streamlit reste volontairement simple : la configuration, le mapping Cityscapes, l’inférence et la visualisation sont progressivement extraits dans `src/` pour garder une base plus lisible sans refonte lourde.

## Données de démonstration

Le dataset complet Cityscapes n’est pas versionné. Le dépôt conserve seulement quelques images et masques de démonstration dans `data/processed/`, afin que l’interface Streamlit puisse être testée sans télécharger tout le dataset.

Les dossiers utilisés par défaut sont :

```text
data/processed/images/test/
data/processed/masks/test/
```

La convention de nommage attendue est documentée dans `data/README.md`.

## Installation locale

Créer un environnement virtuel est recommandé :

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances de l’API :

```bash
pip install -r requirements.txt
```

Installer les dépendances de l’interface Streamlit :

```bash
pip install -r requirements_streamlit.txt
```

Installer les outils de développement :

```bash
pip install -r requirements-dev.txt
```

## Lancement de l’app Streamlit

Mode local recommandé :

```bash
$env:PREDICTION_BACKEND="local"
$env:MODEL_PATH="models\unet_effnetv2b0.keras"
streamlit run app/streamlit_app.py
```

Mode API distante, si un endpoint est disponible :

```bash
$env:PREDICTION_BACKEND="api"
$env:API_URL="https://your-api-host/predict"
streamlit run app/streamlit_app.py
```

## Lancement de l’API FastAPI

```bash
uvicorn main:app --reload
```

L’API charge le modèle Keras attendu dans `models/unet_effnetv2b0.keras`. Si ce fichier est absent, le démarrage de l’API échouera.

## Tests

```bash
pytest
```

Les tests actuels restent légers et vérifient surtout les utilitaires de chargement, d’appel API et de visualisation.

## Déploiement Heroku

Un workflow Heroku existe dans `.github/workflows/deploy-heroku.yml`. L’API distante a été testée, mais son déploiement peut être désactivé ou limité par les ressources disponibles : modèle lourd, cold start et contraintes d’hébergement gratuit.

Le mode recommandé pour évaluer le projet reste donc le lancement local. L’objectif est de montrer l’architecture et le flux d’inférence, pas de maintenir une infrastructure cloud active.

## Limites actuelles

- Le dataset complet Cityscapes n’est pas inclus.
- Les modèles Keras suivis dans Git sont volumineux.
- L’API charge le modèle au démarrage, ce qui peut être coûteux sur une petite infrastructure.
- Les résultats ne sont pas présentés comme un benchmark scientifique complet.
- L’organisation du code reste proche du projet initial pour éviter une refonte prématurée.

## Améliorations possibles

- Séparer proprement l’API dans un dossier `api/`.
- Ajouter une documentation plus détaillée des notebooks.
- Centraliser la configuration des chemins et des modèles.
- Réduire ou externaliser les artefacts modèles volumineux.
- Ajouter des tests supplémentaires sur le prétraitement et le format des prédictions.
- Préparer une démo plus légère avec un modèle optimisé.

## Contexte du projet

Le projet a été initialement développé dans le cadre d’un parcours professionnalisant en Data Science, puis repris comme projet portfolio Computer Vision. Cette version met l’accent sur la lisibilité du dépôt, la compréhension rapide du flux technique et la capacité à relier expérimentation, API et démonstration utilisateur.
