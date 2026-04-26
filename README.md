# Urban Segmenter — Semantic Segmentation for Urban Scenes

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP-0F172A)

## Présentation

Urban Segmenter est un MVP de segmentation sémantique appliqué aux scènes urbaines. Le projet s’appuie sur le dataset Cityscapes, des modèles U-Net entraînés avec des backbones VGG16 et EfficientNetV2B0, ainsi qu’une interface Streamlit pour visualiser les prédictions.

Le dépôt regroupe aussi une preuve de concept d’évolution du modèle : comparaison d’une baseline historique avec une architecture plus moderne, en conservant un protocole de test simple et lisible.

## Objectif

L’objectif est de segmenter une image de rue en grandes classes visuelles exploitables : route, trottoir, bâtiments, objets, végétation, véhicules et arrière-plan. Le projet permet de montrer une démarche complète de Computer Vision : préparation des données, entraînement, évaluation, comparaison de modèles et démonstration locale.

## Contenu actuel

- `app/` : application Streamlit de démonstration.
- `main.py` : API FastAPI de prédiction.
- `notebooks/` : notebooks de préparation, entraînement, évaluation et preuve de concept.
- `src/` : fonctions utilitaires pour les données, les métriques, la visualisation et l’API.
- `models/` : modèles Keras suivis dans le dépôt pour la démonstration.
- `data/processed/` : échantillon d’images et de masques utilisé par l’interface.
- `tests/` : tests légers de cohérence.
- `documents_soutenance_P8/` et `documents_soutenance_p9/` : documents historiques du projet et de la preuve de concept.

## Lancement rapide

Installation minimale pour l’API :

```bash
pip install -r requirements.txt
```

Installation de l’interface Streamlit :

```bash
pip install -r requirements_streamlit.txt
```

Lancer l’interface :

```bash
streamlit run app/streamlit_app.py
```

Lancer l’API :

```bash
uvicorn main:app --reload
```

## Positionnement

Ce dépôt n’est pas présenté comme une solution de production. Il sert de base portfolio pour illustrer un pipeline Computer Vision orienté segmentation urbaine, avec un démonstrateur simple et une preuve de concept d’amélioration de modèle.

Le projet a été initialement développé dans un cadre professionnalisant, puis renommé et préparé pour une remise à niveau progressive sous le nom Urban Segmenter.
