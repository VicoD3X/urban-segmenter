# Comparaison des modèles

Ce document synthétise le rôle des architectures présentes dans le dépôt. Il ne remplace pas les notebooks d’entraînement et ne présente pas de benchmark exhaustif.

## Objectif

Le projet explore plusieurs variantes de U-Net pour la segmentation sémantique Cityscapes. L’objectif est de comparer des backbones classiques et plus récents dans un cadre de laboratoire Computer Vision, sans transformer le dépôt en solution production-ready.

## Architectures

| Modèle | Rôle dans le projet | Lecture rapide |
| --- | --- | --- |
| U-Net mini | Prototype léger | Utile pour valider rapidement la chaîne de données et les sorties de masque. |
| U-Net VGG16 | Baseline historique | Backbone classique, simple à expliquer et utile comme point de comparaison. |
| U-Net MobileNetV2 | Variante légère | Approche orientée efficacité, intéressante pour une démo ou une contrainte d’inférence. |
| U-Net ResNet50 | Variante plus profonde | Backbone plus expressif, mais potentiellement plus coûteux. |
| U-Net EfficientNetV2B0 | Modèle de démonstration principal | Version utilisée par défaut dans `models/unet_effnetv2b0.keras` pour l’inférence locale. |

## Positionnement

Les modèles servent à illustrer une démarche d’expérimentation : partir d’une baseline lisible, tester des backbones plus modernes, puis exposer un modèle utilisable dans une API et une interface Streamlit.

Les résultats précis doivent être consultés dans les notebooks et documents historiques si disponibles. Aucun score additionnel n’est inventé dans cette documentation.

## Limites

- Les modèles sauvegardés peuvent être volumineux.
- Les comparaisons dépendent du protocole d’entraînement, des données disponibles et des ressources matérielles.
- Le dépôt conserve une logique de laboratoire portfolio, pas un benchmark industriel complet.
