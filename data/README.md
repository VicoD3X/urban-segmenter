# Données

Le dataset complet Cityscapes n’est pas versionné dans ce dépôt. Il est trop volumineux pour être inclus proprement dans un projet portfolio GitHub.

Le dépôt conserve uniquement un petit jeu de démonstration utilisé par l’application Streamlit :

```text
data/processed/images/test/
data/processed/masks/test/
```

## Images attendues

Les images doivent suivre la convention Cityscapes :

```text
*_leftImg8bit.png
```

Exemple :

```text
aachen_000001_000019_leftImg8bit.png
```

## Masques attendus

Les masques doivent suivre la convention :

```text
*_gtFine_labelIds.png
```

Exemple :

```text
aachen_000001_000019_gtFine_labelIds.png
```

Pour une image donnée, l’ID de base doit correspondre entre l’image et le masque. L’application Streamlit utilise cette convention pour associer automatiquement les deux fichiers.

## Utilisation par Streamlit

Par défaut, l’interface lit les dossiers suivants :

```text
data/processed/images/test/
data/processed/masks/test/
```

Ces chemins peuvent être surchargés avec les variables d’environnement `IMAGES_DIR` et `MASKS_DIR`.
