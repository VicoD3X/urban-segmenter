# Modèles

Ce dossier contient les artefacts Keras utilisés pour l’inférence locale.

Le modèle attendu par défaut par l’application Streamlit et l’API FastAPI est :

```text
models/unet_effnetv2b0.keras
```

Le dépôt peut aussi contenir des modèles historiques, par exemple une baseline U-Net VGG16. Ces fichiers sont utiles pour comprendre la démarche d’expérimentation, mais ils peuvent devenir volumineux.

## Bonnes pratiques pour ce dépôt

- Ne pas multiplier les poids de modèles dans Git sans raison claire.
- Garder un seul modèle de démonstration principal quand c’est possible.
- Documenter tout nouvel artefact ajouté dans ce dossier.
- Externaliser les modèles lourds si le dépôt devient difficile à cloner.
