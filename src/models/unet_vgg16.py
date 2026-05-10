# ============================================================
# U-Net basé sur VGG16 — Architecture optimisée pour Cityscapes
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore

# Import des métriques et pertes
from src.metrics import (
    dice_loss,
    iou_metric,
    dice_coef,
    pixel_accuracy
)


def unet_vgg16(input_shape=(256, 512, 3), num_classes=8):
    """
    Architecture U-Net utilisant VGG16 comme encodeur.
    Optimisée pour une segmentation 8 classes sur Cityscapes.
    """

    # ---------------------------
    # 1) ENCODER : VGG16
    # ---------------------------
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Geler les poids pour limiter la charge CPU/GPU
    for layer in vgg.layers:
        layer.trainable = False

    # Extraction de 3 niveaux de features
    c1 = vgg.get_layer("block1_conv2").output   # 256x512x64

    c2 = vgg.get_layer("block2_conv2").output   # 128x256x128

    c3 = vgg.get_layer("block3_conv3").output   # 64x128x256

    # ---------------------------
    # 2) BOTTLENECK
    # ---------------------------
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(c3)
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(b)

    # ---------------------------
    # 3) DECODER
    # ---------------------------

    # Up 1 : niveau block3 -> block2
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    x1 = layers.Conv2D(128, 3, padding="same", activation="relu")(u1)
    x1 = layers.Conv2D(128, 3, padding="same", activation="relu")(x1)

    # Up 2 : niveau block2 -> block1
    u2 = layers.UpSampling2D((2, 2))(x1)
    u2 = layers.Concatenate()([u2, c1])
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(u2)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x2)

    # ---------------------------
    # 4) OUTPUT
    # ---------------------------
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x2)

    model = models.Model(inputs=vgg.input, outputs=outputs, name="UNet_VGG16")

    # ---------------------------
    # 5) COMPILATION (Keras 3)
    # ---------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=dice_loss,
        metrics=[iou_metric, dice_coef, pixel_accuracy]
    )

    return model
