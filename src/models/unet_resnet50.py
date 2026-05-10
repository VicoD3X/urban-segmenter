# ============================================================
# U-Net basé sur ResNet50 — Architecture optimisée Cityscapes
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore

# Import des métriques, pertes et outils
from src.metrics import (
    dice_loss,
    iou_metric,
    dice_coef,
    pixel_accuracy
)


def unet_resnet50(input_shape=(256, 512, 3), num_classes=8):
    """
    Architecture U-Net utilisant ResNet50 comme encodeur.
    Optimisée pour un découpage 8 classes sur Cityscapes.
    """

    # ---------------------------
    # 1) ENCODER : ResNet50
    # ---------------------------
    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Gel des poids pour stabiliser et réduire la charge CPU/GPU
    for layer in resnet.layers:
        layer.trainable = False

    # Extraction intermédiaire (trois niveaux utiles)
    c1 = resnet.get_layer("conv1_relu").output         # 128x256x64

    c2 = resnet.get_layer("conv2_block3_out").output   # 64x128x256
    c3 = resnet.get_layer("conv3_block4_out").output   # 32x64x512

    # ---------------------------
    # 2) BOTTLENECK
    # ---------------------------
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(c3)
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(b)

    # ---------------------------
    # 3) DECODER
    # ---------------------------

    # Up 1 : 32x64 -> 64x128
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    x1 = layers.Conv2D(256, 3, padding="same", activation="relu")(u1)
    x1 = layers.Conv2D(256, 3, padding="same", activation="relu")(x1)

    # Up 2 : 64x128 -> 128x256
    u2 = layers.UpSampling2D((2, 2))(x1)
    u2 = layers.Concatenate()([u2, c1])
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(u2)
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(x2)

    # Up 3 : 128x256 -> 256x512
    u3 = layers.UpSampling2D((2, 2))(x2)
    x3 = layers.Conv2D(64, 3, padding="same", activation="relu")(u3)
    x3 = layers.Conv2D(64, 3, padding="same", activation="relu")(x3)

    # ---------------------------
    # 4) OUTPUT
    # ---------------------------
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x3)

    model = models.Model(inputs=resnet.input, outputs=outputs, name="UNet_ResNet50")

    # ---------------------------
    # 5) COMPILATION (Keras 3)
    # ---------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=dice_loss,
        metrics=[iou_metric, dice_coef, pixel_accuracy]
    )

    return model
