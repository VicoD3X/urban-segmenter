# ============================================================
# Metrics, Losses & Inference Time Utilities
# ============================================================

import tensorflow as tf
import time


# ---------------------------
# 1) METRIQUES
# ---------------------------

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    IoU moyen (Jaccard) sur toutes les classes.
    y_true et y_pred sont de forme (B, H, W, C)
    avec y_true en one-hot et y_pred en softmax.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    num_classes = tf.shape(y_true)[-1]

    # Projection du softmax vers labels -> puis one-hot
    y_pred_labels = tf.argmax(y_pred, axis=-1)                     # (B, H, W)
    y_pred_one_hot = tf.one_hot(y_pred_labels, depth=num_classes)  # (B, H, W, C)

    # Mise à plat
    y_true_f = tf.reshape(y_true, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred_one_hot, [-1, num_classes])

    # Intersection et union par classe
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0) - intersection

    iou_per_class = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou_per_class)


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient global entre y_true et y_pred.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def pixel_accuracy(y_true, y_pred):
    """
    Accuracy pixel par pixel (argmax sur les classes).
    """
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    correct = tf.equal(y_true, y_pred)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# ---------------------------
# 2) LOSSES
# ---------------------------

def dice_loss(y_true, y_pred):
    """
    Dice loss : 1 - Dice coefficient.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def balanced_cross_entropy(beta=0.5):
    """
    Version pondérée de la categorical cross-entropy.
    """
    def loss_fn(y_true, y_pred):
        ce = tf.keras.losses.CategoricalCrossentropy()
        return beta * ce(y_true, y_pred)
    return loss_fn


# ---------------------------
# 3) MESURE TEMPS INFERENCE
# ---------------------------

def measure_inference_time(model, sample_input, n_runs=20):
    """
    Mesure le temps d'inférence moyen par image.
    """
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(sample_input, verbose=0)
    end = time.time()
    avg = (end - start) / n_runs
    print(f"Temps d'inférence moyen : {avg:.4f} sec par image")
    return avg
