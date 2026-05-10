import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence # type: ignore

import albumentations as A


# ============================================================
# Mapping 34 -> 8 classes
# ============================================================

CITYSCAPES_34_TO_8 = {
    0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0,      # background
    7:1,                                    # road
    8:2,                                    # sidewalk
    11:3,                                   # building
    12:4, 13:4, 14:4, 15:4, 16:4,            # other construction
    17:5, 18:5, 19:5, 20:5,                  # object
    21:6, 22:6,                              # vegetation
    23:7, 24:7, 25:7, 26:7, 27:7, 28:7, 
    29:7, 30:7, 31:7, 32:7, 33:7             # vehicle
}


# ============================================================
# Fonction : mapping 34 -> 8
# ============================================================

def remap_mask(mask):
    """Applique le mapping 34 classes -> 8 classes."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_id, new_id in CITYSCAPES_34_TO_8.items():
        new_mask[mask == old_id] = new_id
    return new_mask


# ============================================================
# Augmentations Albumentations
# ============================================================

def get_augmentations():
    """Augmentations appliquées uniquement sur les images d'entraînement."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=256, width=512, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.4, border_mode=0)
    ], additional_targets={'mask': 'mask'})


# ============================================================
# Générateur Cityscapes (Keras Sequence)
# ============================================================

class CityscapesSequence(Sequence):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 batch_size=4,
                 target_size=(256, 512),
                 augment=False,
                 n_classes=8,
                 shuffle=True):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size  # (H, W)
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.aug = get_augmentations() if augment else None

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for i in batch_indexes:
            img = cv2.imread(str(self.image_paths[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(self.mask_paths[i]), cv2.IMREAD_UNCHANGED)

            mask = remap_mask(mask)

            # Resize
            H, W = self.target_size
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # Augmentation
            if self.augment:
                augmented = self.aug(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            # Normalisation image
            img = img / 255.0

            # One-hot encoding du mask
            mask = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)

            images.append(img)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)
