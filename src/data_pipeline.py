from pathlib import Path
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "brain_tumor_mri"

TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

NUM_CLASSES = len(CLASS_NAMES)

# Common input size for our first ResNet152 implementation.
IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_image(image):
    """
    Apply augmentation only to training images.
    """

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Paper-described brightness variation
    image = tf.image.random_brightness(
        image,
        max_delta=0.1
    )

    # Paper describes contrast variation approximately 0.8–1.2
    image = tf.image.random_contrast(
        image,
        lower=0.8,
        upper=1.2
    )

    # Keep valid normalized pixel range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_image(image, label):
    """
    Convert image values from uint8 [0,255]
    to float32 [0,1].
    """

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def augment_batch(image, label):
    image = augment_image(image)

    return image, label


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets():

    print("=" * 65)
    print("CREATING BRAIN TUMOR MRI DATASETS")
    print("=" * 65)

    # --------------------------------------------------------
    # TRAINING DATASET
    # --------------------------------------------------------

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    train_ds = train_ds.map(
        normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.map(
        normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # AUGMENT TRAINING ONLY
    # --------------------------------------------------------

    train_ds = train_ds.map(
        augment_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # PERFORMANCE PIPELINE
    # --------------------------------------------------------

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    train_ds, val_ds, test_ds = create_datasets()

    print()
    print("CLASS MAPPING")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index} -> {class_name}")

    print()

    for name, dataset in [
        ("TRAIN", train_ds),
        ("VALIDATION", val_ds),
        ("TEST", test_ds),
    ]:

        images, labels = next(iter(dataset))

        print(f"{name}")
        print("Image batch :", images.shape)
        print("Label batch :", labels.shape)
        print(
            "Pixel range :",
            float(tf.reduce_min(images)),
            "to",
            float(tf.reduce_max(images)),
        )
        print()

    print("=" * 65)
    print("DATA PIPELINE READY")
    print("=" * 65)