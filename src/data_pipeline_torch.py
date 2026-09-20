from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "brain_tumor_mri"

TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_INDICES_PATH = SPLIT_DIR / "train_indices.npy"
VAL_INDICES_PATH = SPLIT_DIR / "val_indices.npy"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    # Dataset contains grayscale, RGB, RGBA and palette images.
    # ResNet152 expects 3-channel RGB input.
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # --------------------------------------------------------
    # PAPER-SPECIFIED AUGMENTATION
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),

    # Convert PIL image to PyTorch tensor
    transforms.ToTensor(),

    # ImageNet normalization for pretrained ResNet152
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

eval_transform = transforms.Compose([

    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# LOAD SAVED SPLIT INDICES
# ============================================================

def load_saved_indices():

    if not TRAIN_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {TRAIN_INDICES_PATH}"
        )

    if not VAL_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {VAL_INDICES_PATH}"
        )

    train_indices = np.load(
        TRAIN_INDICES_PATH
    )

    val_indices = np.load(
        VAL_INDICES_PATH
    )

    return train_indices, val_indices


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    # --------------------------------------------------------
    # Base dataset:
    # Used only for labels / class information.
    # --------------------------------------------------------

    base_train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR
    )

    # --------------------------------------------------------
    # Load permanently saved split
    # --------------------------------------------------------

    train_indices, val_indices = load_saved_indices()

    # --------------------------------------------------------
    # Verify indices match current dataset
    # --------------------------------------------------------

    total_images = len(base_train_dataset)

    all_indices = np.concatenate([
        train_indices,
        val_indices,
    ])

    if len(all_indices) != total_images:
        raise ValueError(
            "Saved split size does not match "
            "the current Training dataset."
        )

    if len(np.unique(all_indices)) != total_images:
        raise ValueError(
            "Saved split contains duplicate "
            "or missing indices."
        )

    if np.min(all_indices) < 0:
        raise ValueError(
            "Saved split contains a negative index."
        )

    if np.max(all_indices) >= total_images:
        raise ValueError(
            "Saved split contains an out-of-range index."
        )

    # --------------------------------------------------------
    # Separate dataset objects are required because
    # training and validation use different transforms.
    # --------------------------------------------------------

    train_full = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_full = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=eval_transform,
    )

    train_dataset = Subset(
        train_full,
        train_indices.tolist(),
    )

    val_dataset = Subset(
        val_full,
        val_indices.tolist(),
    )

    # --------------------------------------------------------
    # Official Kaggle test set remains untouched
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=eval_transform,
    )

    return (
        base_train_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_indices,
        val_indices,
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    (
        base_train_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_indices,
        val_indices,
    ) = create_datasets()

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return (
        base_train_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_indices,
        val_indices,
    )


# ============================================================
# CLASS DISTRIBUTION HELPER
# ============================================================

def print_class_distribution(
    title,
    targets,
    indices,
    class_names,
):

    print(title)

    selected_targets = targets[
        indices
    ]

    for class_index, class_name in enumerate(
        class_names
    ):

        count = np.sum(
            selected_targets == class_index
        )

        print(
            f"{class_index} -> "
            f"{class_name:12s}: "
            f"{count}"
        )

    print()


# ============================================================
# STANDALONE PIPELINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PYTORCH BRAIN TUMOR DATA PIPELINE")
    print("=" * 70)

    (
        base_train_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_indices,
        val_indices,
    ) = create_dataloaders()

    print()

    print(
        "Original training images :",
        len(base_train_dataset)
    )

    print(
        "Training split           :",
        len(train_dataset)
    )

    print(
        "Validation split         :",
        len(val_dataset)
    )

    print(
        "Testing images           :",
        len(test_dataset)
    )

    print()

    print("Saved split files:")

    print(
        "Train:",
        TRAIN_INDICES_PATH
    )

    print(
        "Val  :",
        VAL_INDICES_PATH
    )

    print()

    print("Class mapping:")

    for class_name, class_index in (
        base_train_dataset.class_to_idx.items()
    ):

        print(
            f"{class_index} -> {class_name}"
        )

    print()

    targets = np.array(
        base_train_dataset.targets
    )

    print_class_distribution(
        "TRAIN CLASS DISTRIBUTION",
        targets,
        train_indices,
        CLASS_NAMES,
    )

    print_class_distribution(
        "VALIDATION CLASS DISTRIBUTION",
        targets,
        val_indices,
        CLASS_NAMES,
    )

    # --------------------------------------------------------
    # TEST ONE BATCH FROM EACH SPLIT
    # --------------------------------------------------------

    train_images, train_labels = next(
        iter(train_loader)
    )

    val_images, val_labels = next(
        iter(val_loader)
    )

    test_images, test_labels = next(
        iter(test_loader)
    )

    print("TRAIN BATCH")

    print(
        "Images :",
        train_images.shape
    )

    print(
        "Labels :",
        train_labels.shape
    )

    print()

    print("VALIDATION BATCH")

    print(
        "Images :",
        val_images.shape
    )

    print(
        "Labels :",
        val_labels.shape
    )

    print()

    print("TEST BATCH")

    print(
        "Images :",
        test_images.shape
    )

    print(
        "Labels :",
        test_labels.shape
    )

    print()

    print(
        "CUDA available :",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda:0"
        )

        train_images = train_images.to(
            device,
            non_blocking=True,
        )

        train_labels = train_labels.to(
            device,
            non_blocking=True,
        )

        print(
            "Training batch moved to :",
            train_images.device
        )

    print()

    print("=" * 70)
    print(
        "PYTORCH DATA PIPELINE "
        "WITH SAVED SPLIT READY"
    )
    print("=" * 70)