from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_pipeline_torch import (
    TRAIN_DIR,
    TEST_DIR,
    eval_transform,
    load_saved_indices,
)

from model_resnet152_lstm import ResNet152LSTM


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_resnet152_lstm.pth"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "features"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 32
NUM_WORKERS = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DEEP FEATURE EXTRACTION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)


print("\nCheckpoint loaded:")

print(
    "Epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation accuracy:",
    checkpoint.get("val_accuracy")
)


# ============================================================
# BUILD MODEL
# ============================================================

model = ResNet152LSTM(

    lstm_hidden_size=checkpoint.get(
        "lstm_hidden_size",
        256,
    ),

    dense_features=checkpoint.get(
        "dense_features",
        128,
    ),

    num_classes=4,

    dropout=checkpoint.get(
        "dropout",
        0.3,
    ),
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    "\n✓ Best trained model loaded"
)


# ============================================================
# LOAD SAVED TRAIN / VALIDATION SPLIT
# ============================================================

train_indices, val_indices = (
    load_saved_indices()
)


# ============================================================
# DETERMINISTIC DATASETS
# ============================================================

# Entire original Training directory,
# but using eval_transform so feature extraction
# is deterministic.

train_eval_full = datasets.ImageFolder(

    root=TRAIN_DIR,

    transform=eval_transform,
)


train_dataset = Subset(

    train_eval_full,

    train_indices.tolist(),
)


val_dataset = Subset(

    train_eval_full,

    val_indices.tolist(),
)


# Official untouched Kaggle test set

test_dataset = datasets.ImageFolder(

    root=TEST_DIR,

    transform=eval_transform,
)


print("\nDataset sizes:")

print(
    "Training  :",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Testing   :",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = torch.cuda.is_available()


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

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


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(
    model,
    loader,
    device,
    split_name,
):

    feature_batches = []

    label_batches = []


    print(
        f"\nExtracting {split_name} features..."
    )


    with torch.no_grad():


        for batch_index, (
            images,
            labels,
        ) in enumerate(loader):


            images = images.to(
                device,
                non_blocking=True,
            )


            features = model(
                images,
                return_features=True,
            )


            # Move immediately back to CPU
            # so GPU memory does not accumulate.

            feature_batches.append(
                features.cpu().numpy()
            )


            label_batches.append(
                labels.numpy()
            )


            if (
                (batch_index + 1) % 25 == 0
                or
                (batch_index + 1) == len(loader)
            ):

                print(
                    f"    Batch "
                    f"{batch_index + 1:4d}"
                    f"/{len(loader):4d}"
                )


    features = np.concatenate(
        feature_batches,
        axis=0,
    )


    labels = np.concatenate(
        label_batches,
        axis=0,
    )


    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    assert features.ndim == 2

    assert features.shape[1] == 128

    assert features.shape[0] == labels.shape[0]

    assert np.isfinite(
        features
    ).all()


    print(
        f"\n{split_name} feature shape:",
        features.shape
    )

    print(
        f"{split_name} label shape:",
        labels.shape
    )


    return (
        features,
        labels,
    )


# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================

train_features, train_labels = (
    extract_features(
        model,
        train_loader,
        device,
        "TRAINING",
    )
)


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

val_features, val_labels = (
    extract_features(
        model,
        val_loader,
        device,
        "VALIDATION",
    )
)


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

test_features, test_labels = (
    extract_features(
        model,
        test_loader,
        device,
        "TEST",
    )
)


# ============================================================
# FINAL SHAPE VERIFICATION
# ============================================================

assert train_features.shape == (
    len(train_dataset),
    128,
)

assert val_features.shape == (
    len(val_dataset),
    128,
)

assert test_features.shape == (
    len(test_dataset),
    128,
)


assert train_labels.shape == (
    len(train_dataset),
)

assert val_labels.shape == (
    len(val_dataset),
)

assert test_labels.shape == (
    len(test_dataset),
)


print(
    "\n✓ All feature shapes verified"
)


# ============================================================
# SAVE NUMPY ARRAYS
# ============================================================

np.save(
    FEATURE_DIR
    / "train_features.npy",
    train_features,
)

np.save(
    FEATURE_DIR
    / "train_labels.npy",
    train_labels,
)


np.save(
    FEATURE_DIR
    / "val_features.npy",
    val_features,
)

np.save(
    FEATURE_DIR
    / "val_labels.npy",
    val_labels,
)


np.save(
    FEATURE_DIR
    / "test_features.npy",
    test_features,
)

np.save(
    FEATURE_DIR
    / "test_labels.npy",
    test_labels,
)


# ============================================================
# CLASS DISTRIBUTION CHECK
# ============================================================

def print_class_counts(
    name,
    labels,
):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(
        f"\n{name} class counts:"
    )

    for class_index, count in zip(
        unique,
        counts,
    ):

        print(
            f"    Class "
            f"{int(class_index)}: "
            f"{int(count)}"
        )


print_class_counts(
    "Training",
    train_labels,
)

print_class_counts(
    "Validation",
    val_labels,
)

print_class_counts(
    "Test",
    test_labels,
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ALL DEEP FEATURES EXTRACTED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nSaved feature files:"
)


for filename in [

    "train_features.npy",
    "train_labels.npy",

    "val_features.npy",
    "val_labels.npy",

    "test_features.npy",
    "test_labels.npy",

]:

    path = FEATURE_DIR / filename

    print(
        f"    {filename:22s}"
        f" -> "
        f"{path}"
    )


print(
    "\nNEXT STEP:"
)

print(
    "Load these saved features, fit StandardScaler "
    "on TRAIN only, then fit PCA on TRAIN only."
)