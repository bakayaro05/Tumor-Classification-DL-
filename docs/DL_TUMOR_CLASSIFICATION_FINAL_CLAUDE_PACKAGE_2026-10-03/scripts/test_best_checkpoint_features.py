from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


# ============================================================
# PROJECT PATH SETUP
# ============================================================

# Script:
#
# brain_tumor_project/
# └── actual_test_scripts/
#     └── test_best_checkpoint_features.py

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
print("BEST CHECKPOINT FEATURE EXTRACTION TEST")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK CHECKPOINT EXISTS
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


print("\nCheckpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)


print("\nCheckpoint metadata:")

print(
    "Epoch:",
    checkpoint.get("epoch")
)

print(
    "Train loss:",
    checkpoint.get("train_loss")
)

print(
    "Train accuracy:",
    checkpoint.get("train_accuracy")
)

print(
    "Validation loss:",
    checkpoint.get("val_loss")
)

print(
    "Validation accuracy:",
    checkpoint.get("val_accuracy")
)


# ============================================================
# BUILD MODEL USING CHECKPOINT SETTINGS
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


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)


# VERY IMPORTANT:
#
# Feature extraction must use evaluation mode.
#
# This disables Dropout and tells BatchNorm to use
# the running statistics learned during training.

model.eval()


print(
    "\n✓ Best model weights loaded successfully"
)


# ============================================================
# LOAD SAVED TRAIN/VALIDATION INDICES
# ============================================================

train_indices, val_indices = (
    load_saved_indices()
)


print(
    "\nSaved training indices:",
    len(train_indices)
)

print(
    "Saved validation indices:",
    len(val_indices)
)


# ============================================================
# CREATE DETERMINISTIC TRAINING DATASET
# ============================================================

# IMPORTANT:
#
# We DO NOT use train_transform here.
#
# The normal training transform contains:
#
# RandomHorizontalFlip
# RandomVerticalFlip
# ColorJitter
#
# Those are useful while TRAINING the neural network,
# but NOT while generating the final fixed feature vectors.
#
# For PCA/SVM we need deterministic features.

train_eval_full = datasets.ImageFolder(

    root=TRAIN_DIR,

    transform=eval_transform,
)


train_feature_dataset = Subset(

    train_eval_full,

    train_indices.tolist(),
)


print(
    "\nDeterministic training feature dataset:",
    len(train_feature_dataset)
)


# ============================================================
# CREATE FEATURE EXTRACTION LOADER
# ============================================================

train_feature_loader = DataLoader(

    train_feature_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),
)


print(
    "Feature extraction batches:",
    len(train_feature_loader)
)


# ============================================================
# GET ONE REAL MRI BATCH
# ============================================================

images, labels = next(
    iter(train_feature_loader)
)


print("\nREAL FEATURE-EXTRACTION BATCH")

print(
    "Images:",
    images.shape
)

print(
    "Labels:",
    labels.shape
)

print(
    "Image dtype:",
    images.dtype
)

print(
    "Label dtype:",
    labels.dtype
)


# ============================================================
# MOVE TO GPU
# ============================================================

images = images.to(
    device,
    non_blocking=True,
)

labels = labels.to(
    device,
    non_blocking=True,
)


# ============================================================
# EXTRACT DEEP FEATURES
# ============================================================

with torch.no_grad():

    features = model(

        images,

        return_features=True,
    )


print("\nExtracted feature tensor:")

print(
    features.shape
)


# ============================================================
# VERIFY EXPECTED SHAPE
# ============================================================

expected_shape = (
    images.shape[0],
    128,
)


assert tuple(features.shape) == expected_shape, (
    f"Unexpected feature shape: "
    f"{features.shape}. "
    f"Expected {expected_shape}"
)


assert torch.isfinite(
    features
).all(), (
    "Feature tensor contains NaN or infinity."
)


print(
    "\n✓ Feature shape is correct"
)

print(
    "✓ All extracted features are finite"
)


# ============================================================
# SHOW SMALL FEATURE SAMPLE
# ============================================================

print(
    "\nFirst MRI — first 10 learned features:"
)

print(
    features[
        0,
        :10
    ].detach().cpu().tolist()
)


print(
    "\nFirst batch labels:"
)

print(
    labels.detach().cpu().tolist()
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BEST CHECKPOINT FEATURE EXTRACTION SUCCESSFUL"
)

print(
    "=" * 70
)


print(
    "\nThe trained ResNet152 + LSTM is now ready "
    "to generate fixed 128-D features for:"
)

print(
    "  Training   : 4569 images"
)

print(
    "  Validation : 1143 images"
)

print(
    "  Testing    : 1311 images"
)

print(
    "\nNEXT STEP: extract and save ALL deep features "
    "and labels for PCA + SVM."
)