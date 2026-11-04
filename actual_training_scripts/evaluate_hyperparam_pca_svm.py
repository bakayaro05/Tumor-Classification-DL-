#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import sys
import time

import joblib
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from model_cnn_lstm import CNNLSTM, SUPPORTED_BACKBONES

from src.data_pipeline_torch import (
    TRAIN_DIR,
    eval_transform,
    load_saved_indices,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--backbone",
    required=True,
    choices=SUPPORTED_BACKBONES,
)

parser.add_argument(
    "--run-name",
    required=True,
)

parser.add_argument(
    "--checkpoint-type",
    required=True,
    choices=["best", "epoch30"],
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--pca-components",
    type=int,
    default=64,
)

parser.add_argument(
    "--svm-c",
    type=float,
    default=1.0,
)

args = parser.parse_args()


BACKBONE = args.backbone
RUN_NAME = args.run_name
CHECKPOINT_TYPE = args.checkpoint_type

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
PCA_COMPONENTS = args.pca_components
SVM_C = args.svm_c


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# ============================================================
# EXPERIMENT PATHS
# ============================================================

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "hyperparameter_dl"
    / BACKBONE
    / RUN_NAME
)

MODEL_DIR = (
    EXPERIMENT_ROOT
    / "models"
)

TRAINING_RESULTS_PATH = (
    EXPERIMENT_ROOT
    / "results"
    / "training_results.json"
)


if CHECKPOINT_TYPE == "best":

    CHECKPOINT_FILENAME = "best_val_within_30.pth"
    EVALUATION_NAME = "best_val_within_30"

else:

    CHECKPOINT_FILENAME = "epoch_30_model.pth"
    EVALUATION_NAME = "epoch_30"


CHECKPOINT_PATH = (
    MODEL_DIR
    / CHECKPOINT_FILENAME
)

EVALUATION_ROOT = (
    EXPERIMENT_ROOT
    / "evaluations"
    / EVALUATION_NAME
)

FEATURE_DIR = (
    EVALUATION_ROOT
    / "features"
)

PCA_DIR = (
    EVALUATION_ROOT
    / "pca"
)

SVM_DIR = (
    EVALUATION_ROOT
    / "models"
)

RESULT_DIR = (
    EVALUATION_ROOT
    / "results"
)


for directory in [
    FEATURE_DIR,
    PCA_DIR,
    SVM_DIR,
    RESULT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


SCALER_PATH = (
    PCA_DIR
    / "standard_scaler.joblib"
)

PCA_MODEL_PATH = (
    PCA_DIR
    / "pca_model.joblib"
)

SVM_PATH = (
    SVM_DIR
    / "svm.joblib"
)

FINAL_RESULTS_PATH = (
    RESULT_DIR
    / "final_results.json"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if not TRAINING_RESULTS_PATH.is_file():
    raise FileNotFoundError(
        f"Training results not found:\n"
        f"{TRAINING_RESULTS_PATH}"
    )


# ============================================================
# LOAD TRAINING METADATA
# ============================================================

with TRAINING_RESULTS_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    training_results = json.load(file)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)

print(
    "PHASE 2B FULL HYBRID EVALUATION"
)

print("=" * 80)

print("Backbone        :", BACKBONE)
print("Run             :", RUN_NAME)
print("Checkpoint type :", CHECKPOINT_TYPE)
print("Checkpoint      :", CHECKPOINT_PATH)
print("Device          :", device)

if torch.cuda.is_available():
    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)


# Prefer metadata saved in checkpoint.
# Fall back to training_results.json.

LSTM_HIDDEN_SIZE = int(
    checkpoint.get(
        "lstm_hidden_size",
        training_results.get(
            "lstm_hidden_size",
            256,
        ),
    )
)

DENSE_FEATURES = int(
    checkpoint.get(
        "dense_features",
        training_results.get(
            "dense_features",
            128,
        ),
    )
)

DROPOUT = float(
    checkpoint.get(
        "dropout",
        training_results.get(
            "dropout",
            0.3,
        ),
    )
)

LEARNING_RATE = float(
    checkpoint.get(
        "learning_rate",
        training_results.get(
            "learning_rate",
            1e-4,
        ),
    )
)


print()
print("Checkpoint epoch :", checkpoint.get("epoch"))
print(
    "Checkpoint neural validation accuracy :",
    checkpoint.get("val_accuracy"),
)
print("Learning rate     :", LEARNING_RATE)
print("LSTM hidden size  :", LSTM_HIDDEN_SIZE)
print("Dense features    :", DENSE_FEATURES)
print("Dropout           :", DROPOUT)


# ============================================================
# BUILD MODEL
# ============================================================

model = CNNLSTM(
    backbone_name=BACKBONE,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    dense_features=DENSE_FEATURES,
    num_classes=4,
    dropout=DROPOUT,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("\n✓ CNN-LSTM checkpoint loaded successfully")


# ============================================================
# SAVED TRAIN / VALIDATION SPLIT
# ============================================================

train_indices, val_indices = (
    load_saved_indices()
)


# ============================================================
# DETERMINISTIC DATASETS
# ============================================================

# IMPORTANT:
# No augmentation during feature extraction.

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


print("\nDataset sizes:")

print(
    "Training   :",
    len(train_dataset)
)

print(
    "Validation :",
    len(val_dataset)
)


# ============================================================
# LOADERS
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    model,
    loader,
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

            feature_batches.append(
                features
                .detach()
                .cpu()
                .numpy()
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
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    features = np.concatenate(
        feature_batches,
        axis=0,
    )

    labels = np.concatenate(
        label_batches,
        axis=0,
    )

    assert features.ndim == 2

    assert (
        features.shape[0]
        == labels.shape[0]
    )

    assert (
        features.shape[1]
        == DENSE_FEATURES
    )

    assert np.isfinite(
        features
    ).all()

    print(
        f"{split_name} features:",
        features.shape,
    )

    return features, labels


feature_start = time.time()

train_features, train_labels = (
    extract_features(
        model,
        train_loader,
        "TRAIN",
    )
)

val_features, val_labels = (
    extract_features(
        model,
        val_loader,
        "VALIDATION",
    )
)

feature_seconds = (
    time.time()
    - feature_start
)


assert train_features.shape[0] == 4569
assert val_features.shape[0] == 1143


# ============================================================
# SAVE RAW FEATURES
# ============================================================

np.save(
    FEATURE_DIR / "train_features.npy",
    train_features,
)

np.save(
    FEATURE_DIR / "train_labels.npy",
    train_labels,
)

np.save(
    FEATURE_DIR / "val_features.npy",
    val_features,
)

np.save(
    FEATURE_DIR / "val_labels.npy",
    val_labels,
)

print("\n✓ Raw features saved")


# ============================================================
# STANDARD SCALER
# ============================================================

print(
    "\nFitting StandardScaler "
    "on TRAIN only..."
)

scaler = StandardScaler()

train_scaled = scaler.fit_transform(
    train_features
)

val_scaled = scaler.transform(
    val_features
)

joblib.dump(
    scaler,
    SCALER_PATH,
)

print("✓ StandardScaler complete")


# ============================================================
# PCA
# ============================================================

n_components = min(
    PCA_COMPONENTS,
    train_scaled.shape[1],
    train_scaled.shape[0],
)

print(
    "\nPCA components:",
    n_components,
)

pca = PCA(
    n_components=n_components,
    random_state=42,
)

train_pca = pca.fit_transform(
    train_scaled
)

val_pca = pca.transform(
    val_scaled
)

joblib.dump(
    pca,
    PCA_MODEL_PATH,
)


np.save(
    PCA_DIR / "train_pca.npy",
    train_pca,
)

np.save(
    PCA_DIR / "val_pca.npy",
    val_pca,
)

np.save(
    PCA_DIR / "train_labels.npy",
    train_labels,
)

np.save(
    PCA_DIR / "val_labels.npy",
    val_labels,
)


explained_variance = float(
    np.sum(
        pca.explained_variance_ratio_
    )
)


print(
    "PCA explained variance retained:",
    explained_variance,
)

print("✓ PCA complete")


# ============================================================
# LINEAR SVM
# ============================================================

print(
    "\nTraining Linear SVM..."
)

svm = SVC(
    kernel="linear",
    C=SVM_C,
    probability=False,
    random_state=42,
)

svm_start = time.time()

svm.fit(
    train_pca,
    train_labels,
)

svm_seconds = (
    time.time()
    - svm_start
)

joblib.dump(
    svm,
    SVM_PATH,
)

print(
    f"✓ SVM complete "
    f"({svm_seconds:.2f} sec)"
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print(
    "\nEvaluating VALIDATION set..."
)

val_predictions = svm.predict(
    val_pca
)


val_metrics = {

    "accuracy":
        float(
            accuracy_score(
                val_labels,
                val_predictions,
            )
        ),

    "macro_precision":
        float(
            precision_score(
                val_labels,
                val_predictions,
                average="macro",
                zero_division=0,
            )
        ),

    "macro_recall":
        float(
            recall_score(
                val_labels,
                val_predictions,
                average="macro",
                zero_division=0,
            )
        ),

    "macro_f1":
        float(
            f1_score(
                val_labels,
                val_predictions,
                average="macro",
                zero_division=0,
            )
        ),

    "weighted_f1":
        float(
            f1_score(
                val_labels,
                val_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
}


val_cm = confusion_matrix(
    val_labels,
    val_predictions,
)

val_report = classification_report(
    val_labels,
    val_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)


# ============================================================
# RESULTS
# ============================================================

results = {

    "phase":
        "phase2B_hyperparameter_full_hybrid",

    "backbone":
        BACKBONE,

    "run_name":
        RUN_NAME,

    "checkpoint_type":
        CHECKPOINT_TYPE,

    "checkpoint_file":
        CHECKPOINT_FILENAME,

    "checkpoint_epoch":
        int(
            checkpoint.get(
                "epoch",
                30,
            )
        ),

    "hyperparameters": {

        "learning_rate":
            LEARNING_RATE,

        "lstm_hidden_size":
            LSTM_HIDDEN_SIZE,

        "dense_features":
            DENSE_FEATURES,

        "dropout":
            DROPOUT,
    },

    "neural_checkpoint": {

        "train_accuracy":
            float(
                checkpoint.get(
                    "train_accuracy",
                    0.0,
                )
            ),

        "val_accuracy":
            float(
                checkpoint.get(
                    "val_accuracy",
                    0.0,
                )
            ),
    },

    "deep_features": {

        "dimension":
            int(
                train_features.shape[1]
            ),

        "train_samples":
            int(
                train_features.shape[0]
            ),

        "val_samples":
            int(
                val_features.shape[0]
            ),

        "extraction_time_seconds":
            float(
                feature_seconds
            ),
    },

    "pca": {

        "components":
            int(
                n_components
            ),

        "explained_variance":
            explained_variance,
    },

    "svm": {

        "kernel":
            "linear",

        "C":
            float(
                SVM_C
            ),

        "training_time_seconds":
            float(
                svm_seconds
            ),
    },

    "validation": {

        **val_metrics,

        "confusion_matrix":
            val_cm.tolist(),

        "classification_report":
            val_report,
    },
}


with FINAL_RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("PHASE 2B HYBRID EVALUATION COMPLETE")
print("=" * 80)

print("Backbone       :", BACKBONE)
print("Run            :", RUN_NAME)
print("Checkpoint     :", CHECKPOINT_TYPE)

print(
    f"Neural val acc : "
    f"{float(checkpoint.get('val_accuracy', 0.0)):.2f}%"
)

print(
    f"SVM val acc    : "
    f"{val_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"SVM macro F1   : "
    f"{val_metrics['macro_f1'] * 100:.2f}%"
)

print(
    "\nSaved results:",
    FINAL_RESULTS_PATH,
)   