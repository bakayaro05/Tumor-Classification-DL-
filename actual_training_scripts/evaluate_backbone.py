from pathlib import Path
import argparse
import json
import sys
import time

import joblib
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Subset,
)

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

from model_cnn_lstm import (
    CNNLSTM,
    SUPPORTED_BACKBONES,
)

from src.data_pipeline_torch import (
    TRAIN_DIR,
    TEST_DIR,
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


# ============================================================
# CONFIGURATION
# ============================================================

BACKBONE = args.backbone

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
    / "experiments"
    / BACKBONE
)


MODEL_DIR = (
    EXPERIMENT_ROOT
    / "models"
)


FEATURE_DIR = (
    EXPERIMENT_ROOT
    / "features"
)


PCA_DIR = (
    EXPERIMENT_ROOT
    / "pca"
)


RESULT_DIR = (
    EXPERIMENT_ROOT
    / "results"
)


for directory in [

    MODEL_DIR,

    FEATURE_DIR,

    PCA_DIR,

    RESULT_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_model.pth"
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
    MODEL_DIR
    / "svm.joblib"
)


FINAL_RESULTS_PATH = (
    RESULT_DIR
    / "final_results.json"
)


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
    f"FULL BACKBONE EVALUATION: "
    f"{BACKBONE.upper()}"
)

print("=" * 80)


print(
    "\nDevice:",
    device
)


if torch.cuda.is_available():

    print(

        "GPU:",

        torch.cuda.get_device_name(0)

    )


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(

        f"Checkpoint not found: "
        f"{CHECKPOINT_PATH}"

    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(

    CHECKPOINT_PATH,

    map_location=device,

    weights_only=False,
)


print(
    "\nCheckpoint loaded:"
)


print(
    "Backbone:",
    checkpoint.get(
        "backbone",
        BACKBONE,
    )
)


print(
    "Best epoch:",
    checkpoint.get(
        "epoch"
    )
)


print(
    "NN validation accuracy:",
    checkpoint.get(
        "val_accuracy"
    )
)


# ============================================================
# BUILD MODEL
# ============================================================

model = CNNLSTM(

    backbone_name=BACKBONE,

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

    checkpoint[
        "model_state_dict"
    ]

)


model = model.to(
    device
)


model.eval()


print(
    "\n✓ Best neural checkpoint loaded successfully"
)


# ============================================================
# LOAD SAVED SPLIT
# ============================================================

train_indices, val_indices = (
    load_saved_indices()
)


# ============================================================
# CREATE DETERMINISTIC DATASETS
# ============================================================

# Important:
#
# Feature extraction must NOT use training augmentation.
#
# Therefore training and validation both use eval_transform.

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


test_dataset = datasets.ImageFolder(

    root=TEST_DIR,

    transform=eval_transform,
)


print(
    "\nDataset sizes:"
)


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
        f"\nExtracting "
        f"{split_name} features..."
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
    # VERIFY
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
# FEATURE EXTRACTION
# ============================================================

feature_start = time.time()


train_features, train_labels = (
    extract_features(

        model,

        train_loader,

        device,

        "TRAINING",
    )
)


val_features, val_labels = (
    extract_features(

        model,

        val_loader,

        device,

        "VALIDATION",
    )
)


test_features, test_labels = (
    extract_features(

        model,

        test_loader,

        device,

        "TEST",
    )
)


feature_seconds = (
    time.time()
    - feature_start
)


# ============================================================
# VERIFY EXPECTED SHAPES
# ============================================================

assert train_features.shape == (
    4569,
    128,
)

assert val_features.shape == (
    1143,
    128,
)

assert test_features.shape == (
    1311,
    128,
)


# ============================================================
# SAVE RAW DEEP FEATURES
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


print(
    "\n✓ Raw deep features saved"
)


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


test_scaled = scaler.transform(

    test_features

)


joblib.dump(

    scaler,

    SCALER_PATH,
)


print(
    "✓ StandardScaler complete"
)


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
    n_components
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


test_pca = pca.transform(

    test_scaled

)


joblib.dump(

    pca,

    PCA_MODEL_PATH,
)


# ============================================================
# SAVE PCA FEATURES
# ============================================================

np.save(

    PCA_DIR
    / "train_pca.npy",

    train_pca,
)


np.save(

    PCA_DIR
    / "val_pca.npy",

    val_pca,
)


np.save(

    PCA_DIR
    / "test_pca.npy",

    test_pca,
)


np.save(

    PCA_DIR
    / "train_labels.npy",

    train_labels,
)


np.save(

    PCA_DIR
    / "val_labels.npy",

    val_labels,
)


np.save(

    PCA_DIR
    / "test_labels.npy",

    test_labels,
)


cumulative_variance = np.cumsum(

    pca.explained_variance_ratio_

)


print(
    "PCA explained variance retained:",
    float(
        cumulative_variance[-1]
    )
)


print(
    "✓ PCA complete"
)


# ============================================================
# SVM
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
    "✓ SVM training complete"
)


print(
    f"SVM training time: "
    f"{svm_seconds:.2f} seconds"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    labels,
    predictions,
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "weighted_recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print(
    "\nEvaluating VALIDATION set..."
)


val_predictions = svm.predict(

    val_pca

)


val_metrics = calculate_metrics(

    val_labels,

    val_predictions,
)


val_cm = confusion_matrix(

    val_labels,

    val_predictions,
)


# ============================================================
# TEST EVALUATION
# ============================================================

print(
    "Evaluating untouched TEST set..."
)


test_predictions = svm.predict(

    test_pca

)


test_metrics = calculate_metrics(

    test_labels,

    test_predictions,
)


test_cm = confusion_matrix(

    test_labels,

    test_predictions,
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

val_report = classification_report(

    val_labels,

    val_predictions,

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0,
)


test_report = classification_report(

    test_labels,

    test_predictions,

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "VALIDATION RESULTS"
)

print(
    "=" * 80
)


print(

    f"\nAccuracy        : "
    f"{val_metrics['accuracy'] * 100:.2f}%"

)


print(

    f"Macro Precision : "
    f"{val_metrics['macro_precision'] * 100:.2f}%"

)


print(

    f"Macro Recall    : "
    f"{val_metrics['macro_recall'] * 100:.2f}%"

)


print(

    f"Macro F1        : "
    f"{val_metrics['macro_f1'] * 100:.2f}%"

)


print(
    "\nConfusion matrix:"
)


print(
    val_cm
)


print(
    "\n"
    + "=" * 80
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 80
)


print(

    f"\nAccuracy        : "
    f"{test_metrics['accuracy'] * 100:.2f}%"

)


print(

    f"Macro Precision : "
    f"{test_metrics['macro_precision'] * 100:.2f}%"

)


print(

    f"Macro Recall    : "
    f"{test_metrics['macro_recall'] * 100:.2f}%"

)


print(

    f"Macro F1        : "
    f"{test_metrics['macro_f1'] * 100:.2f}%"

)


print(
    "\nConfusion matrix:"
)


print(
    test_cm
)


# ============================================================
# COMPLETE RESULTS OBJECT
# ============================================================

results = {

    "backbone":
        BACKBONE,

    "neural_network": {

        "best_epoch":
            int(
                checkpoint.get(
                    "epoch"
                )
            ),

        "train_accuracy":
            float(
                checkpoint.get(
                    "train_accuracy"
                )
            ),

        "train_loss":
            float(
                checkpoint.get(
                    "train_loss"
                )
            ),

        "val_accuracy":
            float(
                checkpoint.get(
                    "val_accuracy"
                )
            ),

        "val_loss":
            float(
                checkpoint.get(
                    "val_loss"
                )
            ),
    },


    "deep_features": {

        "dimension":
            128,

        "train_samples":
            int(
                train_features.shape[0]
            ),

        "val_samples":
            int(
                val_features.shape[0]
            ),

        "test_samples":
            int(
                test_features.shape[0]
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
            float(
                cumulative_variance[-1]
            ),
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


    "test": {

        **test_metrics,

        "confusion_matrix":
            test_cm.tolist(),

        "classification_report":
            test_report,
    },
}


# ============================================================
# SAVE RESULTS JSON
# ============================================================

with open(

    FINAL_RESULTS_PATH,

    "w",

    encoding="utf-8",

) as file:


    json.dump(

        results,

        file,

        indent=4,
    )


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BACKBONE EVALUATION COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nBackbone:",
    BACKBONE
)


print(
    f"NN best validation accuracy : "
    f"{checkpoint['val_accuracy']:.2f}%"
)


print(
    f"SVM validation accuracy     : "
    f"{val_metrics['accuracy'] * 100:.2f}%"
)


print(
    f"Final test accuracy         : "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)


print(
    f"Final test macro F1         : "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)


print(
    "\nSaved:"
)


print(
    "Raw features:",
    FEATURE_DIR
)


print(
    "Scaler/PCA:",
    PCA_DIR
)


print(
    "SVM:",
    SVM_PATH
)


print(
    "Final results:",
    FINAL_RESULTS_PATH
)