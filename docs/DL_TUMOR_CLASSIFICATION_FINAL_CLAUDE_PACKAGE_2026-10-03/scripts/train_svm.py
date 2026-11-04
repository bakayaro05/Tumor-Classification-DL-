from pathlib import Path
import sys
import time

import joblib
import numpy as np

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
# PATHS
# ============================================================

PCA_DIR = (
    PROJECT_ROOT
    / "pca_features"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SVM_PATH = (
    MODEL_DIR
    / "resnet152_lstm_pca_svm.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================

SVM_C = 1.0

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# ============================================================
# LOAD TRAIN + VALIDATION PCA FEATURES
# ============================================================

print("=" * 70)
print("LINEAR SVM TRAINING")
print("=" * 70)


train_features = np.load(
    PCA_DIR
    / "train_pca.npy"
)

train_labels = np.load(
    PCA_DIR
    / "train_labels.npy"
)


val_features = np.load(
    PCA_DIR
    / "val_pca.npy"
)

val_labels = np.load(
    PCA_DIR
    / "val_labels.npy"
)


print("\nLoaded PCA features:")

print(
    "Training features  :",
    train_features.shape
)

print(
    "Training labels    :",
    train_labels.shape
)

print(
    "Validation features:",
    val_features.shape
)

print(
    "Validation labels  :",
    val_labels.shape
)


# ============================================================
# VERIFY DATA
# ============================================================

assert train_features.shape == (
    4569,
    64,
)

assert val_features.shape == (
    1143,
    64,
)

assert train_labels.shape == (
    4569,
)

assert val_labels.shape == (
    1143,
)


assert np.isfinite(
    train_features
).all()

assert np.isfinite(
    val_features
).all()


assert train_labels.min() >= 0
assert train_labels.max() <= 3

assert val_labels.min() >= 0
assert val_labels.max() <= 3


print(
    "\n✓ PCA feature arrays verified"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\nTraining class distribution:"
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    count = np.sum(
        train_labels == class_index
    )

    print(
        f"    {class_index} -> "
        f"{class_name:12s}: "
        f"{count}"
    )


# ============================================================
# CREATE LINEAR SVM
# ============================================================

# SVC with kernel="linear" gives us a linear SVM.
#
# probability=True enables predict_proba(), which will
# also be useful later for confidence/probability output.

svm = SVC(

    kernel="linear",

    C=SVM_C,

    probability=True,

    random_state=42,
)


print(
    "\nSVM configuration:"
)

print(
    "Kernel      : linear"
)

print(
    "C           :",
    SVM_C
)

print(
    "Probability : True"
)


# ============================================================
# TRAIN SVM
# ============================================================

print(
    "\nTraining SVM on TRAINING features..."
)


training_start = time.time()


svm.fit(
    train_features,
    train_labels,
)


training_seconds = (
    time.time()
    - training_start
)


print(
    "✓ SVM training complete"
)


print(
    f"Training time: "
    f"{training_seconds:.2f} seconds"
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print(
    "\nEvaluating on VALIDATION set..."
)


val_predictions = svm.predict(
    val_features
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    val_labels,
    val_predictions,
)


precision_macro = precision_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0,
)


recall_macro = recall_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0,
)


f1_macro = f1_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0,
)


precision_weighted = precision_score(
    val_labels,
    val_predictions,
    average="weighted",
    zero_division=0,
)


recall_weighted = recall_score(
    val_labels,
    val_predictions,
    average="weighted",
    zero_division=0,
)


f1_weighted = f1_score(
    val_labels,
    val_predictions,
    average="weighted",
    zero_division=0,
)


# ============================================================
# SUMMARY METRICS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALIDATION RESULTS"
)

print(
    "=" * 70
)


print(
    f"\nAccuracy           : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Macro Precision    : "
    f"{precision_macro * 100:.2f}%"
)


print(
    f"Macro Recall       : "
    f"{recall_macro * 100:.2f}%"
)


print(
    f"Macro F1           : "
    f"{f1_macro * 100:.2f}%"
)


print(
    f"\nWeighted Precision : "
    f"{precision_weighted * 100:.2f}%"
)


print(
    f"Weighted Recall    : "
    f"{recall_weighted * 100:.2f}%"
)


print(
    f"Weighted F1        : "
    f"{f1_weighted * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 70
)


print(

    classification_report(

        val_labels,

        val_predictions,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    val_labels,
    val_predictions,
)


print(
    "=" * 70
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 70
)


print(
    "\nRows    = TRUE class"
)

print(
    "Columns = PREDICTED class\n"
)


print(
    "Class order:"
)

for index, name in enumerate(
    CLASS_NAMES
):

    print(
        f"    {index} = {name}"
    )


print(
    "\n",
    cm
)


# ============================================================
# VALIDATION PROBABILITY CHECK
# ============================================================

val_probabilities = svm.predict_proba(
    val_features
)


assert val_probabilities.shape == (
    len(val_features),
    4,
)


assert np.isfinite(
    val_probabilities
).all()


print(
    "\n✓ Probability output verified:"
)

print(
    val_probabilities.shape
)


print(
    "\nFirst validation sample probabilities:"
)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    print(

        f"    {class_name:12s}: "
        f"{val_probabilities[0, class_index]:.6f}"

    )


# ============================================================
# SAVE TRAINED SVM
# ============================================================

joblib.dump(
    svm,
    SVM_PATH,
)


print(
    "\n"
    + "=" * 70
)

print(
    "SVM MODEL SAVED"
)

print(
    "=" * 70
)


print(
    "\n",
    SVM_PATH
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESNET152 + LSTM + PCA + SVM VALIDATION COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nIMPORTANT:"
)

print(
    "The official TEST set has NOT been used "
    "to train or evaluate the SVM yet."
)


print(
    "\nNEXT STEP:"
)

print(
    "Inspect the validation performance. "
    "If the pipeline is satisfactory, perform the "
    "final evaluation on the untouched TEST set."
)