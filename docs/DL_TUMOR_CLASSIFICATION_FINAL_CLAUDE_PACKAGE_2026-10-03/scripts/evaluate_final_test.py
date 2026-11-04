from pathlib import Path
import sys

import joblib
import numpy as np

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

SVM_PATH = (
    MODEL_DIR
    / "resnet152_lstm_pca_svm.joblib"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# ============================================================
# LOAD FINAL TEST DATA
# ============================================================

print("=" * 70)
print("FINAL UNTOUCHED TEST EVALUATION")
print("=" * 70)


test_features = np.load(
    PCA_DIR
    / "test_pca.npy"
)

test_labels = np.load(
    PCA_DIR
    / "test_labels.npy"
)


print("\nTest features:")
print(test_features.shape)

print("\nTest labels:")
print(test_labels.shape)


# ============================================================
# VERIFY DATA
# ============================================================

assert test_features.shape == (
    1311,
    64,
)

assert test_labels.shape == (
    1311,
)

assert np.isfinite(
    test_features
).all()

assert test_labels.min() >= 0
assert test_labels.max() <= 3


print(
    "\n✓ Test feature arrays verified"
)


# ============================================================
# LOAD TRAINED SVM
# ============================================================

if not SVM_PATH.exists():

    raise FileNotFoundError(
        f"SVM model not found: {SVM_PATH}"
    )


svm = joblib.load(
    SVM_PATH
)


print(
    "\n✓ Trained SVM loaded"
)

print(
    SVM_PATH
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating final TEST predictions..."
)


test_predictions = svm.predict(
    test_features
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions,
)

precision_macro = precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

precision_weighted = precision_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)

f1_weighted = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL TEST RESULTS"
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
        test_labels,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
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
    "Columns = PREDICTED class"
)


print(
    "\nClass order:"
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
# ERROR COUNT
# ============================================================

correct = np.sum(
    test_predictions == test_labels
)

incorrect = len(
    test_labels
) - correct


print(
    "\nCorrect predictions  :",
    int(correct)
)

print(
    "Incorrect predictions:",
    int(incorrect)
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL HYBRID MODEL EVALUATION COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nPipeline:"
)

print(
    "MRI -> ResNet152 -> LSTM -> Dense -> "
    "StandardScaler -> PCA(64) -> Linear SVM"
)


print(
    "\nThis result is from the previously untouched "
    "official TEST set."
)