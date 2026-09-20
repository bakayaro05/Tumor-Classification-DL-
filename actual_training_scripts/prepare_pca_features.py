from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

FEATURE_DIR = (
    PROJECT_ROOT
    / "features"
)

PCA_DIR = (
    PROJECT_ROOT
    / "pca_features"
)

PCA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_PCA_COMPONENTS = 64


# ============================================================
# LOAD SAVED DEEP FEATURES
# ============================================================

print("=" * 70)
print("STANDARD SCALER + PCA PREPARATION")
print("=" * 70)


train_features = np.load(
    FEATURE_DIR
    / "train_features.npy"
)

train_labels = np.load(
    FEATURE_DIR
    / "train_labels.npy"
)


val_features = np.load(
    FEATURE_DIR
    / "val_features.npy"
)

val_labels = np.load(
    FEATURE_DIR
    / "val_labels.npy"
)


test_features = np.load(
    FEATURE_DIR
    / "test_features.npy"
)

test_labels = np.load(
    FEATURE_DIR
    / "test_labels.npy"
)


print("\nOriginal feature shapes:")

print(
    "Train:",
    train_features.shape
)

print(
    "Val  :",
    val_features.shape
)

print(
    "Test :",
    test_features.shape
)


# ============================================================
# VERIFY INPUT FEATURES
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


assert np.isfinite(
    train_features
).all()

assert np.isfinite(
    val_features
).all()

assert np.isfinite(
    test_features
).all()


print(
    "\n✓ Input deep features verified"
)


# ============================================================
# STANDARD SCALER
# ============================================================

print(
    "\nFitting StandardScaler "
    "on TRAINING features only..."
)


scaler = StandardScaler()


# IMPORTANT:
# Fit ONLY on training data.

train_scaled = scaler.fit_transform(
    train_features
)


# Validation and test only use the
# statistics learned from training.

val_scaled = scaler.transform(
    val_features
)


test_scaled = scaler.transform(
    test_features
)


print(
    "✓ StandardScaler fitted successfully"
)


print(
    "\nScaled training feature mean "
    "(approx):",
    float(
        np.mean(train_scaled)
    )
)


print(
    "Scaled training feature std "
    "(approx):",
    float(
        np.std(train_scaled)
    )
)


# ============================================================
# DETERMINE PCA COMPONENT COUNT
# ============================================================

n_components = min(
    MAX_PCA_COMPONENTS,
    train_scaled.shape[1],
    train_scaled.shape[0],
)


print(
    "\nMaximum requested PCA components:",
    MAX_PCA_COMPONENTS
)

print(
    "PCA components being used:",
    n_components
)


# ============================================================
# PCA
# ============================================================

print(
    "\nFitting PCA on TRAINING features only..."
)


pca = PCA(
    n_components=n_components,
    random_state=42,
)


# Fit only on train.

train_pca = pca.fit_transform(
    train_scaled
)


# Apply same learned PCA transformation.

val_pca = pca.transform(
    val_scaled
)


test_pca = pca.transform(
    test_scaled
)


print(
    "✓ PCA fitted successfully"
)


# ============================================================
# PCA RESULTS
# ============================================================

print(
    "\nPCA feature shapes:"
)


print(
    "Train:",
    train_pca.shape
)

print(
    "Val  :",
    val_pca.shape
)

print(
    "Test :",
    test_pca.shape
)


explained_variance = (
    pca.explained_variance_ratio_
)


cumulative_variance = (
    np.cumsum(
        explained_variance
    )
)


print(
    "\nExplained variance:"
)


print(
    "First component:",
    explained_variance[0]
)


print(
    "First 10 components:",
    cumulative_variance[
        min(
            9,
            len(cumulative_variance) - 1
        )
    ]
)


print(
    "First 32 components:",
    cumulative_variance[
        min(
            31,
            len(cumulative_variance) - 1
        )
    ]
)


print(
    "All selected PCA components:",
    cumulative_variance[-1]
)


# ============================================================
# VERIFY PCA OUTPUT
# ============================================================

assert train_pca.shape == (
    4569,
    n_components,
)

assert val_pca.shape == (
    1143,
    n_components,
)

assert test_pca.shape == (
    1311,
    n_components,
)


assert np.isfinite(
    train_pca
).all()

assert np.isfinite(
    val_pca
).all()

assert np.isfinite(
    test_pca
).all()


print(
    "\n✓ PCA outputs verified"
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
    / "train_labels.npy",
    train_labels,
)


np.save(
    PCA_DIR
    / "val_pca.npy",
    val_pca,
)

np.save(
    PCA_DIR
    / "val_labels.npy",
    val_labels,
)


np.save(
    PCA_DIR
    / "test_pca.npy",
    test_pca,
)

np.save(
    PCA_DIR
    / "test_labels.npy",
    test_labels,
)


# ============================================================
# SAVE SCALER + PCA OBJECTS
# ============================================================

joblib.dump(
    scaler,
    PCA_DIR
    / "standard_scaler.joblib",
)


joblib.dump(
    pca,
    PCA_DIR
    / "pca_model.joblib",
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STANDARDIZATION + PCA COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nSaved:"
)


print(
    PCA_DIR
    / "train_pca.npy"
)

print(
    PCA_DIR
    / "val_pca.npy"
)

print(
    PCA_DIR
    / "test_pca.npy"
)

print(
    PCA_DIR
    / "standard_scaler.joblib"
)

print(
    PCA_DIR
    / "pca_model.joblib"
)


print(
    "\nNEXT STEP:"
)

print(
    "Train the Linear SVM using "
    "TRAIN PCA features and evaluate "
    "first on the VALIDATION set."
)