from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

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
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model_cnn_lstm import CNNLSTM, SUPPORTED_BACKBONES
from src.data_pipeline_torch import (
    TRAIN_DIR,
    TEST_DIR,
    eval_transform,
    load_saved_indices,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate the already-trained CNN-LSTM neural classifier head "
        "for one backbone on deterministic train/validation/test splits. "
        "This script does NOT train or modify the model."
    )
)

parser.add_argument(
    "--backbone",
    required=True,
    choices=SUPPORTED_BACKBONES,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help=(
        "Evaluation batch size. If omitted, the batch size saved in the "
        "checkpoint is used so post-training evaluation matches the "
        "original validation protocol as closely as possible."
    ),
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--no-save-predictions",
    action="store_true",
    help="Do not save labels/predictions/softmax scores to NPZ.",
)

args = parser.parse_args()

BACKBONE = args.backbone
REQUESTED_BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
SAVE_PREDICTIONS = not args.no_save_predictions

EXPECTED_SPLIT_SIZES = {
    "train": 4569,
    "validation": 1143,
    "test": 1311,
}


# ============================================================
# PATHS
# ============================================================

EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / BACKBONE
MODEL_DIR = EXPERIMENT_ROOT / "models"
RESULT_DIR = EXPERIMENT_ROOT / "results"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
RESULT_PATH = RESULT_DIR / "neural_head_results.json"
PREDICTION_PATH = RESULT_DIR / "neural_head_predictions.npz"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print(f"NEURAL-HEAD EVALUATION: {BACKBONE.upper()}")
print("=" * 80)
print("\nDevice:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Best checkpoint not found:\n{CHECKPOINT_PATH}\n"
        "This evaluator only uses an existing trained checkpoint."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

saved_backbone = checkpoint.get("backbone", BACKBONE)

if saved_backbone != BACKBONE:
    raise ValueError(
        f"Checkpoint backbone mismatch: checkpoint={saved_backbone!r}, "
        f"requested={BACKBONE!r}"
    )

required_checkpoint_keys = (
    "model_state_dict",
    "epoch",
    "class_to_idx",
)

missing_checkpoint_keys = [
    key for key in required_checkpoint_keys if key not in checkpoint
]

if missing_checkpoint_keys:
    raise KeyError(
        "Checkpoint is missing required keys: "
        + ", ".join(missing_checkpoint_keys)
    )

print("\nCheckpoint:")
print("  Path                 :", CHECKPOINT_PATH)
print("  Best epoch           :", checkpoint.get("epoch"))
print("  Saved train accuracy :", checkpoint.get("train_accuracy"))
print("  Saved val accuracy   :", checkpoint.get("val_accuracy"))

saved_batch_size = int(checkpoint.get("batch_size", 8))
BATCH_SIZE = (
    saved_batch_size
    if REQUESTED_BATCH_SIZE is None
    else int(REQUESTED_BATCH_SIZE)
)

print("  Saved batch size     :", saved_batch_size)
print("  Evaluation batch size:", BATCH_SIZE)

if BATCH_SIZE != saved_batch_size:
    print(
        "  NOTE: evaluation batch size differs from the original "
        "checkpoint batch size; tiny floating-point differences can "
        "occasionally change borderline predictions."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

checkpoint_class_to_idx = {
    str(name): int(index)
    for name, index in checkpoint["class_to_idx"].items()
}

indices = sorted(checkpoint_class_to_idx.values())

if indices != list(range(len(indices))):
    raise ValueError(
        "Checkpoint class indices are not contiguous from 0. "
        f"Found: {indices}"
    )

class_names = [
    name
    for name, index in sorted(
        checkpoint_class_to_idx.items(),
        key=lambda item: item[1],
    )
]

NUM_CLASSES = len(class_names)

if NUM_CLASSES != 4:
    raise ValueError(
        f"Expected a 4-class checkpoint, found {NUM_CLASSES}: {class_names}"
    )

print("\nCheckpoint class mapping:")
for name, index in sorted(
    checkpoint_class_to_idx.items(),
    key=lambda item: item[1],
):
    print(f"  {index} -> {name}")


# ============================================================
# BUILD MODEL
# ============================================================

model = CNNLSTM(
    backbone_name=BACKBONE,
    lstm_hidden_size=int(checkpoint.get("lstm_hidden_size", 256)),
    dense_features=int(checkpoint.get("dense_features", 128)),
    num_classes=NUM_CLASSES,
    dropout=float(checkpoint.get("dropout", 0.3)),
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("\n✓ Best checkpoint weights loaded")
print("✓ Model set to evaluation mode")
print("✓ No training or parameter updates will occur")


# ============================================================
# REBUILD EXACT SAVED SPLITS WITH DETERMINISTIC TRANSFORM
# ============================================================

train_indices, val_indices = load_saved_indices()

if hasattr(train_indices, "tolist"):
    train_indices = train_indices.tolist()

if hasattr(val_indices, "tolist"):
    val_indices = val_indices.tolist()

train_eval_full = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transform,
)

if train_eval_full.class_to_idx != checkpoint_class_to_idx:
    raise ValueError(
        "Training dataset class mapping does not match checkpoint mapping.\n"
        f"Dataset   : {train_eval_full.class_to_idx}\n"
        f"Checkpoint: {checkpoint_class_to_idx}"
    )

if test_dataset.class_to_idx != checkpoint_class_to_idx:
    raise ValueError(
        "Test dataset class mapping does not match checkpoint mapping.\n"
        f"Test      : {test_dataset.class_to_idx}\n"
        f"Checkpoint: {checkpoint_class_to_idx}"
    )

train_dataset = Subset(train_eval_full, train_indices)
val_dataset = Subset(train_eval_full, val_indices)

actual_sizes = {
    "train": len(train_dataset),
    "validation": len(val_dataset),
    "test": len(test_dataset),
}

print("\nDeterministic evaluation dataset sizes:")
for split_name, split_size in actual_sizes.items():
    print(f"  {split_name:10s}: {split_size}")

if actual_sizes != EXPECTED_SPLIT_SIZES:
    raise ValueError(
        "Dataset sizes do not match the locked project split.\n"
        f"Expected: {EXPECTED_SPLIT_SIZES}\n"
        f"Actual  : {actual_sizes}"
    )


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = torch.cuda.is_available()

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "num_workers": NUM_WORKERS,
    "pin_memory": pin_memory,
}

train_loader = DataLoader(train_dataset, **loader_kwargs)
val_loader = DataLoader(val_dataset, **loader_kwargs)
test_loader = DataLoader(test_dataset, **loader_kwargs)


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def evaluate_split(model, loader, device, split_name):
    """
    Deterministic post-training evaluation.

    Important:
    - model.eval() is active.
    - no gradients are created.
    - softmax is used only to express the already-trained logits
      as four class scores for inference/output.
    """
    model.eval()

    # Sum reduction lets us compute an exact average loss over every sample,
    # including the final incomplete evaluation batch.
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    label_batches = []
    prediction_batches = []
    probability_batches = []

    start_time = time.time()

    print(f"\nEvaluating {split_name.upper()}...")

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(
                device,
                non_blocking=True,
            )
            labels_device = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
                raise RuntimeError(
                    f"Unexpected logits shape {tuple(logits.shape)}; "
                    f"expected [batch, {NUM_CLASSES}]"
                )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in {split_name}."
                )

            loss = loss_function(logits, labels_device)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            total_loss += float(loss.item())

            label_batches.append(labels.numpy())
            prediction_batches.append(
                predictions.detach().cpu().numpy()
            )
            probability_batches.append(
                probabilities.detach().cpu().numpy()
            )

            if (
                (batch_index + 1) % 25 == 0
                or (batch_index + 1) == len(loader)
            ):
                print(
                    f"  Batch {batch_index + 1:4d}/{len(loader):4d}"
                )

    labels = np.concatenate(label_batches, axis=0)
    predictions = np.concatenate(prediction_batches, axis=0)
    probabilities = np.concatenate(probability_batches, axis=0)

    if probabilities.shape != (len(labels), NUM_CLASSES):
        raise RuntimeError(
            f"Unexpected softmax array shape: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"Non-finite softmax scores detected in {split_name}."
        )

    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError(
            f"Softmax rows do not sum to 1 in {split_name}."
        )

    metrics = calculate_metrics(labels, predictions)

    confusion = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        digits=6,
        zero_division=0,
    )

    sample_count = int(len(labels))
    correct_count = int(np.sum(labels == predictions))
    incorrect_count = int(sample_count - correct_count)
    average_loss = float(total_loss / sample_count)
    elapsed_seconds = float(time.time() - start_time)

    result = {
        "sample_count": sample_count,
        "cross_entropy_loss": average_loss,
        **metrics,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "evaluation_time_seconds": elapsed_seconds,
    }

    print(f"\n{split_name.upper()} RESULTS")
    print("-" * 80)
    print(f"Cross-entropy loss : {average_loss:.6f}")
    print(f"Accuracy           : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro precision    : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro recall       : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1           : {metrics['macro_f1'] * 100:.2f}%")
    print(
        f"Correct / incorrect: "
        f"{correct_count} / {incorrect_count}"
    )
    print("Confusion matrix:")
    print(confusion)

    return result, labels, predictions, probabilities


# ============================================================
# EVALUATE ALL THREE SPLITS
# ============================================================

overall_start = time.time()

train_result, train_labels, train_predictions, train_probabilities = (
    evaluate_split(
        model,
        train_loader,
        device,
        "train",
    )
)

validation_result, val_labels, val_predictions, val_probabilities = (
    evaluate_split(
        model,
        val_loader,
        device,
        "validation",
    )
)

test_result, test_labels, test_predictions, test_probabilities = (
    evaluate_split(
        model,
        test_loader,
        device,
        "test",
    )
)

overall_seconds = float(time.time() - overall_start)


# ============================================================
# SAVE STRUCTURED RESULTS
# ============================================================

results = {
    "evaluation_type": "cnn_lstm_neural_head_post_training",
    "backbone": BACKBONE,
    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
    "checkpoint": {
        "best_epoch": int(checkpoint["epoch"]),
        "training_time_train_accuracy_percent": (
            None
            if checkpoint.get("train_accuracy") is None
            else float(checkpoint["train_accuracy"])
        ),
        "training_time_train_loss": (
            None
            if checkpoint.get("train_loss") is None
            else float(checkpoint["train_loss"])
        ),
        "validation_selection_accuracy_percent": (
            None
            if checkpoint.get("val_accuracy") is None
            else float(checkpoint["val_accuracy"])
        ),
        "validation_selection_loss": (
            None
            if checkpoint.get("val_loss") is None
            else float(checkpoint["val_loss"])
        ),
        "lstm_hidden_size": int(
            checkpoint.get("lstm_hidden_size", 256)
        ),
        "dense_features": int(
            checkpoint.get("dense_features", 128)
        ),
        "dropout": float(
            checkpoint.get("dropout", 0.3)
        ),
        "learning_rate": (
            None
            if checkpoint.get("learning_rate") is None
            else float(checkpoint["learning_rate"])
        ),
        "training_batch_size": (
            None
            if checkpoint.get("batch_size") is None
            else int(checkpoint["batch_size"])
        ),
        "seed": (
            None
            if checkpoint.get("seed") is None
            else int(checkpoint["seed"])
        ),
    },
    "class_to_idx": checkpoint_class_to_idx,
    "class_names_in_index_order": class_names,
    "dataset": {
        "train_samples": actual_sizes["train"],
        "validation_samples": actual_sizes["validation"],
        "test_samples": actual_sizes["test"],
        "evaluation_transform": "eval_transform",
        "training_augmentation_used_for_this_evaluation": False,
        "test_used_for_checkpoint_selection": False,
    },
    "method": {
        "classifier": "saved 4-class neural linear head",
        "prediction_rule": "argmax(logits)",
        "softmax_usage": (
            "post-training inference scores only; "
            "training used CrossEntropyLoss on raw logits"
        ),
        "model_mode": "eval",
        "gradients_enabled": False,
    },
    "train": train_result,
    "validation": validation_result,
    "test": test_result,
    "total_evaluation_time_seconds": overall_seconds,
}

with RESULT_PATH.open("w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=4)

print("\n✓ Neural-head results saved:")
print(RESULT_PATH)


# ============================================================
# SAVE PAIRED-PREDICTION MATERIAL FOR LATER COMPARISON
# ============================================================

if SAVE_PREDICTIONS:
    np.savez_compressed(
        PREDICTION_PATH,
        class_names=np.asarray(class_names, dtype=str),
        train_labels=train_labels.astype(np.int64),
        train_predictions=train_predictions.astype(np.int64),
        train_softmax=train_probabilities.astype(np.float32),
        validation_labels=val_labels.astype(np.int64),
        validation_predictions=val_predictions.astype(np.int64),
        validation_softmax=val_probabilities.astype(np.float32),
        test_labels=test_labels.astype(np.int64),
        test_predictions=test_predictions.astype(np.int64),
        test_softmax=test_probabilities.astype(np.float32),
    )

    print("✓ Neural-head labels/predictions/softmax saved:")
    print(PREDICTION_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("NEURAL-HEAD EVALUATION COMPLETE")
print("=" * 80)
print("Backbone             :", BACKBONE)
print("Best checkpoint epoch:", checkpoint["epoch"])
print(
    f"Train accuracy       : "
    f"{train_result['accuracy'] * 100:.2f}%"
)
print(
    f"Validation accuracy  : "
    f"{validation_result['accuracy'] * 100:.2f}%"
)
print(
    f"Test accuracy        : "
    f"{test_result['accuracy'] * 100:.2f}%"
)
print(
    f"Test macro F1        : "
    f"{test_result['macro_f1'] * 100:.2f}%"
)
print("\nNo checkpoint was modified or retrained.")
