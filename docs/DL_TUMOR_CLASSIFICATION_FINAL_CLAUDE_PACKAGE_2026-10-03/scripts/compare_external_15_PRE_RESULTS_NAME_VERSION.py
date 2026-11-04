#!/usr/bin/env python3
"""
Freeze and externally validate the 15 pre-specified hybrid pipelines:

  Phase 1  : 5 early-stopping pipelines (one per backbone)
  Phase 2A : 5 standardized 30-epoch pipelines (best internal full-hybrid state per backbone)
  Phase 2B : 5 tuned pipelines (best internal full-hybrid state across 7 configs x 2 retained states)

Selection is performed ONLY from INTERNAL validation metrics.
External labels are never used for model selection.

Recommended workflow:
  1) python actual_training_scripts/compare_external_15.py --freeze-only
  2) Audit an independent external dataset for provenance / overlap / class mapping.
  3) python actual_training_scripts/compare_external_15.py \
         --external-dir /path/to/curated_external_root \
         --confirm-audited

External dataset root must contain exactly:
  glioma/
  meningioma/
  notumor/
  pituitary/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_cnn_lstm import CNNLSTM
from src.data_pipeline_torch import eval_transform


# ============================================================
# CONSTANTS
# ============================================================

BACKBONES = [
    "vgg16",
    "vgg19",
    "resnet50",
    "resnet152",
    "efficientnet_b3",
]

RUN_NAMES = [
    "lr_1e-5",
    "lr_5e-5",
    "lr_5e-4",
    "lstm_128",
    "lstm_512",
    "dropout_0.2",
    "dropout_0.5",
]

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

EXPECTED_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}

EXTERNAL_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "external_validation"
)

MANIFEST_PATH = (
    EXTERNAL_ROOT
    / "frozen_15_manifest.json"
)

SELECTION_CSV_PATH = (
    EXTERNAL_ROOT
    / "frozen_15_selection.csv"
)

RESULTS_ROOT = (
    EXTERNAL_ROOT
    / "comparison_15_models"
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--freeze-only",
    action="store_true",
    help=(
        "Select the 15 pipelines using INTERNAL validation only, "
        "write a frozen SHA-256 manifest, and stop."
    ),
)

parser.add_argument(
    "--external-dir",
    type=Path,
    default=None,
    help=(
        "Curated independent external dataset root containing "
        "glioma/, meningioma/, notumor/, pituitary/."
    ),
)

parser.add_argument(
    "--confirm-audited",
    action="store_true",
    help=(
        "Required for external evaluation. Confirms provenance, "
        "class mapping and source-overlap/duplicate checks were completed."
    ),
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
    "--rerun",
    action="store_true",
    help="Re-run model evaluations even when matching result files already exist.",
)

args = parser.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def absolute_from_manifest(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def safe_float(value, default=None):
    if value is None:
        return default
    return float(value)


def selection_key(candidate: dict) -> tuple:
    """
    Higher is better.

    Primary criterion  : full-hybrid validation accuracy
    Secondary criterion: full-hybrid macro-F1
    Tertiary criterion : prefer best-within-30 on an exact numerical tie
                         (keeps tie handling deterministic)
    """
    checkpoint_preference = (
        1
        if candidate.get("checkpoint_state") == "best_val_within_30"
        else 0
    )

    return (
        float(candidate["internal_validation_accuracy"]),
        float(candidate["internal_validation_macro_f1"]),
        checkpoint_preference,
    )


def require_files(paths: dict[str, Path], label: str) -> None:
    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"\nMissing artifact(s) for {label}:\n"
            + "\n".join(missing)
        )


# ============================================================
# PIPELINE PATH BUILDERS
# ============================================================

def phase1_candidate(backbone: str) -> dict:
    """
    Historical early-stopping pipeline.
    One already-selected pipeline per backbone.
    """

    root = PROJECT_ROOT / "experiments" / backbone

    paths = {
        "checkpoint": root / "models" / "best_model.pth",
        "scaler": root / "pca" / "standard_scaler.joblib",
        "pca": root / "pca" / "pca_model.joblib",
        "svm": root / "models" / "svm.joblib",
        "internal_results": root / "results" / "final_results.json",
    }

    require_files(paths, f"Phase 1 / {backbone}")

    result = load_json(paths["internal_results"])

    val = result["validation"]
    neural = result.get("neural_network", {})

    return {
        "stage": "phase1_early_stopping",
        "backbone": backbone,
        "variant": "early_stopping",
        "checkpoint_state": "best_model",
        "selected_epoch": neural.get("best_epoch"),
        "internal_validation_accuracy": float(val["accuracy"]),
        "internal_validation_macro_f1": float(val["macro_f1"]),
        "paths": paths,
    }


def phase2a_state_candidate(backbone: str, state: str) -> dict:
    """
    Phase 2A standardized 30-epoch pipeline.
    Compare the two retained states using full-hybrid validation metrics.
    """

    root = (
        PROJECT_ROOT
        / "phase2"
        / "experiments_30epoch"
        / backbone
    )

    if state == "best_val_within_30":
        checkpoint_file = "best_val_within_30.pth"
        evaluation_name = "best_val_within_30"
    elif state == "epoch_30":
        checkpoint_file = "epoch_30_model.pth"
        evaluation_name = "epoch_30"
    else:
        raise ValueError(state)

    eval_root = root / "evaluations" / evaluation_name

    paths = {
        "checkpoint": root / "models" / checkpoint_file,
        "scaler": eval_root / "pca" / "standard_scaler.joblib",
        "pca": eval_root / "pca" / "pca_model.joblib",
        "svm": eval_root / "models" / "svm.joblib",
        "internal_results": eval_root / "results" / "final_results.json",
    }

    require_files(paths, f"Phase 2A / {backbone} / {state}")

    result = load_json(paths["internal_results"])
    val = result["validation"]
    neural = result.get("neural_network", {})

    return {
        "stage": "phase2a_standard_30epoch",
        "backbone": backbone,
        "variant": "lr1e-4_lstm256_dropout0.3",
        "checkpoint_state": state,
        "selected_epoch": neural.get("best_epoch"),
        "internal_validation_accuracy": float(val["accuracy"]),
        "internal_validation_macro_f1": float(val["macro_f1"]),
        "paths": paths,
    }


def phase2b_state_candidate(
    backbone: str,
    run_name: str,
    state: str,
) -> dict:
    """
    One Phase 2B changed-hyperparameter pipeline at one retained state.
    """

    root = (
        PROJECT_ROOT
        / "phase2"
        / "hyperparameter_dl"
        / backbone
        / run_name
    )

    if state == "best_val_within_30":
        checkpoint_file = "best_val_within_30.pth"
        evaluation_name = "best_val_within_30"
    elif state == "epoch_30":
        checkpoint_file = "epoch_30_model.pth"
        evaluation_name = "epoch_30"
    else:
        raise ValueError(state)

    eval_root = root / "evaluations" / evaluation_name

    paths = {
        "checkpoint": root / "models" / checkpoint_file,
        "scaler": eval_root / "pca" / "standard_scaler.joblib",
        "pca": eval_root / "pca" / "pca_model.joblib",
        "svm": eval_root / "models" / "svm.joblib",
        "internal_results": eval_root / "results" / "final_results.json",
    }

    require_files(
        paths,
        f"Phase 2B / {backbone} / {run_name} / {state}",
    )

    result = load_json(paths["internal_results"])
    val = result["validation"]

    return {
        "stage": "phase2b_tuned",
        "backbone": backbone,
        "variant": run_name,
        "checkpoint_state": state,
        "selected_epoch": result.get("checkpoint_epoch"),
        "hyperparameters": result.get("hyperparameters", {}),
        "internal_validation_accuracy": float(val["accuracy"]),
        "internal_validation_macro_f1": float(val["macro_f1"]),
        "paths": paths,
    }


# ============================================================
# INTERNAL SELECTION -> EXACTLY 15 PIPELINES
# ============================================================

def select_15_pipelines() -> tuple[list[dict], dict]:
    selected = []
    selection_details = {
        "selection_rule": (
            "Use INTERNAL full-hybrid validation accuracy; "
            "break ties by macro-F1; on an exact tie prefer "
            "best_val_within_30 over epoch_30."
        ),
        "phase2a_candidates_per_backbone": 2,
        "phase2b_candidates_per_backbone": 14,
        "selected": {},
    }

    for backbone in BACKBONES:

        # ----------------------------------------------------
        # PHASE 1: already fixed historical early-stop model
        # ----------------------------------------------------
        p1 = phase1_candidate(backbone)
        selected.append(p1)

        # ----------------------------------------------------
        # PHASE 2A:
        # Compare best-within-30 vs literal epoch-30
        # using full PCA+SVM INTERNAL validation.
        # ----------------------------------------------------
        p2a_candidates = [
            phase2a_state_candidate(
                backbone,
                "best_val_within_30",
            ),
            phase2a_state_candidate(
                backbone,
                "epoch_30",
            ),
        ]

        p2a = max(
            p2a_candidates,
            key=selection_key,
        )

        selected.append(p2a)

        # ----------------------------------------------------
        # PHASE 2B:
        # 7 changed configurations x 2 retained states = 14
        # candidates per backbone.
        # ----------------------------------------------------
        p2b_candidates = []

        for run_name in RUN_NAMES:
            for state in [
                "best_val_within_30",
                "epoch_30",
            ]:
                p2b_candidates.append(
                    phase2b_state_candidate(
                        backbone,
                        run_name,
                        state,
                    )
                )

        p2b = max(
            p2b_candidates,
            key=selection_key,
        )

        selected.append(p2b)

        selection_details["selected"][backbone] = {
            "phase1": {
                "variant": p1["variant"],
                "checkpoint_state": p1["checkpoint_state"],
                "internal_validation_accuracy": p1[
                    "internal_validation_accuracy"
                ],
                "internal_validation_macro_f1": p1[
                    "internal_validation_macro_f1"
                ],
            },
            "phase2a": {
                "variant": p2a["variant"],
                "checkpoint_state": p2a["checkpoint_state"],
                "selected_epoch": p2a["selected_epoch"],
                "internal_validation_accuracy": p2a[
                    "internal_validation_accuracy"
                ],
                "internal_validation_macro_f1": p2a[
                    "internal_validation_macro_f1"
                ],
            },
            "phase2b": {
                "variant": p2b["variant"],
                "checkpoint_state": p2b["checkpoint_state"],
                "selected_epoch": p2b["selected_epoch"],
                "hyperparameters": p2b.get("hyperparameters", {}),
                "internal_validation_accuracy": p2b[
                    "internal_validation_accuracy"
                ],
                "internal_validation_macro_f1": p2b[
                    "internal_validation_macro_f1"
                ],
            },
        }

    assert len(selected) == 15

    return selected, selection_details


# ============================================================
# FREEZE MANIFEST
# ============================================================

def freeze_manifest() -> dict:
    EXTERNAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MANIFEST_PATH.exists():
        print(
            f"\nFrozen manifest already exists:\n{MANIFEST_PATH}\n"
            "It will NOT be overwritten."
        )
        return load_json(MANIFEST_PATH)

    print("=" * 80)
    print("SELECTING 15 PIPELINES USING INTERNAL VALIDATION ONLY")
    print("=" * 80)

    selected, details = select_15_pipelines()

    manifest_candidates = []

    print()
    print(
        f"{'BACKBONE':<18}"
        f"{'STAGE':<28}"
        f"{'VARIANT':<18}"
        f"{'STATE':<22}"
        f"{'VAL ACC':>10}"
        f"{'MACRO F1':>11}"
    )

    print("-" * 107)

    for candidate in selected:

        # Convert paths to project-relative paths and hash exact artifacts.
        path_strings = {
            name: relative_to_project(path)
            for name, path in candidate["paths"].items()
        }

        hashes = {}

        print(
            f"Hashing: "
            f"{candidate['stage']} / "
            f"{candidate['backbone']} / "
            f"{candidate['variant']}"
        )

        for name, path in candidate["paths"].items():
            hashes[name] = sha256_file(path)

        frozen = {
            key: value
            for key, value in candidate.items()
            if key != "paths"
        }

        frozen["paths"] = path_strings
        frozen["sha256"] = hashes

        manifest_candidates.append(frozen)

        print(
            f"{candidate['backbone']:<18}"
            f"{candidate['stage']:<28}"
            f"{candidate['variant']:<18}"
            f"{candidate['checkpoint_state']:<22}"
            f"{candidate['internal_validation_accuracy']*100:>9.3f}%"
            f"{candidate['internal_validation_macro_f1']*100:>10.3f}%"
        )

    manifest = {
        "purpose": (
            "Frozen 15-model architecture-wise external-validation cohort. "
            "All selection was completed using internal validation only."
        ),
        "selection_details": details,
        "class_names": CLASS_NAMES,
        "class_to_idx": EXPECTED_CLASS_TO_IDX,
        "candidate_count": len(manifest_candidates),
        "candidates": manifest_candidates,
    }

    # Write manifest ONCE.
    save_json(
        MANIFEST_PATH,
        manifest,
    )

    # Human-readable selection CSV.
    with SELECTION_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fieldnames = [
            "stage",
            "backbone",
            "variant",
            "checkpoint_state",
            "selected_epoch",
            "internal_validation_accuracy",
            "internal_validation_macro_f1",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for candidate in manifest_candidates:
            writer.writerow({
                name: candidate.get(name)
                for name in fieldnames
            })

    print()
    print("=" * 80)
    print("FROZEN")
    print("=" * 80)
    print("Manifest :", MANIFEST_PATH)
    print("CSV      :", SELECTION_CSV_PATH)
    print()
    print(
        "Do not replace/reselect these models after examining "
        "external results."
    )

    return manifest


def verify_frozen_manifest(manifest: dict) -> None:
    print("\nVerifying frozen model hashes...")

    if manifest.get("candidate_count") != 15:
        raise ValueError(
            "Frozen manifest does not contain exactly 15 candidates."
        )

    for candidate in manifest["candidates"]:

        label = (
            f"{candidate['stage']} / "
            f"{candidate['backbone']} / "
            f"{candidate['variant']}"
        )

        for name, rel_path in candidate["paths"].items():

            path = absolute_from_manifest(rel_path)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Frozen artifact is missing:\n"
                    f"{label}\n{name}: {path}"
                )

            actual = sha256_file(path)
            expected = candidate["sha256"][name]

            if actual != expected:
                raise RuntimeError(
                    "\nFROZEN ARTIFACT HASH MISMATCH\n"
                    f"Pipeline: {label}\n"
                    f"Artifact: {name}\n"
                    f"Path    : {path}\n"
                    "The artifact changed after freezing. "
                    "External evaluation has been stopped."
                )

    print("✓ All frozen hashes match.")


# ============================================================
# EXTERNAL DATASET PREFLIGHT
# ============================================================

def dataset_fingerprint(dataset: datasets.ImageFolder, root: Path) -> str:
    digest = hashlib.sha256()

    for path_string, label in dataset.samples:
        path = Path(path_string)
        relative = path.resolve().relative_to(root.resolve())
        size = path.stat().st_size

        digest.update(
            f"{relative}|{size}|{label}\n".encode("utf-8")
        )

    return digest.hexdigest()


def prepare_external_dataset(external_dir: Path):
    external_dir = external_dir.expanduser().resolve()

    if not external_dir.is_dir():
        raise NotADirectoryError(
            f"External dataset directory not found:\n"
            f"{external_dir}"
        )

    dataset = datasets.ImageFolder(
        root=external_dir,
        transform=eval_transform,
    )

    if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
        raise ValueError(
            "\nExternal class folders do not match the required mapping.\n"
            f"Expected: {EXPECTED_CLASS_TO_IDX}\n"
            f"Found   : {dataset.class_to_idx}\n\n"
            "Curate the external dataset into exactly:\n"
            "glioma/\nmeningioma/\nnotumor/\npituitary/"
        )

    counts = Counter(
        label
        for _, label in dataset.samples
    )

    class_counts = {
        CLASS_NAMES[index]: int(counts[index])
        for index in range(len(CLASS_NAMES))
    }

    if any(count == 0 for count in class_counts.values()):
        raise ValueError(
            f"At least one external class is empty: {class_counts}"
        )

    fingerprint = dataset_fingerprint(
        dataset,
        external_dir,
    )

    metadata = {
        "external_dir": str(external_dir),
        "class_to_idx": dataset.class_to_idx,
        "class_counts": class_counts,
        "total_images": len(dataset),
        "dataset_fingerprint_path_size_label_sha256": fingerprint,
    }

    return dataset, metadata


# ============================================================
# EXTERNAL INFERENCE
# ============================================================

def build_model_from_checkpoint(
    checkpoint_path: Path,
    expected_backbone: str,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_backbone = checkpoint.get(
        "backbone",
        expected_backbone,
    )

    if checkpoint_backbone != expected_backbone:
        raise ValueError(
            f"Checkpoint backbone mismatch: "
            f"expected={expected_backbone}, "
            f"found={checkpoint_backbone}"
        )

    class_to_idx = checkpoint.get("class_to_idx")

    if (
        class_to_idx is not None
        and class_to_idx != EXPECTED_CLASS_TO_IDX
    ):
        raise ValueError(
            "Checkpoint class mapping mismatch:\n"
            f"{class_to_idx}"
        )

    model = CNNLSTM(
        backbone_name=expected_backbone,
        lstm_hidden_size=int(
            checkpoint.get(
                "lstm_hidden_size",
                256,
            )
        ),
        dense_features=int(
            checkpoint.get(
                "dense_features",
                128,
            )
        ),
        num_classes=4,
        dropout=float(
            checkpoint.get(
                "dropout",
                0.3,
            )
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def extract_external_features(
    model,
    loader,
    device,
):
    batches = []
    labels = []

    with torch.no_grad():

        for batch_index, (
            images,
            targets,
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True,
            )

            features = model(
                images,
                return_features=True,
            )

            batches.append(
                features.detach().cpu().numpy()
            )

            labels.append(
                targets.numpy()
            )

            if (
                (batch_index + 1) % 25 == 0
                or
                (batch_index + 1) == len(loader)
            ):
                print(
                    f"    feature batch "
                    f"{batch_index + 1}/{len(loader)}"
                )

    features = np.concatenate(
        batches,
        axis=0,
    )

    labels = np.concatenate(
        labels,
        axis=0,
    )

    if features.ndim != 2:
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Non-finite external deep features detected."
        )

    return features, labels


def calculate_metrics(
    labels,
    predictions,
):
    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
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
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def candidate_id(candidate: dict) -> str:
    return "__".join([
        candidate["stage"],
        candidate["backbone"],
        candidate["variant"],
        candidate["checkpoint_state"],
    ])


def evaluate_candidate(
    candidate: dict,
    dataset: datasets.ImageFolder,
    dataset_metadata: dict,
    device: torch.device,
):
    cid = candidate_id(candidate)

    result_dir = (
        RESULTS_ROOT
        / candidate["stage"]
        / candidate["backbone"]
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        result_dir
        / f"{cid}.json"
    )

    prediction_path = (
        result_dir
        / f"{cid}_predictions.csv"
    )

    # Resume-safe reuse only if same external dataset fingerprint.
    if result_path.is_file() and not args.rerun:

        existing = load_json(result_path)

        if (
            existing.get(
                "external_dataset_fingerprint"
            )
            == dataset_metadata[
                "dataset_fingerprint_path_size_label_sha256"
            ]
        ):
            print("  SKIP: matching external result already exists.")
            return existing

        raise RuntimeError(
            f"Existing result belongs to a different external dataset:\n"
            f"{result_path}\n"
            "Use --rerun only after intentionally clearing/replacing "
            "the comparison results."
        )

    paths = {
        name: absolute_from_manifest(rel_path)
        for name, rel_path in candidate["paths"].items()
    }

    model, checkpoint = build_model_from_checkpoint(
        paths["checkpoint"],
        candidate["backbone"],
        device,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features, labels = extract_external_features(
        model,
        loader,
        device,
    )

    scaler = joblib.load(
        paths["scaler"]
    )

    pca = joblib.load(
        paths["pca"]
    )

    svm = joblib.load(
        paths["svm"]
    )

    # IMPORTANT: external data are transform-only.
    scaled = scaler.transform(
        features
    )

    reduced = pca.transform(
        scaled
    )

    predictions = svm.predict(
        reduced
    )

    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    if not np.array_equal(
        np.sort(np.unique(svm.classes_)),
        np.array([0, 1, 2, 3]),
    ):
        raise ValueError(
            f"Unexpected SVM classes: {svm.classes_}"
        )

    metrics = calculate_metrics(
        labels,
        predictions,
    )

    result = {
        "candidate_id": cid,
        "stage": candidate["stage"],
        "backbone": candidate["backbone"],
        "variant": candidate["variant"],
        "checkpoint_state": candidate["checkpoint_state"],
        "selected_epoch": candidate.get("selected_epoch"),
        "internal_validation_accuracy": candidate[
            "internal_validation_accuracy"
        ],
        "internal_validation_macro_f1": candidate[
            "internal_validation_macro_f1"
        ],
        "external_dataset_fingerprint": dataset_metadata[
            "dataset_fingerprint_path_size_label_sha256"
        ],
        "external_samples": int(len(labels)),
        "checkpoint_epoch_loaded": checkpoint.get("epoch"),
        "external": metrics,
    }

    save_json(
        result_path,
        result,
    )

    # Per-image audit trail.
    with prediction_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "relative_path",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "correct",
        ])

        external_root = Path(
            dataset_metadata["external_dir"]
        ).resolve()

        for (
            sample,
            true_label,
            pred_label,
        ) in zip(
            dataset.samples,
            labels,
            predictions,
        ):
            image_path = Path(sample[0]).resolve()
            relative = image_path.relative_to(
                external_root
            )

            writer.writerow([
                str(relative),
                int(true_label),
                CLASS_NAMES[int(true_label)],
                int(pred_label),
                CLASS_NAMES[int(pred_label)],
                int(true_label == pred_label),
            ])

    return result


# ============================================================
# MASTER COMPARISON OUTPUT
# ============================================================

def save_master_outputs(
    manifest: dict,
    dataset_metadata: dict,
    results: list[dict],
):
    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        RESULTS_ROOT
        / "external_dataset_manifest.json",
        dataset_metadata,
    )

    master_json = {
        "external_dataset": dataset_metadata,
        "selection_manifest": relative_to_project(
            MANIFEST_PATH
        ),
        "model_count": len(results),
        "results": results,
    }

    save_json(
        RESULTS_ROOT
        / "external_comparison_15.json",
        master_json,
    )

    csv_path = (
        RESULTS_ROOT
        / "external_comparison_15.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fieldnames = [
            "stage",
            "backbone",
            "variant",
            "checkpoint_state",
            "selected_epoch",
            "internal_val_accuracy_pct",
            "internal_val_macro_f1_pct",
            "external_accuracy_pct",
            "external_macro_precision_pct",
            "external_macro_recall_pct",
            "external_macro_f1_pct",
            "external_weighted_f1_pct",
            "external_minus_internal_accuracy_pp",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            external = result["external"]

            internal_acc = float(
                result["internal_validation_accuracy"]
            )

            external_acc = float(
                external["accuracy"]
            )

            writer.writerow({
                "stage":
                    result["stage"],

                "backbone":
                    result["backbone"],

                "variant":
                    result["variant"],

                "checkpoint_state":
                    result["checkpoint_state"],

                "selected_epoch":
                    result.get("selected_epoch"),

                "internal_val_accuracy_pct":
                    round(
                        internal_acc * 100,
                        6,
                    ),

                "internal_val_macro_f1_pct":
                    round(
                        float(
                            result[
                                "internal_validation_macro_f1"
                            ]
                        )
                        * 100,
                        6,
                    ),

                "external_accuracy_pct":
                    round(
                        external_acc * 100,
                        6,
                    ),

                "external_macro_precision_pct":
                    round(
                        external["macro_precision"]
                        * 100,
                        6,
                    ),

                "external_macro_recall_pct":
                    round(
                        external["macro_recall"]
                        * 100,
                        6,
                    ),

                "external_macro_f1_pct":
                    round(
                        external["macro_f1"]
                        * 100,
                        6,
                    ),

                "external_weighted_f1_pct":
                    round(
                        external["weighted_f1"]
                        * 100,
                        6,
                    ),

                "external_minus_internal_accuracy_pp":
                    round(
                        (
                            external_acc
                            - internal_acc
                        )
                        * 100,
                        6,
                    ),
            })

    print()
    print("=" * 80)
    print("15-MODEL EXTERNAL COMPARISON COMPLETE")
    print("=" * 80)
    print("Master JSON:", RESULTS_ROOT / "external_comparison_15.json")
    print("Master CSV :", csv_path)


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # FREEZE STAGE
    # --------------------------------------------------------
    if args.freeze_only:

        manifest = freeze_manifest()

        print(
            f"\nFrozen candidate count: "
            f"{manifest['candidate_count']}"
        )

        return


    # --------------------------------------------------------
    # EXTERNAL EVALUATION STAGE
    # --------------------------------------------------------
    if args.external_dir is None:
        parser.error(
            "Choose --freeze-only first, or provide --external-dir."
        )

    if not args.confirm_audited:
        parser.error(
            "External evaluation requires --confirm-audited. "
            "Complete provenance, class-compatibility and "
            "cross-dataset overlap/duplicate checks first."
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "\nFrozen manifest does not exist.\n"
            "Run FIRST:\n"
            "python actual_training_scripts/compare_external_15.py "
            "--freeze-only\n"
            "Then audit the external dataset before evaluation."
        )

    manifest = load_json(
        MANIFEST_PATH
    )

    verify_frozen_manifest(
        manifest
    )

    dataset, dataset_metadata = (
        prepare_external_dataset(
            args.external_dir
        )
    )

    print()
    print("=" * 80)
    print("EXTERNAL DATASET")
    print("=" * 80)
    print("Root       :", dataset_metadata["external_dir"])
    print("Total      :", dataset_metadata["total_images"])
    print("Class count:", dataset_metadata["class_counts"])
    print(
        "Fingerprint:",
        dataset_metadata[
            "dataset_fingerprint_path_size_label_sha256"
        ],
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print(
            "GPU   :",
            torch.cuda.get_device_name(0),
        )

    all_results = []

    for index, candidate in enumerate(
        manifest["candidates"],
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"EXTERNAL MODEL {index}/15"
        )
        print("=" * 80)

        print(
            f"{candidate['stage']} | "
            f"{candidate['backbone']} | "
            f"{candidate['variant']} | "
            f"{candidate['checkpoint_state']}"
        )

        result = evaluate_candidate(
            candidate,
            dataset,
            dataset_metadata,
            device,
        )

        all_results.append(
            result
        )

        print(
            f"  External accuracy : "
            f"{result['external']['accuracy']*100:.3f}%"
        )

        print(
            f"  External macro F1 : "
            f"{result['external']['macro_f1']*100:.3f}%"
        )

    save_master_outputs(
        manifest,
        dataset_metadata,
        all_results,
    )


if __name__ == "__main__":
    main()
