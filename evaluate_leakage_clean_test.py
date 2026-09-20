#!/usr/bin/env python3
"""
Leakage-corrected post-hoc evaluation for the finalized five-backbone project.

Purpose:
- DOES NOT retrain any model.
- DOES NOT refit StandardScaler, PCA, or SVM.
- Uses each experiment's already-saved test PCA features and labels.
- Reconstructs the deterministic ImageFolder test ordering.
- Excludes cross-split test images according to three audit policies:
    1) exact_clean
    2) material_clean  [PRIMARY]
    3) conservative_clean [sensitivity; material + review-only candidates]
- Writes corrected metrics and comparison tables.

Run from project root:
    source venv/bin/activate
    python evaluate_leakage_clean_test.py

Optional:
    python evaluate_leakage_clean_test.py \
        --audit-csv final_project_docs/duplicate_leakage_audit/duplicate_leakage_findings.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BACKBONES = ("vgg16", "vgg19", "resnet50", "resnet152", "efficientnet_b3")
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def normalized_rel(project_root: Path, path_text: str) -> str:
    p = Path(path_text)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(project_root.resolve())
        except ValueError:
            return p.as_posix()
    return p.as_posix()

def test_paths_from_finding(row, project_root):
    out = []
    if row["split_a"] == "test":
        out.append(normalized_rel(project_root, row["path_a"]))
    if row["split_b"] == "test":
        out.append(normalized_rel(project_root, row["path_b"]))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    audit_csv = args.audit_csv or (
        root / "final_project_docs" / "duplicate_leakage_audit" / "duplicate_leakage_findings.csv"
    )
    if not audit_csv.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")

    from src.data_pipeline_torch import TEST_DIR

    test_dir = Path(TEST_DIR).resolve()
    ds = datasets.ImageFolder(root=test_dir)

    # ImageFolder is deterministic and the original evaluator used shuffle=False.
    sample_paths = [normalized_rel(root, p) for p, _ in ds.samples]
    sample_labels = np.asarray([y for _, y in ds.samples], dtype=np.int64)

    if len(sample_paths) != 1311:
        raise RuntimeError(f"Expected 1,311 test images; found {len(sample_paths)}")

    audit = pd.read_csv(audit_csv)

    exact_types = {"exact_file_duplicate", "exact_pixel_duplicate"}
    exact_rows = audit[audit["finding_type"].isin(exact_types)]
    material_rows = audit[audit["material"].astype(bool)]
    conservative_rows = audit[
        audit["material"].astype(bool) | audit["requires_review"].astype(bool)
    ]

    policies = {
        "exact_clean": exact_rows,
        "material_clean": material_rows,
        "conservative_clean": conservative_rows,
    }

    excluded_by_policy = {}
    for name, rows in policies.items():
        excluded = set()
        for _, row in rows.iterrows():
            excluded.update(test_paths_from_finding(row, root))
        excluded_by_policy[name] = excluded

    out_dir = root / "final_project_docs" / "leakage_corrected_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save exact inclusion/exclusion lists for reproducibility.
    path_to_label = dict(zip(sample_paths, sample_labels.tolist()))
    for policy, excluded in excluded_by_policy.items():
        keep = [p for p in sample_paths if p not in excluded]
        pd.DataFrame({"test_path": sorted(excluded)}).to_csv(
            out_dir / f"{policy}_excluded_test_images.csv", index=False
        )
        pd.DataFrame(
            {
                "test_path": keep,
                "class_index": [path_to_label[p] for p in keep],
                "class_name": [CLASS_NAMES[path_to_label[p]] for p in keep],
            }
        ).to_csv(out_dir / f"{policy}_included_test_images.csv", index=False)

    summary_rows = []
    all_results = {}

    for backbone in BACKBONES:
        exp = root / "experiments" / backbone
        pca_dir = exp / "pca"
        model_dir = exp / "models"
        results_dir = exp / "results"

        test_pca_path = pca_dir / "test_pca.npy"
        test_labels_path = pca_dir / "test_labels.npy"
        svm_path = model_dir / "svm.joblib"
        original_path = results_dir / "final_results.json"

        for p in (test_pca_path, test_labels_path, svm_path, original_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing finalized artifact for {backbone}: {p}")

        test_pca = np.load(test_pca_path)
        saved_labels = np.load(test_labels_path)
        svm = joblib.load(svm_path)
        original = json.loads(original_path.read_text())

        if test_pca.shape[0] != len(sample_paths):
            raise RuntimeError(
                f"{backbone}: test_pca rows {test_pca.shape[0]} != ImageFolder rows {len(sample_paths)}"
            )
        if saved_labels.shape[0] != len(sample_paths):
            raise RuntimeError(
                f"{backbone}: test_labels rows {saved_labels.shape[0]} != {len(sample_paths)}"
            )
        if not np.array_equal(saved_labels.astype(np.int64), sample_labels):
            raise RuntimeError(
                f"{backbone}: saved test label order does not match current ImageFolder ordering."
            )

        backbone_result = {
            "backbone": backbone,
            "original_test": original["test"],
            "policies": {},
        }

        for policy, excluded in excluded_by_policy.items():
            keep_mask = np.array([p not in excluded for p in sample_paths], dtype=bool)
            X = test_pca[keep_mask]
            y = saved_labels[keep_mask].astype(np.int64)
            pred = svm.predict(X)

            m = metrics(y, pred)
            cm = confusion_matrix(y, pred, labels=np.arange(4)).tolist()
            report = classification_report(
                y,
                pred,
                labels=np.arange(4),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )

            class_support = {
                cls: int((y == i).sum()) for i, cls in enumerate(CLASS_NAMES)
            }

            result = {
                "policy": policy,
                "n_test": int(len(y)),
                "n_excluded": int(len(sample_paths) - len(y)),
                "class_support": class_support,
                **m,
                "confusion_matrix": cm,
                "classification_report": report,
            }
            backbone_result["policies"][policy] = result

            original_acc = float(original["test"]["accuracy"])
            original_f1 = float(original["test"]["macro_f1"])

            summary_rows.append(
                {
                    "backbone": backbone,
                    "policy": policy,
                    "test_n": len(y),
                    "excluded_n": len(sample_paths) - len(y),
                    "accuracy_pct": m["accuracy"] * 100,
                    "macro_f1_pct": m["macro_f1"] * 100,
                    "macro_precision_pct": m["macro_precision"] * 100,
                    "macro_recall_pct": m["macro_recall"] * 100,
                    "original_accuracy_pct": original_acc * 100,
                    "original_macro_f1_pct": original_f1 * 100,
                    "accuracy_change_pp": (m["accuracy"] - original_acc) * 100,
                    "macro_f1_change_pp": (m["macro_f1"] - original_f1) * 100,
                    "glioma_support": class_support["glioma"],
                    "meningioma_support": class_support["meningioma"],
                    "notumor_support": class_support["notumor"],
                    "pituitary_support": class_support["pituitary"],
                }
            )

        all_results[backbone] = backbone_result

    (out_dir / "leakage_corrected_results.json").write_text(
        json.dumps(all_results, indent=2)
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "leakage_corrected_summary.csv", index=False)

    primary = summary_df[summary_df["policy"] == "material_clean"].copy()
    primary = primary.sort_values("accuracy_pct", ascending=False)
    primary.to_csv(out_dir / "PRIMARY_material_clean_comparison.csv", index=False)

    print("=" * 88)
    print("LEAKAGE-CORRECTED POST-HOC TEST EVALUATION")
    print("=" * 88)
    for policy in ("exact_clean", "material_clean", "conservative_clean"):
        subset = summary_df[summary_df["policy"] == policy]
        n = int(subset.iloc[0]["test_n"])
        excluded = int(subset.iloc[0]["excluded_n"])
        print(f"\n{policy}: {n} images retained; {excluded} excluded")
        print(
            subset[
                [
                    "backbone",
                    "accuracy_pct",
                    "macro_f1_pct",
                    "accuracy_change_pp",
                    "macro_f1_change_pp",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    print("\nPRIMARY RESULT = material_clean")
    print(primary[
        [
            "backbone",
            "test_n",
            "accuracy_pct",
            "macro_f1_pct",
            "accuracy_change_pp",
            "macro_f1_change_pp",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved results to: {out_dir}")
    print(
        "\nIMPORTANT: This is a corrected evaluation of existing trained models, "
        "not a duplicate-group-aware retraining experiment."
    )

if __name__ == "__main__":
    main()
