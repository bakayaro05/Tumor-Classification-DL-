#!/usr/bin/env python3
"""
Phase 2A leakage-corrected post-hoc evaluation.

IMPORTANT:
- Does NOT retrain any model.
- Does NOT refit StandardScaler, PCA, or SVM.
- Uses already-saved Phase-2A PCA test features and trained SVMs.
- Uses the SAME duplicate/leakage audit definitions as Phase 1.
- Evaluates:
    exact_clean
    material_clean [PRIMARY]
    conservative_clean

Primary comparison remains the material-clean test subset.
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


CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)


# ------------------------------------------------------------
# Seven DISTINCT Phase-2A checkpoint evaluations
# ------------------------------------------------------------

EVALUATIONS = [
    {
        "id": "vgg16_epoch30",
        "backbone": "vgg16",
        "evaluation": "best_val_within_30",
        "checkpoint_epoch": 30,
    },
    {
        "id": "vgg19_epoch30",
        "backbone": "vgg19",
        "evaluation": "best_val_within_30",
        "checkpoint_epoch": 30,
    },
    {
        "id": "resnet50_epoch30",
        "backbone": "resnet50",
        "evaluation": "best_val_within_30",
        "checkpoint_epoch": 30,
    },
    {
        "id": "resnet152_best_epoch10",
        "backbone": "resnet152",
        "evaluation": "best_val_within_30",
        "checkpoint_epoch": 10,
    },
    {
        "id": "resnet152_epoch30",
        "backbone": "resnet152",
        "evaluation": "epoch_30",
        "checkpoint_epoch": 30,
    },
    {
        "id": "efficientnet_b3_best_epoch27",
        "backbone": "efficientnet_b3",
        "evaluation": "best_val_within_30",
        "checkpoint_epoch": 27,
    },
    {
        "id": "efficientnet_b3_epoch30",
        "backbone": "efficientnet_b3",
        "evaluation": "epoch_30",
        "checkpoint_epoch": 30,
    },
]


def metrics(y_true, y_pred):

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "weighted_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


def normalized_rel(
    project_root: Path,
    path_text: str,
) -> str:

    p = Path(path_text)

    if p.is_absolute():

        try:

            p = (
                p.resolve()
                .relative_to(
                    project_root.resolve()
                )
            )

        except ValueError:

            return p.as_posix()

    return p.as_posix()


def test_paths_from_finding(
    row,
    project_root,
):

    out = []

    if row["split_a"] == "test":

        out.append(
            normalized_rel(
                project_root,
                row["path_a"],
            )
        )

    if row["split_b"] == "test":

        out.append(
            normalized_rel(
                project_root,
                row["path_b"],
            )
        )

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

    audit_csv = (
        args.audit_csv
        or (
            root
            / "final_project_docs"
            / "duplicate_leakage_audit"
            / "duplicate_leakage_findings.csv"
        )
    )

    if not audit_csv.is_file():

        raise FileNotFoundError(
            f"Audit CSV not found: {audit_csv}"
        )


    # --------------------------------------------------------
    # Reconstruct EXACT deterministic test ordering
    # --------------------------------------------------------

    from src.data_pipeline_torch import TEST_DIR

    test_dir = Path(TEST_DIR).resolve()

    ds = datasets.ImageFolder(
        root=test_dir
    )

    sample_paths = [
        normalized_rel(
            root,
            p,
        )
        for p, _ in ds.samples
    ]

    sample_labels = np.asarray(
        [
            y
            for _, y in ds.samples
        ],
        dtype=np.int64,
    )

    if len(sample_paths) != 1311:

        raise RuntimeError(
            "Expected 1,311 test images; "
            f"found {len(sample_paths)}"
        )


    # --------------------------------------------------------
    # Load ORIGINAL finalized duplicate audit
    # --------------------------------------------------------

    audit = pd.read_csv(
        audit_csv
    )

    exact_types = {
        "exact_file_duplicate",
        "exact_pixel_duplicate",
    }

    exact_rows = (
        audit[
            audit[
                "finding_type"
            ].isin(
                exact_types
            )
        ]
    )

    material_rows = (
        audit[
            audit[
                "material"
            ].astype(bool)
        ]
    )

    conservative_rows = (
        audit[
            audit[
                "material"
            ].astype(bool)
            |
            audit[
                "requires_review"
            ].astype(bool)
        ]
    )

    policies = {
        "exact_clean":
            exact_rows,

        "material_clean":
            material_rows,

        "conservative_clean":
            conservative_rows,
    }


    # --------------------------------------------------------
    # Determine excluded TEST images for each policy
    # --------------------------------------------------------

    excluded_by_policy = {}

    for policy, rows in policies.items():

        excluded = set()

        for _, row in rows.iterrows():

            excluded.update(
                test_paths_from_finding(
                    row,
                    root,
                )
            )

        excluded_by_policy[
            policy
        ] = excluded


    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    out_dir = (
        root
        / "phase2"
        / "results"
        / "leakage_corrected_30epoch"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Verify expected subset sizes
    # --------------------------------------------------------

    expected = {
        "exact_clean": 1095,
        "material_clean": 1061,
        "conservative_clean": 926,
    }

    print(
        "=" * 96
    )

    print(
        "PHASE 2A LEAKAGE-CORRECTED POST-HOC EVALUATION"
    )

    print(
        "=" * 96
    )

    print(
        f"\nSupplied test images: "
        f"{len(sample_paths)}"
    )

    for policy, excluded in excluded_by_policy.items():

        retained = sum(
            p not in excluded
            for p in sample_paths
        )

        excluded_n = (
            len(sample_paths)
            - retained
        )

        print(
            f"{policy:20s}: "
            f"{retained} retained | "
            f"{excluded_n} excluded"
        )

        if retained != expected[policy]:

            raise RuntimeError(
                f"{policy}: expected "
                f"{expected[policy]} retained "
                f"images, but found {retained}. "
                "STOPPING so the Phase-1 and "
                "Phase-2 policies cannot silently differ."
            )


    # --------------------------------------------------------
    # Save exact retained/excluded lists used in Phase 2
    # --------------------------------------------------------

    path_to_label = {
        p: int(y)
        for p, y in zip(
            sample_paths,
            sample_labels,
        )
    }

    for policy, excluded in excluded_by_policy.items():

        removed = [
            p
            for p in sample_paths
            if p in excluded
        ]

        kept = [
            p
            for p in sample_paths
            if p not in excluded
        ]

        pd.DataFrame(
            {
                "test_path":
                    removed,

                "class_index":
                    [
                        path_to_label[p]
                        for p in removed
                    ],

                "class_name":
                    [
                        CLASS_NAMES[
                            path_to_label[p]
                        ]
                        for p in removed
                    ],
            }
        ).to_csv(
            out_dir
            / f"{policy}_excluded_test_images.csv",
            index=False,
        )

        pd.DataFrame(
            {
                "test_path":
                    kept,

                "class_index":
                    [
                        path_to_label[p]
                        for p in kept
                    ],

                "class_name":
                    [
                        CLASS_NAMES[
                            path_to_label[p]
                        ]
                        for p in kept
                    ],
            }
        ).to_csv(
            out_dir
            / f"{policy}_included_test_images.csv",
            index=False,
        )


    # --------------------------------------------------------
    # Evaluate all seven distinct Phase-2A checkpoints
    # --------------------------------------------------------

    summary_rows = []

    all_results = {}

    for item in EVALUATIONS:

        experiment_id = item["id"]

        backbone = item["backbone"]

        evaluation_name = item[
            "evaluation"
        ]

        evaluation_root = (
            root
            / "phase2"
            / "experiments_30epoch"
            / backbone
            / "evaluations"
            / evaluation_name
        )

        pca_dir = (
            evaluation_root
            / "pca"
        )

        model_dir = (
            evaluation_root
            / "models"
        )

        results_dir = (
            evaluation_root
            / "results"
        )

        test_pca_path = (
            pca_dir
            / "test_pca.npy"
        )

        test_labels_path = (
            pca_dir
            / "test_labels.npy"
        )

        svm_path = (
            model_dir
            / "svm.joblib"
        )

        original_path = (
            results_dir
            / "final_results.json"
        )

        for p in (
            test_pca_path,
            test_labels_path,
            svm_path,
            original_path,
        ):

            if not p.is_file():

                raise FileNotFoundError(
                    "Missing Phase-2A artifact "
                    f"for {experiment_id}: {p}"
                )


        print(
            "\n"
            + "-" * 96
        )

        print(
            f"Evaluating: {experiment_id}"
        )

        print(
            f"Backbone: {backbone}"
        )

        print(
            "Checkpoint epoch: "
            f"{item['checkpoint_epoch']}"
        )


        test_pca = np.load(
            test_pca_path
        )

        saved_labels = np.load(
            test_labels_path
        )

        svm = joblib.load(
            svm_path
        )

        original = json.loads(
            original_path.read_text()
        )


        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if test_pca.shape[0] != len(sample_paths):

            raise RuntimeError(
                f"{experiment_id}: "
                f"test_pca rows "
                f"{test_pca.shape[0]} "
                f"!= ImageFolder rows "
                f"{len(sample_paths)}"
            )

        if saved_labels.shape[0] != len(sample_paths):

            raise RuntimeError(
                f"{experiment_id}: "
                f"test_labels rows "
                f"{saved_labels.shape[0]} "
                f"!= {len(sample_paths)}"
            )

        if not np.array_equal(
            saved_labels.astype(
                np.int64
            ),
            sample_labels,
        ):

            raise RuntimeError(
                f"{experiment_id}: "
                "saved test label order does "
                "not match current ImageFolder "
                "ordering."
            )


        experiment_result = {
            "experiment_id":
                experiment_id,

            "backbone":
                backbone,

            "evaluation":
                evaluation_name,

            "checkpoint_epoch":
                item["checkpoint_epoch"],

            "supplied_full_test":
                original["test"],

            "policies":
                {},
        }


        # ----------------------------------------------------
        # Apply each existing leakage-clean policy
        # ----------------------------------------------------

        for policy, excluded in excluded_by_policy.items():

            keep_mask = np.array(
                [
                    p not in excluded
                    for p in sample_paths
                ],
                dtype=bool,
            )

            X = test_pca[
                keep_mask
            ]

            y = saved_labels[
                keep_mask
            ].astype(
                np.int64
            )

            pred = svm.predict(
                X
            )

            m = metrics(
                y,
                pred,
            )

            cm = confusion_matrix(
                y,
                pred,
                labels=np.arange(4),
            ).tolist()

            report = (
                classification_report(
                    y,
                    pred,
                    labels=np.arange(4),
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )
            )

            class_support = {
                cls:
                    int(
                        (
                            y == i
                        ).sum()
                    )

                for i, cls
                in enumerate(
                    CLASS_NAMES
                )
            }


            result = {
                "policy":
                    policy,

                "n_test":
                    int(
                        len(y)
                    ),

                "n_excluded":
                    int(
                        len(sample_paths)
                        - len(y)
                    ),

                "class_support":
                    class_support,

                **m,

                "confusion_matrix":
                    cm,

                "classification_report":
                    report,
            }

            experiment_result[
                "policies"
            ][
                policy
            ] = result


            original_acc = float(
                original[
                    "test"
                ][
                    "accuracy"
                ]
            )

            original_f1 = float(
                original[
                    "test"
                ][
                    "macro_f1"
                ]
            )


            summary_rows.append(
                {
                    "experiment_id":
                        experiment_id,

                    "backbone":
                        backbone,

                    "checkpoint_epoch":
                        item[
                            "checkpoint_epoch"
                        ],

                    "evaluation":
                        evaluation_name,

                    "policy":
                        policy,

                    "test_n":
                        len(y),

                    "excluded_n":
                        len(sample_paths)
                        - len(y),

                    "accuracy_pct":
                        m[
                            "accuracy"
                        ]
                        * 100,

                    "macro_f1_pct":
                        m[
                            "macro_f1"
                        ]
                        * 100,

                    "macro_precision_pct":
                        m[
                            "macro_precision"
                        ]
                        * 100,

                    "macro_recall_pct":
                        m[
                            "macro_recall"
                        ]
                        * 100,

                    "full_test_accuracy_pct":
                        original_acc
                        * 100,

                    "full_test_macro_f1_pct":
                        original_f1
                        * 100,

                    "accuracy_change_pp":
                        (
                            m[
                                "accuracy"
                            ]
                            - original_acc
                        )
                        * 100,

                    "macro_f1_change_pp":
                        (
                            m[
                                "macro_f1"
                            ]
                            - original_f1
                        )
                        * 100,

                    "glioma_support":
                        class_support[
                            "glioma"
                        ],

                    "meningioma_support":
                        class_support[
                            "meningioma"
                        ],

                    "notumor_support":
                        class_support[
                            "notumor"
                        ],

                    "pituitary_support":
                        class_support[
                            "pituitary"
                        ],
                }
            )


        all_results[
            experiment_id
        ] = experiment_result


    # --------------------------------------------------------
    # Save complete outputs
    # --------------------------------------------------------

    (
        out_dir
        / "phase2_leakage_corrected_results.json"
    ).write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        out_dir
        / "phase2_leakage_corrected_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # PRIMARY = material_clean
    # --------------------------------------------------------

    primary = (
        summary_df[
            summary_df[
                "policy"
            ]
            == "material_clean"
        ]
        .copy()
    )

    primary = primary.sort_values(
        "accuracy_pct",
        ascending=False,
    )

    primary.to_csv(
        out_dir
        / "PHASE2A_PRIMARY_material_clean_comparison.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 96
    )

    print(
        "PHASE 2A LEAKAGE-CORRECTED RESULTS"
    )

    print(
        "=" * 96
    )

    for policy in (
        "exact_clean",
        "material_clean",
        "conservative_clean",
    ):

        subset = (
            summary_df[
                summary_df[
                    "policy"
                ]
                == policy
            ]
        )

        n = int(
            subset.iloc[0][
                "test_n"
            ]
        )

        excluded = int(
            subset.iloc[0][
                "excluded_n"
            ]
        )

        print(
            f"\n{policy}: "
            f"{n} retained; "
            f"{excluded} excluded"
        )

        print(
            subset[
                [
                    "experiment_id",
                    "accuracy_pct",
                    "macro_f1_pct",
                    "accuracy_change_pp",
                    "macro_f1_change_pp",
                ]
            ].to_string(
                index=False,
                float_format=
                    lambda x:
                        f"{x:.3f}",
            )
        )


    print(
        "\nPRIMARY RESULT = "
        "material_clean"
    )

    print(
        primary[
            [
                "experiment_id",
                "backbone",
                "checkpoint_epoch",
                "test_n",
                "accuracy_pct",
                "macro_f1_pct",
                "full_test_accuracy_pct",
                "accuracy_change_pp",
            ]
        ].to_string(
            index=False,
            float_format=
                lambda x:
                    f"{x:.3f}",
        )
    )


    print(
        f"\nSaved results to: "
        f"{out_dir}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "- No model was retrained."
    )

    print(
        "- PCA/SVM were NOT refit."
    )

    print(
        "- The existing Phase-1 "
        "duplicate audit definitions "
        "were reused."
    )

    print(
        "- material_clean remains "
        "the primary corrected test policy."
    )


if __name__ == "__main__":
    main()
