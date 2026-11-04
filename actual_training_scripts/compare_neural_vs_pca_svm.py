from pathlib import Path
import csv
import json
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from model_cnn_lstm import SUPPORTED_BACKBONES


OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUTPUT_DIR / "neural_vs_pca_svm_comparison.json"
CSV_PATH = OUTPUT_DIR / "neural_vs_pca_svm_comparison.csv"

SPLITS = ("train", "validation", "test")


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


def paired_outcome_counts(labels, neural_predictions, svm_predictions):
    neural_correct = neural_predictions == labels
    svm_correct = svm_predictions == labels

    return {
        "both_correct": int(
            np.sum(neural_correct & svm_correct)
        ),
        "both_wrong": int(
            np.sum((~neural_correct) & (~svm_correct))
        ),
        "neural_correct_svm_wrong": int(
            np.sum(neural_correct & (~svm_correct))
        ),
        "neural_wrong_svm_correct": int(
            np.sum((~neural_correct) & svm_correct)
        ),
        "prediction_agreement_count": int(
            np.sum(neural_predictions == svm_predictions)
        ),
        "prediction_disagreement_count": int(
            np.sum(neural_predictions != svm_predictions)
        ),
    }


def load_neural_npz(backbone):
    path = (
        PROJECT_ROOT
        / "experiments"
        / backbone
        / "results"
        / "neural_head_predictions.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing neural-head predictions for {backbone}:\n"
            f"{path}\n"
            f"Run evaluate_neural_head.py --backbone {backbone} first "
            f"without --no-save-predictions."
        )

    return path, np.load(path, allow_pickle=False)


def load_svm_inputs(backbone, split):
    experiment_root = PROJECT_ROOT / "experiments" / backbone
    model_path = experiment_root / "models" / "svm.joblib"
    pca_dir = experiment_root / "pca"

    split_filename = {
        "train": "train_pca.npy",
        "validation": "val_pca.npy",
        "test": "test_pca.npy",
    }[split]

    label_filename = {
        "train": "train_labels.npy",
        "validation": "val_labels.npy",
        "test": "test_labels.npy",
    }[split]

    feature_path = pca_dir / split_filename
    label_path = pca_dir / label_filename

    for path in (model_path, feature_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required saved PCA/SVM artifact is missing:\n{path}"
            )

    return model_path, feature_path, label_path


def neural_keys(split):
    prefix = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }[split]

    return (
        f"{prefix}_labels",
        f"{prefix}_predictions",
    )


all_results = {}
summary_rows = []

for backbone in SUPPORTED_BACKBONES:
    print("\n" + "=" * 100)
    print(f"COMPARING NEURAL HEAD VS PCA+SVM: {backbone.upper()}")
    print("=" * 100)

    neural_result_path = (
        PROJECT_ROOT
        / "experiments"
        / backbone
        / "results"
        / "neural_head_results.json"
    )

    final_result_path = (
        PROJECT_ROOT
        / "experiments"
        / backbone
        / "results"
        / "final_results.json"
    )

    if not neural_result_path.is_file():
        raise FileNotFoundError(
            f"Missing {neural_result_path}"
        )

    if not final_result_path.is_file():
        raise FileNotFoundError(
            f"Missing existing hybrid result {final_result_path}"
        )

    with neural_result_path.open("r", encoding="utf-8") as handle:
        neural_result_json = json.load(handle)

    with final_result_path.open("r", encoding="utf-8") as handle:
        hybrid_result_json = json.load(handle)

    neural_npz_path, neural_npz = load_neural_npz(backbone)

    # Load one saved SVM per backbone. It is never refit here.
    svm_path = (
        PROJECT_ROOT
        / "experiments"
        / backbone
        / "models"
        / "svm.joblib"
    )
    svm = joblib.load(svm_path)

    backbone_result = {
        "backbone": backbone,
        "neural_prediction_source": str(
            neural_npz_path.relative_to(PROJECT_ROOT)
        ),
        "svm_model_source": str(
            svm_path.relative_to(PROJECT_ROOT)
        ),
        "comparison_scope": (
            "Same saved CNN-LSTM experiment; neural 4-class head versus "
            "saved StandardScaler/PCA(64)/linear-SVM downstream branch. "
            "No retraining or refitting performed by this script."
        ),
        "splits": {},
    }

    for split in SPLITS:
        model_path, pca_feature_path, pca_label_path = (
            load_svm_inputs(backbone, split)
        )

        svm_features = np.load(pca_feature_path)
        svm_labels = np.load(pca_label_path)

        label_key, prediction_key = neural_keys(split)
        neural_labels = neural_npz[label_key]
        neural_predictions = neural_npz[prediction_key]

        if not np.array_equal(neural_labels, svm_labels):
            raise ValueError(
                f"Label/order mismatch for {backbone} {split}. "
                "The neural-head and SVM branches are not aligned sample-for-sample."
            )

        svm_predictions = svm.predict(svm_features)

        neural_metrics = calculate_metrics(
            neural_labels,
            neural_predictions,
        )

        svm_metrics = calculate_metrics(
            svm_labels,
            svm_predictions,
        )

        deltas = {
            metric: float(
                svm_metrics[metric] - neural_metrics[metric]
            )
            for metric in neural_metrics
        }

        class_names = [
            str(item)
            for item in neural_npz["class_names"].tolist()
        ]

        neural_cm = confusion_matrix(
            neural_labels,
            neural_predictions,
            labels=list(range(len(class_names))),
        )

        svm_cm = confusion_matrix(
            svm_labels,
            svm_predictions,
            labels=list(range(len(class_names))),
        )

        neural_report = classification_report(
            neural_labels,
            neural_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        svm_report = classification_report(
            svm_labels,
            svm_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        paired = paired_outcome_counts(
            neural_labels,
            neural_predictions,
            svm_predictions,
        )

        split_result = {
            "sample_count": int(len(svm_labels)),
            "neural_head": {
                **neural_metrics,
                "confusion_matrix": neural_cm.tolist(),
                "classification_report": neural_report,
            },
            "pca_svm": {
                **svm_metrics,
                "confusion_matrix": svm_cm.tolist(),
                "classification_report": svm_report,
            },
            "delta_pca_svm_minus_neural": deltas,
            "paired_outcomes": paired,
        }

        backbone_result["splits"][split] = split_result

        print(f"\n{split.upper()}")
        print(
            f"  Neural accuracy : "
            f"{neural_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"  PCA+SVM accuracy: "
            f"{svm_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"  Accuracy delta  : "
            f"{deltas['accuracy'] * 100:+.2f} percentage points"
        )
        print(
            f"  Neural macro F1 : "
            f"{neural_metrics['macro_f1'] * 100:.2f}%"
        )
        print(
            f"  PCA+SVM macro F1: "
            f"{svm_metrics['macro_f1'] * 100:.2f}%"
        )
        print(
            f"  Macro-F1 delta  : "
            f"{deltas['macro_f1'] * 100:+.2f} percentage points"
        )

    # Cross-check the recomputed saved SVM val/test metrics against
    # standardized final_results.json so an artifact mismatch is caught.
    for split_key, final_key in (
        ("validation", "validation"),
        ("test", "test"),
    ):
        recomputed = (
            backbone_result["splits"][split_key]["pca_svm"]
        )
        stored = hybrid_result_json[final_key]

        if not np.isclose(
            recomputed["accuracy"],
            float(stored["accuracy"]),
            atol=1e-12,
        ):
            raise ValueError(
                f"Saved SVM artifact and final_results.json disagree for "
                f"{backbone} {split_key} accuracy."
            )

        if not np.isclose(
            recomputed["macro_f1"],
            float(stored["macro_f1"]),
            atol=1e-12,
        ):
            raise ValueError(
                f"Saved SVM artifact and final_results.json disagree for "
                f"{backbone} {split_key} macro F1."
            )

    all_results[backbone] = backbone_result

    val = backbone_result["splits"]["validation"]
    test = backbone_result["splits"]["test"]

    summary_rows.append(
        {
            "backbone": backbone,
            "best_epoch": int(
                neural_result_json["checkpoint"]["best_epoch"]
            ),
            "neural_validation_accuracy_percent":
                val["neural_head"]["accuracy"] * 100.0,
            "pca_svm_validation_accuracy_percent":
                val["pca_svm"]["accuracy"] * 100.0,
            "validation_accuracy_delta_pp":
                val["delta_pca_svm_minus_neural"]["accuracy"] * 100.0,
            "neural_test_accuracy_percent":
                test["neural_head"]["accuracy"] * 100.0,
            "pca_svm_test_accuracy_percent":
                test["pca_svm"]["accuracy"] * 100.0,
            "test_accuracy_delta_pp":
                test["delta_pca_svm_minus_neural"]["accuracy"] * 100.0,
            "neural_test_macro_f1_percent":
                test["neural_head"]["macro_f1"] * 100.0,
            "pca_svm_test_macro_f1_percent":
                test["pca_svm"]["macro_f1"] * 100.0,
            "test_macro_f1_delta_pp":
                test["delta_pca_svm_minus_neural"]["macro_f1"] * 100.0,
            "test_neural_correct_svm_wrong":
                test["paired_outcomes"]["neural_correct_svm_wrong"],
            "test_neural_wrong_svm_correct":
                test["paired_outcomes"]["neural_wrong_svm_correct"],
        }
    )


output = {
    "description": (
        "Direct downstream-classifier comparison: the saved CNN-LSTM "
        "neural head versus the already-trained StandardScaler/PCA/SVM "
        "branch for each backbone. Deltas are PCA+SVM minus neural head."
    ),
    "important_interpretation": (
        "This is not a PCA-only ablation. StandardScaler, PCA and SVM "
        "form the alternative downstream classification branch."
    ),
    "backbones": all_results,
    "summary": summary_rows,
}

with JSON_PATH.open("w", encoding="utf-8") as handle:
    json.dump(output, handle, indent=4)


fieldnames = list(summary_rows[0].keys())

with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)


print("\n" + "=" * 100)
print("NEURAL HEAD VS PCA+SVM COMPARISON COMPLETE")
print("=" * 100)
print("Saved:")
print(JSON_PATH)
print(CSV_PATH)
print(
    "\nPositive delta = PCA+SVM higher; "
    "negative delta = neural head higher."
)
print(
    "No CNN-LSTM checkpoint, scaler, PCA model or SVM was retrained/refit."
)
