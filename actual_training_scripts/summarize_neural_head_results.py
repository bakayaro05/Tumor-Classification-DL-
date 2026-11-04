from pathlib import Path
import csv
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from model_cnn_lstm import SUPPORTED_BACKBONES


OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUTPUT_DIR / "neural_head_comparison.json"
CSV_PATH = OUTPUT_DIR / "neural_head_comparison.csv"


def percent(value):
    return float(value) * 100.0


rows = []

for backbone in SUPPORTED_BACKBONES:
    result_path = (
        PROJECT_ROOT
        / "experiments"
        / backbone
        / "results"
        / "neural_head_results.json"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Missing neural-head result for {backbone}:\n"
            f"{result_path}\n"
            f"Run evaluate_neural_head.py --backbone {backbone} first."
        )

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if result.get("backbone") != backbone:
        raise ValueError(
            f"Backbone mismatch in {result_path}: "
            f"{result.get('backbone')!r}"
        )

    row = {
        "backbone": backbone,
        "best_epoch": int(result["checkpoint"]["best_epoch"]),
        "train_accuracy_percent": percent(
            result["train"]["accuracy"]
        ),
        "validation_accuracy_percent": percent(
            result["validation"]["accuracy"]
        ),
        "test_accuracy_percent": percent(
            result["test"]["accuracy"]
        ),
        "test_macro_precision_percent": percent(
            result["test"]["macro_precision"]
        ),
        "test_macro_recall_percent": percent(
            result["test"]["macro_recall"]
        ),
        "test_macro_f1_percent": percent(
            result["test"]["macro_f1"]
        ),
        "test_weighted_f1_percent": percent(
            result["test"]["weighted_f1"]
        ),
    }

    rows.append(row)


summary = {
    "description": (
        "Five-backbone deterministic post-training evaluation of the "
        "saved CNN-LSTM 4-class neural classifier heads before "
        "StandardScaler/PCA/SVM."
    ),
    "metric_units": "percent for fields ending in _percent",
    "rows": rows,
}


with JSON_PATH.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=4)


fieldnames = list(rows[0].keys())

with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


print("=" * 110)
print("CNN-LSTM NEURAL-HEAD COMPARISON")
print("=" * 110)

header = (
    f"{'Backbone':16s}"
    f"{'Epoch':>7s}"
    f"{'Train Acc':>12s}"
    f"{'Val Acc':>11s}"
    f"{'Test Acc':>12s}"
    f"{'Test Macro F1':>15s}"
)

print(header)
print("-" * len(header))

for row in rows:
    print(
        f"{row['backbone']:16s}"
        f"{row['best_epoch']:7d}"
        f"{row['train_accuracy_percent']:11.2f}%"
        f"{row['validation_accuracy_percent']:10.2f}%"
        f"{row['test_accuracy_percent']:11.2f}%"
        f"{row['test_macro_f1_percent']:14.2f}%"
    )

print("\nSaved:")
print(JSON_PATH)
print(CSV_PATH)
