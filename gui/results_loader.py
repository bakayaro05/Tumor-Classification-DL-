"""Tolerant loaders for saved evaluation and training JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .model_registry import all_artifacts


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}


def _percent(value: Any, already_percent: bool = False) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number if already_percent else number * 100, 2)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def comparison_table(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifacts in all_artifacts(project_root):
        result = load_json(artifacts.final_results)
        if result is None:
            continue
        neural = result.get("neural_network", {})
        validation = result.get("validation", {})
        test = result.get("test", {})
        pca = result.get("pca", {})
        rows.append(
            {
                "Model": artifacts.display_name,
                "Backbone": artifacts.backbone,
                "Best epoch": neural.get("best_epoch"),
                "NN validation accuracy (%)": _percent(neural.get("val_accuracy"), already_percent=True),
                "SVM validation accuracy (%)": _percent(validation.get("accuracy")),
                "Test accuracy (%)": _percent(test.get("accuracy")),
                "Test macro precision (%)": _percent(test.get("macro_precision")),
                "Test macro recall (%)": _percent(test.get("macro_recall")),
                "Test macro F1 (%)": _percent(test.get("macro_f1")),
                "PCA components": pca.get("components"),
                "PCA variance retained (%)": _percent(pca.get("explained_variance")),
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(result: dict[str, Any], split: str) -> pd.DataFrame | None:
    matrix = result.get(split, {}).get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(CLASS_NAMES):
        return None
    labels = [CLASS_DISPLAY[name] for name in CLASS_NAMES]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def classification_report_frame(result: dict[str, Any], split: str) -> pd.DataFrame:
    report = result.get(split, {}).get("classification_report", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return pd.DataFrame()
    for class_name in CLASS_NAMES:
        values = report.get(class_name)
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "Class": CLASS_DISPLAY[class_name],
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1-score": values.get("f1-score"),
                "Support": values.get("support"),
            }
        )
    return pd.DataFrame(rows)


def training_history_frame(result: dict[str, Any] | None) -> pd.DataFrame:
    if not result:
        return pd.DataFrame()
    history = result.get("history", {})
    if not isinstance(history, dict):
        return pd.DataFrame()
    columns = {
        "Train loss": history.get("train_loss", []),
        "Validation loss": history.get("val_loss", []),
        "Train accuracy (%)": history.get("train_accuracy", []),
        "Validation accuracy (%)": history.get("val_accuracy", []),
    }
    length = max((len(value) for value in columns.values() if isinstance(value, list)), default=0)
    if length == 0:
        return pd.DataFrame()
    normalized = {
        key: (value + [None] * length)[:length] if isinstance(value, list) else [None] * length
        for key, value in columns.items()
    }
    frame = pd.DataFrame(normalized)
    frame.index = range(1, length + 1)
    frame.index.name = "Epoch"
    return frame


def _percent(value: Any, *, already_percent: bool = False) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) if already_percent else float(value) * 100.0, 2)
