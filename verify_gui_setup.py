"""Read-only preflight check for the Streamlit GUI."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKBONES = ("vgg16", "vgg19", "resnet50", "resnet152", "efficientnet_b3")
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "streamlit",
    "altair",
    "pandas",
    "PIL",
    "joblib",
    "sklearn",
)


def main() -> int:
    problems: list[str] = []

    print("GUI preflight check")
    print(f"Project root: {PROJECT_ROOT}")

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] package {package}: {version}")
        except ImportError:
            problems.append(f"Missing Python package: {package}")
            print(f"[MISSING] package {package}")

    for source in ("model_cnn_lstm.py", "src/data_pipeline_torch.py"):
        path = PROJECT_ROOT / source
        if path.is_file():
            print(f"[OK] source {source}")
        else:
            problems.append(f"Missing project source: {source}")
            print(f"[MISSING] source {source}")

    for backbone in BACKBONES:
        base = PROJECT_ROOT / "experiments" / backbone
        required = (
            base / "models" / "best_model.pth",
            base / "models" / "svm.joblib",
            base / "pca" / "standard_scaler.joblib",
            base / "pca" / "pca_model.joblib",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            problems.extend(f"Missing artifact: {path}" for path in missing)
            print(f"[MISSING] {backbone}: {len(missing)} prediction artifact(s)")
        else:
            print(f"[OK] {backbone}: prediction bundle")

        result_path = base / "results" / "final_results.json"
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            if result.get("backbone") != backbone:
                problems.append(f"Backbone mismatch in {result_path}")
                print(f"[INVALID] {backbone}: final_results backbone mismatch")
            else:
                print(f"[OK] {backbone}: final results")
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"Unreadable final results {result_path}: {error}")
            print(f"[MISSING/INVALID] {backbone}: final results")

    print()
    if problems:
        print(f"Preflight failed with {len(problems)} problem(s):")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Preflight passed. Start the GUI with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
