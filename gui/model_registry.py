"""Artifact paths and display metadata for the five validated backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKBONE_ORDER = (
    "vgg16",
    "vgg19",
    "resnet50",
    "resnet152",
    "efficientnet_b3",
)

DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "efficientnet_b3": "EfficientNet-B3",
}


@dataclass(frozen=True)
class ModelArtifacts:
    backbone: str
    display_name: str
    checkpoint: Path
    svm: Path
    scaler: Path
    pca: Path
    final_results: Path
    training_results: Path

    def required_for_prediction(self) -> tuple[Path, ...]:
        return self.checkpoint, self.svm, self.scaler, self.pca

    def missing_prediction_files(self) -> list[Path]:
        return [path for path in self.required_for_prediction() if not path.is_file()]

    def prediction_ready(self) -> bool:
        return not self.missing_prediction_files()


def get_artifacts(project_root: Path, backbone: str) -> ModelArtifacts:
    if backbone not in BACKBONE_ORDER:
        raise ValueError(f"Unsupported backbone: {backbone}")

    experiment = project_root / "experiments" / backbone
    return ModelArtifacts(
        backbone=backbone,
        display_name=DISPLAY_NAMES[backbone],
        checkpoint=experiment / "models" / "best_model.pth",
        svm=experiment / "models" / "svm.joblib",
        scaler=experiment / "pca" / "standard_scaler.joblib",
        pca=experiment / "pca" / "pca_model.joblib",
        final_results=experiment / "results" / "final_results.json",
        training_results=experiment / "results" / "training_results.json",
    )


def all_artifacts(project_root: Path) -> list[ModelArtifacts]:
    return [get_artifacts(project_root, name) for name in BACKBONE_ORDER]
