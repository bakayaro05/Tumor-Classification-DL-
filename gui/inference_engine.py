"""Exact single-image inference for the validated CNN-LSTM-PCA-SVM pipeline."""

from __future__ import annotations

import gc
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from .gradcam import GradCAMResult, generate_gradcam
from .occlusion import OcclusionResult, generate_svm_occlusion
from .model_registry import ModelArtifacts, get_artifacts
from .results_loader import CLASS_DISPLAY, CLASS_NAMES


DEFAULT_CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass
class PredictionResult:
    backbone: str
    svm_class_key: str
    svm_class_name: str
    svm_relative_scores: dict[str, float]
    nn_class_key: str
    nn_class_name: str
    nn_scores: dict[str, float]
    models_agree: bool
    elapsed_seconds: float
    device: str
    image_size: tuple[int, int]
    deep_feature_dimension: int
    pca_feature_dimension: int


@dataclass
class LoadedBundle:
    artifacts: ModelArtifacts
    model: torch.nn.Module
    scaler: Any
    pca: Any
    svm: Any
    class_to_idx: dict[str, int]
    device: torch.device


def choose_device(preference: str = "auto") -> torch.device:
    preference = preference.lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_image(image: Image.Image, project_root: Path) -> torch.Tensor:
    """Apply the project's own evaluation transform to a user image."""
    _ensure_project_imports(project_root)
    try:
        from src.data_pipeline_torch import eval_transform
    except ImportError as error:
        raise RuntimeError(
            "Could not import src.data_pipeline_torch.eval_transform. "
            "Place the GUI files in the brain_tumor_project root."
        ) from error

    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 32:
        raise ValueError("The image is too small. Both dimensions must be at least 32 pixels.")
    tensor = eval_transform(image)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise RuntimeError("eval_transform did not return a [C, H, W] tensor.")
    return tensor.unsqueeze(0)


class ModelManager:
    """Keep at most one selected backbone resident in this GUI session."""

    def __init__(self, project_root: Path, device_preference: str = "auto"):
        self.project_root = Path(project_root).resolve()
        self.device_preference = device_preference
        self._bundle: LoadedBundle | None = None
        self._lock = threading.RLock()

    @property
    def loaded_backbone(self) -> str | None:
        return self._bundle.artifacts.backbone if self._bundle else None

    def unload(self) -> None:
        with self._lock:
            if self._bundle is not None:
                self._bundle.model.to("cpu")
                self._bundle = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, backbone: str) -> LoadedBundle:
        with self._lock:
            if self._bundle and self._bundle.artifacts.backbone == backbone:
                return self._bundle

            self.unload()
            artifacts = get_artifacts(self.project_root, backbone)
            missing = artifacts.missing_prediction_files()
            if missing:
                formatted = "\n".join(f"- {path}" for path in missing)
                raise FileNotFoundError(f"Missing prediction artifacts:\n{formatted}")

            _ensure_project_imports(self.project_root)
            from .model_factory import CheckpointCNNLSTM

            device = choose_device(self.device_preference)
            checkpoint = torch.load(
                artifacts.checkpoint,
                map_location=device,
                weights_only=False,
            )
            saved_backbone = checkpoint.get("backbone", backbone)
            if saved_backbone != backbone:
                raise RuntimeError(
                    f"Checkpoint backbone is {saved_backbone!r}, expected {backbone!r}."
                )

            model = CheckpointCNNLSTM(
                backbone_name=backbone,
                lstm_hidden_size=int(checkpoint.get("lstm_hidden_size", 256)),
                dense_features=int(checkpoint.get("dense_features", 128)),
                num_classes=4,
                dropout=float(checkpoint.get("dropout", 0.3)),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device).eval()

            class_to_idx = _validated_class_mapping(checkpoint.get("class_to_idx"))
            bundle = LoadedBundle(
                artifacts=artifacts,
                model=model,
                scaler=joblib.load(artifacts.scaler),
                pca=joblib.load(artifacts.pca),
                svm=joblib.load(artifacts.svm),
                class_to_idx=class_to_idx,
                device=device,
            )
            _validate_bundle(bundle)
            self._bundle = bundle
            return bundle

    def predict(self, image: Image.Image, backbone: str) -> PredictionResult:
        with self._lock:
            started = time.perf_counter()
            bundle = self.load(backbone)
            original_size = image.size
            tensor = prepare_image(image, self.project_root).to(bundle.device)

            with torch.inference_mode():
                debug_output = bundle.model(tensor, return_debug=True)
                deep_features = debug_output["deep_features"]
                logits = bundle.model.classifier(deep_features)

            nn_values = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            deep_numpy = deep_features.detach().cpu().numpy()
            scaled = bundle.scaler.transform(deep_numpy)
            pca_features = bundle.pca.transform(scaled)
            svm_prediction = int(bundle.svm.predict(pca_features)[0])
            decision = np.asarray(bundle.svm.decision_function(pca_features), dtype=float)
            decision = decision.reshape(1, -1)[0]
            svm_classes = [int(value) for value in bundle.svm.classes_]
            relative = _softmax(decision)

            idx_to_key = {index: key for key, index in bundle.class_to_idx.items()}
            nn_prediction = int(np.argmax(nn_values))
            nn_key = idx_to_key[nn_prediction]
            svm_key = idx_to_key[svm_prediction]

            nn_scores = {
                CLASS_DISPLAY[idx_to_key[index]]: float(nn_values[index])
                for index in sorted(idx_to_key)
            }
            svm_scores = {
                CLASS_DISPLAY[idx_to_key[class_index]]: float(score)
                for class_index, score in zip(svm_classes, relative, strict=True)
            }

            return PredictionResult(
                backbone=backbone,
                svm_class_key=svm_key,
                svm_class_name=CLASS_DISPLAY[svm_key],
                svm_relative_scores=svm_scores,
                nn_class_key=nn_key,
                nn_class_name=CLASS_DISPLAY[nn_key],
                nn_scores=nn_scores,
                models_agree=svm_prediction == nn_prediction,
                elapsed_seconds=time.perf_counter() - started,
                device=str(bundle.device),
                image_size=original_size,
                deep_feature_dimension=int(deep_numpy.shape[1]),
                pca_feature_dimension=int(pca_features.shape[1]),
            )

    def explain(
        self,
        image: Image.Image,
        backbone: str,
        target_class_key: str,
    ) -> GradCAMResult:
        """Explain the neural-head class using Grad-CAM on the selected CNN backbone."""
        with self._lock:
            bundle = self.load(backbone)
            tensor = prepare_image(image, self.project_root).to(bundle.device)
            return generate_gradcam(bundle, tensor, target_class_key)

    def explain_svm_occlusion(
        self,
        image: Image.Image,
        backbone: str,
        target_class_key: str | None = None,
        *,
        patch_size: int = 32,
        stride: int = 16,
        batch_size: int = 8,
    ) -> OcclusionResult:
        """Explain a final SVM decision with full-pipeline occlusion sensitivity."""
        with self._lock:
            bundle = self.load(backbone)
            tensor = prepare_image(image, self.project_root).to(bundle.device)
            return generate_svm_occlusion(
                bundle,
                tensor,
                target_class_key,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
            )


def _ensure_project_imports(project_root: Path) -> None:
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _validated_class_mapping(value: Any) -> dict[str, int]:
    mapping = value if isinstance(value, dict) else DEFAULT_CLASS_TO_IDX
    normalized = {str(key): int(index) for key, index in mapping.items()}
    if normalized != DEFAULT_CLASS_TO_IDX:
        raise RuntimeError(
            "Unexpected checkpoint class_to_idx mapping. "
            f"Expected {DEFAULT_CLASS_TO_IDX}, found {normalized}."
        )
    return normalized


def _validate_bundle(bundle: LoadedBundle) -> None:
    if not hasattr(bundle.svm, "decision_function"):
        raise RuntimeError("The loaded SVM does not expose decision_function().")
    svm_classes = [int(value) for value in bundle.svm.classes_]
    if svm_classes != [0, 1, 2, 3]:
        raise RuntimeError(f"Unexpected SVM class order: {svm_classes}")
    dense_features = bundle.model.dense.out_features
    if getattr(bundle.scaler, "n_features_in_", dense_features) != dense_features:
        raise RuntimeError("Scaler input dimension does not match the model's dense features.")
    scaler_output = getattr(bundle.scaler, "n_features_in_", dense_features)
    if getattr(bundle.pca, "n_features_in_", scaler_output) != scaler_output:
        raise RuntimeError("PCA input dimension does not match the scaler output.")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
