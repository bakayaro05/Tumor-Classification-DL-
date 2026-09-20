"""Occlusion sensitivity for the complete CNN-LSTM-PCA-SVM pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .gradcam import IMAGENET_MEAN, IMAGENET_STD
from .results_loader import CLASS_DISPLAY


@dataclass
class OcclusionResult:
    backbone: str
    target_class_key: str
    target_class_name: str
    baseline_prediction_key: str
    baseline_prediction_name: str
    baseline_decision_score: float
    model_input_rgb: np.ndarray
    heatmap: np.ndarray
    signed_score_change: np.ndarray
    patch_size: int
    stride: int
    occlusion_count: int
    prediction_flip_fraction: float
    maximum_score_drop: float
    elapsed_seconds: float


def generate_svm_occlusion(
    bundle: Any,
    input_tensor: torch.Tensor,
    target_class_key: str | None = None,
    *,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 8,
) -> OcclusionResult:
    """Measure regional support for a final SVM class by input occlusion.

    Every masked image is passed through the complete trained pipeline:
    CNN -> LSTM -> dense representation -> saved scaler -> saved PCA -> saved SVM.
    A positive score change means masking that region reduced the target SVM decision
    value, so the unmasked region supported that class.

    Occluded pixels are set to zero in normalized tensor space, corresponding to the
    ImageNet channel means rather than an artificial pure-black square.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Occlusion sensitivity expects one [1, C, H, W] input tensor.")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("patch_size, stride, and batch_size must be positive integers.")

    height, width = (int(value) for value in input_tensor.shape[-2:])
    if patch_size > min(height, width):
        raise ValueError(
            f"patch_size={patch_size} exceeds the transformed input size {width}x{height}."
        )

    started = time.perf_counter()
    model = bundle.model
    model.eval()
    class_to_idx = bundle.class_to_idx
    idx_to_key = {index: key for key, index in class_to_idx.items()}
    svm_classes = [int(value) for value in bundle.svm.classes_]

    baseline_features = _deep_features(model, input_tensor)
    baseline_decision, baseline_prediction = _svm_outputs(bundle, baseline_features)
    baseline_prediction_index = int(baseline_prediction[0])
    baseline_prediction_key = idx_to_key[baseline_prediction_index]

    if target_class_key is None:
        target_class_key = baseline_prediction_key
    if target_class_key not in class_to_idx:
        raise ValueError(f"Unknown occlusion target class: {target_class_key}")

    target_class_index = class_to_idx[target_class_key]
    try:
        target_column = svm_classes.index(target_class_index)
    except ValueError as error:
        raise RuntimeError(
            f"Target class index {target_class_index} is absent from SVM classes {svm_classes}."
        ) from error

    baseline_target_score = float(baseline_decision[0, target_column])
    y_starts = _sliding_starts(height, patch_size, stride)
    x_starts = _sliding_starts(width, patch_size, stride)
    regions = [(y, x) for y in y_starts for x in x_starts]

    score_sum = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    score_drops: list[float] = []
    prediction_flips = 0

    for offset in range(0, len(regions), batch_size):
        batch_regions = regions[offset : offset + batch_size]
        occluded = input_tensor.repeat(len(batch_regions), 1, 1, 1)
        for item, (top, left) in enumerate(batch_regions):
            occluded[item, :, top : top + patch_size, left : left + patch_size] = 0.0

        features = _deep_features(model, occluded)
        decisions, predictions = _svm_outputs(bundle, features)
        drops = baseline_target_score - decisions[:, target_column]
        prediction_flips += int(np.count_nonzero(predictions != baseline_prediction_index))

        for (top, left), drop in zip(batch_regions, drops, strict=True):
            value = float(drop)
            score_drops.append(value)
            score_sum[top : top + patch_size, left : left + patch_size] += value
            coverage[top : top + patch_size, left : left + patch_size] += 1.0

    signed_change = np.divide(
        score_sum,
        coverage,
        out=np.zeros_like(score_sum),
        where=coverage > 0,
    ).astype(np.float32)
    positive_support = np.clip(signed_change, 0.0, None)
    maximum = float(positive_support.max(initial=0.0))
    heatmap = (
        positive_support / maximum
        if maximum > 1e-12
        else np.zeros_like(positive_support, dtype=np.float32)
    )

    return OcclusionResult(
        backbone=bundle.artifacts.backbone,
        target_class_key=target_class_key,
        target_class_name=CLASS_DISPLAY[target_class_key],
        baseline_prediction_key=baseline_prediction_key,
        baseline_prediction_name=CLASS_DISPLAY[baseline_prediction_key],
        baseline_decision_score=baseline_target_score,
        model_input_rgb=_tensor_to_rgb(input_tensor[0]),
        heatmap=heatmap.astype(np.float32),
        signed_score_change=signed_change,
        patch_size=patch_size,
        stride=stride,
        occlusion_count=len(regions),
        prediction_flip_fraction=prediction_flips / len(regions),
        maximum_score_drop=max(score_drops, default=0.0),
        elapsed_seconds=time.perf_counter() - started,
    )


def overlay_occlusion(result: OcclusionResult, opacity: float = 0.55) -> np.ndarray:
    """Overlay positive final-SVM support on the transformed model input."""
    from .gradcam import colorize_heatmap

    opacity = float(np.clip(opacity, 0.0, 1.0))
    heatmap = np.clip(result.heatmap, 0.0, 1.0)
    colored = colorize_heatmap(heatmap).astype(np.float32)
    original = result.model_input_rgb.astype(np.float32)
    alpha = (opacity * heatmap)[..., None]
    blended = original * (1.0 - alpha) + colored * alpha
    return np.clip(blended, 0, 255).round().astype(np.uint8)


def _deep_features(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(tensor, return_debug=True)
        features = output["deep_features"]
    return features.detach().cpu().numpy()


def _svm_outputs(bundle: Any, deep_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = bundle.scaler.transform(deep_features)
    pca_features = bundle.pca.transform(scaled)
    decision = np.asarray(bundle.svm.decision_function(pca_features), dtype=np.float64)
    if decision.ndim == 1:
        decision = decision.reshape(-1, 1)
    prediction = np.asarray(bundle.svm.predict(pca_features), dtype=int)
    return decision, prediction


def _sliding_starts(length: int, patch_size: int, stride: int) -> list[int]:
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    restored = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    restored = restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (restored * 255.0).round().astype(np.uint8)
