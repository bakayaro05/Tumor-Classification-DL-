"""Grad-CAM explanations for the CNN-LSTM neural classifier head."""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .results_loader import CLASS_DISPLAY


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class GradCAMResult:
    backbone: str
    target_class_key: str
    target_class_name: str
    target_layer_name: str
    model_input_rgb: np.ndarray
    heatmap: np.ndarray
    layer_heatmaps: dict[str, np.ndarray]
    elapsed_seconds: float


def generate_gradcam(
    bundle: Any,
    input_tensor: torch.Tensor,
    target_class_key: str,
) -> GradCAMResult:
    """Generate Grad-CAM for one neural-head class on one transformed image."""
    if target_class_key not in bundle.class_to_idx:
        raise ValueError(f"Unknown Grad-CAM target class: {target_class_key}")

    started = time.perf_counter()
    model = bundle.model
    target_layers, target_layer_name = _target_layers(model, bundle.artifacts.backbone)
    captured: dict[str, torch.Tensor] = {}
    lstm_was_training = model.lstm.training

    # cuDNN does not retain the reserve buffer required for RNN backward when
    # an LSTM runs in evaluation mode. Grad-CAM needs that backward pass. Only
    # the LSTM is temporarily switched to training mode; the surrounding model
    # remains in evaluation mode, so BatchNorm and Dropout stay deterministic.
    # This LSTM has one layer and therefore no active recurrent dropout.
    if model.lstm.num_layers != 1 or float(model.lstm.dropout) != 0.0:
        raise RuntimeError(
            "Grad-CAM expected the trained one-layer LSTM with zero recurrent dropout."
        )
    model.lstm.train(True)

    def capture_activation(layer_name: str):
        def hook(_module, _inputs, output):
            captured[layer_name] = output

        return hook

    handles = [
        layer.register_forward_hook(capture_activation(layer_name))
        for layer_name, layer in target_layers
    ]
    try:
        model.zero_grad(set_to_none=True)
        # Force PyTorch's native RNN path for this explanation pass. cuDNN RNNs
        # can reject backward in eval-oriented inference workflows even when the
        # LSTM module is temporarily marked as training. Normal prediction is
        # unaffected and continues to use cuDNN.
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            logits = model(input_tensor)
            target_index = bundle.class_to_idx[target_class_key]
            target_score = logits[0, target_index]
            missing_layers = [name for name, _layer in target_layers if name not in captured]
            if missing_layers:
                raise RuntimeError(
                    "The Grad-CAM target layers did not produce activation maps: "
                    + ", ".join(missing_layers)
                )

            activations = [captured[name] for name, _layer in target_layers]
            for layer_name, activation in zip(
                (name for name, _layer in target_layers), activations
            ):
                if activation.ndim != 4:
                    raise RuntimeError(
                        f"Expected a 4-D activation from {layer_name}, "
                        f"found {tuple(activation.shape)}."
                    )

            gradients = torch.autograd.grad(
                target_score,
                activations,
                retain_graph=False,
                create_graph=False,
            )

        # Normalize each layer before fusion. Otherwise the layer with the
        # largest raw gradient scale can dominate regardless of spatial or
        # semantic usefulness. ResNet uses an equal blend of Layer 3 (14x14)
        # and Layer 4 (7x7); single-layer backbones pass through unchanged.
        layer_cams = [
            _normalized_cam(activation, gradient, input_tensor.shape[-2:])
            for activation, gradient in zip(activations, gradients)
        ]
        cam = torch.stack(layer_cams, dim=0).mean(dim=0)
        cam = _normalize_map(cam, "The fused Grad-CAM map")
        layer_heatmaps = {
            layer_name: layer_cam.detach().cpu().numpy().astype(np.float32)
            for (layer_name, _layer), layer_cam in zip(target_layers, layer_cams)
        }

        return GradCAMResult(
            backbone=bundle.artifacts.backbone,
            target_class_key=target_class_key,
            target_class_name=CLASS_DISPLAY[target_class_key],
            target_layer_name=target_layer_name,
            model_input_rgb=_tensor_to_rgb(input_tensor[0]),
            heatmap=cam.detach().cpu().numpy().astype(np.float32),
            layer_heatmaps=layer_heatmaps,
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        for handle in handles:
            handle.remove()
        model.lstm.train(lstm_was_training)
        model.zero_grad(set_to_none=True)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply a compact blue→cyan→yellow→red colour map without extra dependencies."""
    values = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [0, 0, 70],
            [0, 110, 255],
            [0, 220, 150],
            [255, 225, 0],
            [220, 0, 0],
        ],
        dtype=np.float32,
    )
    channels = [np.interp(values, positions, colors[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def overlay_heatmap(
    result: GradCAMResult,
    opacity: float,
    heatmap: np.ndarray | None = None,
) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    selected_heatmap = result.heatmap if heatmap is None else heatmap
    selected_heatmap = np.clip(np.asarray(selected_heatmap, dtype=np.float32), 0.0, 1.0)
    colored = colorize_heatmap(selected_heatmap).astype(np.float32)
    original = result.model_input_rgb.astype(np.float32)
    # Weight the overlay by activation strength so low-contribution regions remain visible.
    alpha = (opacity * selected_heatmap)[..., None]
    blended = original * (1.0 - alpha) + colored * alpha
    return np.clip(blended, 0, 255).round().astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _target_layers(model: nn.Module, backbone: str) -> tuple[list[tuple[str, nn.Module]], str]:
    if backbone in {"vgg16", "vgg19"}:
        for name, module in reversed(list(model.cnn_backbone.named_modules())):
            if isinstance(module, nn.ReLU):
                layer_name = f"cnn_backbone.{name}"
                return [(layer_name, module)], f"{layer_name} (last activation)"
        raise RuntimeError(f"No convolutional activation layer found for {backbone}.")

    if backbone in {"resnet50", "resnet152"}:
        layer3 = model.cnn_backbone[-2][-1]
        layer4 = model.cnn_backbone[-1][-1]
        return (
            [
                ("cnn_backbone.layer3[-1]", layer3),
                ("cnn_backbone.layer4[-1]", layer4),
            ],
            "cnn_backbone.layer3[-1] + layer4[-1] (equal fusion)",
        )

    if backbone == "efficientnet_b3":
        layer = model.cnn_backbone[-1]
        return [("cnn_backbone.features[-1]", layer)], "cnn_backbone.features[-1]"

    raise ValueError(f"Unsupported Grad-CAM backbone: {backbone}")


def _normalized_cam(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return _normalize_map(cam, "A Grad-CAM layer map")


def _normalize_map(cam: torch.Tensor, description: str) -> torch.Tensor:
    if not bool(torch.isfinite(cam).all().item()):
        raise RuntimeError(f"{description} contained invalid values.")
    minimum = cam.min()
    dynamic_range = cam.max() - minimum
    if dynamic_range.item() <= 1e-12:
        # A constant map contains no spatially discriminative signal. Returning
        # a blank map preserves that meaning without crashing the evaluation.
        return torch.zeros_like(cam)
    return (cam - minimum) / dynamic_range


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    restored = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    restored = restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (restored * 255.0).round().astype(np.uint8)
