"""Final Streamlit dashboard for the DL Tumor Classification research project.

Cumulative project UI:
- Predict MRI is the first tab.
- All 15 frozen pipelines are selectable (5 backbones x Phase 1 / 2A / 2B).
- Live Grad-CAM and full-pipeline SVM occlusion are available for the selected pipeline.
- Base-paper reproduction includes both the Phase 1 reproduction and the standardized
  Phase 2A 30-epoch follow-up.
- Model-card, hyperparameter-sensitivity, external-evaluation, formal-XAI-study,
  and frozen-15 comparison tabs are retained.

Scientific guardrails:
- The formal rated XAI study was performed on Phase 1 ResNet152 only.
- Live XAI for any other frozen pipeline is qualitative demonstration only.
- material_clean is a post-hoc test-subset analysis; it does not repair training leakage.
- BDNeuro balanced is a nested subset of the primary exact-overlap-clean cohort.
- Web-40 is an exploratory source-verified stress test, not clinical validation.
"""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from gui.gradcam import (
    GradCAMResult,
    colorize_heatmap,
    generate_gradcam,
    overlay_heatmap,
)

from gui.occlusion import (
    OcclusionResult,
    generate_svm_occlusion,
    overlay_occlusion,
)

from gui.results_loader import (
    CLASS_DISPLAY,
    CLASS_NAMES,
    confusion_frame,
    load_json,
)

from gui.model_factory import CheckpointCNNLSTM
from src.data_pipeline_torch import eval_transform


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)


# =============================================================================
# DISPLAY / PIPELINE METADATA
# =============================================================================

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

STAGE_ORDER = (
    "Phase 1",
    "Phase 2A",
    "Phase 2B",
)

DEFAULT_CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str

    stage: str

    backbone: str

    display_name: str

    variant: str

    selected_epoch: int

    checkpoint_state: str

    checkpoint: Path

    scaler: Path

    pca: Path

    svm: Path

    final_results: Path

    training_results: Path

    learning_rate: float

    lstm_hidden_size: int

    dropout: float

    final_representative: bool = False

    formal_xai_model: bool = False


    def required_for_prediction(
        self,
    ) -> tuple[Path, ...]:

        return (
            self.checkpoint,
            self.scaler,
            self.pca,
            self.svm,
        )


    def missing_prediction_files(
        self,
    ) -> list[Path]:

        return [

            path

            for path
            in self.required_for_prediction()

            if not path.is_file()
        ]


    def prediction_ready(
        self,
    ) -> bool:

        return not self.missing_prediction_files()


@dataclass(frozen=True)
class BundleArtifacts:

    backbone: str

    display_name: str

    pipeline_id: str


@dataclass
class LoadedPipelineBundle:

    artifacts: BundleArtifacts

    spec: PipelineSpec

    model: torch.nn.Module

    scaler: Any

    pca: Any

    svm: Any

    class_to_idx: dict[str, int]

    device: torch.device


@dataclass
class LivePrediction:

    pipeline_id: str

    stage: str

    backbone: str

    variant: str

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


# =============================================================================
# BUILD THE 15 FROZEN PIPELINES
# =============================================================================

def _phase1_spec(
    backbone: str,
    epoch: int,
) -> PipelineSpec:

    root = (
        PROJECT_ROOT
        / "experiments"
        / backbone
    )

    return PipelineSpec(

        pipeline_id=(
            f"phase1::{backbone}"
        ),

        stage="Phase 1",

        backbone=backbone,

        display_name=(
            DISPLAY_NAMES[
                backbone
            ]
        ),

        variant="early_stopping",

        selected_epoch=epoch,

        checkpoint_state="best_model",

        checkpoint=(
            root
            / "models"
            / "best_model.pth"
        ),

        scaler=(
            root
            / "pca"
            / "standard_scaler.joblib"
        ),

        pca=(
            root
            / "pca"
            / "pca_model.joblib"
        ),

        svm=(
            root
            / "models"
            / "svm.joblib"
        ),

        final_results=(
            root
            / "results"
            / "final_results.json"
        ),

        training_results=(
            root
            / "results"
            / "training_results.json"
        ),

        learning_rate=1e-4,

        lstm_hidden_size=256,

        dropout=0.3,

        formal_xai_model=(
            backbone
            == "resnet152"
        ),
    )


def _phase2a_spec(
    backbone: str,
    epoch: int,
) -> PipelineSpec:

    root = (
        PROJECT_ROOT
        / "phase2"
        / "experiments_30epoch"
        / backbone
    )

    state = (
        "best_val_within_30"
    )

    eval_root = (
        root
        / "evaluations"
        / state
    )

    return PipelineSpec(

        pipeline_id=(
            f"phase2a::{backbone}"
        ),

        stage="Phase 2A",

        backbone=backbone,

        display_name=(
            DISPLAY_NAMES[
                backbone
            ]
        ),

        variant="baseline",

        selected_epoch=epoch,

        checkpoint_state=state,

        checkpoint=(
            root
            / "models"
            / "best_val_within_30.pth"
        ),

        scaler=(
            eval_root
            / "pca"
            / "standard_scaler.joblib"
        ),

        pca=(
            eval_root
            / "pca"
            / "pca_model.joblib"
        ),

        svm=(
            eval_root
            / "models"
            / "svm.joblib"
        ),

        final_results=(
            eval_root
            / "results"
            / "final_results.json"
        ),

        training_results=(
            root
            / "results"
            / "training_results.json"
        ),

        learning_rate=1e-4,

        lstm_hidden_size=256,

        dropout=0.3,

        final_representative=(
            backbone
            == "resnet152"
        ),
    )


def _phase2b_spec(
    backbone: str,
    variant: str,
    state: str,
    epoch: int,
    learning_rate: float,
    lstm_hidden_size: int,
    dropout: float,
) -> PipelineSpec:

    root = (
        PROJECT_ROOT
        / "phase2"
        / "hyperparameter_dl"
        / backbone
        / variant
    )

    eval_root = (
        root
        / "evaluations"
        / state
    )

    checkpoint_name = (

        "best_val_within_30.pth"

        if state
        == "best_val_within_30"

        else "epoch_30_model.pth"
    )

    return PipelineSpec(

        pipeline_id=(
            f"phase2b::{backbone}"
        ),

        stage="Phase 2B",

        backbone=backbone,

        display_name=(
            DISPLAY_NAMES[
                backbone
            ]
        ),

        variant=variant,

        selected_epoch=epoch,

        checkpoint_state=state,

        checkpoint=(
            root
            / "models"
            / checkpoint_name
        ),

        scaler=(
            eval_root
            / "pca"
            / "standard_scaler.joblib"
        ),

        pca=(
            eval_root
            / "pca"
            / "pca_model.joblib"
        ),

        svm=(
            eval_root
            / "models"
            / "svm.joblib"
        ),

        final_results=(
            eval_root
            / "results"
            / "final_results.json"
        ),

        training_results=(
            root
            / "results"
            / "training_results.json"
        ),

        learning_rate=learning_rate,

        lstm_hidden_size=lstm_hidden_size,

        dropout=dropout,
    )


PIPELINE_SPECS: dict[
    str,
    PipelineSpec,
] = {}


for (
    _backbone,
    _epoch,
) in {

    "vgg16":
        16,

    "vgg19":
        18,

    "resnet50":
        7,

    "resnet152":
        8,

    "efficientnet_b3":
        6,

}.items():

    _spec = _phase1_spec(
        _backbone,
        _epoch,
    )

    PIPELINE_SPECS[
        _spec.pipeline_id
    ] = _spec


for (
    _backbone,
    _epoch,
) in {

    "vgg16":
        30,

    "vgg19":
        30,

    "resnet50":
        30,

    "resnet152":
        10,

    "efficientnet_b3":
        27,

}.items():

    _spec = _phase2a_spec(
        _backbone,
        _epoch,
    )

    PIPELINE_SPECS[
        _spec.pipeline_id
    ] = _spec


for _args in [

    (
        "vgg16",
        "lr_5e-4",
        "best_val_within_30",
        27,
        5e-4,
        256,
        0.3,
    ),

    (
        "vgg19",
        "dropout_0.5",
        "epoch_30",
        30,
        1e-4,
        256,
        0.5,
    ),

    (
        "resnet50",
        "dropout_0.2",
        "best_val_within_30",
        28,
        1e-4,
        256,
        0.2,
    ),

    (
        "resnet152",
        "lstm_512",
        "best_val_within_30",
        26,
        1e-4,
        512,
        0.3,
    ),

    (
        "efficientnet_b3",
        "lstm_512",
        "best_val_within_30",
        29,
        1e-4,
        512,
        0.3,
    ),

]:

    _spec = _phase2b_spec(
        *_args
    )

    PIPELINE_SPECS[
        _spec.pipeline_id
    ] = _spec


def pipeline_id_for(
    stage: str,
    backbone: str,
) -> str:

    prefix = {

        "Phase 1":
            "phase1",

        "Phase 2A":
            "phase2a",

        "Phase 2B":
            "phase2b",

    }[
        stage
    ]

    return (
        f"{prefix}::{backbone}"
    )


# =============================================================================
# VERIFIED PROJECT RESULTS
# =============================================================================

BASE_PAPER_REPRODUCTION = pd.DataFrame(
    [
        {
            "Backbone":
                "VGG16",

            "Best epoch":
                16,

            "NN val accuracy (%)":
                94.58,

            "SVM val accuracy (%)":
                94.93,

            "NN test accuracy (%)":
                93.67,

            "Hybrid test accuracy (%)":
                94.97,

            "Test macro F1 (%)":
                94.68,

            "SVM gain (pp)":
                1.30,
        },

        {
            "Backbone":
                "VGG19",

            "Best epoch":
                18,

            "NN val accuracy (%)":
                94.58,

            "SVM val accuracy (%)":
                94.49,

            "NN test accuracy (%)":
                93.14,

            "Hybrid test accuracy (%)":
                93.52,

            "Test macro F1 (%)":
                93.10,

            "SVM gain (pp)":
                0.38,
        },

        {
            "Backbone":
                "ResNet50",

            "Best epoch":
                7,

            "NN val accuracy (%)":
                97.73,

            "SVM val accuracy (%)":
                97.55,

            "NN test accuracy (%)":
                97.33,

            "Hybrid test accuracy (%)":
                98.09,

            "Test macro F1 (%)":
                98.01,

            "SVM gain (pp)":
                0.76,
        },

        {
            "Backbone":
                "ResNet152",

            "Best epoch":
                8,

            "NN val accuracy (%)":
                98.25,

            "SVM val accuracy (%)":
                98.25,

            "NN test accuracy (%)":
                98.17,

            "Hybrid test accuracy (%)":
                98.40,

            "Test macro F1 (%)":
                98.32,

            "SVM gain (pp)":
                0.23,
        },

        {
            "Backbone":
                "EfficientNet-B3",

            "Best epoch":
                6,

            "NN val accuracy (%)":
                98.43,

            "SVM val accuracy (%)":
                98.51,

            "NN test accuracy (%)":
                98.02,

            "Hybrid test accuracy (%)":
                98.47,

            "Test macro F1 (%)":
                98.41,

            "SVM gain (pp)":
                0.46,
        },
    ]
)


PHASE2A_REPRODUCTION = pd.DataFrame(
    [
        {
            "Backbone":
                "VGG16",

            "Selected epoch":
                30,

            "Phase 1 SVM val (%)":
                94.93,

            "Phase 2A SVM val (%)":
                95.888,

            "Phase 2A test (%)":
                96.186,

            "material_clean (%)":
                95.853,
        },

        {
            "Backbone":
                "VGG19",

            "Selected epoch":
                30,

            "Phase 1 SVM val (%)":
                94.49,

            "Phase 2A SVM val (%)":
                95.626,

            "Phase 2A test (%)":
                95.271,

            "material_clean (%)":
                95.099,
        },

        {
            "Backbone":
                "ResNet50",

            "Selected epoch":
                30,

            "Phase 1 SVM val (%)":
                97.55,

            "Phase 2A SVM val (%)":
                98.688,

            "Phase 2A test (%)":
                98.932,

            "material_clean (%)":
                98.963,
        },

        {
            "Backbone":
                "ResNet152",

            "Selected epoch":
                10,

            "Phase 1 SVM val (%)":
                98.25,

            "Phase 2A SVM val (%)":
                98.600,

            "Phase 2A test (%)":
                98.703,

            "material_clean (%)":
                98.869,
        },

        {
            "Backbone":
                "EfficientNet-B3",

            "Selected epoch":
                27,

            "Phase 1 SVM val (%)":
                98.51,

            "Phase 2A SVM val (%)":
                99.300,

            "Phase 2A test (%)":
                99.314,

            "material_clean (%)":
                99.246,
        },
    ]
)


FROZEN_15_ROWS = [

    {
        "Backbone":
            "VGG16",

        "Stage":
            "Phase 1",

        "Variant":
            "early_stopping",

        "State":
            "best_model",

        "Epoch":
            16,

        "Internal val (%)":
            94.925634,

        "BDNeuro full (%)":
            77.609682,

        "BDNeuro balanced (%)":
            77.184466,

        "Web-40 (%)":
            25.0,
    },

    {
        "Backbone":
            "VGG19",

        "Stage":
            "Phase 1",

        "Variant":
            "early_stopping",

        "State":
            "best_model",

        "Epoch":
            18,

        "Internal val (%)":
            94.488189,

        "BDNeuro full (%)":
            75.756430,

        "BDNeuro balanced (%)":
            77.487864,

        "Web-40 (%)":
            35.0,
    },

    {
        "Backbone":
            "ResNet50",

        "Stage":
            "Phase 1",

        "Variant":
            "early_stopping",

        "State":
            "best_model",

        "Epoch":
            7,

        "Internal val (%)":
            97.550306,

        "BDNeuro full (%)":
            81.089259,

        "BDNeuro balanced (%)":
            83.737864,

        "Web-40 (%)":
            30.0,
    },

    {
        "Backbone":
            "ResNet152",

        "Stage":
            "Phase 1",

        "Variant":
            "early_stopping",

        "State":
            "best_model",

        "Epoch":
            8,

        "Internal val (%)":
            98.250219,

        "BDNeuro full (%)":
            81.278366,

        "BDNeuro balanced (%)":
            85.133495,

        "Web-40 (%)":
            35.0,
    },

    {
        "Backbone":
            "EfficientNet-B3",

        "Stage":
            "Phase 1",

        "Variant":
            "early_stopping",

        "State":
            "best_model",

        "Epoch":
            6,

        "Internal val (%)":
            98.512686,

        "BDNeuro full (%)":
            83.093797,

        "BDNeuro balanced (%)":
            85.436893,

        "Web-40 (%)":
            40.0,
    },

    {
        "Backbone":
            "VGG16",

        "Stage":
            "Phase 2A",

        "Variant":
            "baseline",

        "State":
            "best_val_within_30",

        "Epoch":
            30,

        "Internal val (%)":
            95.888014,

        "BDNeuro full (%)":
            80.068079,

        "BDNeuro balanced (%)":
            80.703883,

        "Web-40 (%)":
            30.0,
    },

    {
        "Backbone":
            "VGG19",

        "Stage":
            "Phase 2A",

        "Variant":
            "baseline",

        "State":
            "best_val_within_30",

        "Epoch":
            30,

        "Internal val (%)":
            95.625547,

        "BDNeuro full (%)":
            79.273828,

        "BDNeuro balanced (%)":
            81.128641,

        "Web-40 (%)":
            37.5,
    },

    {
        "Backbone":
            "ResNet50",

        "Stage":
            "Phase 2A",

        "Variant":
            "baseline",

        "State":
            "best_val_within_30",

        "Epoch":
            30,

        "Internal val (%)":
            98.687664,

        "BDNeuro full (%)":
            81.732224,

        "BDNeuro balanced (%)":
            86.771845,

        "Web-40 (%)":
            42.5,
    },

    {
        "Backbone":
            "ResNet152",

        "Stage":
            "Phase 2A",

        "Variant":
            "baseline",

        "State":
            "best_val_within_30",

        "Epoch":
            10,

        "Internal val (%)":
            98.600175,

        "BDNeuro full (%)":
            88.313162,

        "BDNeuro balanced (%)":
            90.473301,

        "Web-40 (%)":
            37.5,
    },

    {
        "Backbone":
            "EfficientNet-B3",

        "Stage":
            "Phase 2A",

        "Variant":
            "baseline",

        "State":
            "best_val_within_30",

        "Epoch":
            27,

        "Internal val (%)":
            99.300087,

        "BDNeuro full (%)":
            86.649017,

        "BDNeuro balanced (%)":
            87.864078,

        "Web-40 (%)":
            42.5,
    },

    {
        "Backbone":
            "VGG16",

        "Stage":
            "Phase 2B",

        "Variant":
            "lr_5e-4",

        "State":
            "best_val_within_30",

        "Epoch":
            27,

        "Internal val (%)":
            96.500437,

        "BDNeuro full (%)":
            77.571861,

        "BDNeuro balanced (%)":
            78.155340,

        "Web-40 (%)":
            30.0,
    },

    {
        "Backbone":
            "VGG19",

        "Stage":
            "Phase 2B",

        "Variant":
            "dropout_0.5",

        "State":
            "epoch_30",

        "Epoch":
            30,

        "Internal val (%)":
            95.538058,

        "BDNeuro full (%)":
            75.000000,

        "BDNeuro balanced (%)":
            77.002427,

        "Web-40 (%)":
            35.0,
    },

    {
        "Backbone":
            "ResNet50",

        "Stage":
            "Phase 2B",

        "Variant":
            "dropout_0.2",

        "State":
            "best_val_within_30",

        "Epoch":
            28,

        "Internal val (%)":
            99.125109,

        "BDNeuro full (%)":
            86.043873,

        "BDNeuro balanced (%)":
            87.864078,

        "Web-40 (%)":
            40.0,
    },

    {
        "Backbone":
            "ResNet152",

        "Stage":
            "Phase 2B",

        "Variant":
            "lstm_512",

        "State":
            "best_val_within_30",

        "Epoch":
            26,

        "Internal val (%)":
            99.300087,

        "BDNeuro full (%)":
            86.535552,

        "BDNeuro balanced (%)":
            89.805825,

        "Web-40 (%)":
            35.0,
    },

    {
        "Backbone":
            "EfficientNet-B3",

        "Stage":
            "Phase 2B",

        "Variant":
            "lstm_512",

        "State":
            "best_val_within_30",

        "Epoch":
            29,

        "Internal val (%)":
            99.387577,

        "BDNeuro full (%)":
            86.573374,

        "BDNeuro balanced (%)":
            88.834951,

        "Web-40 (%)":
            35.0,
    },
]


FINAL_REPRESENTATIVE_RESULTS = {

    "internal_val_accuracy":
        98.600175,

    "internal_val_macro_f1":
        98.545056,

    "internal_test_accuracy":
        98.703,

    "material_clean_accuracy":
        98.869,

    "material_clean_macro_f1":
        98.788,

    "bdneuro_full_accuracy":
        88.313162,

    "bdneuro_full_macro_f1":
        87.854138,

    "bdneuro_balanced_accuracy":
        90.473301,

    "bdneuro_balanced_macro_f1":
        90.496811,

    "web40_accuracy":
        37.5,

    "web40_macro_f1":
        31.664963,
}


WEB40_CLASS_RECALL = pd.DataFrame(
    {
        "Class": [
            "Glioma",
            "Meningioma",
            "No tumour",
            "Pituitary",
        ],

        "Mean recall (%)": [
            12.67,
            30.67,
            90.67,
            7.33,
        ],

        "Mean predicted count / 40": [
            2.07,
            8.33,
            28.27,
            1.33,
        ],
    }
)


HYPERPARAMETER_SENSITIVITY = pd.DataFrame(
    {
        "Backbone": [
            "VGG16",
            "VGG19",
            "ResNet50",
            "ResNet152",
            "EfficientNet-B3",
        ],

        "2A baseline": [
            95.45,
            95.19,
            98.43,
            98.78,
            99.30,
        ],

        "lr_1e-5": [
            90.20,
            89.94,
            96.94,
            97.03,
            96.41,
        ],

        "lr_5e-5": [
            95.28,
            94.66,
            98.60,
            98.95,
            98.95,
        ],

        "lr_5e-4": [
            95.98,
            95.54,
            94.75,
            93.96,
            98.43,
        ],

        "lstm_128": [
            95.80,
            94.66,
            98.43,
            98.95,
            99.21,
        ],

        "lstm_512": [
            95.89,
            95.89,
            99.04,
            99.21,
            99.39,
        ],

        "dropout_0.2": [
            95.63,
            94.84,
            99.13,
            98.60,
            98.95,
        ],

        "dropout_0.5": [
            95.54,
            95.36,
            98.34,
            98.86,
            99.13,
        ],
    }
)


PHASE2B_SELECTED = pd.DataFrame(
    [
        {
            "Backbone":
                "VGG16",

            "Selected variant":
                "lr_5e-4",

            "State":
                "best_val_within_30",

            "Epoch":
                27,

            "Full-hybrid val accuracy (%)":
                96.500437,
        },

        {
            "Backbone":
                "VGG19",

            "Selected variant":
                "dropout_0.5",

            "State":
                "epoch_30",

            "Epoch":
                30,

            "Full-hybrid val accuracy (%)":
                95.538058,
        },

        {
            "Backbone":
                "ResNet50",

            "Selected variant":
                "dropout_0.2",

            "State":
                "best_val_within_30",

            "Epoch":
                28,

            "Full-hybrid val accuracy (%)":
                99.125109,
        },

        {
            "Backbone":
                "ResNet152",

            "Selected variant":
                "lstm_512",

            "State":
                "best_val_within_30",

            "Epoch":
                26,

            "Full-hybrid val accuracy (%)":
                99.300087,
        },

        {
            "Backbone":
                "EfficientNet-B3",

            "Selected variant":
                "lstm_512",

            "State":
                "best_val_within_30",

            "Epoch":
                29,

            "Full-hybrid val accuracy (%)":
                99.387577,
        },
    ]
)


GRADCAM_RATINGS = pd.DataFrame(
    [
        {
            "Grad-CAM map":
                "Layer 3",

            "Good":
                1,

            "Partial":
                9,

            "Poor":
                5,
        },

        {
            "Grad-CAM map":
                "Layer 4",

            "Good":
                3,

            "Partial":
                2,

            "Poor":
                10,
        },

        {
            "Grad-CAM map":
                "Fused",

            "Good":
                2,

            "Partial":
                6,

            "Poor":
                7,
        },
    ]
)


OCCLUSION_XAI_RESULTS = pd.DataFrame(
    [
        {
            "Measure":
                "Tumour-case rating",

            "Result":
                "2 Good / 2 Partial / 5 Poor",
        },

        {
            "Measure":
                "Total occlusion windows",

            "Result":
                "1,859",
        },

        {
            "Measure":
                "Answer-flipping patches",

            "Result":
                "58 / 1,859 (3.12%)",
        },

        {
            "Measure":
                "Cases with ≥1 prediction flip",

            "Result":
                "5 / 11",
        },

        {
            "Measure":
                "Misclassified Te-pi_0078",

            "Result":
                "38 / 169 patches changed prediction",
        },
    ]
)


# =============================================================================
# GENERIC PIPELINE LOADING / INFERENCE
# =============================================================================

def _choose_device() -> torch.device:

    return torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


def _validated_class_mapping(
    value: Any,
) -> dict[str, int]:

    mapping = (

        value

        if isinstance(
            value,
            dict,
        )

        else DEFAULT_CLASS_TO_IDX
    )

    normalized = {

        str(key):
            int(index)

        for key, index
        in mapping.items()
    }

    if (
        normalized
        != DEFAULT_CLASS_TO_IDX
    ):

        raise RuntimeError(
            "Unexpected checkpoint class_to_idx mapping. "
            f"Expected {DEFAULT_CLASS_TO_IDX}, "
            f"found {normalized}."
        )

    return normalized


def _unload_pipeline_bundle() -> None:

    bundle = (
        st.session_state
        .pop(
            "loaded_pipeline_bundle",
            None,
        )
    )

    st.session_state.pop(
        "loaded_pipeline_id",
        None,
    )

    if bundle is not None:

        try:

            bundle.model.to(
                "cpu"
            )

        except Exception:

            pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def _load_pipeline_bundle(
    spec: PipelineSpec,
) -> LoadedPipelineBundle:

    current_id = (
        st.session_state
        .get(
            "loaded_pipeline_id"
        )
    )

    current_bundle = (
        st.session_state
        .get(
            "loaded_pipeline_bundle"
        )
    )

    if (
        current_id
        == spec.pipeline_id

        and

        current_bundle
        is not None
    ):

        return current_bundle

    _unload_pipeline_bundle()

    missing = (
        spec
        .missing_prediction_files()
    )

    if missing:

        formatted = "\n".join(

            f"- {path}"

            for path
            in missing
        )

        raise FileNotFoundError(
            f"Missing prediction artifacts for "
            f"{spec.stage} {spec.display_name}:\n"
            f"{formatted}"
        )

    device = (
        _choose_device()
    )

    checkpoint = torch.load(

        spec.checkpoint,

        map_location=device,

        weights_only=False,
    )

    saved_backbone = (
        checkpoint.get(
            "backbone",
            spec.backbone,
        )
    )

    if (
        saved_backbone
        != spec.backbone
    ):

        raise RuntimeError(
            f"Checkpoint backbone is "
            f"{saved_backbone!r}; "
            f"expected {spec.backbone!r}."
        )

    model = CheckpointCNNLSTM(

        backbone_name=(
            spec.backbone
        ),

        lstm_hidden_size=int(
            checkpoint.get(
                "lstm_hidden_size",
                spec.lstm_hidden_size,
            )
        ),

        dense_features=int(
            checkpoint.get(
                "dense_features",
                128,
            )
        ),

        num_classes=int(
            checkpoint.get(
                "num_classes",
                4,
            )
        ),

        dropout=float(
            checkpoint.get(
                "dropout",
                spec.dropout,
            )
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],

        strict=True,
    )

    model.to(
        device
    ).eval()

    scaler = joblib.load(
        spec.scaler
    )

    pca = joblib.load(
        spec.pca
    )

    svm = joblib.load(
        spec.svm
    )

    class_to_idx = (
        _validated_class_mapping(
            checkpoint.get(
                "class_to_idx"
            )
        )
    )

    if (
        [
            int(value)

            for value
            in svm.classes_
        ]
        != [
            0,
            1,
            2,
            3,
        ]
    ):

        raise RuntimeError(
            f"Unexpected SVM class order: "
            f"{svm.classes_}"
        )

    dense_features = int(
        model.dense.out_features
    )

    if (
        getattr(
            scaler,
            "n_features_in_",
            dense_features,
        )
        != dense_features
    ):

        raise RuntimeError(
            "Scaler input dimension does not "
            "match model dense features."
        )

    if (
        getattr(
            pca,
            "n_features_in_",
            dense_features,
        )
        != dense_features
    ):

        raise RuntimeError(
            "PCA input dimension does not "
            "match scaler/model output."
        )

    bundle = LoadedPipelineBundle(

        artifacts=BundleArtifacts(

            backbone=(
                spec.backbone
            ),

            display_name=(
                spec.display_name
            ),

            pipeline_id=(
                spec.pipeline_id
            ),
        ),

        spec=spec,

        model=model,

        scaler=scaler,

        pca=pca,

        svm=svm,

        class_to_idx=(
            class_to_idx
        ),

        device=device,
    )

    st.session_state[
        "loaded_pipeline_id"
    ] = spec.pipeline_id

    st.session_state[
        "loaded_pipeline_bundle"
    ] = bundle

    return bundle


def _prepare_image(
    image: Image.Image,
) -> torch.Tensor:

    image = (
        ImageOps
        .exif_transpose(
            image
        )
        .convert(
            "RGB"
        )
    )

    if min(
        image.size
    ) < 32:

        raise ValueError(
            "The image is too small. "
            "Both dimensions must be at least 32 pixels."
        )

    tensor = eval_transform(
        image
    )

    if (
        not isinstance(
            tensor,
            torch.Tensor,
        )

        or

        tensor.ndim
        != 3
    ):

        raise RuntimeError(
            "eval_transform did not return "
            "a [C, H, W] tensor."
        )

    return tensor.unsqueeze(
        0
    )


def _softmax_np(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    shifted = (
        values
        - np.max(
            values
        )
    )

    exp = np.exp(
        shifted
    )

    return (
        exp
        / exp.sum()
    )


def _predict_pipeline(
    image: Image.Image,
    spec: PipelineSpec,
) -> LivePrediction:

    started = (
        time.perf_counter()
    )

    bundle = (
        _load_pipeline_bundle(
            spec
        )
    )

    original_size = (
        image.size
    )

    tensor = (
        _prepare_image(
            image
        )
        .to(
            bundle.device
        )
    )

    with torch.inference_mode():

        debug = bundle.model(
            tensor,
            return_debug=True,
        )

        deep_features = (
            debug[
                "deep_features"
            ]
        )

        logits = (
            bundle.model
            .classifier(
                deep_features
            )
        )

    nn_values = (

        torch.softmax(
            logits,
            dim=1,
        )[0]

        .detach()

        .cpu()

        .numpy()
    )

    deep_numpy = (

        deep_features

        .detach()

        .cpu()

        .numpy()
    )

    scaled = (
        bundle.scaler
        .transform(
            deep_numpy
        )
    )

    pca_features = (
        bundle.pca
        .transform(
            scaled
        )
    )

    svm_prediction = int(

        bundle.svm
        .predict(
            pca_features
        )[0]
    )

    decision = np.asarray(

        bundle.svm
        .decision_function(
            pca_features
        ),

        dtype=float,
    )

    decision = (
        decision
        .reshape(
            1,
            -1,
        )[0]
    )

    svm_classes = [

        int(value)

        for value
        in bundle.svm.classes_
    ]

    relative = (
        _softmax_np(
            decision
        )
    )

    idx_to_key = {

        index:
            key

        for key, index
        in bundle.class_to_idx.items()
    }

    nn_prediction = int(
        np.argmax(
            nn_values
        )
    )

    nn_key = (
        idx_to_key[
            nn_prediction
        ]
    )

    svm_key = (
        idx_to_key[
            svm_prediction
        ]
    )

    nn_scores = {

        CLASS_DISPLAY[
            idx_to_key[
                index
            ]
        ]:
            float(
                nn_values[
                    index
                ]
            )

        for index
        in sorted(
            idx_to_key
        )
    }

    svm_scores = {

        CLASS_DISPLAY[
            idx_to_key[
                class_index
            ]
        ]:
            float(
                score
            )

        for (
            class_index,
            score,
        )
        in zip(
            svm_classes,
            relative,
            strict=True,
        )
    }

    return LivePrediction(

        pipeline_id=(
            spec.pipeline_id
        ),

        stage=(
            spec.stage
        ),

        backbone=(
            spec.backbone
        ),

        variant=(
            spec.variant
        ),

        svm_class_key=(
            svm_key
        ),

        svm_class_name=(
            CLASS_DISPLAY[
                svm_key
            ]
        ),

        svm_relative_scores=(
            svm_scores
        ),

        nn_class_key=(
            nn_key
        ),

        nn_class_name=(
            CLASS_DISPLAY[
                nn_key
            ]
        ),

        nn_scores=(
            nn_scores
        ),

        models_agree=(
            svm_prediction
            == nn_prediction
        ),

        elapsed_seconds=(
            time.perf_counter()
            - started
        ),

        device=str(
            bundle.device
        ),

        image_size=(
            original_size
        ),

        deep_feature_dimension=int(
            deep_numpy.shape[
                1
            ]
        ),

        pca_feature_dimension=int(
            pca_features.shape[
                1
            ]
        ),
    )


def _explain_pipeline_gradcam(
    image: Image.Image,
    spec: PipelineSpec,
    target_class_key: str,
) -> GradCAMResult:

    bundle = (
        _load_pipeline_bundle(
            spec
        )
    )

    tensor = (

        _prepare_image(
            image
        )

        .to(
            bundle.device
        )
    )

    return generate_gradcam(

        bundle,

        tensor,

        target_class_key,
    )


def _explain_pipeline_occlusion(
    image: Image.Image,
    spec: PipelineSpec,
    target_class_key: str,
) -> OcclusionResult:

    bundle = (
        _load_pipeline_bundle(
            spec
        )
    )

    tensor = (

        _prepare_image(
            image
        )

        .to(
            bundle.device
        )
    )

    return generate_svm_occlusion(

        bundle,

        tensor,

        target_class_key,

        patch_size=32,

        stride=16,

        batch_size=8,
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _open_image(
    raw: bytes,
) -> Image.Image:

    if (
        len(
            raw
        )
        > 25
        * 1024
        * 1024
    ):

        raise ValueError(
            "The uploaded image exceeds the 25 MB limit."
        )

    try:

        image = Image.open(
            BytesIO(
                raw
            )
        )

        image.load()

        return (

            ImageOps
            .exif_transpose(
                image
            )
            .convert(
                "RGB"
            )
        )

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:

        raise ValueError(
            "The uploaded file is not a readable image."
        ) from error


def _relative_path(
    path: Path,
) -> str:

    try:

        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except ValueError:

        return str(
            path
        )


def _fraction_percent(
    value,
    digits: int = 2,
) -> str:

    if value is None:

        return "—"

    return (
        f"{float(value) * 100:.{digits}f}%"
    )


def _stored_percent(
    value,
    digits: int = 2,
) -> str:

    if value is None:

        return "—"

    number = float(
        value
    )

    if (
        abs(
            number
        )
        <= 1.5
    ):

        number *= 100

    return (
        f"{number:.{digits}f}%"
    )


def _to_percent(
    value,
):

    if value is None:

        return None

    return round(
        float(value)
        * 100,

        2,
    )


def _frozen_row(
    spec: PipelineSpec,
) -> dict:

    for row in FROZEN_15_ROWS:

        if (
            row[
                "Backbone"
            ]
            == spec.display_name

            and

            row[
                "Stage"
            ]
            == spec.stage
        ):

            return row

    raise RuntimeError(
        "Selected pipeline is missing "
        "from FROZEN_15_ROWS."
    )


def _score_chart(
    scores: dict[str, float],
    title: str,
):

    frame = pd.DataFrame(
        {
            "Class":
                list(
                    scores.keys()
                ),

            "Score":
                [
                    value
                    * 100

                    for value
                    in scores.values()
                ],
        }
    )

    return (

        alt.Chart(
            frame
        )

        .mark_bar(
            cornerRadiusTopRight=5,
            cornerRadiusBottomRight=5,
        )

        .encode(

            x=alt.X(
                "Score:Q",

                title=(
                    "Relative score (%)"
                ),

                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            y=alt.Y(
                "Class:N",

                sort="-x",

                title=None,
            ),

            color=alt.Color(
                "Class:N",

                legend=None,
            ),

            tooltip=[
                "Class:N",

                alt.Tooltip(
                    "Score:Q",

                    format=".2f",
                ),
            ],
        )

        .properties(
            title=title,
            height=230,
        )
    )


# =============================================================================
# PREDICTION / LIVE XAI
# =============================================================================

def _render_prediction(
    result: LivePrediction,
) -> None:

    st.subheader(
        "Prediction result"
    )

    (
        primary,
        secondary,
        agreement,
        runtime,
    ) = st.columns(
        4
    )

    primary.metric(
        "Final SVM prediction",
        result.svm_class_name,
    )

    secondary.metric(
        "Neural-head prediction",
        result.nn_class_name,
    )

    agreement.metric(
        "Classifier agreement",

        (
            "Yes"

            if result.models_agree

            else "No"
        ),
    )

    runtime.metric(
        "Inference time",
        f"{result.elapsed_seconds:.2f} s",
    )

    if result.models_agree:

        st.success(
            "Both classifier heads predict "
            f"**{result.svm_class_name}**."
        )

    else:

        st.warning(
            "The neural head and SVM disagree. "
            "The research pipeline's final classification is "
            f"**{result.svm_class_name}** from the SVM."
        )

    left, right = st.columns(
        2
    )

    with left:

        st.altair_chart(

            _score_chart(
                result.svm_relative_scores,
                "SVM relative class scores",
            ),

            use_container_width=True,
        )

        st.caption(
            "Relative decision-value scores, "
            "not calibrated probabilities. "
            "The standardized project SVMs use probability=False."
        )

    with right:

        st.altair_chart(

            _score_chart(
                result.nn_scores,
                "Neural-head softmax scores",
            ),

            use_container_width=True,
        )

        st.caption(
            "Neural softmax scores are model confidence values, "
            "not clinically calibrated probabilities."
        )

    with st.expander(
        "Inference details"
    ):

        st.write(
            {
                "pipeline_id":
                    result.pipeline_id,

                "stage":
                    result.stage,

                "backbone":
                    DISPLAY_NAMES[
                        result.backbone
                    ],

                "variant":
                    result.variant,

                "device":
                    result.device,

                "original_image_size":
                    (
                        f"{result.image_size[0]} "
                        f"× "
                        f"{result.image_size[1]}"
                    ),

                "deep_feature_dimension":
                    result.deep_feature_dimension,

                "pca_feature_dimension":
                    result.pca_feature_dimension,

                "pipeline":
                    (
                        "CNN → spatial sequence → LSTM → Dense128 "
                        "→ StandardScaler → PCA64 → linear SVM"
                    ),
            }
        )


def _render_gradcam(
    explanation: GradCAMResult,
    prediction: LivePrediction,
    spec: PipelineSpec,
    fingerprint: str,
) -> None:

    st.divider()

    st.subheader(
        "Explainability — Grad-CAM"
    )

    st.caption(
        f"{spec.display_name} · "
        f"{spec.stage} · "
        f"{spec.variant}"
    )

    if spec.formal_xai_model:

        st.success(
            "This is the model used in the project's "
            "formally rated Phase 1 XAI study."
        )

    else:

        st.info(
            "Live qualitative XAI demonstration. "
            "The formally rated XAI study was performed on "
            "Phase 1 ResNet152; this map is not an additional "
            "formal XAI experiment."
        )

    if prediction.models_agree:

        st.success(
            "The neural head and final SVM agree on "
            f"**{prediction.svm_class_name}**. "
            "Grad-CAM explains the differentiable neural "
            "component of the classification."
        )

    else:

        st.warning(
            "The neural head predicts "
            f"**{prediction.nn_class_name}**, while the SVM predicts "
            f"**{prediction.svm_class_name}**. "
            "Grad-CAM explains the neural head, "
            "not the external SVM directly."
        )

    opacity = st.slider(

        "Grad-CAM overlay strength",

        min_value=0.0,

        max_value=1.0,

        value=0.55,

        step=0.05,

        key=(
            "gradcam_opacity_"
            f"{spec.pipeline_id}_"
            f"{fingerprint}"
        ),
    )

    layer_heatmaps = getattr(
        explanation,
        "layer_heatmaps",
        {},
    )

    resnet_layers = {
        "cnn_backbone.layer3[-1]",
        "cnn_backbone.layer4[-1]",
    }

    if resnet_layers.issubset(
        layer_heatmaps
    ):

        layer3 = (
            layer_heatmaps[
                "cnn_backbone.layer3[-1]"
            ]
        )

        layer4 = (
            layer_heatmaps[
                "cnn_backbone.layer4[-1]"
            ]
        )

        maps = [
            (
                "Layer 3",
                "14×14 spatial map",
                layer3,
            ),

            (
                "Layer 4",
                "7×7 semantic map",
                layer4,
            ),

            (
                "Fused",
                "50% Layer 3 + 50% Layer 4",
                explanation.heatmap,
            ),
        ]

        (
            c0,
            c1,
            c2,
            c3,
        ) = st.columns(
            4
        )

        c0.image(
            explanation.model_input_rgb,
            caption="Model input",
        )

        for (
            column,
            (
                label,
                description,
                heatmap,
            ),
        ) in zip(
            (
                c1,
                c2,
                c3,
            ),
            maps,
            strict=True,
        ):

            column.image(

                overlay_heatmap(
                    explanation,
                    opacity,
                    heatmap,
                ),

                caption=(
                    f"{label} overlay · "
                    f"{description}"
                ),
            )

        with st.expander(
            "Compare heatmaps without MRI overlay"
        ):

            for (
                column,
                (
                    label,
                    description,
                    heatmap,
                ),
            ) in zip(
                st.columns(
                    3
                ),
                maps,
                strict=True,
            ):

                column.image(

                    colorize_heatmap(
                        heatmap
                    ),

                    caption=(
                        f"{label} · "
                        f"{description}"
                    ),
                )

    else:

        heatmap_rgb = (
            colorize_heatmap(
                explanation.heatmap
            )
        )

        overlay_rgb = (
            overlay_heatmap(
                explanation,
                opacity,
            )
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.image(
            explanation.model_input_rgb,
            caption="Model input",
        )

        c2.image(
            heatmap_rgb,
            caption="Grad-CAM contribution heatmap",
        )

        c3.image(
            overlay_rgb,
            caption=(
                "MRI overlay · "
                f"strength {opacity:.2f}"
            ),
        )

    st.caption(
        f"Grad-CAM time: "
        f"{explanation.elapsed_seconds:.2f} s · "
        "coarse contribution map only — "
        "not a tumour outline, lesion mask "
        "or segmentation boundary."
    )


def _render_occlusion(
    image: Image.Image,
    prediction: LivePrediction,
    spec: PipelineSpec,
    record: dict,
) -> None:

    st.divider()

    st.subheader(
        "Explainability — final-SVM occlusion sensitivity"
    )

    st.caption(
        f"{spec.display_name} · "
        f"{spec.stage} · "
        f"{spec.variant}"
    )

    st.write(
        "This perturbation test masks small image regions and reruns "
        "the complete selected CNN–LSTM → Dense128 → StandardScaler "
        "→ PCA64 → linear-SVM pipeline."
    )

    result: OcclusionResult | None = (
        record.get(
            "occlusion"
        )
    )

    error_text = (
        record.get(
            "occlusion_error"
        )
    )

    if result is None:

        if st.button(

            "Generate final-SVM occlusion map",

            key=(
                "occlusion_"
                f"{spec.pipeline_id}_"
                f"{record['fingerprint']}"
            ),

            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Running the complete frozen pipeline "
                    "across occluded regions…"
                ):

                    result = (
                        _explain_pipeline_occlusion(

                            image,

                            spec,

                            prediction
                            .svm_class_key,
                        )
                    )

                record[
                    "occlusion"
                ] = result

                record[
                    "occlusion_error"
                ] = None

            except Exception as error:

                error_text = str(
                    error
                )

                record[
                    "occlusion_error"
                ] = error_text

    if (
        error_text
        and result is None
    ):

        st.warning(
            "Occlusion sensitivity could not be generated: "
            f"{error_text}"
        )

        return

    if result is None:

        st.caption(
            "Optional. Occlusion is slower than Grad-CAM because "
            "the complete hybrid pipeline is rerun many times."
        )

        return

    opacity = st.slider(

        "Occlusion overlay strength",

        min_value=0.0,

        max_value=1.0,

        value=0.55,

        step=0.05,

        key=(
            "occlusion_opacity_"
            f"{spec.pipeline_id}_"
            f"{record['fingerprint']}"
        ),
    )

    heatmap_rgb = (
        colorize_heatmap(
            result.heatmap
        )
    )

    overlay_rgb = (
        overlay_occlusion(
            result,
            opacity,
        )
    )

    c1, c2, c3 = st.columns(
        3
    )

    c1.image(
        result.model_input_rgb,
        caption="Model input",
    )

    c2.image(
        heatmap_rgb,
        caption="Positive final-SVM support",
    )

    c3.image(
        overlay_rgb,
        caption=(
            "SVM occlusion overlay · "
            f"strength {opacity:.2f}"
        ),
    )

    details = st.columns(
        5
    )

    details[0].metric(
        "Target class",
        result.target_class_name,
    )

    details[1].metric(
        "Occlusion windows",
        result.occlusion_count,
    )

    details[2].metric(
        "Prediction flips",
        f"{result.prediction_flip_fraction * 100:.1f}%",
    )

    details[3].metric(
        "Max score drop",
        f"{result.maximum_score_drop:.4f}",
    )

    details[4].metric(
        "Runtime",
        f"{result.elapsed_seconds:.2f} s",
    )

    if spec.formal_xai_model:

        xai_note = (
            "This pipeline was used in the formal Phase 1 XAI study."
        )

    else:

        xai_note = (
            "This is a live qualitative demonstration; "
            "it was not part of the formally rated Phase 1 XAI protocol."
        )

    st.caption(
        f"Patch {result.patch_size}×"
        f"{result.patch_size} · "
        f"stride {result.stride}. "
        "Masked pixels are replaced by ImageNet channel means in "
        "normalized space. This is coarse sensitivity analysis, "
        "not tumour segmentation. "
        + xai_note
    )


# =============================================================================
# RESULT HELPERS
# =============================================================================

def _render_confusion(
    result: dict,
    split: str,
) -> None:

    frame = confusion_frame(
        result,
        split,
    )

    if frame is None:

        st.info(
            f"No {split} confusion matrix is stored "
            "for this pipeline."
        )

        return

    frame.index = [

        CLASS_DISPLAY[
            name
        ]

        for name
        in CLASS_NAMES
    ]

    frame.columns = [

        CLASS_DISPLAY[
            name
        ]

        for name
        in CLASS_NAMES
    ]

    frame.index.name = (
        "Actual class"
    )

    frame.columns.name = (
        "Predicted class"
    )

    st.dataframe(
        frame,
        use_container_width=True,
    )


def _render_training_curves(
    training: dict | None,
) -> None:

    history = (
        training
        or {}
    ).get(
        "history",
        {},
    )

    if not history:

        st.info(
            "Training-history data are unavailable."
        )

        return

    keys = (
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    )

    lengths = [

        len(
            history.get(
                key,
                [],
            )
        )

        for key
        in keys
    ]

    epoch_count = min(
        lengths,
        default=0,
    )

    if epoch_count == 0:

        st.info(
            "Training-history arrays are empty."
        )

        return

    frame = pd.DataFrame(
        {
            "Epoch":
                range(
                    1,
                    epoch_count
                    + 1,
                ),

            "Training loss":
                history[
                    "train_loss"
                ][:epoch_count],

            "Validation loss":
                history[
                    "val_loss"
                ][:epoch_count],

            "Training accuracy (%)":
                history[
                    "train_accuracy"
                ][:epoch_count],

            "Validation accuracy (%)":
                history[
                    "val_accuracy"
                ][:epoch_count],
        }
    )

    accuracy = frame[
        [
            "Epoch",
            "Training accuracy (%)",
            "Validation accuracy (%)",
        ]
    ].melt(

        id_vars="Epoch",

        var_name="Series",

        value_name="Value",
    )

    loss = frame[
        [
            "Epoch",
            "Training loss",
            "Validation loss",
        ]
    ].melt(

        id_vars="Epoch",

        var_name="Series",

        value_name="Value",
    )

    left, right = st.columns(
        2
    )

    with left:

        chart = (

            alt.Chart(
                accuracy
            )

            .mark_line(
                point=True
            )

            .encode(

                x=alt.X(
                    "Epoch:Q",

                    axis=alt.Axis(
                        tickMinStep=1
                    ),
                ),

                y=alt.Y(
                    "Value:Q",

                    title="Accuracy (%)",
                ),

                color="Series:N",
            )

            .properties(
                title="Accuracy",
                height=300,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with right:

        chart = (

            alt.Chart(
                loss
            )

            .mark_line(
                point=True
            )

            .encode(

                x=alt.X(
                    "Epoch:Q",

                    axis=alt.Axis(
                        tickMinStep=1
                    ),
                ),

                y=alt.Y(
                    "Value:Q",

                    title="Loss",
                ),

                color="Series:N",
            )

            .properties(
                title="Loss",
                height=300,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )


# =============================================================================
# BASE-PAPER REPRODUCTION
# =============================================================================
def _render_base_paper_reproduction() -> None:

    st.subheader(
        "Base-Paper Reproduction and Standardized Follow-up"
    )

    st.caption(
        "Base-paper benchmark → Phase 1 initial reproduction → "
        "Phase 2A standardized 30-epoch reproduction"
    )

    # =========================================================================
    # 1. BASE PAPER
    # =========================================================================

    st.markdown(
        "### 1. Base-paper architecture and reported results"
    )

    st.write(
        "The project was based on the 2026 Springer study by Bérété et al., "
        "which evaluated five pretrained CNN backbones within a hybrid "
        "CNN–LSTM–SVM framework for four-class brain MRI classification."
    )

    st.code(
        "MRI image\n"
        "   ↓\n"
        "Pretrained CNN backbone\n"
        "   ↓\n"
        "Spatial CNN feature map\n"
        "   ↓\n"
        "Feature-map sequence → LSTM\n"
        "   ↓\n"
        "BatchNorm → Dropout → Dense representation\n"
        "   ↓\n"
        "StandardScaler → PCA 64\n"
        "   ↓\n"
        "Linear SVM\n"
        "   ↓\n"
        "Glioma / Meningioma / No tumour / Pituitary",
        language=None,
    )

    st.markdown(
        "#### Results reported by the base paper — RMSprop"
    )

    base_paper_results = pd.DataFrame(
        [
            {
                "Backbone": "VGG16",
                "Accuracy (%)": 94.00,
                "Precision (%)": 94.00,
                "Recall (%)": 93.50,
                "F1-score (%)": 93.50,
            },
            {
                "Backbone": "VGG19",
                "Accuracy (%)": 96.00,
                "Precision (%)": 95.50,
                "Recall (%)": 95.25,
                "F1-score (%)": 95.25,
            },
            {
                "Backbone": "ResNet50",
                "Accuracy (%)": 99.00,
                "Precision (%)": 99.00,
                "Recall (%)": 98.75,
                "F1-score (%)": 98.75,
            },
            {
                "Backbone": "ResNet152",
                "Accuracy (%)": 99.00,
                "Precision (%)": 99.00,
                "Recall (%)": 99.00,
                "F1-score (%)": 99.00,
            },
            {
                "Backbone": "EfficientNet-B3",
                "Accuracy (%)": 98.00,
                "Precision (%)": 98.00,
                "Recall (%)": 97.75,
                "F1-score (%)": 98.00,
            },
        ]
    )

    st.dataframe(
        base_paper_results,
        hide_index=True,
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best paper backbone",
        "ResNet152",
    )

    c2.metric(
        "Paper accuracy",
        "99%",
    )

    c3.metric(
        "Paper F1-score",
        "99%",
    )

    st.info(
        "The base paper's headline result was "
        "**ResNet152 + LSTM + SVM with RMSprop**, reporting "
        "**99% accuracy, precision, recall and F1-score**."
    )


    # =========================================================================
    # 2. OUR IMPLEMENTATION
    # =========================================================================

    st.markdown(
        "### 2. Our reproduced hybrid pipeline"
    )

    st.code(
        "MRI\n"
        " ↓\n"
        "RGB conversion + 224×224 resize\n"
        " ↓\n"
        "ImageNet normalization\n"
        " ↓\n"
        "CNN backbone\n"
        " ↓\n"
        "Spatial feature map\n"
        " ↓\n"
        "Spatial positions reshaped as sequence\n"
        " ↓\n"
        "LSTM 256\n"
        " ↓\n"
        "BatchNorm → Dropout 0.3\n"
        " ↓\n"
        "Dense 128 + ReLU\n"
        " ↓\n"
        "StandardScaler → PCA 64\n"
        " ↓\n"
        "Linear SVM \n"
        " ↓\n"
        "Four-class prediction",
        language=None,
    )

    st.caption(
        "The LSTM processes spatial positions from the CNN feature map "
        "of a single 2D MRI image; it does not process MRI slices as a "
        "temporal sequence."
    )

    st.warning(
        "**Implementation difference:** full VGG16/VGG19 fine-tuning "
        "proved unstable in our implementation. Their retained models "
        "therefore used frozen pretrained convolutional backbones, while "
        "ResNet50, ResNet152 and EfficientNet-B3 remained fine-tuned. "
        "The project should therefore be described as a reproduction of "
        "the hybrid framework rather than an exact replication."
    )


    # =========================================================================
    # 3. DATASET
    # =========================================================================

    st.markdown(
        "### 3. Dataset and evaluation setup"
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Total images",
        "7,023",
    )

    d2.metric(
        "Training folder",
        "5,712",
    )

    d3.metric(
        "Supplied test",
        "1,311",
    )

    d4.metric(
        "Classes",
        "4",
    )

    st.markdown(
        """
The supplied 5,712-image training directory was divided using a fixed,
stratified seed-42 split:

- **Training:** 4,569 images
- **Validation:** 1,143 images
- **Supplied test:** 1,311 images

Training augmentation was applied only to the training partition.
Validation, test and feature extraction used deterministic preprocessing.

StandardScaler, PCA and the linear SVM were fitted using training features only.
"""
    )


    # =========================================================================
    # 4. PHASE 1
    # =========================================================================

    st.markdown(
        "### 4. Phase 1 — Initial reproduction"
    )

    st.write(
        "Phase 1 tested whether the hybrid CNN–LSTM–PCA–SVM design "
        "could reproduce the high internal performance reported by "
        "the base paper."
    )

    st.dataframe(
        BASE_PAPER_REPRODUCTION[
            [
                "Backbone",
                "Best epoch",
                "NN val accuracy (%)",
                "SVM val accuracy (%)",
                "Hybrid test accuracy (%)",
                "Test macro F1 (%)",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    comparison_frame = (
        BASE_PAPER_REPRODUCTION[
            [
                "Backbone",
                "NN test accuracy (%)",
                "Hybrid test accuracy (%)",
            ]
        ]
        .melt(
            id_vars="Backbone",
            var_name="Classifier",
            value_name="Test accuracy (%)",
        )
    )

    comparison_frame["Baseline"] = 90.0

    phase1_chart = (
        alt.Chart(
            comparison_frame
        )
        .mark_bar(
            size=28
        )
        .encode(
            x=alt.X(
                "Backbone:N",
                sort=[
                    "VGG16",
                    "VGG19",
                    "ResNet50",
                    "ResNet152",
                    "EfficientNet-B3",
                ],
                title=None,
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),

            y=alt.Y(
                "Test accuracy (%):Q",
                title="Test accuracy (%)",
                scale=alt.Scale(
                    domain=[90, 100],
                    zero=False,
                ),
            ),

            y2=alt.Y2(
                "Baseline:Q"
            ),

            color=alt.Color(
                "Classifier:N",
                title=None,
            ),

            xOffset="Classifier:N",

            tooltip=[
                "Backbone:N",
                "Classifier:N",
                alt.Tooltip(
                    "Test accuracy (%):Q",
                    format=".2f",
                ),
            ],
        )
        .properties(
            title=(
                "Phase 1 neural-head vs PCA+SVM "
                "downstream-classifier performance"
            ),
            height=350,
        )
    )

    st.altair_chart(
        phase1_chart,
        use_container_width=True,
    )

    p1, p2 = st.columns(2)

    with p1:

        st.success(
            """
**Phase 1 reproduced high internal performance**

- VGG16: **94.97%**
- VGG19: **93.52%**
- ResNet50: **98.09%**
- ResNet152: **98.40%**
- EfficientNet-B3: **98.47%**
"""
        )

    with p2:

        st.success(
            """
**PCA+SVM improved test accuracy for all five backbones**

- VGG16: **+1.30 pp**
- VGG19: **+0.38 pp**
- ResNet50: **+0.76 pp**
- ResNet152: **+0.23 pp**
- EfficientNet-B3: **+0.46 pp**
"""
        )

    st.caption(
        "This is a neural-head versus PCA+SVM downstream-classifier "
        "comparison, not a PCA-only ablation."
    )


    # =========================================================================
    # 5. PHASE 2A MOTIVATION
    # =========================================================================

    st.markdown(
        "### 5. Why was Phase 2A performed?"
    )

    st.write(
        "Phase 1 used the original early-stopping procedure. "
        "This raised the possibility that some backbones, particularly "
        "the VGG models, had been stopped before reaching their later "
        "validation optimum."
    )

    st.info(
        "**Phase 2A research question:** "
        "was early stopping hiding further convergence?"
    )

    st.write(
        "To test this, every backbone was trained for exactly "
        "**30 epochs with no early termination**, using the same "
        "baseline hyperparameters."
    )

    st.markdown(
        """
For each backbone, two states were retained:

1. **best_val_within_30** — the epoch with the highest neural validation
   accuracy during the 30-epoch run.
2. **epoch_30_model** — the final state after epoch 30.

Both retained states were evaluated through the complete
StandardScaler → PCA64 → linear-SVM downstream pipeline.

The representative Phase 2A results below use **best_val_within_30**.
"""
    )


    # =========================================================================
    # 6. PHASE 2A RESULTS
    # =========================================================================

    st.markdown(
        "### 6. Phase 2A — Standardized 30-Epoch Results"
    )

    st.dataframe(
        PHASE2A_REPRODUCTION,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "VGG16, VGG19 and ResNet50 reached their best-within-30 "
        "state at epoch 30. ResNet152 peaked at epoch 10 and "
        "EfficientNet-B3 at epoch 27."
    )

    phase_compare = (
        PHASE2A_REPRODUCTION[
            [
                "Backbone",
                "Phase 1 SVM val (%)",
                "Phase 2A SVM val (%)",
            ]
        ]
        .melt(
            id_vars="Backbone",
            var_name="Protocol",
            value_name="SVM validation accuracy (%)",
        )
    )

    phase_chart = (
        alt.Chart(
            phase_compare
        )
        .mark_point(
            filled=True,
            size=170,
        )
        .encode(
            x=alt.X(
                "SVM validation accuracy (%):Q",
                title="Full-hybrid SVM validation accuracy (%)",
                scale=alt.Scale(
                    domain=[93, 100],
                    zero=False,
                ),
            ),

            y=alt.Y(
                "Backbone:N",
                sort=[
                    "VGG16",
                    "VGG19",
                    "ResNet50",
                    "ResNet152",
                    "EfficientNet-B3",
                ],
                title=None,
            ),

            color=alt.Color(
                "Protocol:N",
                title=None,
            ),

            tooltip=[
                "Backbone:N",
                "Protocol:N",
                alt.Tooltip(
                    "SVM validation accuracy (%):Q",
                    format=".3f",
                ),
            ],
        )
        .properties(
            title="Phase 1 vs Phase 2A validation performance",
            height=300,
        )
    )

    st.altair_chart(
        phase_chart,
        use_container_width=True,
    )

    p1, p2 = st.columns(2)

    with p1:

        st.success(
            "**Every backbone improved over its Phase 1 SVM-validation "
            "result.** The strongest evidence of premature stopping was "
            "seen in the VGG models, which continued improving through "
            "the fixed 30-epoch protocol."
        )

    with p2:

        st.success(
            "**ResNet152 moved closer to the base-paper result.**\n\n"
            "Phase 1 supplied-test accuracy: **98.40%**  \n"
            "Phase 2A supplied-test accuracy: **98.703%**  \n"
            "Base-paper ResNet152 accuracy: **99%**"
        )


    # =========================================================================
    # DIRECT BASE-PAPER COMPARISON
    # =========================================================================

    st.markdown(
        "### Direct comparison with the base-paper results"
    )

    direct_comparison = pd.DataFrame(
        [
            {
                "Backbone": "VGG16",
                "Base paper accuracy (%)": 94.00,
                "Our Phase 1 test (%)": 94.97,
                "Our Phase 2A test (%)": 96.186,
            },
            {
                "Backbone": "VGG19",
                "Base paper accuracy (%)": 96.00,
                "Our Phase 1 test (%)": 93.52,
                "Our Phase 2A test (%)": 95.271,
            },
            {
                "Backbone": "ResNet50",
                "Base paper accuracy (%)": 99.00,
                "Our Phase 1 test (%)": 98.09,
                "Our Phase 2A test (%)": 98.932,
            },
            {
                "Backbone": "ResNet152",
                "Base paper accuracy (%)": 99.00,
                "Our Phase 1 test (%)": 98.40,
                "Our Phase 2A test (%)": 98.703,
            },
            {
                "Backbone": "EfficientNet-B3",
                "Base paper accuracy (%)": 98.00,
                "Our Phase 1 test (%)": 98.47,
                "Our Phase 2A test (%)": 99.314,
            },
        ]
    )

    st.dataframe(
        direct_comparison,
        hide_index=True,
        use_container_width=True,
    )

    comparison_long = (
        direct_comparison
        .melt(
            id_vars="Backbone",
            var_name="Study / phase",
            value_name="Accuracy (%)",
        )
    )

    comparison_chart = (
        alt.Chart(
            comparison_long
        )
        .mark_point(
            filled=True,
            size=180,
        )
        .encode(
            x=alt.X(
                "Accuracy (%):Q",
                scale=alt.Scale(
                    domain=[92, 100],
                    zero=False,
                ),
                title="Accuracy (%)",
            ),

            y=alt.Y(
                "Backbone:N",
                sort=[
                    "VGG16",
                    "VGG19",
                    "ResNet50",
                    "ResNet152",
                    "EfficientNet-B3",
                ],
                title=None,
            ),

            color=alt.Color(
                "Study / phase:N",
                title=None,
            ),

            shape=alt.Shape(
                "Study / phase:N",
                title=None,
            ),

            tooltip=[
                "Backbone:N",
                "Study / phase:N",
                alt.Tooltip(
                    "Accuracy (%):Q",
                    format=".3f",
                ),
            ],
        )
        .properties(
            title=(
                "Base-paper accuracy vs our Phase 1 "
                "and Phase 2A supplied-test accuracy"
            ),
            height=320,
        )
    )

    st.altair_chart(
        comparison_chart,
        use_container_width=True,
    )

    st.info(
        "**Like-for-like ResNet152 comparison:** "
        "the base paper reported **99%**, our Phase 1 pipeline achieved "
        "**98.40%**, and the standardized Phase 2A best-within-30 "
        "pipeline achieved **98.703%**."
    )

    st.info(
        "**Highest numerical Phase 2A result:** EfficientNet-B3 achieved "
        "**99.314%** on the supplied test split. Because this is a "
        "different backbone, it should not replace the ResNet152-to-ResNet152 "
        "comparison when discussing reproduction of the paper's headline model."
    )

    st.success(
        "**Reproduction conclusion:** the base paper's high internal "
        "performance was successfully reproduced at a very similar level. "
        "The standardized Phase 2A protocol produced our closest "
        "like-for-like ResNet152 result at **98.703%**, compared with the "
        "paper's **99%**. Subsequent tabs examine the reliability, "
        "sensitivity and external generalizability of these high "
        "within-dataset results."
    )


# =============================================================================
# MODEL CARD
# =============================================================================

def _render_model_card(
    spec: PipelineSpec,
) -> None:

    st.subheader(
        f"{spec.display_name} — "
        f"{spec.stage} model card"
    )

    if spec.final_representative:

        st.success(
            "★ Final project representative"
        )

    if spec.formal_xai_model:

        st.info(
            "This Phase 1 ResNet152 pipeline was used "
            "in the formal completed XAI study."
        )

    st.markdown(
        f"""
**Stage:** {spec.stage}  
**Variant:** `{spec.variant}`  
**Selected checkpoint:** `{spec.checkpoint_state}`  
**Selected epoch:** {spec.selected_epoch}  
**Learning rate:** {spec.learning_rate:g}  
**LSTM hidden size:** {spec.lstm_hidden_size}  
**Dropout:** {spec.dropout:g}  
**Final classifier:** StandardScaler → PCA64 → linear SVM (C=1)
"""
    )

    frozen = (
        _frozen_row(
            spec
        )
    )

    result = (

        load_json(
            spec.final_results
        )

        if spec.final_results.is_file()

        else None
    )

    training = (

        load_json(
            spec.training_results
        )

        if spec.training_results.is_file()

        else None
    )

    neural_val = None

    if result:

        neural_val = (
            result
            .get(
                "neural_network",
                {},
            )
            .get(
                "val_accuracy"
            )
        )

    if (
        neural_val is None
        and training
    ):

        neural_val = (
            training.get(
                "best_val_accuracy"
            )
        )

    test_accuracy = None

    if result:

        test_accuracy = (
            result
            .get(
                "test",
                {},
            )
            .get(
                "accuracy"
            )
        )

    metrics = st.columns(
        6
    )

    metrics[0].metric(
        "Selected epoch",
        spec.selected_epoch,
    )

    metrics[1].metric(
        "NN val",

        (
            _stored_percent(
                neural_val
            )

            if neural_val
            is not None

            else "—"
        ),
    )

    metrics[2].metric(
        "SVM val",
        f"{frozen['Internal val (%)']:.3f}%",
    )

    metrics[3].metric(
        "Internal test",

        (
            _fraction_percent(
                test_accuracy,
                3,
            )

            if test_accuracy
            is not None

            else "Not used"
        ),
    )

    metrics[4].metric(
        "BDNeuro full",
        f"{frozen['BDNeuro full (%)']:.2f}%",
    )

    metrics[5].metric(
        "Web-40",
        f"{frozen['Web-40 (%)']:.1f}%",
    )

    if (
        spec.stage
        == "Phase 2B"
    ):

        st.info(
            "Phase 2B configuration selection used validation only. "
            "The internal test set was deliberately not used during "
            "the 35-run sensitivity search."
        )

    if result:

        (
            val_tab,
            test_tab,
            class_tab,
            curve_tab,
        ) = st.tabs(
            [
                "Validation confusion matrix",
                "Test confusion matrix",
                "Per-class test report",
                "Training curves",
            ]
        )

        with val_tab:

            _render_confusion(
                result,
                "validation",
            )

        with test_tab:

            _render_confusion(
                result,
                "test",
            )

        with class_tab:

            report = (

                result

                .get(
                    "test",
                    {},
                )

                .get(
                    "classification_report",
                    {},
                )
            )

            if report:

                rows = []

                for key in CLASS_NAMES:

                    values = (
                        report.get(
                            key,
                            {},
                        )
                    )

                    rows.append(
                        {
                            "Class":
                                CLASS_DISPLAY[
                                    key
                                ],

                            "Precision (%)":
                                _to_percent(
                                    values.get(
                                        "precision"
                                    )
                                ),

                            "Recall (%)":
                                _to_percent(
                                    values.get(
                                        "recall"
                                    )
                                ),

                            "F1-score (%)":
                                _to_percent(
                                    values.get(
                                        "f1-score"
                                    )
                                ),

                            "Support":
                                values.get(
                                    "support"
                                ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        rows
                    ),

                    hide_index=True,

                    use_container_width=True,
                )

            else:

                st.info(
                    "No internal test classification report "
                    "is stored for this pipeline."
                )

        with curve_tab:

            _render_training_curves(
                training
            )

    else:

        st.warning(
            "The selected result JSON was not found. "
            "Prediction can still work if the saved bundle is complete."
        )

    st.divider()

    st.markdown(
        "#### Frozen external/stress record"
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Evaluation":
                        "Internal validation",

                    "Accuracy (%)":
                        frozen[
                            "Internal val (%)"
                        ],
                },

                {
                    "Evaluation":
                        "BDNeuro full",

                    "Accuracy (%)":
                        frozen[
                            "BDNeuro full (%)"
                        ],
                },

                {
                    "Evaluation":
                        "BDNeuro balanced",

                    "Accuracy (%)":
                        frozen[
                            "BDNeuro balanced (%)"
                        ],
                },

                {
                    "Evaluation":
                        "Web-40",

                    "Accuracy (%)":
                        frozen[
                            "Web-40 (%)"
                        ],
                },
            ]
        ),

        hide_index=True,

        use_container_width=True,
    )

    with st.expander(
        "Artifact provenance"
    ):

        rows = []

        for (
            label,
            path,
        ) in [

            (
                "Checkpoint",
                spec.checkpoint,
            ),

            (
                "StandardScaler",
                spec.scaler,
            ),

            (
                "PCA64",
                spec.pca,
            ),

            (
                "Linear SVM",
                spec.svm,
            ),

            (
                "Result JSON",
                spec.final_results,
            ),

            (
                "Training history",
                spec.training_results,
            ),

        ]:

            rows.append(
                {
                    "Artifact":
                        label,

                    "Status":
                        (
                            "Ready"

                            if path.is_file()

                            else "Missing"
                        ),

                    "Path":
                        _relative_path(
                            path
                        ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                rows
            ),

            hide_index=True,

            use_container_width=True,
        )

    st.caption(
        "Original internal validation/test partitions were later shown "
        "to contain material cross-split duplication. External results "
        "are therefore kept separate from the original internal metrics."
    )


# =============================================================================
# HYPERPARAMETER SENSITIVITY
# =============================================================================

def _render_hyperparameter_sensitivity() -> None:

    st.subheader(
        "Which Training Choice Matters Most?"
    )

    st.caption(
        "Phase 2B · one factor at a time around the 30-epoch baseline · "
        "5 backbones × 7 changed configurations = 35 runs"
    )

    st.info(
        "Baseline: RMSprop · LR 1e-4 · LSTM 256 · dropout 0.3 · "
        "Dense128 · batch 8 · seed 42."
    )

    c1, c2, c3 = st.columns(
        3
    )

    c1.metric(
        "Learning-rate spread",
        "4.62 pp",
    )

    c1.caption(
        "Mean spread across learning-rate settings"
    )

    c2.metric(
        "LSTM-width spread",
        "0.58 pp",
    )

    c2.caption(
        "Mean spread across LSTM-width settings"
    )

    c3.metric(
        "Dropout spread",
        "0.42 pp",
    )

    c3.caption(
        "Mean spread across dropout settings"
    )

    st.markdown(
        "### Peak neural validation accuracy"
    )

    parameter_order = [
        "2A baseline",
        "lr_1e-5",
        "lr_5e-5",
        "lr_5e-4",
        "lstm_128",
        "lstm_512",
        "dropout_0.2",
        "dropout_0.5",
    ]

    backbone_order = [
        "VGG16",
        "VGG19",
        "ResNet50",
        "ResNet152",
        "EfficientNet-B3",
    ]

    heatmap_frame = (
        HYPERPARAMETER_SENSITIVITY
        .melt(

            id_vars="Backbone",

            var_name="Setting",

            value_name=(
                "Peak validation accuracy (%)"
            ),
        )
    )

    base = (

        alt.Chart(
            heatmap_frame
        )

        .encode(

            x=alt.X(
                "Setting:N",

                sort=parameter_order,

                title=None,

                axis=alt.Axis(
                    labelAngle=0
                ),
            ),

            y=alt.Y(
                "Backbone:N",

                sort=backbone_order,

                title=None,
            ),
        )
    )

    heatmap = (

        base

        .mark_rect()

        .encode(

            color=alt.Color(
                "Peak validation accuracy (%):Q",

                scale=alt.Scale(
                    domain=[
                        88,
                        100,
                    ],

                    scheme="oranges",
                ),

                title=(
                    "Validation accuracy (%)"
                ),
            ),

            tooltip=[
                "Backbone:N",
                "Setting:N",

                alt.Tooltip(
                    "Peak validation accuracy (%):Q",

                    format=".2f",
                ),
            ],
        )
    )

    labels = (

        base

        .mark_text(
            fontSize=13
        )

        .encode(

            text=alt.Text(
                "Peak validation accuracy (%):Q",

                format=".2f",
            ),

            color=alt.condition(
                (
                    "datum['Peak validation accuracy (%)'] "
                    ">= 97"
                ),

                alt.value(
                    "white"
                ),

                alt.value(
                    "black"
                ),
            ),
        )
    )

    st.altair_chart(
        (
            heatmap
            + labels
        )
        .properties(
            height=360
        ),

        use_container_width=True,
    )

    st.caption(
        "These values describe neural-training sensitivity. "
        "The Phase 2B frozen representatives were selected later "
        "using the complete downstream PCA+SVM validation result."
    )

    with st.expander(
        "Exactly what was varied?"
    ):

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Factor":
                            "Learning rate",

                        "Baseline":
                            "1e-4",

                        "Changed values":
                            "1e-5, 5e-5, 5e-4",

                        "Runs / backbone":
                            3,
                    },

                    {
                        "Factor":
                            "LSTM hidden size",

                        "Baseline":
                            "256",

                        "Changed values":
                            "128, 512",

                        "Runs / backbone":
                            2,
                    },

                    {
                        "Factor":
                            "Dropout",

                        "Baseline":
                            "0.3",

                        "Changed values":
                            "0.2, 0.5",

                        "Runs / backbone":
                            2,
                    },
                ]
            ),

            hide_index=True,

            use_container_width=True,
        )

        st.write(
            "Each of the 35 runs completed 30 epochs. "
            "Both **best-within-30** and **epoch-30** states were "
            "evaluated downstream, giving **70 full-hybrid PCA+SVM evaluations**."
        )

        st.write(
            "The internal test set was not used for "
            "Phase 2B configuration selection."
        )

    st.markdown(
        "### Main interpretation"
    )

    left, right = st.columns(
        2
    )

    with left:

        st.success(
            "**Learning rate was by far the strongest tested sensitivity.**\n\n"
            "The `1e-5` setting under-trained the frozen VGGs, while "
            "`5e-4` substantially damaged the fine-tuned ResNets. "
            "The preferred learning rate therefore depended on the "
            "backbone/training regime."
        )

    with right:

        st.warning(
            "**LSTM width and dropout had much smaller effects.**\n\n"
            "The strongest tuned full-hybrid states improved only about "
            "**0.09–0.70 pp** over the corresponding Phase 2A baseline "
            "internally."
        )

    st.markdown(
        "### Phase 2B representatives carried forward"
    )

    st.dataframe(
        PHASE2B_SELECTED,

        hide_index=True,

        use_container_width=True,
    )

    st.error(
        "**External finding:** the small Phase 2B internal gains did not "
        "consistently transfer. Phase 2A had stronger stage-mean performance "
        "on BDNeuro full, the balanced nested subset, and Web-40."
    )

    st.success(
        "**Takeaway:** learning rate dominated the tested training "
        "sensitivity, but internal tuning gains were not evidence "
        "of improved external robustness."
    )


# =============================================================================
# EXTERNAL EVALUATION
# =============================================================================

def _render_external_heatmaps() -> None:

    st.subheader(
        "How Well Do the Frozen Models Transfer?"
    )

    st.caption(
        "All 15 frozen pipelines · BDNeuro exact-overlap-clean cohort "
        "(n=2,644) · balanced nested subset (n=1,648) · "
        "Web-40 exploratory stress test (n=40)"
    )

    st.info(
        "The 15 model states were frozen before external performance "
        "evaluation. No BDNeuro or Web-40 result was used to retrain, "
        "refit or recalibrate a frozen pipeline."
    )

    frame = pd.DataFrame(
        FROZEN_15_ROWS
    ).copy()

    frame[
        "Stage short"
    ] = (
        frame[
            "Stage"
        ]
        .map(
            {
                "Phase 1":
                    "P1",

                "Phase 2A":
                    "P2A",

                "Phase 2B":
                    "P2B",
            }
        )
    )

    backbone_order = [
        "VGG16",
        "VGG19",
        "ResNet50",
        "ResNet152",
        "EfficientNet-B3",
    ]

    stage_order = [
        "P1",
        "P2A",
        "P2B",
    ]

    def build(
        metric: str,
        title: str,
        domain: list[float],
        scheme: str,
    ):

        base = (

            alt.Chart(
                frame
            )

            .encode(

                x=alt.X(
                    "Stage short:N",

                    sort=stage_order,

                    title=None,

                    axis=alt.Axis(
                        labelAngle=0
                    ),
                ),

                y=alt.Y(
                    "Backbone:N",

                    sort=backbone_order,

                    title=None,
                ),
            )
        )

        cells = (

            base

            .mark_rect(
                stroke="white",

                strokeWidth=1,
            )

            .encode(

                color=alt.Color(
                    metric,

                    type="quantitative",

                    scale=alt.Scale(
                        domain=domain,

                        scheme=scheme,
                    ),

                    legend=None,
                ),

                tooltip=[
                    "Backbone:N",
                    "Stage:N",
                    "Variant:N",
                    "Epoch:Q",

                    alt.Tooltip(
                        metric,

                        type="quantitative",

                        format=".2f",
                    ),
                ],
            )
        )

        threshold = (
            domain[0]
            +
            (
                domain[1]
                - domain[0]
            )
            * 0.55
        )

        labels = (

            base

            .mark_text(
                fontSize=14,

                fontWeight="bold",
            )

            .encode(

                text=alt.Text(
                    metric,

                    type="quantitative",

                    format=".2f",
                ),

                color=alt.condition(
                    f"datum['{metric}'] >= {threshold}",

                    alt.value(
                        "white"
                    ),

                    alt.value(
                        "black"
                    ),
                ),
            )
        )

        highlight = (

            base

            .transform_filter(
                (
                    "datum.Backbone == 'ResNet152' "
                    "&& datum['Stage short'] == 'P2A'"
                )
            )

            .mark_rect(
                fillOpacity=0,

                stroke="#ff8c00",

                strokeWidth=4,
            )
        )

        return (
            cells
            + labels
            + highlight
        ).properties(
            title=title,

            width=300,

            height=300,
        )

    combined = (

        alt.hconcat(

            build(
                "BDNeuro full (%)",
                "BDNeuro full (n=2,644)",
                [
                    74,
                    91,
                ],
                "oranges",
            ),

            build(
                "BDNeuro balanced (%)",
                "BDNeuro balanced, nested (n=1,648)",
                [
                    74,
                    92,
                ],
                "oranges",
            ),

            build(
                "Web-40 (%)",
                "Web-40 exploratory (n=40)",
                [
                    20,
                    45,
                ],
                "yelloworangebrown",
            ),

            spacing=14,
        )

        .resolve_scale(
            color="independent"
        )
    )

    st.altair_chart(
        combined,
        use_container_width=True,
    )

    st.caption(
        "Orange outline = final project representative. "
        "The balanced BDNeuro cohort is a nested sensitivity subset, "
        "not an independent second external dataset."
    )

    c1, c2, c3 = st.columns(
        3
    )

    c1.success(
        "**BDNeuro full:** 75.00–88.31% "
        "across the frozen 15 pipelines."
    )

    c2.success(
        "**Strongest structured-external result:** "
        "ResNet152 Phase 2A — 88.31% full / 90.47% balanced."
    )

    c3.warning(
        "**Web-40:** 25.0–42.5%, exposing severe sensitivity "
        "to heterogeneous presentation/domain shift."
    )


def _render_external_evaluation() -> None:

    _render_external_heatmaps()

    st.markdown(
        "### Interpretation across stages"
    )

    st.markdown(
        """
- **Phase 2A had the strongest stage-level transfer overall.**
  Mean accuracy was approximately **83.21% on BDNeuro full**,
  **85.39% on the balanced nested subset**, and **38.0% on Web-40**.

- **Phase 2B tuning gains did not consistently transfer.**
  Several tuned states improved internally, but those gains were not
  reliably preserved externally.

- **Highest internal validation did not guarantee highest external performance.**
  EfficientNet-B3 Phase 2B had the highest frozen internal validation result;
  ResNet152 Phase 2A had the strongest BDNeuro result.
"""
    )

    st.subheader(
        "Final Representative — Evaluation Trajectory"
    )

    final_rows = pd.DataFrame(
        {
            "Evaluation": [
                "Internal validation*",
                "material_clean†",
                "BDNeuro full",
                "BDNeuro balanced‡",
                "Web-40§",
            ],

            "Accuracy (%)": [
                98.600175,
                98.869,
                88.313162,
                90.473301,
                37.5,
            ],

            "Macro F1 (%)": [
                98.545056,
                98.788,
                87.854138,
                90.496811,
                31.664963,
            ],
        }
    )

    st.dataframe(
        final_rows,

        hide_index=True,

        use_container_width=True,
    )

    trajectory = (

        alt.Chart(
            final_rows
        )

        .mark_bar()

        .encode(

            x=alt.X(
                "Evaluation:N",

                sort=None,

                title=None,
            ),

            y=alt.Y(
                "Accuracy (%):Q",

                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            tooltip=[
                "Evaluation:N",

                alt.Tooltip(
                    "Accuracy (%):Q",

                    format=".2f",
                ),

                alt.Tooltip(
                    "Macro F1 (%):Q",

                    format=".2f",
                ),
            ],
        )

        .properties(
            title=(
                "ResNet152 Phase 2A: "
                "internal-to-external trajectory"
            ),

            height=330,
        )
    )

    st.altair_chart(
        trajectory,

        use_container_width=True,
    )

    st.markdown(
        "### Data-integrity context"
    )

    left, right = st.columns(
        2
    )

    with left:

        st.markdown(
            "#### Nickparvar leakage audit"
        )

        st.write(
            "**562 material cross-split pair relationships** were identified: "
            "464 exact/file-or-pixel relationships plus "
            "98 high-confidence near-duplicate relationships."
        )

        st.write(
            "These relationships implicated **250 unique test images**. "
            "The primary post-hoc `material_clean` test subset excluded "
            "those 250 test images, leaving **1,061**."
        )

        st.caption(
            "562 is a count of pair relationships, not unique images. "
            "Training/validation contamination was not repaired."
        )

    with right:

        st.markdown(
            "#### BDNeuro overlap audit"
        )

        st.write(
            "Raw MRI cohort: **5,941** images. "
            "**3,297 unique images** had exact decoded-pixel overlap "
            "with the development dataset."
        )

        st.write(
            "Primary exact-overlap-clean cohort: **2,644**. "
            "Balanced nested sensitivity subset: "
            "**1,648** (412 per class)."
        )

    frame = pd.DataFrame(
        FROZEN_15_ROWS
    )

    stage_means = (

        frame

        .groupby(
            "Stage",

            as_index=False,
        )[
            [
                "BDNeuro full (%)",
                "BDNeuro balanced (%)",
                "Web-40 (%)",
            ]
        ]

        .mean()
    )

    stage_long = (
        stage_means
        .melt(

            id_vars="Stage",

            var_name="Evaluation",

            value_name=(
                "Mean accuracy (%)"
            ),
        )
    )

    stage_chart = (

        alt.Chart(
            stage_long
        )

        .mark_bar()

        .encode(

            x=alt.X(
                "Stage:N",

                sort=[
                    "Phase 1",
                    "Phase 2A",
                    "Phase 2B",
                ],
            ),

            y=alt.Y(
                "Mean accuracy (%):Q",

                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            color="Evaluation:N",

            xOffset="Evaluation:N",

            tooltip=[
                "Stage:N",
                "Evaluation:N",

                alt.Tooltip(
                    "Mean accuracy (%):Q",

                    format=".2f",
                ),
            ],
        )

        .properties(
            title=(
                "Stage-mean external/stress accuracy "
                "across five backbones"
            ),

            height=330,
        )
    )

    st.altair_chart(
        stage_chart,

        use_container_width=True,
    )

    st.markdown(
        "### Web-40 class behaviour"
    )

    st.dataframe(
        WEB40_CLASS_RECALL,

        hide_index=True,

        use_container_width=True,
    )

    st.warning(
        "Across the frozen pipelines, Web-40 showed a strong "
        "no-tumour bias: mean recall was high for no-tumour "
        "(90.67%) but very low for glioma (12.67%) and "
        "pituitary (7.33%). The 40-image test is exploratory; "
        "one image changes accuracy by 2.5 percentage points."
    )

    st.success(
        "**Overall external finding:** high performance on the original "
        "public dataset did not guarantee robust "
        "out-of-distribution performance."
    )

    st.caption(
        "* Original internal split, later shown to contain cross-split "
        "duplication. † Post-hoc test subset only; training contamination "
        "is not repaired. ‡ Nested subset of BDNeuro full. "
        "§ Exploratory source-verified stress test, not clinical validation."
    )


# =============================================================================
# XAI STUDY
# =============================================================================

def _render_xai_study() -> None:

    st.subheader(
        "Does Correct Classification Mean Lesion-Centred Reasoning?"
    )

    st.caption(
        "Formal rated Phase 1 ResNet152 XAI study · "
        "Grad-CAM + final-SVM occlusion sensitivity"
    )

    st.info(
        "The formal rated XAI study used **Phase 1 ResNet152**. "
        "Live maps generated for the other 14 frozen pipelines are "
        "qualitative demonstrations only."
    )

    st.markdown(
        "### Why was XAI evaluated?"
    )

    st.write(
        "The project already had very high classification accuracy, "
        "but accuracy alone could not establish whether the model was "
        "actually focusing on the visible tumour region. "
        "The XAI study therefore tested whether correct classifications "
        "were accompanied by anatomically plausible, lesion-centred "
        "explanation maps."
    )

    left, right = st.columns(
        2
    )

    with left:

        st.markdown(
            "#### Grad-CAM protocol"
        )

        st.write(
            "**20 test images** were sampled with seed 42: "
            "5 per class. The **15 tumour-containing cases** "
            "were rated for lesion correspondence."
        )

        st.markdown(
            """
- ResNet Layer 3: **14×14**
- ResNet Layer 4: **7×7**
- Fused: equal Layer 3 / Layer 4 blend
"""
        )

    with right:

        st.markdown(
            "#### Final-SVM occlusion protocol"
        )

        st.write(
            "**11 selected images** were evaluated through the full "
            "saved CNN–LSTM–StandardScaler–PCA–SVM pipeline."
        )

        st.markdown(
            """
- Patch: **32×32**
- Stride: **16**
- Windows/image: **169**
- Total perturbations: **1,859**
"""
        )

    st.markdown(
        "### Rating definitions"
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rating":
                        "Good",

                    "Interpretation":
                        (
                            "Strongest activation/support mainly aligned "
                            "with the visible tumour region."
                        ),
                },

                {
                    "Rating":
                        "Partial",

                    "Interpretation":
                        (
                            "Some lesion correspondence, but substantial "
                            "support also outside the lesion."
                        ),
                },

                {
                    "Rating":
                        "Poor",

                    "Interpretation":
                        (
                            "Little useful lesion correspondence or dominance "
                            "of skull/border/facial/orbital/non-lesion regions."
                        ),
                },
            ]
        ),

        hide_index=True,

        use_container_width=True,
    )

    st.caption(
        "These were qualitative lesion-correspondence ratings, "
        "not segmentation metrics such as Dice or IoU."
    )

    st.markdown(
        "## Grad-CAM results"
    )

    g1, g2, g3 = st.columns(
        3
    )

    g1.metric(
        "Formal cases",
        "20",
    )

    g2.metric(
        "Tumour cases rated",
        "15",
    )

    g3.metric(
        "Both heads jointly correct",
        "19 / 20",
    )

    st.dataframe(
        GRADCAM_RATINGS,

        hide_index=True,

        use_container_width=True,
    )

    gradcam_long = (
        GRADCAM_RATINGS

        .melt(
            id_vars="Grad-CAM map",

            var_name="Rating",

            value_name="Cases",
        )
    )

    gradcam_chart = (

        alt.Chart(
            gradcam_long
        )

        .mark_bar()

        .encode(

            x=alt.X(
                "Grad-CAM map:N",

                sort=[
                    "Layer 3",
                    "Layer 4",
                    "Fused",
                ],

                title=None,
            ),

            y=alt.Y(
                "Cases:Q",

                title=(
                    "Number of tumour cases"
                ),

                scale=alt.Scale(
                    domain=[
                        0,
                        15,
                    ]
                ),
            ),

            color=alt.Color(
                "Rating:N",

                sort=[
                    "Good",
                    "Partial",
                    "Poor",
                ],
            ),

            xOffset="Rating:N",

            tooltip=[
                "Grad-CAM map:N",
                "Rating:N",
                "Cases:Q",
            ],
        )

        .properties(
            title=(
                "Qualitative lesion-correspondence ratings"
            ),

            height=330,
        )
    )

    st.altair_chart(
        gradcam_chart,

        use_container_width=True,
    )

    c1, c2 = st.columns(
        2
    )

    c1.success(
        "**Layer 3:** 1 Good, 9 Partial, 5 Poor. "
        "Ten of 15 cases showed at least some lesion correspondence, "
        "but fully convincing localization was uncommon."
    )

    c2.warning(
        "**Layer 4:** 3 Good, 2 Partial, 10 Poor. "
        "Deeper semantic activations were frequently not centred "
        "on the visible lesion."
    )

    st.write(
        "The fused map remained mixed: "
        "**2 Good, 6 Partial, 7 Poor**."
    )

    st.markdown(
        "## Final-SVM occlusion results"
    )

    o1, o2, o3, o4 = st.columns(
        4
    )

    o1.metric(
        "Images evaluated",
        "11",
    )

    o2.metric(
        "Occlusions / image",
        "169",
    )

    o3.metric(
        "Total perturbations",
        "1,859",
    )

    o4.metric(
        "Prediction-changing patches",
        "58 (3.12%)",
    )

    st.dataframe(
        OCCLUSION_XAI_RESULTS,

        hide_index=True,

        use_container_width=True,
    )

    st.write(
        "**58 / 1,859** means that across all masked variants, "
        "58 local occlusions caused the final SVM to output a different class. "
        "These flips were concentrated in **5 of 11 cases**."
    )

    st.write(
        "For the shared misclassified case **Te-pi_0078**, "
        "**38 of 169** occlusions changed the output class, "
        "indicating unusually high local sensitivity. "
        "A flip does not necessarily mean the new prediction became correct."
    )

    st.warning(
        "Occlusion tumour-case ratings were "
        "**2 Good / 2 Partial / 5 Poor**. "
        "Final-SVM occlusion was therefore not consistently more "
        "lesion-focused than Grad-CAM."
    )

    st.markdown(
        "## Overall XAI finding"
    )

    st.error(
        "**High classification accuracy could not be equated with "
        "anatomically plausible or consistently lesion-centred "
        "model reasoning.**"
    )

    st.markdown(
        """
The study showed that correct classifications could coexist with only
partial lesion correspondence; deeper Grad-CAM maps frequently emphasized
non-lesion regions; and final-SVM occlusion was not consistently more
lesion-focused.

This **does not prove shortcut learning**, does not identify the exact
abstract features learned by the network, and does not provide tumour
segmentation.
"""
    )

    st.success(
        "**Takeaway:** the formal Phase 1 XAI experiment answered the "
        "project's explainability question sufficiently; a second formal "
        "Phase 2 XAI study was not required for the current project objective."
    )


# =============================================================================
# FROZEN 15 COMPARISON
# =============================================================================

def _render_frozen_15() -> None:

    st.subheader(
        "Frozen 15-Pipeline Comparison"
    )

    st.caption(
        "Five CNN backbones × three retained project stages. "
        "The external comparison was performed after the frozen set "
        "had been locked."
    )

    frame = pd.DataFrame(
        FROZEN_15_ROWS
    )

    frame[
        "Final representative"
    ] = np.where(

        (
            frame[
                "Backbone"
            ]
            == "ResNet152"
        )

        &

        (
            frame[
                "Stage"
            ]
            == "Phase 2A"
        ),

        "★",

        "",
    )

    f1, f2 = st.columns(
        2
    )

    with f1:

        stages = st.multiselect(

            "Stage filter",

            options=list(
                STAGE_ORDER
            ),

            default=list(
                STAGE_ORDER
            ),
        )

    with f2:

        backbones = st.multiselect(

            "Backbone filter",

            options=[
                DISPLAY_NAMES[
                    name
                ]

                for name
                in BACKBONE_ORDER
            ],

            default=[
                DISPLAY_NAMES[
                    name
                ]

                for name
                in BACKBONE_ORDER
            ],
        )

    filtered = frame[

        frame[
            "Stage"
        ].isin(
            stages
        )

        &

        frame[
            "Backbone"
        ].isin(
            backbones
        )

    ].copy()

    st.dataframe(
        filtered,

        hide_index=True,

        use_container_width=True,
    )

    metric = st.selectbox(

        "Chart metric",

        [
            "Internal val (%)",
            "BDNeuro full (%)",
            "BDNeuro balanced (%)",
            "Web-40 (%)",
        ],

        index=1,
    )

    chart = (

        alt.Chart(
            filtered
        )

        .mark_bar()

        .encode(

            x=alt.X(
                "Backbone:N",

                title=None,
            ),

            y=alt.Y(
                metric,

                type="quantitative",

                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            color="Stage:N",

            xOffset="Stage:N",

            tooltip=[
                "Backbone:N",
                "Stage:N",
                "Variant:N",
                "State:N",
                "Epoch:Q",

                alt.Tooltip(
                    metric,

                    type="quantitative",

                    format=".2f",
                ),
            ],
        )

        .properties(
            height=360
        )
    )

    st.altair_chart(
        chart,

        use_container_width=True,
    )

    st.markdown(
        "#### Stage means"
    )

    stage_means = (

        frame

        .groupby(
            "Stage",

            as_index=False,
        )[
            [
                "BDNeuro full (%)",
                "BDNeuro balanced (%)",
                "Web-40 (%)",
            ]
        ]

        .mean()
    )

    st.dataframe(
        stage_means,

        hide_index=True,

        use_container_width=True,
    )


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "Brain Tumor MRI Classification — Final Project Dashboard"
)

st.caption(
    "Hybrid CNN–LSTM–PCA–SVM research project · "
    "base-paper reproduction · "
    "15 frozen pipelines · "
    "Phase 2 optimization · "
    "overlap-clean external validation · "
    "Web-40 stress testing · explainability"
)

st.warning(
    "Research and educational use only. "
    "This application is not a medical device and must not be used "
    "to diagnose, exclude or treat disease."
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header(
        "Frozen model selection"
    )

    stage = st.selectbox(

        "Project stage",

        STAGE_ORDER,

        index=1,
    )

    backbone = st.selectbox(

        "CNN backbone",

        BACKBONE_ORDER,

        index=3,

        format_func=lambda name:
            DISPLAY_NAMES[
                name
            ],
    )

    pipeline_id = (
        pipeline_id_for(
            stage,
            backbone,
        )
    )

    spec = (
        PIPELINE_SPECS[
            pipeline_id
        ]
    )

    st.divider()

    st.markdown(
        f"### {spec.display_name}"
    )

    st.caption(
        f"{spec.stage} · "
        f"{spec.variant}"
    )

    st.caption(
        f"`{spec.checkpoint_state}` · "
        f"epoch {spec.selected_epoch}"
    )

    if spec.final_representative:

        st.success(
            "★ Final project representative"
        )

    if spec.formal_xai_model:

        st.info(
            "Formal XAI study model"
        )

    if spec.prediction_ready():

        st.success(
            "Prediction bundle ready"
        )

    else:

        st.error(
            "Prediction bundle incomplete"
        )

        for missing in (
            spec
            .missing_prediction_files()
        ):

            st.caption(
                _relative_path(
                    missing
                )
            )

    st.caption(
        "Only one frozen pipeline is kept resident "
        "in memory at a time."
    )

    st.divider()

    if st.button(
        "Unload cached model",

        use_container_width=True,
    ):

        _unload_pipeline_bundle()

        st.session_state.pop(
            "prediction_record",
            None,
        )

        st.success(
            "Cached model unloaded."
        )


# =============================================================================
# NAVIGATION
# =============================================================================

(
    predict_tab,
    reproduction_tab,
    model_card_tab,
    sensitivity_tab,
    external_tab,
    xai_tab,
    comparison_tab,
) = st.tabs(
    [
        "Predict MRI",
        "Base-paper reproduction",
        "Model card",
        "Hyperparameter sensitivity",
        "External evaluation",
        "XAI study",
        "Frozen 15 comparison",
    ]
)


# =============================================================================
# PREDICT MRI
# =============================================================================

with predict_tab:

    st.subheader(
        "Upload one MRI image"
    )

    selected = st.columns(
        [
            1.2,
            1,
            1,
            1,
        ]
    )

    selected[0].metric(
        "Selected model",
        spec.display_name,
    )

    selected[1].metric(
        "Stage",
        spec.stage,
    )

    selected[2].metric(
        "Epoch",
        spec.selected_epoch,
    )

    selected[3].metric(
        "Variant",
        spec.variant,
    )

    uploaded = st.file_uploader(

        (
            "Supported formats: PNG, JPG/JPEG, BMP, TIFF "
            "(no DICOM in this prototype)"
        ),

        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
        ],
    )

    if uploaded is None:

        st.info(
            "Upload one MRI image to begin."
        )

    else:

        raw = (
            uploaded
            .getvalue()
        )

        fingerprint = (
            hashlib
            .sha256(
                raw
            )
            .hexdigest()
        )

        try:

            image = (
                _open_image(
                    raw
                )
            )

            (
                image_col,
                info_col,
            ) = st.columns(
                [
                    1,
                    1.6,
                ]
            )

            with image_col:

                st.image(

                    image,

                    caption=(
                        f"{uploaded.name} · "
                        f"{image.width} × "
                        f"{image.height}"
                    ),
                )

            with info_col:

                st.markdown(
                    f"""
### Selected frozen pipeline

**Model:** {spec.display_name}  
**Stage:** {spec.stage}  
**Variant:** `{spec.variant}`  
**Checkpoint:** `{spec.checkpoint_state}`  
**Selected epoch:** {spec.selected_epoch}  
**LSTM hidden size:** {spec.lstm_hidden_size}  
**Dropout:** {spec.dropout:g}  
**Final classifier:** StandardScaler → PCA64 → linear SVM  
**Preprocessing:** deterministic project evaluation transform
"""
                )

                if spec.final_representative:

                    st.success(
                        "This is the final project representative."
                    )

            if spec.formal_xai_model:

                st.info(
                    "This pipeline is the formal "
                    "Phase 1 ResNet152 XAI-study model."
                )

            else:

                st.info(
                    "Grad-CAM and occlusion generated here are live "
                    "qualitative demonstrations. The formally rated "
                    "XAI study used Phase 1 ResNet152."
                )

            analyze = st.button(

                (
                    "Analyze MRI with "
                    f"{spec.display_name} · "
                    f"{spec.stage}"
                ),

                type="primary",

                disabled=(
                    not spec
                    .prediction_ready()
                ),

                use_container_width=True,
            )

            if analyze:

                try:

                    with st.spinner(
                        "Loading the selected frozen pipeline "
                        "and running inference…"
                    ):

                        prediction = (
                            _predict_pipeline(
                                image,
                                spec,
                            )
                        )

                        explanation = None

                        explanation_error = None

                        try:

                            explanation = (
                                _explain_pipeline_gradcam(

                                    image,

                                    spec,

                                    prediction
                                    .nn_class_key,
                                )
                            )

                        except Exception as error:

                            explanation_error = (
                                str(
                                    error
                                )
                            )

                    st.session_state[
                        "prediction_record"
                    ] = {
                        "fingerprint":
                            fingerprint,

                        "pipeline_id":
                            spec.pipeline_id,

                        "result":
                            prediction,

                        "explanation":
                            explanation,

                        "explanation_error":
                            explanation_error,

                        "occlusion":
                            None,

                        "occlusion_error":
                            None,
                    }

                except Exception as error:

                    st.error(
                        str(
                            error
                        )
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.exception(
                            error
                        )

            record = (
                st.session_state
                .get(
                    "prediction_record"
                )
            )

            if (
                record

                and

                record.get(
                    "fingerprint"
                )
                == fingerprint

                and

                record.get(
                    "pipeline_id"
                )
                == spec.pipeline_id
            ):

                prediction = (
                    record[
                        "result"
                    ]
                )

                _render_prediction(
                    prediction
                )

                if (
                    record.get(
                        "explanation"
                    )
                    is not None
                ):

                    _render_gradcam(

                        record[
                            "explanation"
                        ],

                        prediction,

                        spec,

                        fingerprint,
                    )

                else:

                    st.divider()

                    st.subheader(
                        "Explainability — Grad-CAM"
                    )

                    st.warning(
                        "Classification completed, but Grad-CAM "
                        "could not be generated: "
                        f"{record.get('explanation_error', 'unknown error')}"
                    )

                _render_occlusion(

                    image,

                    prediction,

                    spec,

                    record,
                )

        except Exception as error:

            st.error(
                str(
                    error
                )
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(
                    error
                )


# =============================================================================
# OTHER TABS
# =============================================================================

with reproduction_tab:

    _render_base_paper_reproduction()


with model_card_tab:

    _render_model_card(
        spec
    )


with sensitivity_tab:

    _render_hyperparameter_sensitivity()


with external_tab:

    _render_external_evaluation()


with xai_tab:

    _render_xai_study()


with comparison_tab:

    _render_frozen_15()