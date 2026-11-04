"""Artifact registry for all 15 frozen CNN-LSTM-PCA-SVM pipelines."""

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

STAGE_ORDER = (
    "Phase 1",
    "Phase 2A",
    "Phase 2B",
)

DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "efficientnet_b3": "EfficientNet-B3",
}


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    backbone: str
    stage: str
    variant: str
    checkpoint_state: str
    selected_epoch: int
    learning_rate: float
    lstm_hidden_size: int
    dropout: float
    final_representative: bool = False
    formal_xai_model: bool = False


@dataclass(frozen=True)
class ModelArtifacts:
    pipeline_id: str

    backbone: str
    display_name: str

    stage: str
    variant: str
    checkpoint_state: str
    selected_epoch: int

    learning_rate: float
    lstm_hidden_size: int
    dropout: float

    final_representative: bool
    formal_xai_model: bool

    checkpoint: Path
    svm: Path
    scaler: Path
    pca: Path
    final_results: Path
    training_results: Path

    @property
    def label(self) -> str:
        return f"{self.display_name} · {self.stage}"

    def required_for_prediction(self) -> tuple[Path, ...]:
        return (
            self.checkpoint,
            self.svm,
            self.scaler,
            self.pca,
        )

    def missing_prediction_files(self) -> list[Path]:
        return [
            path
            for path in self.required_for_prediction()
            if not path.is_file()
        ]

    def prediction_ready(self) -> bool:
        return not self.missing_prediction_files()


def _pipeline_id(backbone: str, stage: str) -> str:

    suffix = {
        "Phase 1": "phase1",
        "Phase 2A": "phase2a",
        "Phase 2B": "phase2b",
    }[stage]

    return f"{backbone}_{suffix}"


PIPELINE_SPECS: dict[str, PipelineSpec] = {}


# ============================================================================
# PHASE 1
# ============================================================================
#
# Original standardized early-stopping representatives.
#

PHASE1_EPOCHS = {
    "vgg16": 16,
    "vgg19": 18,
    "resnet50": 7,
    "resnet152": 8,
    "efficientnet_b3": 6,
}


for backbone, epoch in PHASE1_EPOCHS.items():

    pipeline_id = _pipeline_id(
        backbone,
        "Phase 1",
    )

    PIPELINE_SPECS[pipeline_id] = PipelineSpec(
        pipeline_id=pipeline_id,
        backbone=backbone,
        stage="Phase 1",
        variant="early_stopping",
        checkpoint_state="best_model",
        selected_epoch=epoch,
        learning_rate=1e-4,
        lstm_hidden_size=256,
        dropout=0.3,

        # The formally rated XAI study was specifically
        # performed on Phase 1 ResNet152.
        formal_xai_model=(
            backbone == "resnet152"
        ),
    )


# ============================================================================
# PHASE 2A
# ============================================================================
#
# Fixed 30-epoch baseline.
#
# Every backbone completed all 30 epochs.
#
# These are the retained full-hybrid representatives chosen from
# best_val_within_30 vs epoch_30 using SVM validation performance.
#

PHASE2A_EPOCHS = {
    "vgg16": 30,
    "vgg19": 30,
    "resnet50": 30,
    "resnet152": 10,
    "efficientnet_b3": 27,
}


for backbone, epoch in PHASE2A_EPOCHS.items():

    pipeline_id = _pipeline_id(
        backbone,
        "Phase 2A",
    )

    PIPELINE_SPECS[pipeline_id] = PipelineSpec(
        pipeline_id=pipeline_id,
        backbone=backbone,
        stage="Phase 2A",
        variant="baseline",
        checkpoint_state="best_val_within_30",
        selected_epoch=epoch,
        learning_rate=1e-4,
        lstm_hidden_size=256,
        dropout=0.3,

        final_representative=(
            backbone == "resnet152"
        ),
    )


# ============================================================================
# PHASE 2B
# ============================================================================
#
# Strongest changed configuration per backbone.
#
# These were selected using full-hybrid internal validation performance
# before external evaluation.
#

PHASE2B_SELECTION = {

    "vgg16": {
        "variant": "lr_5e-4",
        "checkpoint_state": "best_val_within_30",
        "epoch": 27,
        "learning_rate": 5e-4,
        "lstm_hidden_size": 256,
        "dropout": 0.3,
    },

    "vgg19": {
        "variant": "dropout_0.5",
        "checkpoint_state": "epoch_30",
        "epoch": 30,
        "learning_rate": 1e-4,
        "lstm_hidden_size": 256,
        "dropout": 0.5,
    },

    "resnet50": {
        "variant": "dropout_0.2",
        "checkpoint_state": "best_val_within_30",
        "epoch": 28,
        "learning_rate": 1e-4,
        "lstm_hidden_size": 256,
        "dropout": 0.2,
    },

    "resnet152": {
        "variant": "lstm_512",
        "checkpoint_state": "best_val_within_30",
        "epoch": 26,
        "learning_rate": 1e-4,
        "lstm_hidden_size": 512,
        "dropout": 0.3,
    },

    "efficientnet_b3": {
        "variant": "lstm_512",
        "checkpoint_state": "best_val_within_30",
        "epoch": 29,
        "learning_rate": 1e-4,
        "lstm_hidden_size": 512,
        "dropout": 0.3,
    },
}


for backbone, configuration in PHASE2B_SELECTION.items():

    pipeline_id = _pipeline_id(
        backbone,
        "Phase 2B",
    )

    PIPELINE_SPECS[pipeline_id] = PipelineSpec(
        pipeline_id=pipeline_id,
        backbone=backbone,
        stage="Phase 2B",
        variant=configuration["variant"],
        checkpoint_state=configuration["checkpoint_state"],
        selected_epoch=configuration["epoch"],
        learning_rate=configuration["learning_rate"],
        lstm_hidden_size=configuration["lstm_hidden_size"],
        dropout=configuration["dropout"],
    )


PIPELINE_ORDER = tuple(

    _pipeline_id(
        backbone,
        stage,
    )

    for stage in STAGE_ORDER
    for backbone in BACKBONE_ORDER
)


# ============================================================================
# HELPERS
# ============================================================================

def pipeline_id_for(
    stage: str,
    backbone: str,
) -> str:

    pipeline_id = _pipeline_id(
        backbone,
        stage,
    )

    if pipeline_id not in PIPELINE_SPECS:

        raise ValueError(
            f"No frozen pipeline exists for "
            f"stage={stage!r}, backbone={backbone!r}."
        )

    return pipeline_id


def resolve_pipeline_id(
    identifier: str,
) -> str:
    """
    Accept either:

    - one of the new 15-pipeline IDs, or
    - an old bare backbone name.

    Bare backbone names are mapped to Phase 1 for backward compatibility
    with the formal Grad-CAM / occlusion evaluation apps.
    """

    if identifier in PIPELINE_SPECS:
        return identifier

    if identifier in BACKBONE_ORDER:
        return pipeline_id_for(
            "Phase 1",
            identifier,
        )

    raise ValueError(
        f"Unknown model identifier: {identifier!r}"
    )


def get_pipeline_spec(
    identifier: str,
) -> PipelineSpec:

    pipeline_id = resolve_pipeline_id(
        identifier
    )

    return PIPELINE_SPECS[
        pipeline_id
    ]


def _checkpoint_filename(
    checkpoint_state: str,
) -> str:

    if checkpoint_state == "best_model":
        return "best_model.pth"

    if checkpoint_state == "best_val_within_30":
        return "best_val_within_30.pth"

    if checkpoint_state == "epoch_30":
        return "epoch_30_model.pth"

    raise ValueError(
        f"Unsupported checkpoint state: "
        f"{checkpoint_state}"
    )


# ============================================================================
# ARTIFACT PATH CONSTRUCTION
# ============================================================================

def get_pipeline_artifacts(
    project_root: Path,
    identifier: str,
) -> ModelArtifacts:

    project_root = Path(
        project_root
    ).resolve()

    spec = get_pipeline_spec(
        identifier
    )


    # ------------------------------------------------------------------------
    # PHASE 1
    # ------------------------------------------------------------------------

    if spec.stage == "Phase 1":

        experiment = (
            project_root
            / "experiments"
            / spec.backbone
        )

        checkpoint = (
            experiment
            / "models"
            / "best_model.pth"
        )

        scaler = (
            experiment
            / "pca"
            / "standard_scaler.joblib"
        )

        pca = (
            experiment
            / "pca"
            / "pca_model.joblib"
        )

        svm = (
            experiment
            / "models"
            / "svm.joblib"
        )

        final_results = (
            experiment
            / "results"
            / "final_results.json"
        )

        training_results = (
            experiment
            / "results"
            / "training_results.json"
        )


    # ------------------------------------------------------------------------
    # PHASE 2A
    # ------------------------------------------------------------------------

    elif spec.stage == "Phase 2A":

        experiment = (
            project_root
            / "phase2"
            / "experiments_30epoch"
            / spec.backbone
        )

        evaluation = (
            experiment
            / "evaluations"
            / spec.checkpoint_state
        )

        checkpoint = (
            experiment
            / "models"
            / _checkpoint_filename(
                spec.checkpoint_state
            )
        )

        scaler = (
            evaluation
            / "pca"
            / "standard_scaler.joblib"
        )

        pca = (
            evaluation
            / "pca"
            / "pca_model.joblib"
        )

        svm = (
            evaluation
            / "models"
            / "svm.joblib"
        )

        final_results = (
            evaluation
            / "results"
            / "final_results.json"
        )

        training_results = (
            experiment
            / "results"
            / "training_results.json"
        )


    # ------------------------------------------------------------------------
    # PHASE 2B
    # ------------------------------------------------------------------------

    elif spec.stage == "Phase 2B":

        experiment = (
            project_root
            / "phase2"
            / "hyperparameter_dl"
            / spec.backbone
            / spec.variant
        )

        evaluation = (
            experiment
            / "evaluations"
            / spec.checkpoint_state
        )

        checkpoint = (
            experiment
            / "models"
            / _checkpoint_filename(
                spec.checkpoint_state
            )
        )

        scaler = (
            evaluation
            / "pca"
            / "standard_scaler.joblib"
        )

        pca = (
            evaluation
            / "pca"
            / "pca_model.joblib"
        )

        svm = (
            evaluation
            / "models"
            / "svm.joblib"
        )

        final_results = (
            evaluation
            / "results"
            / "final_results.json"
        )

        training_results = (
            experiment
            / "results"
            / "training_results.json"
        )


    else:

        raise ValueError(
            f"Unsupported stage: {spec.stage}"
        )


    return ModelArtifacts(

        pipeline_id=spec.pipeline_id,

        backbone=spec.backbone,
        display_name=DISPLAY_NAMES[
            spec.backbone
        ],

        stage=spec.stage,
        variant=spec.variant,
        checkpoint_state=spec.checkpoint_state,
        selected_epoch=spec.selected_epoch,

        learning_rate=spec.learning_rate,
        lstm_hidden_size=spec.lstm_hidden_size,
        dropout=spec.dropout,

        final_representative=(
            spec.final_representative
        ),

        formal_xai_model=(
            spec.formal_xai_model
        ),

        checkpoint=checkpoint,
        svm=svm,
        scaler=scaler,
        pca=pca,

        final_results=final_results,
        training_results=training_results,
    )


# ============================================================================
# BACKWARD-COMPATIBLE PHASE 1 API
# ============================================================================
#
# Keep these functions because:
#
# - results_loader.py uses them
# - the original Grad-CAM evaluation app uses ModelManager with bare backbones
# - the formal occlusion evaluation app uses bare backbones
#

def get_artifacts(
    project_root: Path,
    backbone: str,
) -> ModelArtifacts:

    return get_pipeline_artifacts(
        project_root,
        pipeline_id_for(
            "Phase 1",
            backbone,
        ),
    )


def all_artifacts(
    project_root: Path,
) -> list[ModelArtifacts]:

    return [
        get_artifacts(
            project_root,
            backbone,
        )
        for backbone in BACKBONE_ORDER
    ]


def all_pipeline_artifacts(
    project_root: Path,
) -> list[ModelArtifacts]:

    return [
        get_pipeline_artifacts(
            project_root,
            pipeline_id,
        )
        for pipeline_id in PIPELINE_ORDER
    ]