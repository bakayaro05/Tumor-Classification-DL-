from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_pipeline_torch import create_datasets
from model_cnn_lstm import (
    CNNLSTM,
    SUPPORTED_BACKBONES,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--backbone",
    required=True,
    choices=SUPPORTED_BACKBONES,
)

parser.add_argument(
    "--epochs",
    type=int,
    default=30,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "--lstm-hidden-size",
    type=int,
    default=256,
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.3,
)

parser.add_argument(
    "--run-name",
    required=True,
    help="Unique Phase-2B experiment name, e.g. lr_1e-5",
)

parser.add_argument(
    "--patience",
    type=int,
    default=3,
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Resume training from the existing best checkpoint "
        "for this backbone."
    ),
)

args = parser.parse_args()
RESUME = args.resume


# ============================================================
# CONFIGURATION
# ============================================================

BACKBONE = args.backbone

EPOCHS = args.epochs

BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate

LSTM_HIDDEN_SIZE = args.lstm_hidden_size

DROPOUT = args.dropout

RUN_NAME = args.run_name

# Phase 2A: early stopping is intentionally disabled.
# Setting patience above the total epoch count means the
# legacy early-stopping branch can never terminate the run.
PATIENCE = EPOCHS + 1

NUM_WORKERS = args.num_workers


DENSE_FEATURES = 128

NUM_CLASSES = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# EXPERIMENT DIRECTORIES
# ============================================================

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "hyperparameter_dl"
    / BACKBONE
    / RUN_NAME
)

MODEL_DIR = (
    EXPERIMENT_ROOT
    / "models"
)

RESULT_DIR = (
    EXPERIMENT_ROOT
    / "results"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_val_within_30.pth"
)

FINAL_CHECKPOINT_PATH = (
    MODEL_DIR
    / "epoch_30_model.pth"
)

TRAINING_RESULTS_PATH = (
    RESULT_DIR
    / "training_results.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)

print(
    f"COMPARATIVE TRAINING: "
    f"{BACKBONE.upper()}"
)

print("=" * 80)


print(
    "\nDevice:",
    device
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print(
    "\nRandom seed:",
    SEED
)


# ============================================================
# DATASETS
# ============================================================

(
    base_train_dataset,
    train_dataset,
    val_dataset,
    test_dataset,
    train_indices,
    val_indices,
) = create_datasets()


print(
    "\nDataset sizes:"
)

print(
    "Training  :",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Testing   :",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = torch.cuda.is_available()


# Seeded generator for reproducible shuffle order.

loader_generator = torch.Generator()

loader_generator.manual_seed(
    SEED
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory,

    drop_last=True,

    generator=loader_generator,
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory,
)


print(
    "\nBatch size:",
    BATCH_SIZE
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)

print(
    "Training samples used per epoch:",
    len(train_loader) * BATCH_SIZE
)

print(
    "Incomplete samples dropped:",
    len(train_dataset)
    - len(train_loader) * BATCH_SIZE
)


# ============================================================
# MODEL
# ============================================================

model = CNNLSTM(

    backbone_name=BACKBONE,

    lstm_hidden_size=LSTM_HIDDEN_SIZE,

    dense_features=DENSE_FEATURES,

    num_classes=NUM_CLASSES,

    dropout=DROPOUT,
)

# ============================================================
# VGG TRANSFER-LEARNING STABILIZATION
# ============================================================

if BACKBONE in [
    "vgg16",
    "vgg19",
]:

    print(
        "\nFreezing pretrained VGG backbone "
        "for stable transfer learning..."
    )

    for parameter in model.cnn_backbone.parameters():
        parameter.requires_grad = False

    print(
        "✓ VGG convolutional backbone frozen"
    )

model = model.to(
    device
)


print(
    "\nCNN output channels:",
    model.cnn_output_channels
)


print(
    "Model moved to:",
    device
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()


trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.RMSprop(
    trainable_parameters,
    lr=LEARNING_RATE,
)


trainable_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_count = sum(
    p.numel()
    for p in model.parameters()
    if not p.requires_grad
)

print(
    "\nTrainable parameters:",
    f"{trainable_count:,}"
)

print(
    "Frozen parameters:",
    f"{frozen_count:,}"
)


print(
    "\nPhase 2B run:",
    RUN_NAME
)

print(
    "LSTM hidden size:",
    LSTM_HIDDEN_SIZE
)

print(
    "Dropout:",
    DROPOUT
)

print(
    "\nOptimizer: RMSprop"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Maximum epochs:",
    EPOCHS
)

print(
    "Early stopping:",
    "DISABLED for Phase 2A fixed-duration training"
)

print(
    "Resume requested:",
    RESUME
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for batch_index, (
        images,
        labels,
    ) in enumerate(loader):


        images = images.to(
            device,
            non_blocking=True,
        )


        labels = labels.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels,
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1,
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        if (
            (batch_index + 1) % 50 == 0
            or
            (batch_index + 1) == len(loader)
        ):

            running_accuracy = (
                100.0
                * correct
                / total
            )


            print(

                f"    Train batch "
                f"{batch_index + 1:4d}"
                f"/{len(loader):4d}"

                f" | Loss: "
                f"{loss.item():.4f}"

                f" | Running Acc: "
                f"{running_accuracy:.2f}%"

            )


    epoch_loss = (
        running_loss
        / total
    )


    epoch_accuracy = (
        100.0
        * correct
        / total
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():


        for images, labels in loader:


            images = images.to(
                device,
                non_blocking=True,
            )


            labels = labels.to(
                device,
                non_blocking=True,
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels,
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    epoch_loss = (
        running_loss
        / total
    )


    epoch_accuracy = (
        100.0
        * correct
        / total
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# HISTORY
# ============================================================

# ============================================================
# HISTORY / RESUME STATE
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_time_seconds": [],
}


best_val_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0

start_epoch = 0


# ============================================================
# RESUME FROM BEST CHECKPOINT
# ============================================================

if RESUME:

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"--resume was requested, but checkpoint "
            f"does not exist:\n{CHECKPOINT_PATH}"
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESUMING EXISTING EXPERIMENT"
    )

    print(
        "=" * 80
    )


    resume_checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )


    saved_backbone = resume_checkpoint.get(
        "backbone"
    )


    if saved_backbone != BACKBONE:

        raise ValueError(
            f"Checkpoint backbone is "
            f"{saved_backbone}, "
            f"but requested backbone is "
            f"{BACKBONE}."
        )


    # --------------------------------------------------------
    # LOAD MODEL WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(
        resume_checkpoint[
            "model_state_dict"
        ]
    )


    # --------------------------------------------------------
    # LOAD OPTIMIZER STATE
    # --------------------------------------------------------

    optimizer.load_state_dict(
        resume_checkpoint[
            "optimizer_state_dict"
        ]
    )


    # --------------------------------------------------------
    # RESTORE TRAINING STATE
    # --------------------------------------------------------

    start_epoch = int(
        resume_checkpoint[
            "epoch"
        ]
    )


    best_epoch = start_epoch


    best_val_accuracy = float(
        resume_checkpoint[
            "val_accuracy"
        ]
    )


    saved_history = resume_checkpoint.get(
        "history"
    )


    if saved_history is not None:

        history = saved_history


    # We are resuming from the BEST checkpoint,
    # therefore the non-improvement counter resets.

    epochs_without_improvement = 0


    print(
        "\n✓ Model weights restored"
    )

    print(
        "✓ Optimizer state restored"
    )

    print(
        "✓ Training history restored"
    )


    print(
        "\nResume checkpoint epoch:",
        start_epoch
    )


    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )


    print(
        "Target total epochs:",
        EPOCHS
    )


    if start_epoch >= EPOCHS:

        raise ValueError(
            f"Checkpoint is already at epoch "
            f"{start_epoch}, while requested "
            f"maximum epochs is {EPOCHS}. "
            f"Use a larger --epochs value."
        )

# ============================================================
# TRAINING
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 80
)


training_start = time.time()


for epoch in range(
    start_epoch,
    EPOCHS,
):


    epoch_start = time.time()


    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )


    print(
        "-" * 80
    )


    train_loss, train_accuracy = (
        train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device,
        )
    )


    val_loss, val_accuracy = (
        validate(

            model,

            val_loader,

            criterion,

            device,
        )
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    history[
        "train_loss"
    ].append(
        train_loss
    )


    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )


    history[
        "val_loss"
    ].append(
        val_loss
    )


    history[
        "val_accuracy"
    ].append(
        val_accuracy
    )


    history[
        "epoch_time_seconds"
    ].append(
        epoch_time
    )


    print(
        "\nEpoch results:"
    )


    print(
        f"    Train Loss : "
        f"{train_loss:.4f}"
    )


    print(
        f"    Train Acc  : "
        f"{train_accuracy:.2f}%"
    )


    print(
        f"    Val Loss   : "
        f"{val_loss:.4f}"
    )


    print(
        f"    Val Acc    : "
        f"{val_accuracy:.2f}%"
    )


    print(
        f"    Epoch Time : "
        f"{epoch_time / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:


        best_val_accuracy = (
            val_accuracy
        )


        best_epoch = (
            epoch + 1
        )


        epochs_without_improvement = 0


        checkpoint = {

            "backbone":
                BACKBONE,

            "run_name":
                RUN_NAME,

            "epoch":
                best_epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "class_to_idx":
                base_train_dataset.class_to_idx,

            "lstm_hidden_size":
                LSTM_HIDDEN_SIZE,

            "dense_features":
                DENSE_FEATURES,

            "dropout":
                DROPOUT,

            "learning_rate":
                LEARNING_RATE,

            "batch_size":
                BATCH_SIZE,

            "seed":
                SEED,

            "history":
                history,

            "phase":
                "phase2B_dl_hyperparameter",
        }


        torch.save(

            checkpoint,

            CHECKPOINT_PATH,
        )


        print(
            "\n✓ New best model saved"
        )


        print(
            CHECKPOINT_PATH
        )


        print(
            f"✓ Best validation accuracy: "
            f"{best_val_accuracy:.2f}%"
        )


    else:


        epochs_without_improvement += 1


        print(

            f"\nNo validation improvement "
            f"for "
            f"{epochs_without_improvement} "
            f"epoch(s)."

        )


        if (
            epochs_without_improvement
            >= PATIENCE
        ):


            print(
                "\nEARLY STOPPING TRIGGERED"
            )


            break


# ============================================================
# SAVE FINAL EPOCH CHECKPOINT
# ============================================================

# With early stopping disabled, the model currently contains
# the weights from the requested final epoch.

final_checkpoint = {
    "backbone": BACKBONE,
    "run_name": RUN_NAME,
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),

    "train_loss": train_loss,
    "train_accuracy": train_accuracy,

    "val_loss": val_loss,
    "val_accuracy": val_accuracy,

    "class_to_idx":
        base_train_dataset.class_to_idx,

    "lstm_hidden_size":
        LSTM_HIDDEN_SIZE,

    "dense_features":
        DENSE_FEATURES,

    "dropout":
        DROPOUT,

    "learning_rate":
        LEARNING_RATE,

    "batch_size":
        BATCH_SIZE,

    "seed":
        SEED,

    "history":
        history,

    "phase":
        "phase2B_dl_hyperparameter",
}

torch.save(
    final_checkpoint,
    FINAL_CHECKPOINT_PATH,
)

print(
    "\n✓ Final epoch checkpoint saved"
)

print(
    FINAL_CHECKPOINT_PATH
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

training_seconds = (
    time.time()
    - training_start
)


epochs_completed = len(
    history[
        "train_loss"
    ]
)


results = {

    "backbone":
        BACKBONE,

    "run_name":
        RUN_NAME,

    "seed":
        SEED,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        epochs_completed,

    "best_epoch":
        best_epoch,

    "best_val_accuracy":
        best_val_accuracy,

    "training_time_seconds":
        training_seconds,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "lstm_hidden_size":
        LSTM_HIDDEN_SIZE,

    "dense_features":
        DENSE_FEATURES,

    "dropout":
        DROPOUT,

    "history":
        history,
}


with open(
    TRAINING_RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


print(
    "\n"
    + "=" * 80
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nBackbone:",
    BACKBONE
)


print(
    "Epochs completed:",
    epochs_completed
)


print(
    "Best epoch:",
    best_epoch
)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)


print(
    f"Total training time: "
    f"{training_seconds / 60:.2f} minutes"
)


print(
    "\nBest-within-30 checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print(
    "\nEpoch-30 checkpoint:"
)

print(
    FINAL_CHECKPOINT_PATH
)


print(
    "\nTraining results:"
)

print(
    TRAINING_RESULTS_PATH
)


print(
    "\nNEXT STEP:"
)

print(
    f"Extract deterministic deep features "
    f"from the best {BACKBONE} checkpoint."
)