from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATH SETUP
# ============================================================

# Current script:
#
# brain_tumor_project/
# └── actual_training_scripts/
#     └── train_resnet152_lstm.py
#
# Therefore:
#
# .parent        = actual_training_scripts
# .parent.parent = brain_tumor_project

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Add project root to Python import path
if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_pipeline_torch import create_dataloaders
from model_resnet152_lstm import ResNet152LSTM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_resnet152_lstm.pth"
)


# ------------------------------------------------------------
# MULTI-EPOCH TRAINING
# ------------------------------------------------------------

EPOCHS = 10

TRAIN_BATCH_SIZE = 8

NUM_WORKERS = 4

LEARNING_RATE = 1e-4


# ------------------------------------------------------------
# EARLY STOPPING
# ------------------------------------------------------------

# Stop if validation accuracy fails to improve
# for 3 consecutive epochs.

PATIENCE = 3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("RESNET152 + LSTM TRAINING")
print("=" * 70)


print(
    "\nDevice:",
    device,
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# LOAD EXISTING DATA PIPELINE
# ============================================================

(
    base_train_dataset,
    train_dataset,
    val_dataset,
    test_dataset,
    original_train_loader,
    original_val_loader,
    test_loader,
    train_indices,
    val_indices,
) = create_dataloaders()


print("\nDataset sizes:")


print(
    "Training  :",
    len(train_dataset),
)


print(
    "Validation:",
    len(val_dataset),
)


print(
    "Testing   :",
    len(test_dataset),
)


# ============================================================
# TRAINING-SPECIFIC DATALOADERS
# ============================================================

# We reuse the EXACT SAME datasets and saved split.
#
# Only the batch size is changed to 8 for training.

pin_memory = torch.cuda.is_available()


train_loader = DataLoader(

    train_dataset,

    batch_size=TRAIN_BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory,

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Training dataset = 4569 images
    # Batch size       = 8
    #
    # 4569 = 571 × 8 + 1
    #
    # Without drop_last=True, the final training batch
    # contains only one sample.
    #
    # BatchNorm1d cannot calculate training batch statistics
    # from a batch containing only one sample.
    #
    # Therefore the incomplete final batch is discarded.
    #
    # Because shuffle=True, the same MRI is not permanently
    # excluded across epochs.
    # --------------------------------------------------------

    drop_last=True,
)


val_loader = DataLoader(

    val_dataset,

    batch_size=TRAIN_BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory,
)


print(
    "\nTraining batch size:",
    TRAIN_BATCH_SIZE,
)


print(
    "Training batches:",
    len(train_loader),
)


print(
    "Training samples used per epoch:",
    len(train_loader) * TRAIN_BATCH_SIZE,
)


print(
    "Incomplete training samples dropped:",
    len(train_dataset)
    - (
        len(train_loader)
        * TRAIN_BATCH_SIZE
    ),
)


print(
    "Validation batches:",
    len(val_loader),
)


# ============================================================
# MODEL
# ============================================================

model = ResNet152LSTM(

    lstm_hidden_size=256,

    dense_features=128,

    num_classes=4,

    dropout=0.3,
)


model = model.to(device)


print(
    "\nModel moved to:",
    device,
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.RMSprop(

    model.parameters(),

    lr=LEARNING_RATE,
)


print(
    "Optimizer: RMSprop"
)


print(
    "Learning rate:",
    LEARNING_RATE,
)


print(
    "Maximum epochs:",
    EPOCHS,
)


print(
    "Early stopping patience:",
    PATIENCE,
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


        # ----------------------------------------------------
        # MOVE BATCH TO GPU
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True,
        )


        labels = labels.to(
            device,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # CLEAR PREVIOUS GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        logits = model(images)


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels,
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE MODEL PARAMETERS
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # ACCUMULATE LOSS
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=1,
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        # ----------------------------------------------------
        # TRAINING PROGRESS
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 50 == 0
            or
            (batch_index + 1) == len(loader)
        ):

            current_accuracy = (
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
                f"{current_accuracy:.2f}%"

            )


    # --------------------------------------------------------
    # EPOCH METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # NO GRADIENTS DURING VALIDATION
    # --------------------------------------------------------

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


            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            logits = model(images)


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1,
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

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
# TRAINING HISTORY
# ============================================================

# Store metrics from every completed epoch.
#
# We will later use these values to inspect:
#
# training loss
# validation loss
# training accuracy
# validation accuracy

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],
}


# ============================================================
# TRAINING LOOP STATE
# ============================================================

best_val_accuracy = 0.0


epochs_without_improvement = 0


print(
    "\n"
    + "=" * 70
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 70
)


training_start = time.time()


# ============================================================
# MULTI-EPOCH TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):


    epoch_start = time.time()


    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )


    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device,
        )
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    val_loss, val_accuracy = (
        validate(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device,
        )
    )


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_seconds = (
        time.time()
        - epoch_start
    )


    # ========================================================
    # STORE EPOCH HISTORY
    # ========================================================

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


    # ========================================================
    # DISPLAY EPOCH RESULTS
    # ========================================================

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
        f"{epoch_seconds / 60:.2f} minutes"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:


        # ----------------------------------------------------
        # VALIDATION IMPROVED
        # ----------------------------------------------------

        best_val_accuracy = val_accuracy


        epochs_without_improvement = 0


        checkpoint = {

            "epoch":
                epoch + 1,

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
                256,

            "dense_features":
                128,

            "dropout":
                0.3,

            "learning_rate":
                LEARNING_RATE,

            "train_batch_size":
                TRAIN_BATCH_SIZE,

            "history":
                history,
        }


        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )


        print(
            "\n✓ New best model saved:"
        )


        print(
            CHECKPOINT_PATH
        )


        print(
            f"✓ Best validation accuracy "
            f"is now {best_val_accuracy:.2f}%"
        )


    else:


        # ----------------------------------------------------
        # VALIDATION DID NOT IMPROVE
        # ----------------------------------------------------

        epochs_without_improvement += 1


        print(
            f"\nNo validation improvement "
            f"for "
            f"{epochs_without_improvement} "
            f"epoch(s)."
        )


        print(
            f"Best validation accuracy "
            f"remains "
            f"{best_val_accuracy:.2f}%"
        )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):


            print(
                "\n"
                + "=" * 70
            )


            print(
                "EARLY STOPPING TRIGGERED"
            )


            print(
                "=" * 70
            )


            print(
                f"\nValidation accuracy "
                f"did not improve for "
                f"{PATIENCE} consecutive epochs."
            )


            print(
                "\nStopping training early "
                "to avoid unnecessary "
                "over-training."
            )


            break


# ============================================================
# FINAL SUMMARY
# ============================================================

total_seconds = (
    time.time()
    - training_start
)


epochs_completed = len(
    history["train_loss"]
)


print(
    "\n"
    + "=" * 70
)


print(
    "TRAINING RUN COMPLETE"
)


print(
    "=" * 70
)


print(
    f"\nEpochs completed: "
    f"{epochs_completed}"
)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)


print(
    f"Total time: "
    f"{total_seconds / 60:.2f} minutes"
)


print(
    "\nBest model checkpoint:"
)


print(
    CHECKPOINT_PATH
)


# ============================================================
# PRINT TRAINING HISTORY
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "TRAINING HISTORY"
)


print(
    "=" * 70
)


for index in range(
    epochs_completed
):


    print(

        f"Epoch {index + 1:2d}"

        f" | Train Loss: "
        f"{history['train_loss'][index]:.4f}"

        f" | Train Acc: "
        f"{history['train_accuracy'][index]:.2f}%"

        f" | Val Loss: "
        f"{history['val_loss'][index]:.4f}"

        f" | Val Acc: "
        f"{history['val_accuracy'][index]:.2f}%"

    )


print(
    "\nNEXT STEP: inspect the "
    "multi-epoch training results "
    "and select the best checkpoint "
    "before deep-feature extraction "
    "for PCA + SVM."
)