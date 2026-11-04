#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path.home() / "brain_tumor_project"

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "actual_training_scripts"
    / "train_backbone_hyperparams.py"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "hyperparameter_dl"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "logs"
    / "hyperparameter_dl"
)

SUMMARY_PATH = (
    LOG_ROOT
    / "run_summary.txt"
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIVE BACKBONES
# ============================================================

BACKBONES = [
    "vgg16",
    "vgg19",
    "resnet50",
    "resnet152",
    "efficientnet_b3",
]


# ============================================================
# SEVEN NEW DL HYPERPARAMETER CONFIGURATIONS
#
# Baseline already exists:
# LR=1e-4, LSTM=256, Dropout=0.3
# ============================================================

CONFIGS = [
    {
        "run_name": "lr_1e-5",
        "learning_rate": "1e-5",
        "lstm_hidden_size": 256,
        "dropout": 0.3,
    },

    {
        "run_name": "lr_5e-5",
        "learning_rate": "5e-5",
        "lstm_hidden_size": 256,
        "dropout": 0.3,
    },

    {
        "run_name": "lr_5e-4",
        "learning_rate": "5e-4",
        "lstm_hidden_size": 256,
        "dropout": 0.3,
    },

    {
        "run_name": "lstm_128",
        "learning_rate": "1e-4",
        "lstm_hidden_size": 128,
        "dropout": 0.3,
    },

    {
        "run_name": "lstm_512",
        "learning_rate": "1e-4",
        "lstm_hidden_size": 512,
        "dropout": 0.3,
    },

    {
        "run_name": "dropout_0.2",
        "learning_rate": "1e-4",
        "lstm_hidden_size": 256,
        "dropout": 0.2,
    },

    {
        "run_name": "dropout_0.5",
        "learning_rate": "1e-4",
        "lstm_hidden_size": 256,
        "dropout": 0.5,
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def append_summary(text):

    with SUMMARY_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TRAIN_SCRIPT.is_file():

        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}"
        )


    total_runs = (
        len(BACKBONES)
        * len(CONFIGS)
    )

    print(
        "=" * 80
    )

    print(
        "PHASE 2B - DL HYPERPARAMETER STUDY"
    )

    print(
        "=" * 80
    )

    print(
        f"Python: {sys.executable}"
    )

    print(
        f"Total planned runs: {total_runs}"
    )

    print(
        f"Started: {timestamp()}"
    )


    append_summary(
        "=" * 80
    )

    append_summary(
        "PHASE 2B DL HYPERPARAMETER STUDY"
    )

    append_summary(
        f"Started: {timestamp()}"
    )

    append_summary(
        f"Python: {sys.executable}"
    )

    append_summary(
        f"Total planned runs: {total_runs}"
    )

    append_summary(
        "=" * 80
    )


    current_run = 0


    for backbone in BACKBONES:

        for config in CONFIGS:

            current_run += 1

            run_name = config[
                "run_name"
            ]

            learning_rate = config[
                "learning_rate"
            ]

            lstm_hidden_size = config[
                "lstm_hidden_size"
            ]

            dropout = config[
                "dropout"
            ]


            # ------------------------------------------------
            # Expected output artifacts
            # ------------------------------------------------

            experiment_root = (
                OUTPUT_ROOT
                / backbone
                / run_name
            )

            best_checkpoint = (
                experiment_root
                / "models"
                / "best_val_within_30.pth"
            )

            epoch30_checkpoint = (
                experiment_root
                / "models"
                / "epoch_30_model.pth"
            )

            training_results = (
                experiment_root
                / "results"
                / "training_results.json"
            )


            # ------------------------------------------------
            # Log path
            # ------------------------------------------------

            backbone_log_dir = (
                LOG_ROOT
                / backbone
            )

            backbone_log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            log_path = (
                backbone_log_dir
                / f"{run_name}.log"
            )


            print()
            print(
                "=" * 80
            )

            print(
                f"RUN {current_run}/{total_runs}"
            )

            print(
                f"Backbone       : {backbone}"
            )

            print(
                f"Run            : {run_name}"
            )

            print(
                f"Learning rate  : {learning_rate}"
            )

            print(
                f"LSTM hidden    : {lstm_hidden_size}"
            )

            print(
                f"Dropout        : {dropout}"
            )

            print(
                f"Started        : {timestamp()}"
            )

            print(
                "=" * 80
            )


            # ------------------------------------------------
            # Resume-safe skip
            # ------------------------------------------------

            if (
                best_checkpoint.is_file()
                and
                epoch30_checkpoint.is_file()
                and
                training_results.is_file()
            ):

                print(
                    "SKIPPED - complete artifacts already exist."
                )

                append_summary(
                    f"{timestamp()} | "
                    f"SKIPPED | "
                    f"{backbone} | "
                    f"{run_name} | "
                    f"LR={learning_rate} | "
                    f"LSTM={lstm_hidden_size} | "
                    f"dropout={dropout}"
                )

                continue


            # ------------------------------------------------
            # Training command
            # ------------------------------------------------

            command = [
                sys.executable,
                str(
                    TRAIN_SCRIPT
                ),

                "--backbone",
                backbone,

                "--run-name",
                run_name,

                "--epochs",
                "30",

                "--batch-size",
                "8",

                "--learning-rate",
                str(
                    learning_rate
                ),

                "--lstm-hidden-size",
                str(
                    lstm_hidden_size
                ),

                "--dropout",
                str(
                    dropout
                ),

                "--num-workers",
                "4",
            ]


            # ------------------------------------------------
            # Execute while showing terminal output AND
            # simultaneously saving it to log file.
            # ------------------------------------------------

            with log_path.open(
                "w",
                encoding="utf-8",
                buffering=1,
            ) as log_file:

                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                assert process.stdout is not None

                for line in process.stdout:

                    print(
                        line,
                        end="",
                        flush=True,
                    )

                    log_file.write(
                        line
                    )

                return_code = (
                    process.wait()
                )


            # ------------------------------------------------
            # Verify result
            # ------------------------------------------------

            if return_code != 0:

                print(
                    f"\nFAILED with exit code "
                    f"{return_code}"
                )

                append_summary(
                    f"{timestamp()} | "
                    f"FAILED(code={return_code}) | "
                    f"{backbone} | "
                    f"{run_name} | "
                    f"LR={learning_rate} | "
                    f"LSTM={lstm_hidden_size} | "
                    f"dropout={dropout}"
                )

                # Continue to next run.
                continue


            if (
                best_checkpoint.is_file()
                and
                epoch30_checkpoint.is_file()
                and
                training_results.is_file()
            ):

                print(
                    "\nSUCCESS - all expected artifacts found."
                )

                append_summary(
                    f"{timestamp()} | "
                    f"SUCCESS | "
                    f"{backbone} | "
                    f"{run_name} | "
                    f"LR={learning_rate} | "
                    f"LSTM={lstm_hidden_size} | "
                    f"dropout={dropout}"
                )

            else:

                print(
                    "\nWARNING - process finished but "
                    "expected artifacts are incomplete."
                )

                append_summary(
                    f"{timestamp()} | "
                    f"INCOMPLETE_ARTIFACTS | "
                    f"{backbone} | "
                    f"{run_name} | "
                    f"LR={learning_rate} | "
                    f"LSTM={lstm_hidden_size} | "
                    f"dropout={dropout}"
                )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "PHASE 2B TRAINING QUEUE FINISHED"
    )

    print(
        f"Finished: {timestamp()}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        "=" * 80
    )


    append_summary(
        "=" * 80
    )

    append_summary(
        "PHASE 2B TRAINING QUEUE FINISHED"
    )

    append_summary(
        f"Finished: {timestamp()}"
    )

    append_summary(
        "=" * 80
    )


if __name__ == "__main__":
    main()
