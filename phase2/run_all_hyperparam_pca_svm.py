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

EVALUATOR = (
    PROJECT_ROOT
    / "actual_training_scripts"
    / "evaluate_hyperparam_pca_svm.py"
)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "hyperparameter_dl"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "phase2"
    / "logs"
    / "hyperparameter_pca_svm"
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
# EXPERIMENT MATRIX
# ============================================================

BACKBONES = [
    "vgg16",
    "vgg19",
    "resnet50",
    "resnet152",
    "efficientnet_b3",
]

RUN_NAMES = [
    "lr_1e-5",
    "lr_5e-5",
    "lr_5e-4",
    "lstm_128",
    "lstm_512",
    "dropout_0.2",
    "dropout_0.5",
]

CHECKPOINT_TYPES = [
    "best",
    "epoch30",
]


# ============================================================
# HELPERS
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
        f.write(text + "\n")


def evaluation_name(checkpoint_type):

    if checkpoint_type == "best":
        return "best_val_within_30"

    return "epoch_30"


def checkpoint_filename(checkpoint_type):

    if checkpoint_type == "best":
        return "best_val_within_30.pth"

    return "epoch_30_model.pth"


# ============================================================
# PREFLIGHT
# ============================================================

def preflight():

    print("=" * 80)
    print("PHASE 2B PCA+SVM PREFLIGHT")
    print("=" * 80)

    missing = []

    for backbone in BACKBONES:

        for run_name in RUN_NAMES:

            for checkpoint_type in CHECKPOINT_TYPES:

                checkpoint = (
                    EXPERIMENT_ROOT
                    / backbone
                    / run_name
                    / "models"
                    / checkpoint_filename(
                        checkpoint_type
                    )
                )

                if not checkpoint.is_file():

                    missing.append(
                        checkpoint
                    )

    if missing:

        print(
            f"\nERROR: "
            f"{len(missing)} checkpoint(s) missing."
        )

        for path in missing:
            print(path)

        raise FileNotFoundError(
            "Preflight failed. "
            "No evaluations started."
        )

    print(
        "\n✓ All 70 source checkpoints exist."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not EVALUATOR.is_file():

        raise FileNotFoundError(
            f"Evaluator not found:\n"
            f"{EVALUATOR}"
        )

    preflight()

    total_runs = (
        len(BACKBONES)
        * len(RUN_NAMES)
        * len(CHECKPOINT_TYPES)
    )

    print()
    print("=" * 80)
    print("PHASE 2B FULL HYBRID EVALUATION QUEUE")
    print("=" * 80)

    print(
        f"Total planned evaluations: "
        f"{total_runs}"
    )

    print(
        f"Started: {timestamp()}"
    )

    append_summary(
        "=" * 80
    )

    append_summary(
        "PHASE 2B FULL HYBRID "
        "PCA+SVM EVALUATION"
    )

    append_summary(
        f"Started: {timestamp()}"
    )

    append_summary(
        f"Total evaluations: {total_runs}"
    )

    append_summary(
        "=" * 80
    )


    current_run = 0

    completed = 0
    skipped = 0
    failed = 0


    for backbone in BACKBONES:

        for run_name in RUN_NAMES:

            for checkpoint_type in CHECKPOINT_TYPES:

                current_run += 1

                eval_name = evaluation_name(
                    checkpoint_type
                )

                experiment_root = (
                    EXPERIMENT_ROOT
                    / backbone
                    / run_name
                )

                final_results = (
                    experiment_root
                    / "evaluations"
                    / eval_name
                    / "results"
                    / "final_results.json"
                )


                print()
                print("=" * 80)

                print(
                    f"EVALUATION "
                    f"{current_run}/{total_runs}"
                )

                print(
                    f"Backbone   : {backbone}"
                )

                print(
                    f"Run        : {run_name}"
                )

                print(
                    f"Checkpoint : "
                    f"{checkpoint_type}"
                )

                print(
                    f"Started    : {timestamp()}"
                )

                print("=" * 80)


                # ============================================
                # RESUME-SAFE SKIP
                # ============================================

                if final_results.is_file():

                    print(
                        "SKIPPED - final_results.json "
                        "already exists."
                    )

                    skipped += 1

                    append_summary(
                        f"{timestamp()} | "
                        f"SKIPPED | "
                        f"{backbone} | "
                        f"{run_name} | "
                        f"{checkpoint_type}"
                    )

                    continue


                # ============================================
                # LOG FILE
                # ============================================

                log_dir = (
                    LOG_ROOT
                    / backbone
                    / run_name
                )

                log_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log_path = (
                    log_dir
                    / f"{checkpoint_type}.log"
                )


                # ============================================
                # COMMAND
                # ============================================

                command = [

                    sys.executable,

                    str(EVALUATOR),

                    "--backbone",
                    backbone,

                    "--run-name",
                    run_name,

                    "--checkpoint-type",
                    checkpoint_type,

                    "--batch-size",
                    "32",

                    "--num-workers",
                    "4",

                    "--pca-components",
                    "64",

                    "--svm-c",
                    "1.0",
                ]


                # ============================================
                # EXECUTE
                # ============================================

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

                    assert (
                        process.stdout
                        is not None
                    )

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


                # ============================================
                # VERIFY
                # ============================================

                if (
                    return_code == 0
                    and
                    final_results.is_file()
                ):

                    completed += 1

                    print(
                        "\nSUCCESS - evaluation "
                        "artifact verified."
                    )

                    append_summary(
                        f"{timestamp()} | "
                        f"SUCCESS | "
                        f"{backbone} | "
                        f"{run_name} | "
                        f"{checkpoint_type}"
                    )

                else:

                    failed += 1

                    print(
                        "\nFAILED - evaluation "
                        "did not complete correctly."
                    )

                    append_summary(
                        f"{timestamp()} | "
                        f"FAILED | "
                        f"{backbone} | "
                        f"{run_name} | "
                        f"{checkpoint_type} | "
                        f"return_code="
                        f"{return_code}"
                    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 2B EVALUATION QUEUE FINISHED")
    print("=" * 80)

    print(
        f"Newly completed : {completed}"
    )

    print(
        f"Already existing: {skipped}"
    )

    print(
        f"Failed          : {failed}"
    )

    print(
        f"Total accounted : "
        f"{completed + skipped + failed}"
    )

    print(
        f"Finished        : "
        f"{timestamp()}"
    )

    print(
        f"Summary log     : "
        f"{SUMMARY_PATH}"
    )


    append_summary(
        "=" * 80
    )

    append_summary(
        "QUEUE FINISHED"
    )

    append_summary(
        f"Newly completed={completed}"
    )

    append_summary(
        f"Already existing={skipped}"
    )

    append_summary(
        f"Failed={failed}"
    )

    append_summary(
        f"Finished={timestamp()}"
    )

    append_summary(
        "=" * 80
    )


if __name__ == "__main__":
    main()