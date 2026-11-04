#!/usr/bin/env python3
"""
Create one deterministic class-balanced subset from the already-audited
BDNeuro exact-overlap-clean cohort.

Source cohort (current project):
    external_data/BDNeuro_V8_exact_clean/
        glioma/
        meningioma/
        notumor/
        pituitary/

Current counts:
    glioma      1159
    meningioma   472
    notumor      412
    pituitary    601

Therefore the default balanced subset uses the smallest class size:
    412 images/class = 1,648 images total.

The source 2,644-image cohort is never modified.

Selection is deterministic:
    SHA256(f"{seed}|{class}/{filename}")
Images are ranked by that value within each class, then the first N are copied.

Run:
    cd ~/brain_tumor_project
    source venv/bin/activate
    python actual_training_scripts/build_bdneuro_balanced_subset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "external_data"
    / "BDNeuro_V8_exact_clean"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "external_data"
    / "BDNeuro_V8_exact_clean_balanced"
)

DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT
    / "phase2"
    / "external_validation"
    / "bdneuro_balanced_subset"
)

CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


parser = argparse.ArgumentParser()

parser.add_argument(
    "--source-dir",
    type=Path,
    default=DEFAULT_SOURCE,
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT,
)

parser.add_argument(
    "--manifest-dir",
    type=Path,
    default=DEFAULT_MANIFEST_DIR,
)

parser.add_argument(
    "--per-class",
    type=int,
    default=None,
    help=(
        "Images per class. If omitted, automatically uses the "
        "smallest source-class count."
    ),
)

parser.add_argument(
    "--seed",
    type=str,
    default="bdneuro-balanced-v1-seed42",
)

parser.add_argument(
    "--rebuild",
    action="store_true",
    help=(
        "Delete and rebuild only the derived balanced subset and its "
        "manifest directory. The 2,644-image source cohort is never touched."
    ),
)

args = parser.parse_args()


def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def project_relative(path: Path) -> str:
    return str(
        path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        )
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def selection_rank(
    class_name: str,
    filename: str,
    seed: str,
) -> str:
    text = (
        f"{seed}|{class_name}/{filename}"
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def save_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def main() -> None:
    print("=" * 80)
    print("BUILD DETERMINISTIC BALANCED BDNEURO EXTERNAL SUBSET")
    print("=" * 80)

    if not args.source_dir.is_dir():
        raise FileNotFoundError(
            f"Source cohort not found: {args.source_dir}"
        )

    source_by_class = {}

    for class_name in CLASS_NAMES:
        class_dir = (
            args.source_dir
            / class_name
        )

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        files = sorted(
            path
            for path in class_dir.rglob("*")
            if is_image(path)
        )

        source_by_class[
            class_name
        ] = files

    counts = {
        class_name: len(files)
        for class_name, files
        in source_by_class.items()
    }

    print("\nSOURCE COUNTS")
    for class_name in CLASS_NAMES:
        print(
            f"{class_name:12s}: "
            f"{counts[class_name]}"
        )

    smallest = min(
        counts.values()
    )

    n_per_class = (
        smallest
        if args.per_class is None
        else args.per_class
    )

    if n_per_class <= 0:
        raise ValueError(
            "--per-class must be positive."
        )

    if any(
        n_per_class > count
        for count in counts.values()
    ):
        raise ValueError(
            f"Requested {n_per_class} per class but "
            f"smallest class contains only {smallest}."
        )

    print(
        "\nSelected images per class:",
        n_per_class,
    )

    print(
        "Balanced total            :",
        n_per_class * len(CLASS_NAMES),
    )

    if args.rebuild:
        if args.output_dir.exists():
            shutil.rmtree(
                args.output_dir
            )

        if args.manifest_dir.exists():
            shutil.rmtree(
                args.manifest_dir
            )

    else:
        if args.output_dir.exists():
            raise RuntimeError(
                "\nOutput directory already exists:\n"
                f"{args.output_dir}\n\n"
                "Nothing was changed. Use --rebuild only if you intentionally "
                "want to recreate the derived balanced subset."
            )

        if args.manifest_dir.exists():
            raise RuntimeError(
                "\nManifest directory already exists:\n"
                f"{args.manifest_dir}\n\n"
                "Nothing was changed. Use --rebuild only if you intentionally "
                "want to recreate the derived balanced subset."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_rows = []

    for class_name in CLASS_NAMES:
        destination_class = (
            args.output_dir
            / class_name
        )

        destination_class.mkdir(
            parents=True,
            exist_ok=True,
        )

        ranked = sorted(
            source_by_class[class_name],
            key=lambda path: (
                selection_rank(
                    class_name,
                    path.name,
                    args.seed,
                ),
                path.name,
            ),
        )

        selected = ranked[
            :n_per_class
        ]

        for path in selected:
            destination = (
                destination_class
                / path.name
            )

            shutil.copy2(
                path,
                destination,
            )

            source_hash = sha256_file(
                path
            )

            destination_hash = sha256_file(
                destination
            )

            if (
                source_hash
                != destination_hash
            ):
                raise RuntimeError(
                    "Copy verification failed:\n"
                    f"{path}\n"
                    f"{destination}"
                )

            selected_rows.append({
                "class": class_name,
                "source_path": project_relative(
                    path
                ),
                "destination_path": project_relative(
                    destination
                ),
                "selection_rank_sha256": selection_rank(
                    class_name,
                    path.name,
                    args.seed,
                ),
                "file_sha256": source_hash,
            })

    # Verify final counts.
    final_counts = Counter()

    for class_name in CLASS_NAMES:
        final_counts[class_name] = len([
            p
            for p in (
                args.output_dir
                / class_name
            ).rglob("*")
            if is_image(p)
        ])

    expected_total = (
        n_per_class
        * len(CLASS_NAMES)
    )

    actual_total = sum(
        final_counts.values()
    )

    if actual_total != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} balanced images, "
            f"found {actual_total}."
        )

    if any(
        final_counts[class_name]
        != n_per_class
        for class_name in CLASS_NAMES
    ):
        raise RuntimeError(
            "Final class counts are not balanced."
        )

    manifest_csv = (
        args.manifest_dir
        / "balanced_selection_manifest.csv"
    )

    with manifest_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "source_path",
                "destination_path",
                "selection_rank_sha256",
                "file_sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(
            selected_rows
        )

    summary = {
        "source_cohort": project_relative(
            args.source_dir
        ),
        "source_counts": counts,
        "selection_method": (
            "Deterministic SHA-256 ranking within each class. "
            "No model outputs or performance values were used."
        ),
        "selection_seed": args.seed,
        "images_per_class": n_per_class,
        "balanced_total": expected_total,
        "balanced_counts": {
            class_name: final_counts[class_name]
            for class_name in CLASS_NAMES
        },
        "output_dir": project_relative(
            args.output_dir
        ),
        "source_cohort_modified": False,
    }

    save_json(
        args.manifest_dir
        / "balanced_subset_summary.json",
        summary,
    )

    print()
    print("=" * 80)
    print("FINAL BALANCED SUBSET")
    print("=" * 80)

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:12s}: "
            f"{final_counts[class_name]}"
        )

    print(
        f"{'TOTAL':12s}: "
        f"{actual_total}"
    )

    print("\nOutput:")
    print(args.output_dir)

    print("\nManifest:")
    print(manifest_csv)

    print(
        "\nSUCCESS: balanced subset created deterministically "
        "without modifying the full 2,644-image cohort."
    )


if __name__ == "__main__":
    main()
