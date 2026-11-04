#!/usr/bin/env python3
"""
Build an exact-overlap-clean BDNeuro V8 external cohort.

This script DOES NOT delete or modify the raw BDNeuro dataset.

It:
1) reads the cross-dataset exact decoded-pixel overlap audit CSV;
2) excludes every unique BDNeuro image listed there;
3) keeps all other BDNeuro images, including non-zero pHash similarity candidates;
4) merges BDNeuro train/val/test into one ImageFolder-compatible external cohort;
5) maps BDNeuro "no_tumor" -> project class name "notumor";
6) writes traceability manifests and verifies expected counts.

Expected from the completed audit:
- Raw BDNeuro MRI cohort: 5,941 images
- Unique exact-pixel-overlap BDNeuro images: 3,297
- Exact-pixel-clean external cohort: 2,644 images

Run:
    cd ~/brain_tumor_project
    source venv/bin/activate
    python actual_training_scripts/build_bdneuro_exact_clean_cohort.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BDROOT = (
    PROJECT_ROOT
    / "external_data"
    / "BDNeuro_V8_raw"
    / "Brain_Tumor_MRI_Dataset_Final"
)

DEFAULT_AUDIT_CSV = (
    PROJECT_ROOT
    / "phase2"
    / "external_validation"
    / "bdneuro_overlap_audit"
    / "cross_exact_pixel_duplicates.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "external_data"
    / "BDNeuro_V8_exact_clean"
)

DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT
    / "phase2"
    / "external_validation"
    / "bdneuro_exact_clean_cohort"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

SPLITS = ("train", "val", "test")

CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "notumor",
    "pituitary": "pituitary",
}

EXPECTED_RAW_TOTAL = 5941
EXPECTED_EXCLUDED_UNIQUE = 3297
EXPECTED_CLEAN_TOTAL = 2644

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bdroot",
    type=Path,
    default=DEFAULT_BDROOT,
)

parser.add_argument(
    "--audit-csv",
    type=Path,
    default=DEFAULT_AUDIT_CSV,
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
    "--rebuild",
    action="store_true",
    help=(
        "Delete and rebuild ONLY the derived exact-clean output directory "
        "and its manifest directory. The raw BDNeuro dataset is never touched."
    ),
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

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


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


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


def collect_bdneuro_images(
    bdroot: Path,
) -> list[dict]:
    records = []

    for split in SPLITS:
        split_root = bdroot / split

        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Missing BDNeuro split: {split_root}"
            )

        for source_class, mapped_class in CLASS_MAP.items():
            class_dir = split_root / source_class

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing BDNeuro class folder: {class_dir}"
                )

            for path in sorted(
                class_dir.rglob("*")
            ):
                if not is_image(path):
                    continue

                records.append({
                    "source_path_obj": path,
                    "source_path": project_relative(path),
                    "source_split": split,
                    "source_class": source_class,
                    "mapped_class": mapped_class,
                })

    return records


def load_excluded_paths(
    audit_csv: Path,
) -> set[str]:
    if not audit_csv.is_file():
        raise FileNotFoundError(
            f"Audit CSV not found: {audit_csv}"
        )

    with audit_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            f"Audit CSV contains no rows: {audit_csv}"
        )

    if "bdneuro_path" not in rows[0]:
        raise RuntimeError(
            "Audit CSV does not contain the expected "
            "'bdneuro_path' column."
        )

    return {
        row["bdneuro_path"]
        for row in rows
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("BUILD BDNEURO V8 EXACT-PIXEL-CLEAN EXTERNAL COHORT")
    print("=" * 80)

    print("\nRaw BDNeuro :")
    print(args.bdroot)

    print("\nExact-overlap audit CSV:")
    print(args.audit_csv)

    print("\nDerived clean cohort:")
    print(args.output_dir)

    print("\nManifest directory:")
    print(args.manifest_dir)

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------

    if not args.bdroot.is_dir():
        raise FileNotFoundError(
            f"BDNeuro root not found: {args.bdroot}"
        )

    # --------------------------------------------------------
    # Rebuild behavior
    # --------------------------------------------------------

    if args.rebuild:
        if args.output_dir.exists():
            print(
                "\nRebuild requested: removing derived "
                "clean cohort only..."
            )

            shutil.rmtree(
                args.output_dir
            )

        if args.manifest_dir.exists():
            print(
                "Rebuild requested: removing previous "
                "manifest directory..."
            )

            shutil.rmtree(
                args.manifest_dir
            )

    else:
        if args.output_dir.exists():
            raise RuntimeError(
                "\nOutput directory already exists:\n"
                f"{args.output_dir}\n\n"
                "Nothing was changed. If you intentionally want "
                "to rebuild the DERIVED cohort, rerun with --rebuild."
            )

        if args.manifest_dir.exists():
            raise RuntimeError(
                "\nManifest directory already exists:\n"
                f"{args.manifest_dir}\n\n"
                "Nothing was changed. If you intentionally want "
                "to rebuild the DERIVED cohort, rerun with --rebuild."
            )

    # --------------------------------------------------------
    # Collect source cohort + exact exclusions
    # --------------------------------------------------------

    source_records = collect_bdneuro_images(
        args.bdroot
    )

    excluded_paths = load_excluded_paths(
        args.audit_csv
    )

    print("\nSource MRI images        :", len(source_records))
    print("Unique exact exclusions  :", len(excluded_paths))

    if len(source_records) != EXPECTED_RAW_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_RAW_TOTAL} source MRIs, "
            f"found {len(source_records)}."
        )

    if len(excluded_paths) != EXPECTED_EXCLUDED_UNIQUE:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED_UNIQUE} unique exact-pixel "
            f"exclusions, found {len(excluded_paths)}."
        )

    source_path_set = {
        record["source_path"]
        for record in source_records
    }

    exclusions_not_found = (
        excluded_paths - source_path_set
    )

    if exclusions_not_found:
        examples = "\n".join(
            sorted(exclusions_not_found)[:10]
        )

        raise RuntimeError(
            "Some audited BDNeuro exclusion paths were not found "
            "in the current raw dataset.\n"
            f"Missing count: {len(exclusions_not_found)}\n"
            f"Examples:\n{examples}"
        )

    # --------------------------------------------------------
    # Create class directories
    # --------------------------------------------------------

    for mapped_class in CLASS_MAP.values():
        (
            args.output_dir
            / mapped_class
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    args.manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build derived cohort
    # --------------------------------------------------------

    kept_rows = []
    excluded_rows = []

    kept_counts_by_class = Counter()
    kept_counts_by_split = Counter()
    excluded_counts_by_class = Counter()
    excluded_counts_by_split = Counter()

    used_destination_paths = set()

    for index, record in enumerate(
        source_records,
        1,
    ):
        source_path = record["source_path_obj"]
        source_rel = record["source_path"]

        if source_rel in excluded_paths:
            excluded_counts_by_class[
                record["mapped_class"]
            ] += 1

            excluded_counts_by_split[
                record["source_split"]
            ] += 1

            excluded_rows.append({
                "source_path": source_rel,
                "source_split": record["source_split"],
                "source_class": record["source_class"],
                "mapped_class": record["mapped_class"],
                "exclusion_reason": (
                    "exact_decoded_pixel_match_to_internal_dataset"
                ),
            })

            continue

        # Stable short identifier based on original project-relative path.
        short_id = hashlib.sha256(
            source_rel.encode("utf-8")
        ).hexdigest()[:12]

        destination_name = (
            f"{record['source_split']}__"
            f"{source_path.stem}__"
            f"{short_id}"
            f"{source_path.suffix.lower()}"
        )

        destination_path = (
            args.output_dir
            / record["mapped_class"]
            / destination_name
        )

        destination_rel = project_relative(
            destination_path
        )

        if destination_rel in used_destination_paths:
            raise RuntimeError(
                f"Destination collision detected: {destination_path}"
            )

        used_destination_paths.add(
            destination_rel
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        source_sha256 = sha256_file(
            source_path
        )

        destination_sha256 = sha256_file(
            destination_path
        )

        if source_sha256 != destination_sha256:
            raise RuntimeError(
                "Copy verification failed:\n"
                f"Source: {source_path}\n"
                f"Destination: {destination_path}"
            )

        kept_counts_by_class[
            record["mapped_class"]
        ] += 1

        kept_counts_by_split[
            record["source_split"]
        ] += 1

        kept_rows.append({
            "source_path": source_rel,
            "source_split": record["source_split"],
            "source_class": record["source_class"],
            "mapped_class": record["mapped_class"],
            "destination_path": destination_rel,
            "sha256": source_sha256,
        })

        if len(kept_rows) % 500 == 0:
            print(
                f"Copied clean images: {len(kept_rows)}"
            )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    clean_files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if is_image(path)
    )

    print("\n" + "=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    print("\nRaw MRI total            :", len(source_records))
    print("Exact-pixel excluded     :", len(excluded_rows))
    print("Clean cohort kept        :", len(kept_rows))
    print("Files physically present :", len(clean_files))

    if len(excluded_rows) != EXPECTED_EXCLUDED_UNIQUE:
        raise RuntimeError(
            f"Expected to exclude {EXPECTED_EXCLUDED_UNIQUE}, "
            f"excluded {len(excluded_rows)}."
        )

    if len(kept_rows) != EXPECTED_CLEAN_TOTAL:
        raise RuntimeError(
            f"Expected clean cohort size {EXPECTED_CLEAN_TOTAL}, "
            f"found {len(kept_rows)}."
        )

    if len(clean_files) != EXPECTED_CLEAN_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_CLEAN_TOTAL} copied image files, "
            f"found {len(clean_files)}."
        )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    write_csv(
        args.manifest_dir
        / "kept_images_manifest.csv",
        kept_rows,
        [
            "source_path",
            "source_split",
            "source_class",
            "mapped_class",
            "destination_path",
            "sha256",
        ],
    )

    write_csv(
        args.manifest_dir
        / "excluded_exact_pixel_overlap_manifest.csv",
        excluded_rows,
        [
            "source_path",
            "source_split",
            "source_class",
            "mapped_class",
            "exclusion_reason",
        ],
    )

    summary = {
        "source_dataset": "BDNeuro-MRI Version 8",
        "source_mri_total": len(source_records),
        "cross_dataset_reference": (
            "Original Nickparvar 7,023-image development dataset"
        ),
        "exclusion_rule": (
            "Exclude only BDNeuro images with an exact decoded-pixel "
            "match to at least one image in the original development "
            "dataset, as recorded by the completed overlap audit."
        ),
        "perceptual_hash_policy": (
            "Non-identical pHash similarity candidates are retained. "
            "pHash similarity alone is not treated as proof of duplication."
        ),
        "unique_exact_pixel_images_excluded": len(excluded_rows),
        "exact_pixel_clean_images_kept": len(kept_rows),
        "mapped_class_names": CLASS_MAP,
        "kept_by_class": dict(
            sorted(
                kept_counts_by_class.items()
            )
        ),
        "kept_by_original_bdneuro_split": dict(
            sorted(
                kept_counts_by_split.items()
            )
        ),
        "excluded_by_class": dict(
            sorted(
                excluded_counts_by_class.items()
            )
        ),
        "excluded_by_original_bdneuro_split": dict(
            sorted(
                excluded_counts_by_split.items()
            )
        ),
        "raw_dataset_preserved": True,
        "derived_cohort_path": project_relative(
            args.output_dir
        ),
    }

    save_json(
        args.manifest_dir
        / "cohort_summary.json",
        summary,
    )

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print("\nKEPT BY CLASS")
    for cls in (
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ):
        print(
            f"{cls:12s}: "
            f"{kept_counts_by_class[cls]}"
        )

    print("\nEXCLUDED BY CLASS")
    for cls in (
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ):
        print(
            f"{cls:12s}: "
            f"{excluded_counts_by_class[cls]}"
        )

    print("\nKEPT BY ORIGINAL BDNEURO SPLIT")
    for split in SPLITS:
        print(
            f"{split:8s}: "
            f"{kept_counts_by_split[split]}"
        )

    print("\nOutput cohort:")
    print(args.output_dir)

    print("\nCohort summary:")
    print(
        args.manifest_dir
        / "cohort_summary.json"
    )

    print(
        "\nSUCCESS: Exact-overlap-clean BDNeuro external "
        "cohort built without modifying the raw dataset."
    )


if __name__ == "__main__":
    main()
