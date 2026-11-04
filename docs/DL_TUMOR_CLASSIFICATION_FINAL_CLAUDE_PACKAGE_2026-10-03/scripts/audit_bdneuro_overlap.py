#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fftpack import dct

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline_torch import TRAIN_DIR, TEST_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
INTERNAL_CLASS_NAMES = {"glioma", "meningioma", "notumor", "pituitary"}
BD_SPLITS = ("train", "val", "test")
BD_CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "notumor",
    "pituitary": "pituitary",
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bdroot",
    type=Path,
    default=(
        PROJECT_ROOT
        / "external_data"
        / "BDNeuro_V8_raw"
        / "Brain_Tumor_MRI_Dataset_Final"
    ),
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=(
        PROJECT_ROOT
        / "phase2"
        / "external_validation"
        / "bdneuro_overlap_audit"
    ),
)

parser.add_argument(
    "--near-threshold",
    type=int,
    default=6,
)

parser.add_argument(
    "--strong-threshold",
    type=int,
    default=4,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)

args = parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def decoded_pixel_hash(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        array = np.asarray(image, dtype=np.uint8)

    h = hashlib.sha256()
    h.update(f"{width}x{height}|RGB|".encode("utf-8"))
    h.update(array.tobytes())
    return h.hexdigest(), width, height


def phash64(path: Path) -> np.uint64:
    with Image.open(path) as source:
        image = source.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)

    transformed = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = transformed[:8, :8]
    median = np.median(low)
    bits = (low >= median).reshape(-1)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return np.uint64(value)


def collect_internal_images() -> list[dict]:
    records = []

    for split, root in [("Training", Path(TRAIN_DIR)), ("Testing", Path(TEST_DIR))]:
        if not root.is_dir():
            raise FileNotFoundError(f"Internal dataset folder not found: {root}")

        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cls = class_dir.name.lower()

            if cls not in INTERNAL_CLASS_NAMES:
                continue

            for path in sorted(class_dir.rglob("*")):
                if is_image(path):
                    records.append({
                        "dataset": "nickparvar",
                        "split": split,
                        "source_class": cls,
                        "mapped_class": cls,
                        "path": path,
                    })

    return records


def collect_bdneuro_images(bdroot: Path) -> list[dict]:
    records = []

    if not bdroot.is_dir():
        raise FileNotFoundError(f"BDNeuro root not found: {bdroot}")

    for split in BD_SPLITS:
        split_root = bdroot / split

        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing BDNeuro split: {split_root}")

        for source_class, mapped_class in BD_CLASS_MAP.items():
            class_dir = split_root / source_class

            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing BDNeuro class folder: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if is_image(path):
                    records.append({
                        "dataset": "bdneuro_v8",
                        "split": split,
                        "source_class": source_class,
                        "mapped_class": mapped_class,
                        "path": path,
                    })

    return records


def add_hashes(records: list[dict], label: str) -> None:
    total = len(records)

    for index, record in enumerate(records, 1):
        path = record["path"]

        try:
            record["sha256"] = sha256_file(path)

            pixel_hash, width, height = decoded_pixel_hash(path)
            record["pixel_sha256"] = pixel_hash
            record["width"] = width
            record["height"] = height
            record["phash64"] = phash64(path)

        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(f"Could not process image:\n{path}\n{error}") from error

        if index % 500 == 0 or index == total:
            print(f"  {label}: {index}/{total}")


def cross_exact_pairs(left: list[dict], right: list[dict], key: str) -> list[dict]:
    lookup = defaultdict(list)

    for record in left:
        lookup[record[key]].append(record)

    rows = []

    for r in right:
        for l in lookup.get(r[key], []):
            rows.append({
                "internal_path": rel(l["path"]),
                "internal_split": l["split"],
                "internal_class": l["mapped_class"],
                "bdneuro_path": rel(r["path"]),
                "bdneuro_split": r["split"],
                "bdneuro_source_class": r["source_class"],
                "bdneuro_mapped_class": r["mapped_class"],
                "same_class": int(l["mapped_class"] == r["mapped_class"]),
                "match_hash": str(r[key]),
            })

    return rows


def within_exact_pairs(records: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)

    for record in records:
        groups[record[key]].append(record)

    rows = []

    for hash_value, group in groups.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]

                rows.append({
                    "path_a": rel(a["path"]),
                    "split_a": a["split"],
                    "class_a": a["mapped_class"],
                    "path_b": rel(b["path"]),
                    "split_b": b["split"],
                    "class_b": b["mapped_class"],
                    "same_class": int(a["mapped_class"] == b["mapped_class"]),
                    "match_hash": str(hash_value),
                })

    return rows


BYTE_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a[:, None], b[None, :])

    byte_view = xor.view(np.uint8).reshape(
        xor.shape[0],
        xor.shape[1],
        8,
    )

    return BYTE_POPCOUNT[byte_view].sum(axis=2)


def cross_phash_candidates(
    internal: list[dict],
    external: list[dict],
    threshold: int,
    strong_threshold: int,
    batch_size: int,
) -> list[dict]:
    internal_hashes = np.array([r["phash64"] for r in internal], dtype=np.uint64)
    external_hashes = np.array([r["phash64"] for r in external], dtype=np.uint64)

    rows = []

    print("\nCross-dataset pHash screening...")

    for start in range(0, len(external), batch_size):
        end = min(start + batch_size, len(external))

        distances = hamming_matrix(
            external_hashes[start:end],
            internal_hashes,
        )

        ext_idx, int_idx = np.where(distances <= threshold)

        for local_external_index, internal_index in zip(ext_idx, int_idx):
            external_index = start + int(local_external_index)

            e = external[external_index]
            i = internal[int(internal_index)]
            distance = int(distances[local_external_index, internal_index])

            rows.append({
                "phash_distance": distance,
                "candidate_level": (
                    "STRONG_CANDIDATE"
                    if distance <= strong_threshold
                    else "REVIEW_CANDIDATE"
                ),
                "internal_path": rel(i["path"]),
                "internal_split": i["split"],
                "internal_class": i["mapped_class"],
                "bdneuro_path": rel(e["path"]),
                "bdneuro_split": e["split"],
                "bdneuro_source_class": e["source_class"],
                "bdneuro_mapped_class": e["mapped_class"],
                "same_class": int(i["mapped_class"] == e["mapped_class"]),
                "exact_file_match": int(i["sha256"] == e["sha256"]),
                "exact_pixel_match": int(i["pixel_sha256"] == e["pixel_sha256"]),
            })

        print(f"  BDNeuro screened: {end}/{len(external)}")

    rows.sort(
        key=lambda row: (
            row["phash_distance"],
            row["bdneuro_path"],
            row["internal_path"],
        )
    )

    return rows


def within_phash_candidates(
    records: list[dict],
    threshold: int,
    strong_threshold: int,
    batch_size: int,
) -> list[dict]:
    hashes = np.array([r["phash64"] for r in records], dtype=np.uint64)

    rows = []

    print("\nWithin-BDNeuro pHash screening...")

    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))

        distances = hamming_matrix(
            hashes[start:end],
            hashes,
        )

        local_idx, other_idx = np.where(distances <= threshold)

        for local_index, j in zip(local_idx, other_idx):
            i = start + int(local_index)
            j = int(j)

            if j <= i:
                continue

            a = records[i]
            b = records[j]
            distance = int(distances[local_index, j])

            rows.append({
                "phash_distance": distance,
                "candidate_level": (
                    "STRONG_CANDIDATE"
                    if distance <= strong_threshold
                    else "REVIEW_CANDIDATE"
                ),
                "path_a": rel(a["path"]),
                "split_a": a["split"],
                "class_a": a["mapped_class"],
                "path_b": rel(b["path"]),
                "split_b": b["split"],
                "class_b": b["mapped_class"],
                "same_class": int(a["mapped_class"] == b["mapped_class"]),
                "cross_split": int(a["split"] != b["split"]),
                "exact_file_match": int(a["sha256"] == b["sha256"]),
                "exact_pixel_match": int(a["pixel_sha256"] == b["pixel_sha256"]),
            })

        print(f"  BDNeuro screened: {end}/{len(records)}")

    rows.sort(
        key=lambda row: (
            row["phash_distance"],
            row["path_a"],
            row["path_b"],
        )
    )

    return rows


def main():
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BDNEURO V8 EXTERNAL-OVERLAP AUDIT")
    print("=" * 80)

    print("\nInternal Training:", TRAIN_DIR)
    print("Internal Testing :", TEST_DIR)
    print("BDNeuro root     :", args.bdroot)
    print("Output           :", args.output_dir)

    internal = collect_internal_images()
    bdneuro = collect_bdneuro_images(args.bdroot)

    internal_counts = Counter(
        (r["split"], r["mapped_class"])
        for r in internal
    )

    bd_counts = Counter(
        (r["split"], r["source_class"])
        for r in bdneuro
    )

    print("\nCollected:")
    print("Internal images :", len(internal))
    print("BDNeuro images  :", len(bdneuro))

    if len(internal) != 7023:
        raise RuntimeError(
            f"Expected 7,023 internal images, found {len(internal)}."
        )

    if len(bdneuro) != 5941:
        raise RuntimeError(
            f"Expected 5,941 BDNeuro cohort images, found {len(bdneuro)}."
        )

    counts_payload = {
        "internal_total": len(internal),
        "bdneuro_total": len(bdneuro),
        "internal_split_class_counts": {
            f"{split}/{cls}": count
            for (split, cls), count in sorted(internal_counts.items())
        },
        "bdneuro_split_class_counts": {
            f"{split}/{cls}": count
            for (split, cls), count in sorted(bd_counts.items())
        },
    }

    save_json(
        args.output_dir / "dataset_counts.json",
        counts_payload,
    )

    print("\nComputing exact + pixel + pHash fingerprints...")
    add_hashes(internal, "Internal")
    add_hashes(bdneuro, "BDNeuro")

    cross_file = cross_exact_pairs(internal, bdneuro, "sha256")
    cross_pixel = cross_exact_pairs(internal, bdneuro, "pixel_sha256")

    write_csv(
        args.output_dir / "cross_exact_file_duplicates.csv",
        cross_file,
        [
            "internal_path",
            "internal_split",
            "internal_class",
            "bdneuro_path",
            "bdneuro_split",
            "bdneuro_source_class",
            "bdneuro_mapped_class",
            "same_class",
            "match_hash",
        ],
    )

    write_csv(
        args.output_dir / "cross_exact_pixel_duplicates.csv",
        cross_pixel,
        [
            "internal_path",
            "internal_split",
            "internal_class",
            "bdneuro_path",
            "bdneuro_split",
            "bdneuro_source_class",
            "bdneuro_mapped_class",
            "same_class",
            "match_hash",
        ],
    )

    bd_file = within_exact_pairs(bdneuro, "sha256")
    bd_pixel = within_exact_pairs(bdneuro, "pixel_sha256")

    write_csv(
        args.output_dir / "bdneuro_internal_exact_file_duplicates.csv",
        bd_file,
        [
            "path_a",
            "split_a",
            "class_a",
            "path_b",
            "split_b",
            "class_b",
            "same_class",
            "match_hash",
        ],
    )

    write_csv(
        args.output_dir / "bdneuro_internal_exact_pixel_duplicates.csv",
        bd_pixel,
        [
            "path_a",
            "split_a",
            "class_a",
            "path_b",
            "split_b",
            "class_b",
            "same_class",
            "match_hash",
        ],
    )

    cross_near = cross_phash_candidates(
        internal,
        bdneuro,
        args.near_threshold,
        args.strong_threshold,
        args.batch_size,
    )

    write_csv(
        args.output_dir / "cross_phash_candidates.csv",
        cross_near,
        [
            "phash_distance",
            "candidate_level",
            "internal_path",
            "internal_split",
            "internal_class",
            "bdneuro_path",
            "bdneuro_split",
            "bdneuro_source_class",
            "bdneuro_mapped_class",
            "same_class",
            "exact_file_match",
            "exact_pixel_match",
        ],
    )

    bd_near = within_phash_candidates(
        bdneuro,
        args.near_threshold,
        args.strong_threshold,
        args.batch_size,
    )

    write_csv(
        args.output_dir / "bdneuro_internal_phash_candidates.csv",
        bd_near,
        [
            "phash_distance",
            "candidate_level",
            "path_a",
            "split_a",
            "class_a",
            "path_b",
            "split_b",
            "class_b",
            "same_class",
            "cross_split",
            "exact_file_match",
            "exact_pixel_match",
        ],
    )

    cross_strong = sum(
        row["candidate_level"] == "STRONG_CANDIDATE"
        for row in cross_near
    )

    bd_strong = sum(
        row["candidate_level"] == "STRONG_CANDIDATE"
        for row in bd_near
    )

    summary = {
        "internal_images": len(internal),
        "bdneuro_images": len(bdneuro),
        "near_threshold_phash_hamming": args.near_threshold,
        "strong_threshold_phash_hamming": args.strong_threshold,
        "cross_dataset": {
            "exact_file_duplicate_pairs": len(cross_file),
            "exact_pixel_duplicate_pairs": len(cross_pixel),
            "phash_candidates_le_threshold": len(cross_near),
            "phash_strong_candidates": int(cross_strong),
        },
        "within_bdneuro": {
            "exact_file_duplicate_pairs": len(bd_file),
            "exact_pixel_duplicate_pairs": len(bd_pixel),
            "phash_candidates_le_threshold": len(bd_near),
            "phash_strong_candidates": int(bd_strong),
            "cross_split_phash_candidates": int(
                sum(row["cross_split"] for row in bd_near)
            ),
        },
        "important_note": (
            "pHash matches are screening candidates, not automatic proof of "
            "duplicate identity. Review near-duplicate candidates before "
            "excluding images or making scientific claims."
        ),
    }

    save_json(
        args.output_dir / "summary.json",
        summary,
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print("\nCROSS-DATASET:")
    print("Exact file pairs  :", len(cross_file))
    print("Exact pixel pairs :", len(cross_pixel))
    print(
        f"pHash <= {args.near_threshold} candidates:",
        len(cross_near),
    )
    print(
        f"pHash <= {args.strong_threshold} strong:",
        cross_strong,
    )

    print("\nWITHIN BDNEURO:")
    print("Exact file pairs  :", len(bd_file))
    print("Exact pixel pairs :", len(bd_pixel))
    print(
        f"pHash <= {args.near_threshold} candidates:",
        len(bd_near),
    )
    print(
        f"pHash <= {args.strong_threshold} strong:",
        bd_strong,
    )

    print("\nSaved:")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
