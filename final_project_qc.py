#!/usr/bin/env python3
"""Final QC, leakage audit, and conditional experimental lock.

Run from the brain_tumor_project root (or pass --project-root).
This script is deliberately conservative: it only writes the experimental lock
when no exact cross-split duplicates or unresolved high-similarity near-duplicate
pairs are detected.

It does NOT retrain any model or alter saved checkpoints, scaler, PCA, SVM, split
indices, evaluation results, Grad-CAM CSVs, or occlusion CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_COUNTS = {"train": 4569, "validation": 1143, "test": 1311}
BACKBONES = ("vgg16", "vgg19", "resnet50", "resnet152", "efficientnet_b3")
DISPLAY = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "efficientnet_b3": "EfficientNet-B3",
}
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ImageRecord:
    split: str
    class_name: str
    path: str
    relative_path: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    pixel_sha256: str
    phash64: str
    dhash64: str


@dataclass(frozen=True)
class PairFinding:
    finding_type: str
    split_a: str
    split_b: str
    class_a: str
    class_b: str
    path_a: str
    path_b: str
    sha256_equal: bool
    pixel_equal: bool
    phash_distance: int
    dhash_distance: int
    gray_mae: float
    gray_correlation: float
    material: bool
    requires_review: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--near-phash", type=int, default=4)
    parser.add_argument("--near-dhash", type=int, default=6)
    return parser.parse_args()


def _dct_basis(size: int = 32, low: int = 8) -> np.ndarray:
    n = np.arange(size, dtype=np.float64)
    k = np.arange(low, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (2.0 * n + 1.0) * k / (2.0 * size))
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


DCT8 = _dct_basis()


def _bits_to_u64(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(bool).ravel():
        value = (value << 1) | int(bit)
    return value


def phash64(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64)
    coeff = DCT8 @ arr @ DCT8.T
    flat = coeff.ravel()
    median = np.median(flat[1:])
    return _bits_to_u64(flat > median)


def dhash64(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    return _bits_to_u64(arr[:, 1:] > arr[:, :-1])


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    h = hashlib.sha256()
    h.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def gray_array(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32)


def gray_metrics(path_a: Path, path_b: Path, cache: dict[str, np.ndarray]) -> tuple[float, float]:
    def get(path: Path) -> np.ndarray:
        key = str(path)
        if key not in cache:
            cache[key] = gray_array(path)
        return cache[key]

    a = get(path_a).ravel()
    b = get(path_b).ravel()
    mae = float(np.mean(np.abs(a - b)))
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-8 or b_std < 1e-8:
        corr = 1.0 if mae == 0.0 else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return mae, corr


def make_records(project_root: Path) -> tuple[list[ImageRecord], dict[str, list[int]]]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from torchvision import datasets
        from src.data_pipeline_torch import TRAIN_DIR, TEST_DIR, load_saved_indices
    except Exception as error:
        raise RuntimeError(
            "Could not import the project's data pipeline. Run this script from the "
            "brain_tumor_project root with the project virtual environment active."
        ) from error

    train_dir = Path(TRAIN_DIR).expanduser().resolve()
    test_dir = Path(TEST_DIR).expanduser().resolve()
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directories are unavailable. Training={train_dir}; Testing={test_dir}"
        )

    base_train = datasets.ImageFolder(root=train_dir)
    base_test = datasets.ImageFolder(root=test_dir)
    train_indices, val_indices = load_saved_indices()
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    all_indices = np.concatenate([train_indices, val_indices])
    if len(all_indices) != len(base_train):
        raise RuntimeError("Saved train/validation split does not cover the Training directory.")
    if len(np.unique(all_indices)) != len(base_train):
        raise RuntimeError("Saved train/validation indices overlap or omit Training images.")
    if all_indices.min() < 0 or all_indices.max() >= len(base_train):
        raise RuntimeError("Saved split contains out-of-range indices.")

    index_map: dict[str, list[int]] = {
        "train": train_indices.tolist(),
        "validation": val_indices.tolist(),
        "test": list(range(len(base_test))),
    }

    counts = {k: len(v) for k, v in index_map.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected split sizes: {counts}; expected {EXPECTED_COUNTS}")

    records: list[ImageRecord] = []

    def build(split: str, samples, classes, indices: list[int], root: Path) -> None:
        for n, idx in enumerate(indices, start=1):
            path_text, class_index = samples[idx]
            path = Path(path_text)
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    pixel_hash = decoded_pixel_sha256(image)
                    p_hash = phash64(image)
                    d_hash = dhash64(image)
            except (UnidentifiedImageError, OSError) as error:
                raise RuntimeError(f"Unreadable image: {path}") from error
            try:
                relative = path.relative_to(project_root).as_posix()
            except ValueError:
                relative = os.path.relpath(path, project_root).replace(os.sep, "/")
            records.append(
                ImageRecord(
                    split=split,
                    class_name=classes[class_index],
                    path=str(path),
                    relative_path=relative,
                    size_bytes=path.stat().st_size,
                    width=width,
                    height=height,
                    sha256=file_sha256(path),
                    pixel_sha256=pixel_hash,
                    phash64=f"{p_hash:016x}",
                    dhash64=f"{d_hash:016x}",
                )
            )
            if n % 500 == 0:
                print(f"  hashed {split}: {n}/{len(indices)}")

    print("Hashing saved training split...")
    build("train", base_train.samples, base_train.classes, index_map["train"], train_dir)
    print("Hashing saved validation split...")
    build("validation", base_train.samples, base_train.classes, index_map["validation"], train_dir)
    print("Hashing official held-out test set...")
    build("test", base_test.samples, base_test.classes, index_map["test"], test_dir)
    return records, index_map


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a[:, None], b[None, :])
    byte_view = xor.view(np.uint8).reshape(xor.shape + (8,))
    return POPCOUNT[byte_view].sum(axis=-1, dtype=np.uint8)


def exact_findings(records: list[ImageRecord]) -> list[PairFinding]:
    findings: list[PairFinding] = []
    by_sha: dict[str, list[int]] = defaultdict(list)
    by_pixel: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r.sha256].append(i)
        by_pixel[r.pixel_sha256].append(i)

    seen: set[tuple[int, int]] = set()
    for mapping, ftype in ((by_sha, "exact_file_duplicate"), (by_pixel, "exact_pixel_duplicate")):
        for group in mapping.values():
            if len(group) < 2:
                continue
            for ai in range(len(group)):
                for bi in range(ai + 1, len(group)):
                    i, j = group[ai], group[bi]
                    a, b = records[i], records[j]
                    if a.split == b.split:
                        continue
                    key = tuple(sorted((i, j)))
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        PairFinding(
                            finding_type=ftype,
                            split_a=a.split,
                            split_b=b.split,
                            class_a=a.class_name,
                            class_b=b.class_name,
                            path_a=a.relative_path,
                            path_b=b.relative_path,
                            sha256_equal=a.sha256 == b.sha256,
                            pixel_equal=a.pixel_sha256 == b.pixel_sha256,
                            phash_distance=(int(a.phash64, 16) ^ int(b.phash64, 16)).bit_count(),
                            dhash_distance=(int(a.dhash64, 16) ^ int(b.dhash64, 16)).bit_count(),
                            gray_mae=0.0,
                            gray_correlation=1.0,
                            material=True,
                            requires_review=False,
                        )
                    )
    return findings


def near_findings(
    records: list[ImageRecord],
    exact_pairs: set[frozenset[str]],
    near_phash: int,
    near_dhash: int,
) -> list[PairFinding]:
    groups = {split: [r for r in records if r.split == split] for split in EXPECTED_COUNTS}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    findings: list[PairFinding] = []
    cache: dict[str, np.ndarray] = {}

    for split_a, split_b in pairs:
        aa = groups[split_a]
        bb = groups[split_b]
        p_a = np.array([int(r.phash64, 16) for r in aa], dtype=np.uint64)
        p_b = np.array([int(r.phash64, 16) for r in bb], dtype=np.uint64)
        d_a = np.array([int(r.dhash64, 16) for r in aa], dtype=np.uint64)
        d_b = np.array([int(r.dhash64, 16) for r in bb], dtype=np.uint64)
        print(f"Comparing perceptual hashes: {split_a} ↔ {split_b}")
        chunk = 192
        for start in range(0, len(aa), chunk):
            stop = min(start + chunk, len(aa))
            pd = hamming_matrix(p_a[start:stop], p_b)
            dd = hamming_matrix(d_a[start:stop], d_b)
            rows, cols = np.where((pd <= near_phash) & (dd <= near_dhash))
            for rr, cc in zip(rows.tolist(), cols.tolist()):
                a = aa[start + rr]
                b = bb[cc]
                pair_key = frozenset((a.relative_path, b.relative_path))
                if pair_key in exact_pairs:
                    continue
                mae, corr = gray_metrics(Path(a.path), Path(b.path), cache)
                p_dist = int(pd[rr, cc])
                d_dist = int(dd[rr, cc])
                # Very high-confidence materially equivalent images.
                material = (
                    (p_dist <= 1 and d_dist <= 1 and mae <= 4.0 and corr >= 0.9995)
                    or (p_dist <= 2 and d_dist <= 2 and mae <= 2.0 and corr >= 0.9998)
                )
                # Conservative unresolved zone: similar enough to require visual inspection.
                review = (not material) and (mae <= 12.0 and corr >= 0.995)
                if not material and not review:
                    continue
                findings.append(
                    PairFinding(
                        finding_type="high_confidence_near_duplicate" if material else "near_duplicate_review",
                        split_a=a.split,
                        split_b=b.split,
                        class_a=a.class_name,
                        class_b=b.class_name,
                        path_a=a.relative_path,
                        path_b=b.relative_path,
                        sha256_equal=False,
                        pixel_equal=False,
                        phash_distance=p_dist,
                        dhash_distance=d_dist,
                        gray_mae=round(mae, 6),
                        gray_correlation=round(corr, 8),
                        material=material,
                        requires_review=review,
                    )
                )
    return findings


def save_contact_sheets(project_root: Path, output_dir: Path, findings: list[PairFinding]) -> None:
    review = [f for f in findings if f.requires_review]
    if not review:
        return
    review_dir = output_dir / "near_duplicate_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for idx, finding in enumerate(review, start=1):
        pa = (project_root / finding.path_a).resolve()
        pb = (project_root / finding.path_b).resolve()
        with Image.open(pa) as ia, Image.open(pb) as ib:
            ia = ia.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
            ib = ib.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (660, 380), "white")
        canvas.paste(ia, (10, 10))
        canvas.paste(ib, (330, 10))
        draw = ImageDraw.Draw(canvas)
        text = (
            f"pHash={finding.phash_distance} dHash={finding.dhash_distance} "
            f"MAE={finding.gray_mae:.3f} corr={finding.gray_correlation:.6f}"
        )
        draw.text((10, 340), text, fill="black")
        draw.text((10, 360), f"A: {finding.path_a}", fill="black")
        # filename contains pair index only to avoid path characters
        canvas.save(review_dir / f"candidate_{idx:03d}.jpg", quality=92)


def write_csv(path: Path, findings: list[PairFinding]) -> None:
    fields = list(PairFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def patch_methodology_sentence(project_root: Path, docs_dir: Path) -> str:
    app_path = project_root / "app.py"
    if not app_path.is_file():
        return "app.py not found; methodology sentence not patched."
    old = "2. Extract a spatial feature map with the selected pretrained-and-fine-tuned CNN backbone."
    new = (
        "2. Extract a spatial feature map using the selected pretrained CNN backbone, "
        "following its model-specific freezing or fine-tuning configuration."
    )
    text = app_path.read_text(encoding="utf-8")
    if old not in text:
        return "app.py methodology sentence was already updated or differs from the expected text."
    backup = docs_dir / "app.py.prelock_backup"
    shutil.copy2(app_path, backup)
    app_path.write_text(text.replace(old, new), encoding="utf-8")
    return f"Updated app.py methodology wording; backup saved at {backup.relative_to(project_root)}."


def final_results_table(project_root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for backbone in BACKBONES:
        base = project_root / "experiments" / backbone / "results"
        final_path = base / "final_results.json"
        training_path = base / "training_results.json"
        if not final_path.is_file() or not training_path.is_file():
            missing.append(backbone)
            continue
        final = json.loads(final_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "backbone": backbone,
                "model": DISPLAY[backbone],
                "best_epoch": int(training["best_epoch"]),
                "nn_val_accuracy_pct": round(float(training["best_val_accuracy"]), 2),
                "svm_val_accuracy_pct": round(float(final["validation"]["accuracy"]) * 100, 2),
                "test_accuracy_pct": round(float(final["test"]["accuracy"]) * 100, 2),
                "test_macro_precision_pct": round(float(final["test"]["macro_precision"]) * 100, 2),
                "test_macro_recall_pct": round(float(final["test"]["macro_recall"]) * 100, 2),
                "test_macro_f1_pct": round(float(final["test"]["macro_f1"]) * 100, 2),
                "pca_components": int(final["pca"]["components"]),
                "pca_variance_retained_pct": round(float(final["pca"]["explained_variance"]) * 100, 2),
            }
        )
    return rows, missing


def write_report_source(project_root: Path, docs_dir: Path, audit_status: str, audit_summary: dict) -> None:
    rows, missing = final_results_table(project_root)
    lines = [
        "# Report Source of Truth",
        "",
        "This file is generated from the standardized `experiments/<backbone>/results/` artifacts.",
        "Do not mix these values with older root-level ResNet152 preliminary artifacts.",
        "",
        "## Experimental status",
        "",
        f"- Duplicate/leakage audit status: **{audit_status}**",
        f"- Saved split: 4,569 training; 1,143 validation; 1,311 official held-out test images.",
        "- Five standardized backbones: VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3.",
        "- Final classifier for each backbone: CNN → LSTM → 128-D Dense → training-only StandardScaler → training-only PCA(64) → Linear SVM (C=1.0).",
        "- Formal explainability evaluation: ResNet152 only.",
        "",
        "## Standardized five-backbone results",
        "",
        "| Model | Best epoch | NN val acc (%) | SVM val acc (%) | Test acc (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) | PCA | PCA variance (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['best_epoch']} | {r['nn_val_accuracy_pct']:.2f} | "
            f"{r['svm_val_accuracy_pct']:.2f} | {r['test_accuracy_pct']:.2f} | "
            f"{r['test_macro_precision_pct']:.2f} | {r['test_macro_recall_pct']:.2f} | "
            f"{r['test_macro_f1_pct']:.2f} | {r['pca_components']} | "
            f"{r['pca_variance_retained_pct']:.2f} |"
        )
    lines += [
        "",
        "EfficientNet-B3 has the highest observed test accuracy in the standardized comparison; the difference from ResNet152 is one correctly classified test image and should not be described as statistically superior without a formal paired test.",
        "",
        "## Explainability scope and locked interpretation",
        "",
        "- Formal Grad-CAM protocol: ResNet152, seed 42, five images per class, 20 total; Layer 3 (14×14), Layer 4 (7×7), and 50/50 fused map; fixed opacity 0.55.",
        "- Tumor-image Grad-CAM Good-or-Partial: Layer 3 10/15 (66.7%); Layer 4 5/15 (33.3%); fused 8/15 (53.3%).",
        "- Final-SVM occlusion protocol: fixed 11 cases, patch 32×32, stride 16, 169 occlusions/image; nine tumor and two no-tumor cases.",
        "- Tumor-image occlusion: Good 2/9; Partial 2/9; Poor 5/9; Good-or-Partial 4/9 (44.4%).",
        "- Direct comparison: Grad-CAM more anatomically relevant 4/9; occlusion more relevant 3/9; both predominantly off-lesion 2/9; both clearly lesion-focused 0/9.",
        "- Across 1,859 occlusions: 58 prediction-changing occlusions (3.12% pooled); median per-image flip rate 0%; five of 11 images had at least one flip.",
        "- Do not describe Grad-CAM or occlusion maps as segmentation masks, tumor boundaries, or proof of clinically valid reasoning.",
        "- Strong classification performance is a dataset-level result and does not establish consistently lesion-based reasoning or clinical readiness.",
        "",
        "## Leakage audit statement",
        "",
    ]
    if audit_status == "CLEAN":
        lines.append(
            "No exact or unresolved high-similarity near-duplicate images were detected across Train↔Validation, Train↔Test, or Validation↔Test by the final audit. This does not establish patient-level independence because reliable patient identifiers are unavailable."
        )
    else:
        lines.append(
            "The experimental lock is pending because the duplicate/leakage audit did not return CLEAN. Do not finalize the no-duplication statement until the blocker is resolved."
        )
    if missing:
        lines += ["", f"WARNING: missing standardized result files for: {', '.join(missing)}"]
    (docs_dir / "REPORT_SOURCE_OF_TRUTH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_outline(docs_dir: Path) -> None:
    text = """# Final Report Outline

## 1. Introduction
- Brain tumor MRI classification problem.
- Motivation for hybrid CNN–LSTM–SVM classification.
- Base-paper relationship and project objective.
- Research/educational scope; no clinical-deployment claim.

## 2. Dataset and Experimental Protocol
- Masoud Nickparvar Brain Tumor MRI dataset, 7,023 images.
- Original Training directory: 5,712; official held-out Testing directory: 1,311.
- Fixed stratified 80/20 split of original Training directory using seed 42: 4,569 train and 1,143 validation.
- Four classes: glioma, meningioma, no tumor, pituitary.
- Final duplicate/leakage audit and its result.

## 3. Methodology
- Deterministic evaluation preprocessing and training augmentation.
- Five pretrained CNN backbones.
- Model-specific transfer learning: VGG16/VGG19 convolutional backbones frozen; other finalized backbones follow their fine-tuning configuration.
- CNN feature map → spatial sequence → LSTM(256) → BatchNorm → Dropout → Dense(128).
- Neural head used for neural training/monitoring; final deployed classifier is PCA–SVM.
- StandardScaler fitted only on training features.
- PCA fitted only on training features, 64 components.
- Linear SVM C=1.0, probability=False in the standardized pipeline.

## 4. Standardized Five-Backbone Results
- Main comparison table from REPORT_SOURCE_OF_TRUTH.md.
- Training/validation curves.
- Confusion matrices and per-class performance.
- Note that EfficientNet-B3 achieved the highest observed test accuracy; do not overstate the one-image difference versus ResNet152.

## 5. Explainability Methodology
- Explainability experiment restricted to ResNet152.
- Grad-CAM: Layer 3, Layer 4, fused map; fixed 20-image protocol.
- Explain that Grad-CAM targets the neural head, not the external SVM.
- Final-SVM occlusion: fixed 11-case subset, 32×32 patch, stride 16, full pipeline rerun.
- Explain within-image heatmap normalization limitation.

## 6. Explainability Results
- Grad-CAM qualitative counts.
- Occlusion localization counts.
- Grad-CAM-versus-occlusion comparison.
- Prediction-flip analysis, including the misclassified pituitary case.

## 7. Discussion
- Strong classification performance versus inconsistent lesion-focused reasoning.
- Possible shortcut features: cropping, orientation, contrast, scanner/acquisition patterns, borders, skull/facial anatomy, and dataset-specific correlations.
- Why Grad-CAM and SVM occlusion can disagree: they explain different decision mechanisms.
- Classification success does not establish causal or clinically valid reasoning.

## 8. Limitations
- Single public dataset and dataset-specific correlations.
- Patient-level independence cannot be verified without reliable patient IDs.
- Explainability assessment is qualitative and deliberately limited in sample size.
- Grad-CAM explains the neural head; occlusion is computationally coarse.
- Per-image normalization can make negligible absolute effects appear visually strong.
- No segmentation ground truth; maps are not tumor boundaries.
- No external clinical validation or prospective evaluation.

## 9. Conclusion
- Five-backbone hybrid pipeline successfully implemented and standardized.
- High held-out classification performance.
- Explainability analysis reveals that high accuracy does not guarantee lesion-centred reasoning.
- System remains a research/educational prototype, not a diagnostic medical device.

## Appendices
- Exact software environment.
- Final artifact manifest.
- Duplicate/leakage audit report.
- Full per-class metrics and confusion matrices.
- Fixed Grad-CAM and occlusion case tables.
"""
    (docs_dir / "REPORT_OUTLINE.md").write_text(text, encoding="utf-8")


def write_legacy_note(project_root: Path, docs_dir: Path) -> None:
    candidates = [
        "models/best_resnet152_lstm.pth",
        "models/resnet152_lstm_pca_svm.joblib",
        "features/",
        "pca_features/",
        "actual_training_scripts/train_resnet152_lstm.py",
        "actual_training_scripts/train_svm.py",
        "actual_training_scripts/prepare_pca_features.py",
    ]
    present = [c for c in candidates if (project_root / c.rstrip("/")).exists()]
    text = [
        "# Legacy Artifact Note",
        "",
        "The authoritative final five-backbone artifacts live under `experiments/<backbone>/`.",
        "Older root-level ResNet152 artifacts are retained only for provenance and should not be used for final metrics, GUI inference, or report values.",
        "",
        "## Legacy paths detected",
    ]
    text.extend(f"- `{p}`" for p in present)
    if not present:
        text.append("- None of the expected legacy paths were detected.")
    (docs_dir / "LEGACY_ARTIFACTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_manifest(project_root: Path, docs_dir: Path) -> None:
    ignored_parts = {"venv", ".venv", ".git", "__pycache__", ".pytest_cache"}
    raw_data_prefix = (project_root / "data" / "brain_tumor_mri").resolve()
    items: list[str] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        try:
            path.resolve().relative_to(raw_data_prefix)
            continue
        except ValueError:
            pass
        try:
            rel = path.relative_to(project_root).as_posix()
        except ValueError:
            continue
        items.append(rel)
    items.sort()
    (docs_dir / "FINAL_PROJECT_MANIFEST.txt").write_text("\n".join(items) + "\n", encoding="utf-8")


def write_environment(docs_dir: Path) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        content = result.stdout
    except Exception as error:
        content = f"Could not capture pip freeze: {error}\n"
    (docs_dir / "environment_pip_freeze.txt").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or (project_root / "final_project_docs" / "duplicate_leakage_audit")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = output_dir.parent
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL BRAIN-TUMOR PROJECT QC")
    print("=" * 80)
    print("Project root:", project_root)

    records, index_map = make_records(project_root)
    print(f"Hashed {len(records)} images across the fixed splits.")

    exact = exact_findings(records)
    exact_pairs = {frozenset((f.path_a, f.path_b)) for f in exact}
    near = near_findings(records, exact_pairs, args.near_phash, args.near_dhash)
    findings = exact + near

    material = [f for f in findings if f.material]
    review = [f for f in findings if f.requires_review]
    status = "MATERIAL_DUPLICATION" if material else ("REVIEW_REQUIRED" if review else "CLEAN")

    write_csv(output_dir / "duplicate_leakage_findings.csv", findings)
    save_contact_sheets(project_root, output_dir, findings)

    summary = {
        "protocol_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "status": status,
        "split_counts": {k: len(v) for k, v in index_map.items()},
        "images_hashed": len(records),
        "cross_split_exact_or_pixel_duplicate_pairs": sum(
            1 for f in findings if f.finding_type.startswith("exact_")
        ),
        "high_confidence_near_duplicate_pairs": sum(
            1 for f in findings if f.finding_type == "high_confidence_near_duplicate"
        ),
        "near_duplicate_review_pairs": len(review),
        "material_pairs": len(material),
        "thresholds": {
            "candidate_phash_hamming_max": args.near_phash,
            "candidate_dhash_hamming_max": args.near_dhash,
            "material_rule": "exact file/pixel duplicate OR very-high-similarity perceptual pair",
            "review_rule": "pHash/dHash candidate with grayscale MAE <=12 and correlation >=0.995",
        },
        "patient_level_note": (
            "This image-level audit cannot establish patient-level independence because "
            "reliable patient identifiers are not available in the project dataset."
        ),
    }
    (output_dir / "duplicate_leakage_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Duplicate / Leakage Audit",
        "",
        f"Status: **{status}**",
        "",
        f"Images hashed: {summary['images_hashed']}",
        f"- Train: {summary['split_counts']['train']}",
        f"- Validation: {summary['split_counts']['validation']}",
        f"- Test: {summary['split_counts']['test']}",
        "",
        f"Exact/pixel duplicate cross-split pairs: {summary['cross_split_exact_or_pixel_duplicate_pairs']}",
        f"High-confidence near-duplicate pairs: {summary['high_confidence_near_duplicate_pairs']}",
        f"Unresolved review pairs: {summary['near_duplicate_review_pairs']}",
        "",
        "Comparisons performed: Train↔Validation, Train↔Test, Validation↔Test.",
        "The audit uses file SHA-256, decoded-pixel SHA-256, 64-bit pHash, 64-bit dHash, and grayscale pixel similarity for perceptual candidates.",
        "",
        summary["patient_level_note"],
    ]
    (output_dir / "DUPLICATE_LEAKAGE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Documentation/report package is generated regardless of audit result.
    write_report_source(project_root, docs_dir, status, summary)
    write_report_outline(docs_dir)
    write_legacy_note(project_root, docs_dir)
    write_manifest(project_root, docs_dir)
    write_environment(docs_dir)

    if status == "CLEAN":
        patch_note = patch_methodology_sentence(project_root, docs_dir)
        lock_text = f"""# FINAL EXPERIMENTAL PROJECT LOCK\n\nLocked at: {datetime.now(timezone.utc).isoformat()}\n\n## Lock basis\n- Fixed dataset protocol preserved: 4,569 training, 1,143 validation, 1,311 official held-out test images.\n- Five standardized backbones only: VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3.\n- Standardized five-backbone evaluation complete.\n- ResNet152 Grad-CAM and final-SVM occlusion evaluations complete.\n- Duplicate/leakage audit: CLEAN. No exact or unresolved high-similarity near-duplicate cross-split images detected by the final image-level audit.\n- Patient-level independence is not claimed because reliable patient identifiers are unavailable.\n\n## Experimental freeze\nNo retraining, additional backbones, additional Grad-CAM layers, SVM retuning, PCA retuning, split changes, or alternative occlusion settings are required for the finalized study. Any future experimentation must be explicitly labeled post-lock and must not replace the locked results.\n\n## Documentation action\n{patch_note}\n\n## Next phase\nFinal report, tables/figures, presentation, and viva preparation only.\n"""
        (docs_dir / "FINAL_EXPERIMENTAL_LOCK.md").write_text(lock_text, encoding="utf-8")
        print("\nAUDIT CLEAN — EXPERIMENTAL PROJECT LOCKED.")
        print("Lock file:", docs_dir / "FINAL_EXPERIMENTAL_LOCK.md")
        return 0

    blocker = [
        "# FINAL QC BLOCKER",
        "",
        f"The duplicate/leakage audit returned **{status}**; therefore the experimental project was NOT locked.",
        "",
    ]
    if material:
        blocker.append("Material cross-split duplicate pairs were detected. Review `duplicate_leakage_findings.csv` before any lock decision.")
    if review:
        blocker.append("Potential near-duplicate pairs require visual review. Contact sheets are in `duplicate_leakage_audit/near_duplicate_review/`.")
    (docs_dir / "FINAL_QC_BLOCKER.md").write_text("\n".join(blocker) + "\n", encoding="utf-8")
    print(f"\nAudit status: {status}. Experimental lock withheld.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
