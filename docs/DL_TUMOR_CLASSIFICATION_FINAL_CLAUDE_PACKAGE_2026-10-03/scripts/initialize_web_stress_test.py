#!/usr/bin/env python3
from pathlib import Path
import csv

PROJECT_ROOT = Path.home() / "brain_tumor_project"
DATA_ROOT = PROJECT_ROOT / "external_data" / "web_stress_40_raw"
META_ROOT = PROJECT_ROOT / "phase2" / "external_validation" / "web_stress_40"
MANIFEST = META_ROOT / "web_stress_40_manifest.csv"

CLASSES = [
    ("glioma", "GLI"),
    ("meningioma", "MEN"),
    ("notumor", "NOR"),
    ("pituitary", "PIT"),
]

FIELDS = [
    "image_id",
    "class",
    "local_filename",
    "source_page_url",
    "source_domain",
    "source_title",
    "source_type",
    "source_reported_diagnosis",
    "license_or_access_note",
    "mri_sequence_if_known",
    "orientation_if_known",
    "annotation_status",
    "date_accessed",
    "eligibility_status",
    "notes",
]

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    META_ROOT.mkdir(parents=True, exist_ok=True)

    for cls, _ in CLASSES:
        (DATA_ROOT / cls).mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        raise RuntimeError(
            f"Manifest already exists:\n{MANIFEST}\n"
            "Nothing was overwritten."
        )

    rows = []
    for cls, prefix in CLASSES:
        for i in range(1, 11):
            rows.append({
                "image_id": f"{prefix}_{i:02d}",
                "class": cls,
                "local_filename": "",
                "source_page_url": "",
                "source_domain": "",
                "source_title": "",
                "source_type": "",
                "source_reported_diagnosis": "",
                "license_or_access_note": "",
                "mri_sequence_if_known": "",
                "orientation_if_known": "",
                "annotation_status": "",
                "date_accessed": "",
                "eligibility_status": "PENDING",
                "notes": "",
            })

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("WEB STRESS TEST 40 — INITIALIZED")
    print("=" * 72)
    print("\nImage folders:")
    for cls, _ in CLASSES:
        print(DATA_ROOT / cls)
    print("\nManifest:")
    print(MANIFEST)
    print("\nLocked target: 10 source-verified images/class = 40 total.")
    print("No model outputs should be inspected before the 40-image set is frozen.")

if __name__ == "__main__":
    main()
