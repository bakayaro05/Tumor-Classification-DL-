from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("data/brain_tumor_mri")

SPLITS = ["Training", "Testing"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

print("=" * 60)
print("BRAIN TUMOR MRI DATASET INSPECTION")
print("=" * 60)

total_images = 0
corrupted = []

sizes = {}
modes = {}

for split in SPLITS:

    split_path = ROOT / split

    print(f"\n{'=' * 20} {split} {'=' * 20}")

    split_total = 0

    for class_dir in sorted(split_path.iterdir()):

        if not class_dir.is_dir():
            continue

        class_count = 0

        for image_path in class_dir.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in VALID_EXTENSIONS
            ):

                try:
                    with Image.open(image_path) as img:

                        size = img.size
                        mode = img.mode

                        sizes[size] = sizes.get(size, 0) + 1
                        modes[mode] = modes.get(mode, 0) + 1

                        class_count += 1

                except Exception as e:

                    corrupted.append((str(image_path), str(e)))

        split_total += class_count

        print(
            f"{class_dir.name:15s}: "
            f"{class_count:4d} images"
        )

    print(f"{'TOTAL':15s}: {split_total}")

    total_images += split_total


print("\n" + "=" * 60)

print(f"TOTAL DATASET IMAGES: {total_images}")

print("\nIMAGE MODES:")
for mode, count in sorted(modes.items()):
    print(f"{mode:10s}: {count}")

print("\nMOST COMMON IMAGE SIZES:")

for size, count in sorted(
    sizes.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(f"{str(size):20s}: {count}")


print("\nCORRUPTED IMAGES:", len(corrupted))

if corrupted:

    for path, error in corrupted[:20]:

        print("\nFile :", path)
        print("Error:", error)

print("\nDATASET INSPECTION COMPLETE")
print("=" * 60)