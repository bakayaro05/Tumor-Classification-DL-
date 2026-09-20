from pathlib import Path

import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "brain_tumor_mri"
    / "Training"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

VALIDATION_SPLIT = 0.20
SEED = 42


def main():

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = datasets.ImageFolder(
        root=TRAIN_DIR
    )

    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        shuffle=True,
        stratify=targets,
    )

    np.save(
        SPLIT_DIR / "train_indices.npy",
        train_indices,
    )

    np.save(
        SPLIT_DIR / "val_indices.npy",
        val_indices,
    )

    print("Saved:")
    print(SPLIT_DIR / "train_indices.npy")
    print(SPLIT_DIR / "val_indices.npy")

    print()
    print("Training images  :", len(train_indices))
    print("Validation images:", len(val_indices))

    print()
    print("TRAIN CLASS COUNTS")

    train_targets = targets[train_indices]

    for class_index, class_name in enumerate(dataset.classes):
        count = np.sum(train_targets == class_index)
        print(f"{class_index} -> {class_name:12s}: {count}")

    print()
    print("VALIDATION CLASS COUNTS")

    val_targets = targets[val_indices]

    for class_index, class_name in enumerate(dataset.classes):
        count = np.sum(val_targets == class_index)
        print(f"{class_index} -> {class_name:12s}: {count}")

    print()
    print("SPLIT FILES SAVED SUCCESSFULLY")


if __name__ == "__main__":
    main()