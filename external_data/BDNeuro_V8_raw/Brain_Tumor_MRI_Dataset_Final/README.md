# Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)

## Overview
- **Task**: Multi-class brain tumor classification from MRI images
- **Classes**: Glioma, Meningioma, No_tumor, Pituitary
- **Total images**: 5941 (after exact- and near-duplicate removal)
- **Split**: 70% Train / 15% Validation / 15% Test, stratified per class
- **Source**: Epic & CSCR Hospital [add full attribution / consent statement here]
- **License**: [FILL IN — e.g. CC BY-NC 4.0]
- **Citation**: [FILL IN — how you'd like this dataset cited]

## Preprocessing & integrity
1. Combined the original Train/Test images into a single pool.
2. Removed exact byte-identical duplicate files (MD5 hash).
3. Removed near-duplicate images via perceptual hashing (`imagehash.phash`,
   Hamming distance <= 5).
4. Re-split into Train/Val/Test with a per-class stratified split (fixed seed = 42)
   so every split preserves the overall class distribution.
5. Verified **zero image-content overlap** between Train/Val/Test.
6. Renamed all files to anonymized, sequential names (`<class>_<split>_<index>.<ext>`).

## Class distribution

|            |   train |   val |   test |   total |
|:-----------|--------:|------:|-------:|--------:|
| glioma     |    1534 |   329 |    328 |    2191 |
| meningioma |    1035 |   222 |    221 |    1478 |
| no_tumor   |     551 |   118 |    118 |     787 |
| pituitary  |    1040 |   223 |    222 |    1485 |
| TOTAL      |    4160 |   892 |    889 |    5941 |

## Baseline model benchmark
Reported on the Test split described above.

| Architecture   | Paradigm          |   Test Acc. |   Test F1 |
|:---------------|:------------------|------------:|----------:|
| CNN            | From scratch      |      0.847  |    0.8457 |
| ViT            | From scratch      |      0.8256 |    0.824  |
| Hybrid CNN–ViT | From scratch      |      0.8189 |    0.8201 |
| ResNet50       | Frozen pretrained |      0.8031 |    0.8022 |
| DenseNet121    | Frozen pretrained |      0.8706 |    0.8703 |
| ResNet50       | Fine-tuned        |      0.9708 |    0.9708 |
| DenseNet121    | Fine-tuned        |      0.9831 |    0.9831 |

## Directory structure
```
Brain_Tumor_MRI_Dataset_Final/
    train/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    val/
        ...
    test/
        ...
```

## Known limitations
- Duplicate/near-duplicate removal is image-level. If multiple images come from the same
  patient/study and are not visually near-identical, patient-level leakage may still be
  possible unless the patient-level split option was used when building this release.
