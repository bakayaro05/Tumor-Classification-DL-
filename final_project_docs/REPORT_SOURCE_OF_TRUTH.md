# Report Source of Truth

This file is generated from the standardized `experiments/<backbone>/results/` artifacts.
Do not mix these values with older root-level ResNet152 preliminary artifacts.

## Experimental status

- Duplicate/leakage audit status: **MATERIAL_DUPLICATION**
- Saved split: 4,569 training; 1,143 validation; 1,311 official held-out test images.
- Five standardized backbones: VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3.
- Final classifier for each backbone: CNN → LSTM → 128-D Dense → training-only StandardScaler → training-only PCA(64) → Linear SVM (C=1.0).
- Formal explainability evaluation: ResNet152 only.

## Standardized five-backbone results

| Model | Best epoch | NN val acc (%) | SVM val acc (%) | Test acc (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) | PCA | PCA variance (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 16 | 94.58 | 94.93 | 94.97 | 94.86 | 94.57 | 94.68 | 64 | 99.18 |
| VGG19 | 18 | 94.58 | 94.49 | 93.52 | 93.28 | 93.02 | 93.10 | 64 | 99.33 |
| ResNet50 | 7 | 97.73 | 97.55 | 98.09 | 98.06 | 97.98 | 98.01 | 64 | 99.45 |
| ResNet152 | 8 | 98.25 | 98.25 | 98.40 | 98.35 | 98.28 | 98.32 | 64 | 99.39 |
| EfficientNet-B3 | 6 | 98.43 | 98.51 | 98.47 | 98.43 | 98.40 | 98.41 | 64 | 99.33 |

EfficientNet-B3 has the highest observed test accuracy in the standardized comparison; the difference from ResNet152 is one correctly classified test image and should not be described as statistically superior without a formal paired test.

## Explainability scope and locked interpretation

- Formal Grad-CAM protocol: ResNet152, seed 42, five images per class, 20 total; Layer 3 (14×14), Layer 4 (7×7), and 50/50 fused map; fixed opacity 0.55.
- Tumor-image Grad-CAM Good-or-Partial: Layer 3 10/15 (66.7%); Layer 4 5/15 (33.3%); fused 8/15 (53.3%).
- Final-SVM occlusion protocol: fixed 11 cases, patch 32×32, stride 16, 169 occlusions/image; nine tumor and two no-tumor cases.
- Tumor-image occlusion: Good 2/9; Partial 2/9; Poor 5/9; Good-or-Partial 4/9 (44.4%).
- Direct comparison: Grad-CAM more anatomically relevant 4/9; occlusion more relevant 3/9; both predominantly off-lesion 2/9; both clearly lesion-focused 0/9.
- Across 1,859 occlusions: 58 prediction-changing occlusions (3.12% pooled); median per-image flip rate 0%; five of 11 images had at least one flip.
- Do not describe Grad-CAM or occlusion maps as segmentation masks, tumor boundaries, or proof of clinically valid reasoning.
- Strong classification performance is a dataset-level result and does not establish consistently lesion-based reasoning or clinical readiness.

## Leakage audit statement

The experimental lock is pending because the duplicate/leakage audit did not return CLEAN. Do not finalize the no-duplication statement until the blocker is resolved.
