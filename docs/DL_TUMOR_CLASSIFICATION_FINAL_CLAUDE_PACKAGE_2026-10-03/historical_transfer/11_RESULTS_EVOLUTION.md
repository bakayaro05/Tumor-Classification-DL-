# Results Evolution

> **Hostile-review qualification:** Saved best-epoch fields are zero-based; displayed epochs are one-based. Grad-CAM used a fixed 20-image sample. Occlusion used 11 deliberately selected cases and independently normalized each map. A pre-audit DenseNet121 claim is unsupported by supplied source/manifests/results and is excluded pending recovery.

### Exact explanation audit facts recovered

- Grad-CAM Layer 3 ratings: 1 Good, 9 Partial, 5 Poor; Layer 4: 3/2/10; 50/50 fusion: 2/6/7. Layer 3 was more relevant in 9/15 tumor cases, Layer 4 in 6/15, fusion in 0/15.
- Neural and SVM heads agreed on 20/20 samples and were both correct on 19/20. Both predicted glioma for pituitary `Te-pi_0078.jpg`.
- Occlusion: 32×32 patch, stride 16, 169 masks/image, 1,859 total. There were 58 flips pooled; median per-image flip rate was 0%; 5/11 had any flip. `Te-pi_0078.jpg` accounted for 38/169 flips and maximum drop 2.532; correct cases had 20/1,690 flips (~1.18%).
- Independent heatmap normalization can exaggerate negligible effects: recorded maxima include 0.0016 (`Te-me_0012.jpg`), 0.0045 (`Te-me_0137.jpg`), and 0.0051 (`Te-glTr_0002.jpg`).

## Interpretation rule

The numbers below are genuine recorded outputs. The 20 September 2026 leakage audit changed their interpretation: they are now **historical results on a contaminated split**, not valid clean-test estimates. No post-audit retraining result exists in the available record.

## Stage 1 — First ResNet152 baseline

The first full pipeline established technical feasibility:

- best neural checkpoint: epoch 8;
- neural validation accuracy: 97.99%;
- neural validation loss: 0.0677;
- SVM validation accuracy: 98.08%;
- test accuracy: 97.64%;
- test macro F1: 97.50%;
- test weighted F1: 97.63%;
- PCA: 64 components; approximately 99.47% explained variance recorded for this early root-level pipeline.

This run used legacy root-level artifacts and was later superseded by standardized per-backbone experiments.

## Stage 2 — VGG16 failure and recovery

The initial fully unfrozen VGG16 run collapsed around 27.9% validation accuracy. After freezing the convolutional backbone:

- best neural validation accuracy: 94.58%;
- SVM validation accuracy: 94.93%;
- test accuracy: 94.97%;
- test macro F1: 94.68%.

This was a large recovery but remained below residual/EfficientNet models.

## Stage 3 — Standardized five-backbone results

| Backbone | Best epoch | NN val acc. | SVM val acc. | Test acc. | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 16 | 94.58% | 94.93% | 94.97% | 94.86% | 94.57% | 94.68% | 94.94% |
| VGG19 | 18 | 94.58% | 94.49% | 93.52% | 93.28% | 93.02% | 93.10% | 93.48% |
| ResNet50 | 7 | 97.73% | 97.55% | 98.09% | 98.06% | 97.98% | 98.01% | 98.09% |
| ResNet152 | 8 | 98.25% | 98.25% | 98.40% | 98.35% | 98.28% | 98.32% | 98.40% |
| EfficientNet-B3 | 6 | 98.43% | 98.51% | 98.47% | 98.43% | 98.40% | 98.41% | 98.47% |

EfficientNet-B3 was the numerical leader on test accuracy and macro F1. ResNet152 was approximately 0.08 percentage points lower in test accuracy.

## Stage 4 — Per-class results

### VGG16 test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 95.37% | 89.33% | 92.25% | 300 |
| Meningioma | 89.07% | 90.52% | 89.79% | 306 |
| No tumor | 96.65% | 99.75% | 98.18% | 405 |
| Pituitary | 98.34% | 98.67% | 98.50% | 300 |

### VGG19 test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 93.48% | 86.00% | 89.58% | 300 |
| Meningioma | 86.08% | 88.89% | 87.46% | 306 |
| No tumor | 96.88% | 99.51% | 98.17% | 405 |
| Pituitary | 96.70% | 97.67% | 97.18% | 300 |

### ResNet50 test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 98.31% | 97.00% | 97.65% | 300 |
| Meningioma | 97.03% | 96.08% | 96.55% | 306 |
| No tumor | 98.53% | 99.51% | 99.02% | 405 |
| Pituitary | 98.35% | 99.33% | 98.84% | 300 |

### ResNet152 test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 98.32% | 97.33% | 97.82% | 300 |
| Meningioma | 97.08% | 97.71% | 97.39% | 306 |
| No tumor | 99.02% | 99.75% | 99.38% | 405 |
| Pituitary | 98.99% | 98.33% | 98.66% | 300 |

### EfficientNet-B3 test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 98.02% | 99.00% | 98.51% | 300 |
| Meningioma | 98.00% | 96.08% | 97.03% | 306 |
| No tumor | 99.02% | 99.51% | 99.26% | 405 |
| Pituitary | 98.67% | 99.00% | 98.84% | 300 |

The no-tumor class was consistently easiest on the historical split. VGG models showed their largest weakness in glioma/meningioma discrimination.

## Stage 5 — Test confusion matrices

Rows are true classes and columns are predicted classes in order: glioma, meningioma, notumor, pituitary.

- **VGG16:** `[[268,31,0,1],[13,277,12,4],[0,1,404,0],[0,2,2,296]]`
- **VGG19:** `[[258,37,0,5],[16,272,13,5],[0,2,403,0],[2,5,0,293]]`
- **ResNet50:** `[[291,7,0,2],[4,294,6,2],[1,0,403,1],[0,2,0,298]]`
- **ResNet152:** `[[292,7,0,1],[2,299,3,2],[0,1,404,0],[3,1,1,295]]`
- **EfficientNet-B3:** `[[297,3,0,0],[5,294,4,3],[1,0,403,1],[0,3,0,297]]`

## Stage 6 — Explainability results changed the qualitative conclusion

The Grad-CAM sample showed that correct class predictions often coexisted with poor lesion localization. Among tumor images, fused Grad-CAM was Good for 2/15, Partial for 6/15, and Poor for 7/15. Final-SVM occlusion was Good for 2/9 tumor cases, Partial for 2/9, and Poor for 5/9. These findings argue against claiming that the system reliably localizes tumor tissue.

## Stage 7 — Leakage audit changed the scientific conclusion

Audit result:

- 464 exact/pixel duplicate cross-split pairs;
- 98 high-confidence near-duplicate pairs;
- 562 material pairs total;
- 481 review candidates.

Thus the result evolution is not simply “accuracy improved.” The final and most important evolution is:

`high historical scores → explanation audit reveals spurious focus → split audit reveals material leakage → clean performance currently unknown`.

## Metrics not established

No authoritative project-level values were found for ROC-AUC, sensitivity, specificity, confidence intervals, calibration error, external-dataset accuracy, or statistical significance between backbones. Do not manufacture them. Training curves exist in result JSON/GUI pathways but are not reproduced numerically in this transfer.
