# Experiment History

All metrics below are actual recorded outputs unless explicitly labeled **proposed**. Because the 20 September 2026 audit found material cross-split duplication, classification metrics from the original split must be described as **historical contaminated-split results**.

## Successful and completed experiments

### E0 — Dataset readability and mode audit

- **Dataset:** 7,023/7,023 images readable; 0 corrupt.
- **Modes:** 3,093 grayscale `L`, 3,926 `RGB`, 3 `RGBA`, and 1 palette `P`; dimensions varied.
- **Implemented normalization:** Convert to RGB and resize to 224×224.

### E1 — Tensor-shape and module verification

- **Phase:** Late August / early September 2026
- **Objective:** Build the hybrid model safely in small pieces.
- **Configuration:** ResNet152 feature map `[B,2048,7,7]`; reshape to `[B,49,2048]`; LSTM hidden size 256; BatchNorm; Dropout 0.3; Dense 128; four logits.
- **Procedure:** Dedicated scripts tested ResNet features, LSTM section, post-LSTM layers, complete model, real MRI batches, and one training step.
- **Result:** Forward shapes, deep features, logits, loss, gradients, and parameter updates were verified.
- **Learning:** The LSTM sequence comprises rasterized spatial positions, not time or MRI slices.

### E2 — Early paper-inspired ResNet152 baseline

- **Phase:** Early September 2026
- **Objective:** Establish one working end-to-end implementation before generalization.
- **Dataset:** 4,569 train; 1,143 validation; 1,311 test.
- **Model:** ImageNet ResNet152→LSTM(256)→BN→Dropout→Dense(128)→temporary neural head; RMSprop `1e-4`.
- **Feature classifier:** StandardScaler→PCA(64)→linear SVM `C=1`.
- **Actual result:** Best neural checkpoint at epoch 8; validation accuracy 97.99%, validation loss 0.0677. SVM validation accuracy 98.08%. Final test accuracy 97.64%, macro F1 97.50%, weighted F1 97.63%.
- **Influence:** Confirmed the hybrid pipeline and motivated the five-backbone framework.
- **Status:** Completed but superseded by standardized experiment artifacts; split later found contaminated.

### E3 — Standardized five-backbone comparison

- **Phase:** 6–12 September 2026
- **Objective:** Compare alternative CNN feature extractors under a common pipeline.
- **Common configuration:** Seed 42; LSTM 256; Dense 128; Dropout 0.3; RMSprop `1e-4`; StandardScaler and PCA fitted on training only; PCA 64; linear SVM `C=1`; deterministic extracted features.
- **Backbone policy:** VGG16/VGG19 convolutional blocks frozen; ResNet50/ResNet152/EfficientNet-B3 fine-tuned.

| Backbone | Best epoch | NN val acc. | SVM val acc. | Test acc. | Test macro F1 | PCA variance retained |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | 16 | 94.58% | 94.93% | 94.97% | 94.68% | 99.1796% |
| VGG19 | 18 | 94.58% | 94.49% | 93.52% | 93.10% | 99.3344% |
| ResNet50 | 7 | 97.73% | 97.55% | 98.09% | 98.01% | 99.4493% |
| ResNet152 | 8 | 98.25% | 98.25% | 98.40% | 98.32% | 99.3901% |
| EfficientNet-B3 | 6 | 98.43% | 98.51% | 98.47% | 98.41% | 99.3340% |

- **Interpretation at the time:** EfficientNet-B3 achieved the best test accuracy; ResNet152 remained extremely close and best aligned with the base paper.
- **Current interpretation:** Ranking and scores may be affected by cross-split duplicate leakage. They require clean reruns.
- **Run-record caveat:** Best-epoch fields are zero-based; displayed epochs are one-based. Recorded durations: ResNet50 3.83 min, ResNet152 8.37 min, VGG19 7.04 min, VGG16 2.88 min initially plus 2.59 min resumed. These are not controlled efficiency measurements.

### E4 — Grad-CAM frozen qualitative evaluation

- **Phase/date:** 20 September 2026 evaluation records
- **Objective:** Determine whether high-performing ResNet152 focused on visually plausible tumor regions.
- **Protocol:** Seed 42; five samples per class; 20 test images; layer 3, layer 4, and fused maps; fixed 0.55 overlay opacity; target was the neural-head predicted class.
- **Prediction result:** 19/20 sampled cases were correctly classified by both heads; one pituitary image was classified as glioma by both.
- **Localization result for 15 tumor images:**

| Map | Good | Partial | Poor |
|---|---:|---:|---:|
| Layer 3 | 1 | 9 | 5 |
| Layer 4 | 3 | 2 | 10 |
| Fused | 2 | 6 | 7 |

- **Unexpected result:** Many maps were dominated by inferior skull, face, border, or extracranial hotspots. One no-tumor sample (`Te-no_0216.jpg`) appeared to contain a conspicuous rounded hyperintense region and was flagged for expert review.
- **Learning:** Classification correctness does not guarantee anatomically plausible saliency. Earlier layers sometimes helped, but no map was consistently best.
- **Additional comparison:** Layer 3 was judged more relevant than Layer 4 on 9/15 tumor cases; Layer 4 on 6/15; fusion on 0/15. Both classifier heads agreed on all 20 and were jointly correct on 19. The shared error was `Te-pi_0078.jpg`, predicted glioma. The sample is small and fixed, not representative.

### E5 — Final-SVM occlusion comparison

- **Phase/date:** 20 September 2026 evaluation records
- **Objective:** Explain the actual SVM decision rather than only the neural head.
- **Protocol:** 11 frozen samples derived from the Grad-CAM set; patch 32×32; stride 16; 169 occlusions per image; masked values replaced by ImageNet channel means.
- **Tumor-image localization ratings (9 images):** Good 2, Partial 2, Poor 5.
- **Comparison:** Occlusion more anatomically relevant in 3; Grad-CAM more relevant in 4; both predominantly off-lesion in 2. Two no-tumor cases were not applicable for lesion localization.
- **Learning:** Occlusion can reveal evidence used by the final SVM, but it is not automatically more plausible than Grad-CAM and must not be called segmentation.
- **Decision stability:** 1,859 occlusions produced 58 class flips (3.12% pooled); median per-image flip rate was 0%; 5/11 images had any flip. The misclassified `Te-pi_0078.jpg` accounted for 38/169 flips (22.5%, maximum target-score drop 2.532); correct cases had 20/1,690 flips (~1.18%).
- **Normalization warning:** Each heatmap was normalized independently. Bright red can therefore mean a negligible absolute score change: maxima included 0.0016 (`Te-me_0012.jpg`), 0.0045 (`Te-me_0137.jpg`), and 0.0051 (`Te-glTr_0002.jpg`).
- **Selection warning:** The 11 images were deliberately selected from the earlier Grad-CAM review; they are not a random or representative test sample.

### E6 — Cross-split duplicate/leakage audit

- **Date:** 20 September 2026
- **Objective:** Test whether train, validation, and test were independent at image level.
- **Dataset:** All 7,023 files; split counts 4,569/1,143/1,311.
- **Protocol:** Exact-file/pixel matching plus perceptual candidates using pHash and dHash; candidate thresholds pHash Hamming ≤4 and dHash Hamming ≤6; review rule included grayscale MAE ≤12 and correlation ≥0.995.
- **Actual result:** `MATERIAL_DUPLICATION`; 464 exact or pixel-duplicate cross-split pairs; 98 additional high-confidence near-duplicate pairs; 562 material pairs; 481 further review candidates.
- **Learning:** The original evaluation is compromised. Image-level auditing cannot establish patient-level independence.
- **Influence:** Prevented final lock and created the required clean-rerun phase.

## Failed experiments

### F1 — TensorFlow GPU route

- **Symptom:** GPU detection and trivial matrix multiplication succeeded, but real execution failed with unsupported PTX/`sm_120` errors.
- **Result:** Abandoned TensorFlow; migrated to PyTorch.

### F2 — Fully unfrozen VGG16 training

- **Objective:** Fine-tune VGG16 like the other backbones.
- **Result:** Validation collapsed around 27.9%, close to majority/chance-like behavior for four classes.
- **Interpretation:** The configuration was unstable; exact single root cause was not conclusively isolated in the available record.
- **Diagnostic evidence:** Trained features were effectively constant and PCA produced NaNs; direct pretrained activations remained healthy.
- **Influence:** VGG16 and VGG19 backbones were frozen.

## Partial or limited experiments

### P1 — Explainability evaluation

Grad-CAM and occlusion were implemented and qualitatively reviewed on small fixed samples. There was no expert lesion mask, quantitative localization benchmark, inter-rater reliability analysis, or external dataset. These are not clinical validation studies.

### P2 — GUI testing

Source and verification scripts exist, and the GUI was iteratively developed. This transfer did not rerun the full GUI with every heavy artifact, GPU path, and image format. Treat operational completeness as codebase-dependent.

## Proposed but not completed

- Full base-paper optimizer matrix (five backbones × Adam/SGD/AdaGrad/RMSprop = 20 runs). Only the RMSprop branch is verified in this project; this is unexecuted scope, not a rejected scientific option.
- DenseNet121 remains contradictory: one pre-audit narrative says it was evaluated, but supplied source/manifests/results do not substantiate it.

- Duplicate-group-aware resplitting and full retraining of all five models (**blocking**).
- External-dataset or institution-level validation.
- Patient-level split using reliable identifiers (currently impossible with known metadata).
- Formal statistical comparison, confidence intervals, repeated seeds, or cross-validation.
- Quantitative explainability with expert masks and radiologist review.
- Calibration analysis for class probabilities/confidence.
- Efficiency comparison using parameters, VRAM, latency, and energy.
- Attention mechanisms, ensembles, or lightweight deployment models.
- DICOM support and clinically appropriate preprocessing.
- True benign/malignant or grade classification using a properly labeled dataset.
