# MASTER FINAL PROJECT CONTEXT — DL TUMOR CLASSIFICATION

**Authoritative status date:** 3 October 2026  
**Purpose:** self-contained context for Claude, ChatGPT, report writing, viva preparation, code maintenance, and final project handoff.

---

## 1. Final one-paragraph project summary

DL TUMOR CLASSIFICATION is a four-class brain MRI classification project based on the 2026 Springer paper *Hybrid CNN+LSTM+SVM model for advanced brain tumor classification*. The final implementation uses one of five alternative ImageNet-pretrained CNN backbones — VGG16, VGG19, ResNet50, ResNet152, or EfficientNet-B3 — followed by spatial feature-map flattening into a sequence, a one-layer LSTM, BatchNorm, Dropout, a 128-dimensional Dense/ReLU representation, and a temporary four-class neural head. The learned 128-D features are also passed through a StandardScaler fitted on training data only, PCA to 64 components fitted on training data only, and a linear SVM with C=1. The project grew from a faithful paper-inspired baseline into a rigorous evaluation study covering convergence, hyperparameter sensitivity, data leakage, external validation, explainability, GUI deployment, and out-of-distribution stress testing.

---

## 2. Final project question

The mature project is not merely “how high is the classification accuracy?” It asks:

> Can a hybrid CNN-LSTM-PCA-SVM brain MRI classifier achieve strong four-class performance, and how trustworthy are those results after examining training behavior, duplicate leakage, external generalization, and model explanation behavior?

The four classes are:
- glioma
- meningioma
- notumor
- pituitary

---

## 3. Project origin and evolution

### 3.1 Earliest classical direction — July 2026

The project initially considered a classical MRI pipeline: resizing/grayscale conversion, median filtering and contrast enhancement, skull/background removal, K-means segmentation, morphology, shape/GLCM features, standardization/PCA, and RBF-SVM. Candidate papers included Mandle et al., Rao/Karunakara, Saad et al., and Malathi/Sinthia.

Important early cautions:
- a K-means cluster index is arbitrary and cannot automatically be called “tumor”;
- HGG/LGG must not be silently re-labelled malignant/benign;
- the classification labels must be supported by the actual dataset.

This direction was superseded when a recent deep-learning base paper was required.

### 3.2 Base-paper selection — August 2026

The project selected the 2026 Springer paper *Hybrid CNN+LSTM+SVM model for advanced brain tumor classification*. The paper provided:
- five CNN backbones;
- CNN feature map → LSTM → Dense representation;
- StandardScaler/PCA/linear-SVM hybrid classification;
- a 4-class Kaggle MRI dataset;
- 30 epochs;
- LR 1e-4;
- multiple optimizers.

The project did **not** reproduce the complete five-backbone × four-optimizer 20-run paper grid. RMSprop became the project optimizer. The work should therefore be described as a **faithful/adapted implementation of the core architecture**, not an exact replication of every paper experiment.

### 3.3 TensorFlow failure and PyTorch migration

The first GPU route used TensorFlow under Ubuntu/WSL2 on an RTX 5070 Laptop GPU. TensorFlow could detect the GPU and perform simple operations but failed during real workloads with an `INVALID_PTX` / Blackwell `sm_120` compatibility problem. Nightly TensorFlow did not provide a reliable path.

Final framework:
- Ubuntu under WSL2
- Python 3.12.14
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- CUDA available
- NVIDIA RTX 5070 Laptop GPU, 8 GB VRAM

All authoritative project model artifacts are therefore PyTorch/scikit-learn based.

---

## 4. Dataset and split

### 4.1 Original development dataset

Masoud Nickparvar Kaggle Brain Tumor MRI dataset:
- total: 7,023
- supplied Training: 5,712
- supplied Testing: 1,311

Classes in the complete dataset:
- glioma: 1,621
- meningioma: 1,645
- notumor: 2,000
- pituitary: 1,757

The supplied Training directory was split once with stratification and seed 42:
- train: 4,569
- validation: 1,143
- official test: 1,311

The same saved split was reused across all backbones for controlled comparison.

### 4.2 Dataset inspection

All 7,023 files were readable. Historical inspection found heterogeneous image modes and spatial dimensions; conversion to RGB was necessary for ImageNet-pretrained CNN inputs.

### 4.3 Preprocessing

Authoritative path:
- convert image to RGB;
- resize to 224×224;
- ImageNet normalization;
- random augmentation only during training;
- deterministic evaluation transform for validation, test, feature extraction, external data and uploaded GUI images.

Training augmentation included horizontal/vertical flips and photometric perturbation. Saturation/hue were paper-specified, but exact presence in every historical PyTorch snapshot should not be reconstructed from memory if `src/data_pipeline_torch.py` is available — inspect the source.

---

## 5. Final model architecture

The five backbones are **alternative pipelines**, not an ensemble.

General architecture:

`MRI → pretrained CNN → spatial feature map → flatten H×W positions into a sequence → LSTM → last LSTM output → BatchNorm → Dropout → Dense128 → ReLU`

Two branches then use the 128-D learned features:

1. **Neural head**  
   `Dense128 → Linear(4 logits)`  
   CrossEntropyLoss is applied directly to raw logits during training. Softmax is inference-only.

2. **Hybrid final classifier**  
   `Dense128 features → StandardScaler(train only) → PCA64(train only) → Linear SVM C=1`

Baseline hyperparameters:
- optimizer: RMSprop
- learning rate: 1e-4
- batch size: 8
- seed: 42
- LSTM hidden size: 256
- Dense features: 128
- dropout: 0.3
- four output classes

For ResNet152, the feature map example is:
`[B, 2048, 7, 7] → [B, 49, 2048] → LSTM → [B, 49, 256] → last step [B,256] → Dense128`.

The LSTM therefore models a rasterized **spatial sequence inside one 2D image**. It does not process time or a patient’s ordered MRI slices.

---

## 6. Backbone policy and VGG stabilization

ResNet50, ResNet152 and EfficientNet-B3 were trained as the fine-tuned family.

VGG16 initially collapsed when fully unfrozen:
- validation performance fell to about 28%;
- learned downstream features became nearly constant;
- PCA produced NaNs;
- diagnostic checks showed pretrained convolutional activations themselves were healthy.

Decision:
- freeze pretrained VGG16 convolutional backbone;
- use the same frozen-backbone policy for VGG19;
- train the LSTM/post-LSTM/classifier layers.

This means cross-backbone comparison is a practical pipeline comparison, not a perfectly symmetric optimization experiment.

---

## 7. Phase 1 — baseline implementation and five-backbone comparison

Phase 1 established that the complete hybrid pipeline worked for all five architectures.

### 7.1 Neural-head versus PCA+SVM results

| Backbone | Best epoch | Neural test acc | Neural macro F1 | SVM val acc | PCA+SVM test acc | PCA+SVM macro F1 | Test acc change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vgg16 | 16 | 93.67% | 93.32% | 94.93% | 94.97% | 94.68% | +1.30 pp |
| vgg19 | 18 | 93.14% | 92.72% | 94.49% | 93.52% | 93.10% | +0.38 pp |
| resnet50 | 7 | 97.33% | 97.19% | 97.55% | 98.09% | 98.01% | +0.76 pp |
| resnet152 | 8 | 98.17% | 98.06% | 98.25% | 98.40% | 98.32% | +0.23 pp |
| efficientnet_b3 | 6 | 98.02% | 97.94% | 98.51% | 98.47% | 98.41% | +0.45 pp |

Correct interpretation:
- PCA+SVM improved Phase 1 **test** accuracy and macro-F1 for all five backbones;
- validation did not uniformly improve;
- therefore do not claim a universal PCA benefit;
- because both dimensionality reduction and classifier family change, call this a **downstream-classifier comparison**, not a PCA-only ablation.

### 7.2 Historical status of these internal scores

These are real recorded outputs but the later leakage audit showed the saved split is not image-level independent. They must be described as **historical contaminated-split internal results**, not clean generalization estimates.

---

## 8. Explainability and GUI work

### 8.1 Streamlit GUI

The GUI was implemented after the standardized five-backbone pipeline was complete. It supports:
- upload ordinary raster MRI images;
- select any of the five backbones;
- final SVM prediction;
- neural-head prediction and agreement status;
- class scores;
- validation/test performance;
- confusion matrices and per-class metrics;
- model comparison;
- Grad-CAM;
- final-SVM occlusion analysis.

Safeguards:
- uses deterministic evaluation preprocessing;
- loads saved scaler/PCA/SVM rather than fitting on uploaded images;
- SVM decision values are relative scores, not calibrated clinical probabilities;
- research/educational prototype only;
- no DICOM support in the first GUI version;
- not a medical device.

### 8.2 Grad-CAM

Frozen formal evaluation:
- ResNet152
- seed 42
- 20 official-test images
- 5 per class
- Layer 3, Layer 4, 50:50 fused map
- fixed opacity 0.55
- 15 tumor images rated Good/Partial/Poor

Results:
- Layer 3: Good 1, Partial 9, Poor 5; chosen most relevant 9/15
- Layer 4: Good 3, Partial 2, Poor 10; chosen most relevant 6/15
- fused: Good 2, Partial 6, Poor 7; chosen most relevant 0/15
- neural and SVM heads agreed on all 20;
- both were correct on 19/20;
- shared error: `Te-pi_0078.jpg`, true pituitary, predicted glioma.

Finding: strong classification accuracy can coexist with poor/off-lesion attribution. Grad-CAM is a coarse neural-head contribution map, not segmentation and not proof of clinically valid reasoning.

### 8.3 Final-SVM occlusion

Protocol:
- 11 fixed cases;
- 9 tumor + 2 no-tumor controls;
- patch 32×32;
- stride 16;
- 169 occlusions/image;
- complete CNN→LSTM→Scaler→PCA→SVM rerun.

Tumor localization:
- Good 2
- Partial 2
- Poor 5

Comparison:
- Grad-CAM more anatomically relevant: 4 cases
- occlusion more relevant: 3
- both predominantly off-lesion: 2

Stability:
- 1,859 total occlusions
- 58 class flips = 3.12% pooled
- median per-image flip rate 0%
- 5/11 had any class flip
- `Te-pi_0078.jpg`: 38/169 flips = 22.5%

Finding: occlusion directly probes the final SVM decision but remains qualitative and coarse.

---

## 9. Phase 2D — data integrity audit changed the interpretation

Before final lock, the project performed a train↔validation, train↔test and validation↔test exact/perceptual duplicate audit.

Authoritative audit:
- 7,023 images hashed
- 464 cross-split exact-file or decoded-pixel duplicate pairs
- 98 additional high-confidence near-duplicate pairs
- 562 material cross-split pairs total
- 481 additional perceptual review candidates
- 250 unique official-test images implicated
- patient-level independence cannot be established without reliable patient IDs

Status: `MATERIAL_DUPLICATION`.

### 9.1 Material-clean post-hoc test

The implicated 250 test images were excluded from a **post-hoc** evaluation:
- official test: 1,311
- material_clean: 1,061

| Backbone | material_clean acc | material_clean macro F1 | Original test acc |
| --- | --- | --- | --- |
| vgg16 | 94.816% | 94.302% | 94.966% |
| vgg19 | 93.025% | 92.415% | 93.516% |
| resnet50 | 98.303% | 98.134% | 98.093% |
| resnet152 | 98.398% | 98.315% | 98.398% |
| efficientnet_b3 | 98.680% | 98.511% | 98.474% |

Interpretation:
- performance did not collapse after removing directly implicated test images;
- however this does **not** undo the fact that training/validation were built from a contaminated image-level split;
- it does not establish patient-level independence;
- it is not equivalent to a newly group-separated training run.

### 9.2 Methodology decision after the audit

Immediately after discovery, an earlier transfer snapshot treated duplicate-group-aware re-splitting/retraining as a blocking next step. The later project roadmap refined that position:

- preserve original metrics as historical contaminated-split outputs;
- use material_clean as a narrow post-hoc leakage-aware test;
- continue controlled Phase 2A/2B studies with the leakage caveat;
- add independently audited external validation;
- keep full duplicate-group-aware re-splitting/retraining as a **stronger optional future remediation**, required only for a stronger clean internal-generalization claim.

This later decision is authoritative.

---

## 10. Phase 2A — common fixed 30-epoch study

Purpose:
- remove early stopping as a training-horizon confounder;
- compare convergence, late-epoch stability and overfitting under a common 30-epoch horizon;
- retain both the best-validation checkpoint and the epoch-30 state.

All five baselines:
- LR 1e-4
- LSTM256
- dropout0.3
- batch8
- seed42
- 30/30 epochs completed.

Full-hybrid PCA+SVM state selection by internal validation:

| Backbone | State | Epoch | SVM val acc | Macro F1 | Internal test acc* |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | best_val_within_30 | 27 | 99.300% | 99.270% | 99.314% |
| resnet152 | best_val_within_30 | 10 | 98.600% | 98.545% | 98.703% |
| resnet50 | epoch_30 | 30 | 98.688% | 98.644% | 98.932% |
| vgg16 | epoch_30 | 30 | 95.888% | 95.746% | 96.186% |
| vgg19 | epoch_30 | 30 | 95.626% | 95.462% | 95.271% |

`*` The Phase 2A result JSONs include original internal-test metrics; they remain contaminated-split historical outputs and were not used for external model selection.

Important distinction:
- the training-history “best neural validation accuracy” and the post-training **PCA+SVM validation accuracy** are different metrics;
- final retained-state selection for external comparison used the full-hybrid validation result.

---

## 11. Phase 2B — controlled hyperparameter sensitivity

One-factor-at-a-time design around baseline LR 1e-4 / LSTM256 / dropout0.3.

Seven changed configurations per backbone:
- LR 1e-5
- LR 5e-5
- LR 5e-4
- LSTM128
- LSTM512
- dropout0.2
- dropout0.5

Five backbones × seven variants = **35/35 completed neural training runs**.

Both retained states were evaluated downstream:
- best_val_within_30
- epoch_30

Therefore Phase 2B produced **70/70 downstream PCA+SVM validation evaluations**.

Best changed-config full-hybrid state per backbone:

| Backbone | Selected changed config | State | Epoch | SVM val acc | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% |
| resnet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% |
| resnet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% |
| vgg16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% |
| vgg19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% |

### 11.1 Main Phase 2B sensitivity findings

Across the 40 matched neural-training runs (5 Phase2A baselines + 35 Phase2B variants):
- mean within-backbone peak-validation range from learning-rate changes: **4.62 pp**
- LSTM-width range: **0.58 pp**
- dropout range: **0.42 pp**

Learning rate was therefore the dominant tested sensitivity.

LSTM512 increased peak neural validation versus LSTM256 baseline for all five backbones, but only by roughly +0.09 to +0.70 pp and not always with stable epoch-30 behavior.

The clearest unstable setting was ResNet50 LR5e-4:
- best neural validation ≈94.75% at epoch25;
- epoch30 ≈90.99%;
- about -3.76 pp drift.

Because there was one seed and one contaminated validation split, these are controlled sensitivity findings, not statistically established superiority claims.

### 11.2 Best-state versus epoch30 downstream comparison

Across the 35 changed configurations:
- best-neural retained state gave higher SVM validation in 20 cases;
- epoch30 gave higher SVM validation in 8;
- identical in 7.

Only two retained states per run were evaluated downstream, so do not claim the absolute best SVM epoch from all 30 epochs was found.

---

## 12. Pre-external freeze — 15 pipelines

To prevent external-data tuning, exactly 15 full hybrid pipelines were frozen before external evaluation:
- five Phase1 early-stopping pipelines;
- five Phase2A standardized representatives;
- five Phase2B changed-config representatives.

The freeze manifest hashes checkpoint, scaler, PCA, SVM and internal result artifacts. After external results were generated, model selection was not regenerated.

Phase2A representatives:
- VGG16 epoch30
- VGG19 epoch30
- ResNet50 epoch30
- ResNet152 best-within-30 epoch10
- EfficientNet-B3 best-within-30 epoch27

Phase2B representatives:
- VGG16 LR5e-4, best, epoch27
- VGG19 dropout0.5, epoch30
- ResNet50 dropout0.2, best, epoch28
- ResNet152 LSTM512, best, epoch26
- EfficientNet-B3 LSTM512, best, epoch29

---

## 13. Phase 2C — BDNeuro external validation

### 13.1 Why BDNeuro was chosen

The project needed a structurally compatible four-class external MRI source to test frozen pipelines without tuning. BDNeuro-MRI V8 had the same broad four-class task and documented Bangladeshi clinical provenance.

### 13.2 Raw BDNeuro inventory

True MRI cohort: **5,941** images. A naive recursive total of 5,943 included two root-level plot PNGs, which were not class MRIs.

Class totals:
- glioma 2,191
- meningioma 1,478
- notumor 787
- pituitary 1,485

Split totals:
- train 4,160
- val 892
- test 889

### 13.3 Cross-dataset audit

BDNeuro was audited against all 7,023 development images using:
- SHA256 exact bytes;
- decoded RGB pixel hash;
- 64-bit pHash.

Findings:
- exact file pairs: 3,421
- exact pixel pairs: 3,519
- unique BDNeuro images with exact-file overlap: 3,295
- unique BDNeuro images with exact-pixel overlap: **3,297**
- pHash <=6 candidate pairs: 9,981
- pHash <=4 strong pairs: 6,664

Within BDNeuro:
- exact file duplicate pairs: 0
- exact pixel duplicate pairs: 0
- pHash <=6 candidates: 1,588
- pHash <=4 strong candidates: 65

Documented provenance differences were not allowed to override direct pixel identity. Exact decoded-pixel identity is objective evidence that an image is reused.

### 13.4 Exclusion decision

Final rule:
- exclude **exact decoded-pixel cross-dataset matches only**;
- do **not** automatically remove non-identical pHash candidates.

Reason: MRI anatomy can create strong perceptual similarity; pHash similarity alone does not prove that two scans are the same underlying image.

### 13.5 Primary exact-overlap-clean cohort

5,941 raw − 3,297 unique exact-pixel-overlap images = **2,644** retained.

Class counts:
- glioma 1,159
- meningioma 472
- notumor 412
- pituitary 601

Original BDNeuro split origin of retained images:
- train 1,845
- val 364
- test 435

All retained images were merged into one external cohort because the project used no BDNeuro subset for training or model selection.

### 13.6 Balanced sensitivity subset

The natural clean cohort is imbalanced. A secondary deterministic sensitivity subset selected the smallest class count:
- 412/class
- total 1,648

Selection used deterministic SHA256 ranking and no model outputs.

The full 2,644 cohort is the **primary** external validation. The balanced 1,648 subset is nested inside it and is **not a second independent dataset**.

### 13.7 Full-cohort results

| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 77.61% | 75.50% | -17.32 pp |
| vgg16 | Phase2A | 80.07% | 78.71% | -15.82 pp |
| vgg16 | Phase2B | 77.57% | 76.01% | -18.93 pp |
| vgg19 | Phase1 | 75.76% | 74.80% | -18.73 pp |
| vgg19 | Phase2A | 79.27% | 78.12% | -16.35 pp |
| vgg19 | Phase2B | 75.00% | 74.27% | -20.54 pp |
| resnet50 | Phase1 | 81.09% | 79.93% | -16.46 pp |
| resnet50 | Phase2A | 81.73% | 81.42% | -16.96 pp |
| resnet50 | Phase2B | 86.04% | 84.83% | -13.08 pp |
| resnet152 | Phase1 | 81.28% | 80.84% | -16.97 pp |
| resnet152 | Phase2A | 88.31% | 87.85% | -10.29 pp |
| resnet152 | Phase2B | 86.54% | 85.67% | -12.76 pp |
| efficientnet_b3 | Phase1 | 83.09% | 81.93% | -15.42 pp |
| efficientnet_b3 | Phase2A | 86.65% | 85.12% | -12.65 pp |
| efficientnet_b3 | Phase2B | 86.57% | 85.53% | -12.81 pp |

Stage averages:

| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 79.77% | 78.60% |
| Phase2A | 83.21% | 82.25% |
| Phase2B | 82.34% | 81.26% |

### 13.8 Balanced sensitivity results

| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 77.18% | 76.82% | -17.74 pp |
| vgg16 | Phase2A | 80.70% | 80.62% | -15.18 pp |
| vgg16 | Phase2B | 78.16% | 77.88% | -18.35 pp |
| vgg19 | Phase1 | 77.49% | 77.64% | -17.00 pp |
| vgg19 | Phase2A | 81.13% | 81.24% | -14.50 pp |
| vgg19 | Phase2B | 77.00% | 77.15% | -18.54 pp |
| resnet50 | Phase1 | 83.74% | 83.80% | -13.81 pp |
| resnet50 | Phase2A | 86.77% | 86.90% | -11.92 pp |
| resnet50 | Phase2B | 87.86% | 87.87% | -11.26 pp |
| resnet152 | Phase1 | 85.13% | 85.13% | -13.12 pp |
| resnet152 | Phase2A | 90.47% | 90.50% | -8.13 pp |
| resnet152 | Phase2B | 89.81% | 89.76% | -9.49 pp |
| efficientnet_b3 | Phase1 | 85.44% | 85.60% | -13.08 pp |
| efficientnet_b3 | Phase2A | 87.86% | 87.79% | -11.44 pp |
| efficientnet_b3 | Phase2B | 88.83% | 88.96% | -10.55 pp |

Stage averages:

| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 81.80% | 81.80% |
| Phase2A | 85.39% | 85.41% |
| Phase2B | 84.33% | 84.33% |

### 13.9 External interpretation

Key conclusions:
- every frozen pipeline dropped relative to internal validation;
- Phase2A generalized better on average than Phase2B on both full and balanced cohorts;
- ResNet50 benefited strongly from Phase2B dropout0.2;
- VGG16/VGG19 tuned variants generalized worse than Phase2A;
- ResNet152 Phase2A produced the strongest primary external result;
- EfficientNet-B3 Phase2A/2B remained competitive.

ResNet152 Phase2A:
- internal full-hybrid validation accuracy 98.60%
- BDNeuro full accuracy 88.31%, macro F1 87.85%
- BDNeuro balanced accuracy 90.47%, macro F1 90.50%
- internal→full drop -10.29 pp
- internal→balanced drop -8.13 pp

This is the strongest structured generalization evidence in the project.

---

## 14. In-the-wild 40-image web stress test

### 14.1 Why it was added

The guide-level practical question was: “What if I upload an MRI image that is not from the development dataset, for example an image found through the web?”

BDNeuro answers structured external-dataset generalization. The web stress test was added to probe a harder condition: heterogeneous publication/web images with major acquisition and presentation shift.

### 14.2 Cohort design

Exactly 40 source-verified MRI images:
- 10 glioma
- 10 meningioma
- 10 notumor/healthy
- 10 pituitary

Sources were medical/open-access pages and case reports. Search engines could be used to discover images, but the original source page established the diagnosis.

Before inference:
- source diagnosis documented;
- panel selected/cropped without changing anatomy;
- obvious CT, histology, segmentation maps, ambiguous labels and heavily annotated unsuitable images rejected;
- no failed image could be replaced after model predictions were seen.

### 14.3 Pre-inference overlap audit

Initial audit:
- 0 exact file duplicates within web set
- 0 exact pixel duplicates within web set
- 0 exact file/pixel overlaps against Nickparvar or BDNeuro
- two pHash candidates: NOR_01 distance0 and MEN_02 distance6.

Visual review:
- `NOR_01` was visibly the same underlying slice as Nickparvar `Te-no_0227.jpg` despite non-identical pixels; it was rejected and replaced **before inference**.
- `MEN_02` was clearly a different scan/orientation from its pHash candidate and was retained.

Final rerun:
- exact file duplicates: 0
- exact pixel duplicates: 0
- cross-dataset exact overlap: 0
- pHash <=4: 0
- one remaining pHash distance6 candidate, already reviewed.

The cohort was then locked. No image was changed after predictions.

### 14.4 Web stress results

| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 25.00% | 14.58% | -69.93 pp |
| vgg16 | Phase2A | 30.00% | 23.23% | -65.89 pp |
| vgg16 | Phase2B | 30.00% | 21.59% | -66.50 pp |
| vgg19 | Phase1 | 35.00% | 28.89% | -59.49 pp |
| vgg19 | Phase2A | 37.50% | 27.78% | -58.13 pp |
| vgg19 | Phase2B | 35.00% | 25.45% | -60.54 pp |
| resnet50 | Phase1 | 30.00% | 26.49% | -67.55 pp |
| resnet50 | Phase2A | 42.50% | 38.18% | -56.19 pp |
| resnet50 | Phase2B | 40.00% | 33.91% | -59.13 pp |
| resnet152 | Phase1 | 35.00% | 28.08% | -63.25 pp |
| resnet152 | Phase2A | 37.50% | 31.66% | -61.10 pp |
| resnet152 | Phase2B | 35.00% | 25.74% | -64.30 pp |
| efficientnet_b3 | Phase1 | 40.00% | 34.78% | -58.51 pp |
| efficientnet_b3 | Phase2A | 42.50% | 31.97% | -56.80 pp |
| efficientnet_b3 | Phase2B | 35.00% | 28.07% | -64.39 pp |

Stage averages:

| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 33.00% | 26.56% |
| Phase2A | 38.00% | 30.56% |
| Phase2B | 35.00% | 26.95% |

Best web accuracy:
- ResNet50 Phase2A: 42.5% = 17/40
- EfficientNet-B3 Phase2A: 42.5% = 17/40

Because n=40, one image = 2.5 percentage points. Small numerical differences are not stable enough to support fine ranking.

### 14.5 Strong prediction bias

Across all 15 web evaluations, mean per-class recall was approximately:
- glioma: 12.7%
- meningioma: 30.7%
- notumor: 90.7%
- pituitary: 7.3%

Average prediction counts per 40 images:
- glioma ≈2.1
- meningioma ≈8.3
- notumor ≈28.3
- pituitary ≈1.3

The dominant failure mode was therefore not random error. Under strong web-domain shift, models heavily over-predicted **notumor**.

### 14.6 Correct interpretation

Do not report “42.5% clinical accuracy.” The correct wording is:

> In a small, source-verified 40-image in-the-wild web MRI stress test with substantial acquisition, sequence, orientation and presentation heterogeneity, frozen model accuracy ranged from 25% to 42.5%, with particularly poor glioma and pituitary recall and a strong bias toward the no-tumor class.

This demonstrates sensitivity to strong distribution shift.

---

## 15. Generalization gradient across the project

The project now contains progressively harder evidence:

1. internal validation: roughly 95–99%
2. historical official/material-clean internal testing: roughly 93–99%
3. structured BDNeuro exact-overlap-clean external evaluation: roughly 75–88% on the full cohort
4. source-verified web stress test: 25–42.5%

This is one of the main scientific findings:

> High internal performance does not imply robustness to increasingly dissimilar external MRI domains.

---

## 16. Final Phase 3 candidate selection

The user’s intended meaning of “final candidate selection” is simple: identify the model that gave the strongest overall project evidence.

**Selected final project representative:**
### ResNet152 — Phase 2A standardized 30-epoch pipeline
- LR 1e-4
- LSTM 256
- dropout 0.3
- Dense128
- best retained state at epoch10
- StandardScaler → PCA64 → linear SVM C=1

Evidence:
- internal full-hybrid val accuracy: 98.60%
- internal macro F1: 98.55%
- BDNeuro full: 88.31% accuracy, 87.85% macro F1
- BDNeuro balanced: 90.47% accuracy, 90.50% macro F1
- smallest internal→BDNeuro accuracy gap among the 15 frozen candidates on both full and balanced evaluations
- web stress: 37.5% accuracy, showing that even the selected model is not robust to unrestricted domain shift

ResNet50 Phase2A and EfficientNet-B3 Phase2A were higher on the tiny web set (42.5%), but the 40-image stress test is exploratory and does not outweigh the larger structured external evidence.

This selection is retrospective project synthesis. It was **not** used to alter or retune any model after external testing.

---

## 17. PPT roadmap closure

The Review-I PPT prescribed:
- Phase1 baseline system;
- Phase2A fixed 30 epochs;
- Phase2B hyperparameter analysis;
- Phase2C external validation;
- Phase2D data integrity/robustness;
- Phase3 comparative analysis and final model selection;
- Phase4 Grad-CAM + occlusion;
- Phase5 prototype application;
- Phase6 final validation/report/demonstration.

All prescribed experimental phases are complete. The web stress test is additional work beyond the original PPT plan.

---

## 18. What failed, what was rejected, and what was deliberately not done

Failed:
- TensorFlow GPU route due PTX/sm_120 compatibility.
- Fully unfrozen VGG16 training due collapse/constant learned features.

Rejected/avoided:
- benign/malignant relabeling from tumor class;
- interpreting the five backbones as an ensemble;
- claiming LSTM models temporal MRI information;
- calling Grad-CAM segmentation or explanation of the external SVM;
- treating SVM relative scores as calibrated probabilities;
- calling material_clean a newly independent test set;
- tuning models using BDNeuro or web outcomes;
- replacing web images after model predictions;
- adding DenseNet121 without a verified experiment bundle;
- claiming statistically significant hyperparameter superiority from one seed.

Not reproduced from the base paper:
- four-optimizer comparison;
- full 20-combination experiment grid;
- batch size 32 as the main project training batch.

---

## 19. Limitations that remain

- Original train/validation/test split has material cross-split duplication.
- material_clean is post-hoc and does not repair contaminated training.
- reliable patient IDs are unavailable, so patient-level independence cannot be proven.
- Phase2A/2B use one seed.
- VGG backbones are frozen while the residual/EfficientNet family was fine-tuned.
- only LR, LSTM width and dropout were varied one factor at a time.
- XAI evaluation is small and qualitative.
- BDNeuro exact-clean image-level independence from Nickparvar is improved by exact-overlap removal, but patient-level independence still cannot be established from public metadata.
- the web stress test is only 40 source-verified convenience images and is not a population-level clinical evaluation.
- no clinical deployment/safety claim is supported.

Optional stronger future work:
- construct exact/near-duplicate groups across Nickparvar;
- split groups instead of images;
- retrain all five models from scratch;
- refit all scaler/PCA/SVM objects on the new training partition;
- repeat internal evaluation;
- still avoid claiming patient-level independence without IDs.

---

## 20. Current project end-state

The experimental project is complete.

Completed:
- baseline architecture;
- five-backbone training;
- neural-head evaluation;
- downstream PCA+SVM evaluation;
- leakage audit;
- material-clean re-evaluation;
- Phase2A;
- Phase2B;
- all 80 Phase2 retained-state full-hybrid evaluations (10 Phase2A + 70 Phase2B);
- GUI;
- Grad-CAM;
- final-SVM occlusion;
- 15-pipeline pre-external freeze;
- BDNeuro overlap audit;
- BDNeuro exact-clean primary external evaluation;
- BDNeuro balanced sensitivity evaluation;
- 40-image web source collection;
- web overlap audit and pHash visual review;
- web cohort lock;
- 600 web predictions;
- final candidate selection;
- final consolidated reports and Claude transfer package.

Use this file as the highest-level authoritative project context.
