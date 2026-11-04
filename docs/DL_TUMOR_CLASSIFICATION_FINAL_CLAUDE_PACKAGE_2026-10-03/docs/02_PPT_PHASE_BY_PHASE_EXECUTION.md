# PPT ROADMAP — PHASE-BY-PHASE EXECUTION AND CLOSURE

This document maps the **actual completed project** to the phases prescribed in `Brain_Tumor_Classification_Review_I_FINAL [Autosaved].pptx`.

The PPT was a Review-I roadmap. Its “upcoming” labels are historical and must not be treated as current status.

---

## Phase 1 — Base-Paper Implementation — COMPLETE

### PPT intention
Build:
`MRI → preprocessing → CNN → spatial sequence → LSTM → Dense128 → StandardScaler → PCA64 → linear SVM → initial evaluation`

### What was actually done
- acquired and inspected the 7,023-image Nickparvar dataset;
- created one seed-42 stratified 4,569/1,143 train/validation split from the 5,712-image training folder;
- retained the supplied 1,311-image testing folder;
- implemented deterministic evaluation preprocessing and training-only augmentation;
- incrementally verified CNN output, sequence reshape, LSTM, post-LSTM layers and full training step;
- generalized the ResNet152 prototype into a reusable `CNNLSTM` model supporting five backbones;
- stabilized VGG16/VGG19 by freezing pretrained CNN convolutional blocks;
- trained all five backbones with the project RMSprop configuration;
- extracted 128-D deep features;
- fit StandardScaler on training features only;
- fit PCA64 on standardized training features only;
- trained linear SVM C=1;
- evaluated neural head and downstream hybrid classifier.

### Why
Phase 1 had to establish a faithful/adapted baseline before adding robustness or explainability extensions.

### Findings
| Backbone | Best epoch | Neural test acc | Neural macro F1 | SVM val acc | PCA+SVM test acc | PCA+SVM macro F1 | Test acc change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vgg16 | 16 | 93.67% | 93.32% | 94.93% | 94.97% | 94.68% | +1.30 pp |
| vgg19 | 18 | 93.14% | 92.72% | 94.49% | 93.52% | 93.10% | +0.38 pp |
| resnet50 | 7 | 97.33% | 97.19% | 97.55% | 98.09% | 98.01% | +0.76 pp |
| resnet152 | 8 | 98.17% | 98.06% | 98.25% | 98.40% | 98.32% | +0.23 pp |
| efficientnet_b3 | 6 | 98.02% | 97.94% | 98.51% | 98.47% | 98.41% | +0.45 pp |

### Status
**COMPLETE.**

---

## Phase 2A — Fixed 30-Epoch Training Protocol — COMPLETE

### PPT intention
Use a common 30-epoch horizon so early stopping does not hide convergence, late-epoch deterioration or overfitting. Compare best-within-30 and epoch-30 states.

### What was done
- trained all five baseline configurations for exactly 30 epochs;
- no early termination;
- saved `best_val_within_30.pth`;
- saved `epoch_30_model.pth`;
- evaluated both retained states through the full StandardScaler→PCA→SVM path;
- produced 10 full-hybrid retained-state evaluations.

### Why
A best-checkpoint-only summary can hide whether a model deteriorates after its peak.

### Findings
| Backbone | State | Epoch | SVM val acc | Macro F1 | Internal test acc* |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | best_val_within_30 | 27 | 99.300% | 99.270% | 99.314% |
| resnet152 | best_val_within_30 | 10 | 98.600% | 98.545% | 98.703% |
| resnet50 | epoch_30 | 30 | 98.688% | 98.644% | 98.932% |
| vgg16 | epoch_30 | 30 | 95.888% | 95.746% | 96.186% |
| vgg19 | epoch_30 | 30 | 95.626% | 95.462% | 95.271% |

Important training-history observations:
- VGG16/VGG19 kept improving late into training;
- ResNet152 peaked early and stayed close;
- EfficientNet-B3 peaked near the end;
- fixed-horizon trajectories exposed instability hidden by early stopping.

### Status
**COMPLETE.**

---

## Phase 2B — Hyperparameter Analysis — COMPLETE

### PPT intention
Controlled one-factor sensitivity for learning rate, LSTM width and dropout.

### What was done
Seven changed configurations for each backbone:
- LR1e-5
- LR5e-5
- LR5e-4
- LSTM128
- LSTM512
- dropout0.2
- dropout0.5

Total neural runs: **35/35**.  
Both retained checkpoint states were then evaluated downstream: **70/70** PCA+SVM validations.

### Why
The project wanted interpretable sensitivity evidence, not a black-box hyperparameter search.

### Findings
- mean learning-rate sensitivity range: **4.62 pp**
- LSTM-width range: **0.58 pp**
- dropout range: **0.42 pp**
- learning rate was architecture-dependent and by far the strongest sensitivity;
- LSTM512 raised peak neural validation slightly for every backbone but not always stably;
- dropout effects were smaller and inconsistent;
- ResNet50 LR5e-4 showed the strongest late-epoch deterioration.

Best changed-config full-hybrid representatives:

| Backbone | Selected changed config | State | Epoch | SVM val acc | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% |
| resnet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% |
| resnet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% |
| vgg16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% |
| vgg19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% |

### Status
**COMPLETE.**

---

## Phase 2C — External Validation — COMPLETE

### PPT intention
Use an independent external brain-MRI dataset, document provenance/class mapping, audit overlap, freeze CNN-LSTM/scaler/PCA/SVM, do not tune externally, and report accuracy, macro metrics, per-class metrics, confusion matrix and internal→external shift.

### What was done
- selected BDNeuro-MRI V8;
- inventoried 5,941 class MRIs;
- audited against all 7,023 development images;
- discovered **3,297 unique exact decoded-pixel overlaps**;
- excluded only objectively confirmed exact-pixel overlaps;
- retained 2,644 exact-overlap-clean images;
- built a deterministic 1,648-image balanced sensitivity subset;
- froze 15 pipelines before external evaluation;
- verified artifact hashes before inference;
- ran all 15 pipelines on the full and balanced cohorts;
- did not retrain/refit/recalibrate on BDNeuro.

### Why
The original split had leakage, and strong internal performance alone could not establish generalization.

### Primary findings
ResNet152 Phase2A:
- 88.31% full-cohort accuracy, 87.85% macro F1;
- 90.47% balanced accuracy, 90.50% macro F1;
- smallest internal→BDNeuro accuracy gap among frozen candidates.

Phase2A generalized better on average than Phase2B.

### Additional work beyond PPT
The project added a 40-image source-verified web stress test after BDNeuro to examine much stronger domain shift. Accuracy fell to 25–42.5%, with strong `notumor` bias.

### Status
**COMPLETE, exceeded original plan.**

---

## Phase 2D — Data Integrity / Robustness — COMPLETE

### PPT intention
Check exact duplicates, perceptual near duplicates and cross-split leakage, then ensure performance is not merely duplicate-image overlap.

### What was done
Original Nickparvar audit:
- 7,023 images;
- 464 exact/pixel cross-split duplicate pairs;
- 98 high-confidence near-duplicate pairs;
- 562 material pairs;
- 481 further review candidates;
- 250 official-test images implicated.

A material_clean 1,061-image post-hoc test was evaluated.

BDNeuro and web datasets were separately audited before external inference.

### Why
This was the key threat to the credibility of very high internal results.

### Findings
The original split is not image-level independent. Patient-level independence cannot be established. External validation therefore became essential.

### Status
**COMPLETE.** Full duplicate-group-aware retraining is retained as optional stronger future remediation, not an unfinished audit.

---

## Phase 3 — Comparative Analysis & Final Candidate Selection — COMPLETE

### PPT intention
Combine Phase1/2 evidence across all backbones, compare stability, sensitivity and external generalization, and select a final candidate.

### What was done
- compared all five backbones across Phase1, Phase2A and Phase2B;
- froze 15 representative pipelines;
- compared all 15 on two BDNeuro evaluations;
- compared all 15 on the web stress cohort;
- analyzed generalization gaps and stage-level trends.

### Final selected candidate
**ResNet152 Phase2A standardized 30-epoch pipeline, best state at epoch10.**

Why:
- 98.60% internal full-hybrid validation accuracy;
- strongest BDNeuro full result: 88.31%;
- strongest BDNeuro balanced result: 90.47%;
- smallest structured-external generalization gaps;
- consistent overall evidence.

The web stress result was only 37.5%, showing the limitations of strong domain shift; this does not change the final representative because the web test was a tiny exploratory n=40 stress test.

### Status
**COMPLETE.**

---

## Phase 4 — Explainability Analysis — COMPLETE

### PPT intention
Layer3/Layer4/fused Grad-CAM, final-SVM occlusion, error/disagreement analysis, coarse localization assessment.

### What was done
- formal fixed 20-image ResNet152 Grad-CAM study;
- Layer3, Layer4 and 50:50 fusion;
- final-SVM occlusion on 11 fixed cases;
- rating and comparison protocol;
- score-drop and class-flip analysis.

### Findings
Classification correctness did not guarantee anatomically plausible saliency. Off-lesion skull/border/extracranial activation was common. Neither method is segmentation or proof of clinically valid reasoning.

### Status
**COMPLETE.**

---

## Phase 5 — Prototype Application — COMPLETE

### PPT intention
Upload MRI, choose backbone, classify, show scores/performance/model comparison/explainability.

### What was done
Streamlit GUI:
- five-backbone selector;
- upload PNG/JPEG/BMP/TIFF;
- final SVM class;
- neural-head comparison;
- relative SVM scores and neural softmax;
- metrics/confusion matrices/training curves;
- cross-backbone comparison;
- Grad-CAM;
- SVM occlusion;
- artifact readiness checks.

### Status
**COMPLETE.**

---

## Phase 6 — Final Validation, Report & Demonstration — COMPLETE

### What was done
Final validation now includes:
- historical internal official test;
- material_clean post-hoc test;
- BDNeuro exact-overlap-clean full external cohort;
- BDNeuro balanced sensitivity subset;
- 40-image web stress test.

Documentation:
- comprehensive Phase2/external report;
- dedicated web-stress report;
- this final Claude transfer package.

Demonstration:
- GUI implements the full research prototype.

### Status
**COMPLETE for experimental/project work.** Remaining actions are only user-side submission/presentation logistics.

---

## Bottom line

Every experimental or implementation item prescribed by the PPT has been completed. The web stress test and the extensive cross-dataset overlap audit go beyond the original roadmap.
