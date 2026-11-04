# FULL PROJECT TIMELINE — START TO FINAL LOCK

This timeline preserves both successful work and superseded decisions.

## 16 July 2026 — classical-image-processing project conception
- Considered classical brain-tumor segmentation/classification papers.
- Proposed median filtering, skull/background removal, K-means, morphology, handcrafted shape/GLCM features, scaling/PCA and RBF-SVM.
- Corrected two conceptual risks: arbitrary K-means labels and unsupported benign/malignant relabeling.
- This direction was later superseded.

## 17 August 2026 — recent base-paper requirement
- Goal changed to a recent deep-learning paper implementation.
- Selected the 2026 Springer hybrid CNN+LSTM+SVM paper.
- Five alternative backbones fixed: VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3.
- Four-class task fixed: glioma, meningioma, notumor, pituitary.
- Novelty/Grad-CAM deferred until the base pipeline worked.

## 28 August 2026 — environment setup and TensorFlow failure
- WSL2/Ubuntu, RTX 5070-class GPU.
- TensorFlow detected GPU but real workloads failed with unsupported PTX/sm_120 behavior.
- TensorFlow-nightly path was attempted.
- Final decision: abandon TensorFlow; use PyTorch 2.11.0+cu128.

## Late August 2026 — dataset and deterministic split
- Nickparvar Kaggle dataset: 5,712 training-folder images + 1,311 supplied testing-folder images.
- Saved seed42 stratified split of supplied Training: 4,569 train / 1,143 validation.
- Official 1,311 Testing folder retained.
- RGB conversion, 224×224 input, ImageNet normalization.
- Training-only augmentation; deterministic evaluation transform.

## Late August to early September — architecture built in small verified steps
Dedicated scripts tested:
- ResNet152 feature map;
- spatial reshape;
- LSTM section;
- post-LSTM BN/Dropout/Dense;
- complete forward path;
- real MRI batch;
- backward training step.

Key interpretation fixed: LSTM sees flattened spatial positions of one image, not time or ordered slices.

## Early September — first complete ResNet152 baseline
- Trained ResNet152+LSTM with RMSprop LR1e-4.
- Best early checkpoint at epoch8 in the legacy root-level pipeline.
- Extracted 128-D features.
- StandardScaler→PCA64→linear SVM.
- Early SVM test result around 97.64% accuracy / 97.50% macro F1.
- Proved end-to-end feasibility.
- Later standardized experiment bundle superseded these root-level artifacts.

## 6–12 September — generic five-backbone framework
- Replaced ResNet-specific model with generic `CNNLSTM`.
- Standard experiment bundles under `experiments/<backbone>/`.
- All five backbones trained/evaluated.

### VGG16 failure
- Fully unfrozen VGG16 collapsed near 28% validation.
- Downstream features nearly constant; PCA NaNs.
- Diagnostic script showed pretrained CNN activations themselves were healthy.
- Decision: freeze VGG16 CNN.
- VGG19 assigned same frozen transfer-learning policy.
- ResNet/EfficientNet remained fine-tuned.

## Phase1 standardized comparison
Historical PCA+SVM test:
- VGG16 94.97%
- VGG19 93.52%
- ResNet50 98.09%
- ResNet152 98.40%
- EfficientNet-B3 98.47%

Neural-head evaluation added for all five, enabling downstream-classifier comparison.

## 12 September onward — GUI
- Streamlit prototype added.
- Prediction path uses saved standardized experiment artifacts.
- Final SVM and neural-head outputs separated.
- Relative SVM scores explicitly not called probabilities.
- Grad-CAM integrated as neural-head explanation.
- Prototype labelled research/educational only.

## September — Grad-CAM and SVM occlusion
- Formal Grad-CAM protocol frozen rather than continually changing visualizations.
- 20-image ResNet152 sample, seed42, 5/class.
- Layer3/Layer4/fusion visually rated.
- Found frequent skull/border/extracranial saliency.
- Added final-SVM occlusion to explain the deployed decision path.
- 11-case occlusion comparison completed.
- Conclusion: explanation reliability limited; classification correctness does not validate localization.

## 20 September — pre-lock hostile review
- Project appeared nearly ready to lock.
- Only major scientific verification still missing: cross-split duplicate/leakage audit.
- Audit was run before final lock.

## 20 September — material cross-split duplication discovered
Audit:
- 464 exact/pixel duplicate pairs;
- 98 additional high-confidence near-duplicate pairs;
- 562 material pairs;
- 481 review candidates;
- 250 official-test images implicated.

Immediate consequence:
- old interpretation that internal scores could be treated as clean generalization was withdrawn.

### Evolution of remediation decision
The first post-audit transfer package treated full group-aware resplitting/retraining as blocking. Later project review refined the practical plan:
- preserve old internal metrics as historical contaminated-split outputs;
- perform 1,061-image material_clean post-hoc test;
- continue controlled Phase2 studies with explicit caveat;
- perform genuinely external validation with overlap audit;
- leave group-aware retraining as stronger optional future remediation.

This later decision is authoritative.

## Phase2A — standardized fixed 30 epochs
- five backbones;
- 30 epochs each;
- baseline LR1e-4/LSTM256/dropout0.3;
- saved best-within30 and epoch30;
- evaluated both states downstream;
- 10/10 full-hybrid retained-state evaluations complete.

## Phase2B — sensitivity study
Seven variants × five backbones:
- 35/35 training runs.
Both retained states downstream:
- 70/70 PCA+SVM validations.

Findings:
- learning rate was the strongest sensitivity;
- LSTM512 had small consistent peak-neural gains;
- dropout effects smaller;
- best neural checkpoint was not always best downstream SVM state.

## 2 October — Phase2 analysis consolidation
A 40-run training analysis package and comprehensive report were built. Earlier claims that Phase2B was only 1/35 were superseded by machine-readable evidence showing all 35 complete.

## External-validation planning
Decision:
- do architecture-wise comparison rather than expose only one model;
- freeze 5 Phase1 + 5 Phase2A + 5 Phase2B pipelines = 15;
- selection based only on internal validation before external inference;
- hash frozen checkpoint/scaler/PCA/SVM/result artifacts.

## BDNeuro V8 selection
Raw class MRI count: 5,941.
Cross-dataset audit against Nickparvar discovered:
- 3,297 unique exact decoded-pixel overlaps.

Important reasoning:
- different stated provenance cannot override direct pixel identity;
- no speculation about which public dataset lineage caused reuse;
- pHash similarity alone is not enough to discard non-identical MRIs.

Final external cleaning rule:
- remove exact decoded-pixel overlaps only.

## BDNeuro exact-clean cohort
- 2,644 images retained.
- primary natural distribution.
- balanced sensitivity subset: 1,648 = 412/class.

## 3 October — 15-model BDNeuro evaluation completed
All 15 frozen pipelines evaluated on:
- 2,644-image full exact-clean cohort;
- 1,648-image balanced subset.

Strongest structured external evidence:
- ResNet152 Phase2A 88.31% full;
- ResNet152 Phase2A 90.47% balanced.

Key result:
Phase2B tuning did not consistently transfer. Phase2A generalized better on average.

## 3 October — 40-image web stress test
Motivation: answer the practical “random outside MRI” question.

Collection:
- 10 glioma
- 10 meningioma
- 10 notumor
- 10 pituitary
- source-verified medical/publication images.

Pre-inference audit:
- initial exact overlap 0;
- pHash0 candidate NOR_01 visually proved to be the same underlying Nickparvar slice despite altered pixels;
- NOR_01 replaced before inference;
- MEN_02 pHash6 visually confirmed different and retained;
- final rerun: 0 exact overlaps, pHash<=4 = 0.

Cohort locked.

## Web 15-model evaluation
600 predictions.
Accuracy 25–42.5%.
Strong `notumor` bias.
Mean recall over the 15 pipelines:
- glioma 12.7%
- meningioma 30.7%
- notumor 90.7%
- pituitary 7.3%

Key interpretation:
structured external generalization and unrestricted web-domain robustness are very different questions.

## Final Phase3 candidate selection
Final project representative:
**ResNet152 Phase2A, LR1e-4/LSTM256/dropout0.3, best state epoch10.**

Reason:
strongest overall structured external performance and smallest BDNeuro generalization gaps, while keeping internal performance high.

## Final project closure
All PPT experimental phases complete.
Remaining work is only submission/presentation logistics. Duplicate-group-aware retraining remains optional future scientific strengthening.
