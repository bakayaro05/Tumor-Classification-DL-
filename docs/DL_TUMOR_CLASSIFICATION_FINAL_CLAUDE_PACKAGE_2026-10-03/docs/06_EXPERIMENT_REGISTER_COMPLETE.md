# COMPLETE EXPERIMENT REGISTER

This register covers successful, failed, partial, superseded and proposed experiments.

## E0 — Dataset readability and mode audit
**Objective:** verify that the public MRI corpus could be loaded consistently.  
**Result:** all 7,023 images readable; heterogeneous modes and dimensions detected.  
**Decision influenced:** convert to RGB and use standardized resizing/normalization.

## E1 — CNN tensor-shape verification
**Scripts:** `test_resnet152_features.py`, `test_real_mri_batch.py`.  
**Objective:** establish actual pretrained CNN output shape.  
**Result:** ResNet152 `[B,3,224,224] → [B,2048,7,7]`.

## E2 — LSTM-section verification
**Script:** `test_lstm_section.py`.  
**Result:** `[B,2048,7,7] → [B,49,2048] → LSTM256 → [B,49,256]`.

## E3 — Post-LSTM verification
**Script:** `test_post_lstm_layers.py`.  
**Result:** last LSTM step→BN→Dropout0.3→Dense128→ReLU.

## E4 — Full model forward pass
**Script:** `test_complete_model.py`.  
**Objective:** verify full neural path.

## E5 — Training-step verification
**Script:** `test_training_step.py`.  
**Objective:** verify forward/loss/backward/update without running full training.

## E6 — Early ResNet152 baseline
**Scripts:** `model_resnet152_lstm.py`, `train_resnet152_lstm.py`, `extract_all_features.py`, `prepare_pca_features.py`, `train_svm.py`, `evaluate_final_test.py`.  
**Result:** first working full hybrid; early test around 97.64% accuracy / 97.50% macro F1.  
**Status:** completed but superseded by standardized five-backbone bundles.

## E7 — Generic five-backbone architecture
**Scripts:** `model_cnn_lstm.py`, `test_all_backbones.py`.  
**Result:** common architecture for VGG16/VGG19/ResNet50/ResNet152/EfficientNet-B3.

## F1 — Fully unfrozen VGG16
**Objective:** fine-tune VGG like the residual models.  
**Failure:** ~28% validation, nearly constant learned features, PCA NaNs.  
**Diagnosis:** pretrained CNN activation itself was healthy; failure emerged in the trained end-to-end configuration.  
**Fix:** freeze VGG CNN layers.

## E8 — Standardized Phase1 five-backbone training/evaluation
**Scripts:** `train_backbone.py`, `evaluate_backbone.py`.  
**Result:** complete experiment bundles for all five.  
**Historical internal results:**

| Backbone | Best epoch | Neural test acc | Neural macro F1 | SVM val acc | PCA+SVM test acc | PCA+SVM macro F1 | Test acc change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vgg16 | 16 | 93.67% | 93.32% | 94.93% | 94.97% | 94.68% | +1.30 pp |
| vgg19 | 18 | 93.14% | 92.72% | 94.49% | 93.52% | 93.10% | +0.38 pp |
| resnet50 | 7 | 97.33% | 97.19% | 97.55% | 98.09% | 98.01% | +0.76 pp |
| resnet152 | 8 | 98.17% | 98.06% | 98.25% | 98.40% | 98.32% | +0.23 pp |
| efficientnet_b3 | 6 | 98.02% | 97.94% | 98.51% | 98.47% | 98.41% | +0.45 pp |

**Current interpretation:** contaminated-split internal evidence.

## E9 — Neural-head deterministic evaluation
**Objective:** quantify performance before scaler/PCA/SVM.  
**Result:** all five best checkpoints evaluated with deterministic preprocessing.  
**Consequence:** allowed proper neural-head vs PCA+SVM downstream-classifier comparison.

## E10 — Streamlit GUI
**Scripts:** `app.py`, project `gui/*`, `verify_gui_setup.py`.  
**Result:** five-backbone research prototype implemented.

## E11 — Grad-CAM formal evaluation
**Script:** `gradcam_evaluation_app.py`.  
**Protocol:** ResNet152, seed42, 20 images, 5/class, Layer3/Layer4/fusion.  
**Result:** localization reliability limited; frequent off-lesion hotspots.

## E12 — Final-SVM occlusion
**Script:** `occlusion_evaluation_app.py`.  
**Protocol:** 11 cases; patch32; stride16; 169 perturbations/image.  
**Result:** 58/1,859 flips; qualitative localization mixed.

## E13 — Cross-split duplicate/leakage audit
**Result:** MATERIAL_DUPLICATION.  
- exact/pixel pairs: 464
- high-confidence near pairs: 98
- material pairs: 562
- review candidates: 481
- implicated test images: 250

**Consequence:** historical internal scores no longer treated as clean generalization.

## E14 — material_clean post-hoc test
**Cohort:** 1,061 of 1,311 official test images.  
**Result:**

| Backbone | material_clean acc | material_clean macro F1 | Original test acc |
| --- | --- | --- | --- |
| vgg16 | 94.816% | 94.302% | 94.966% |
| vgg19 | 93.025% | 92.415% | 93.516% |
| resnet50 | 98.303% | 98.134% | 98.093% |
| resnet152 | 98.398% | 98.315% | 98.398% |
| efficientnet_b3 | 98.680% | 98.511% | 98.474% |

**Interpretation:** high performance persists after directly implicated test images are removed, but training contamination remains unresolved.

## E15 — Phase2A five fixed 30-epoch neural runs
**Script:** `train_backbone_30epoch.py`.  
**Count:** 5/5.  
**Output:** training histories plus best-within30 and epoch30 checkpoints.

## E16 — Phase2A retained-state full-hybrid evaluation
**Script:** `evaluate_backbone_30epoch.py`.  
**Count:** 10/10 states.  
**Selection:** by internal PCA+SVM validation.

| Backbone | State | Epoch | SVM val acc | Macro F1 | Internal test acc* |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | best_val_within_30 | 27 | 99.300% | 99.270% | 99.314% |
| resnet152 | best_val_within_30 | 10 | 98.600% | 98.545% | 98.703% |
| resnet50 | epoch_30 | 30 | 98.688% | 98.644% | 98.932% |
| vgg16 | epoch_30 | 30 | 95.888% | 95.746% | 96.186% |
| vgg19 | epoch_30 | 30 | 95.626% | 95.462% | 95.271% |

## E17 — Phase2B 35-run neural hyperparameter study
**Runner:** `run_all_dl_hyperparams.py`; actual project worker `train_backbone_hyperparams.py` (worker raw snapshot is not bundled here).  
**Count:** 35/35.

## E18 — Phase2B 70 retained-state downstream evaluations
**Actual project worker:** `evaluate_hyperparam_pca_svm.py` (not physically available in this transfer).  
**Count:** 70/70.  
**Key selected changed-config states:**

| Backbone | Selected changed config | State | Epoch | SVM val acc | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% |
| resnet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% |
| resnet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% |
| vgg16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% |
| vgg19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% |

## E19 — 40-run Phase2 sensitivity synthesis
**Input:** 5 Phase2A baseline training JSONs + 35 Phase2B training JSONs.  
**Finding:** learning rate dominates tested sensitivity; LSTM width modest; dropout smaller.

## E20 — Freeze 15 external candidates
**Script:** `compare_external_15.py`.  
**Count:** 5 Phase1 + 5 Phase2A + 5 Phase2B.  
**Safeguard:** SHA256 artifact manifest; selection from internal validation only.

## E21 — BDNeuro inventory
**Raw class MRIs:** 5,941.  
**Purpose:** external four-class source.

## E22 — BDNeuro cross-dataset overlap audit
**Script:** `audit_bdneuro_overlap.py`.  
**Finding:** 3,297 unique exact decoded-pixel overlaps with the development dataset.

## E23 — BDNeuro exact-clean cohort build
**Script:** `build_bdneuro_exact_clean_cohort.py`.  
**Result:** n=2,644.

## E24 — BDNeuro balanced sensitivity subset
**Script:** `build_bdneuro_balanced_subset.py`.  
**Result:** 412/class, n=1,648.

## E25 — BDNeuro full 15-pipeline evaluation
**Script:** repaired `compare_external_15.py` with `--results-name`.  
**Result range:** 75.0–88.31% accuracy.  
**Strongest:** ResNet152 Phase2A 88.31%.

## E26 — BDNeuro balanced 15-pipeline evaluation
**Result range:** 77.0–90.47% accuracy.  
**Strongest:** ResNet152 Phase2A 90.47%.

## E27 — Web stress-test source collection
**Target:** 40 source-verified MRIs, 10/class.  
**Support files:** source-list workbook + manifest.

## E28 — Web pre-inference audit
**Script:** `audit_web_stress_40.py`.  
**Initial notable candidates:** NOR_01 pHash0, MEN_02 pHash6.  
**Visual decision:** NOR_01 same underlying Nickparvar slice → replace; MEN_02 different → keep.

## E29 — Final web audit
**Result:** 0 exact file/pixel overlaps; no pHash<=4 candidates; one reviewed distance6 candidate.

## E30 — Web 15-pipeline stress evaluation
**Count:** 600 predictions.  
**Accuracy range:** 25–42.5%.  
**Finding:** strong notumor prediction bias, poor glioma/pituitary recall.

## E31 — Final overall candidate synthesis
**Selected:** ResNet152 Phase2A epoch10.  
**Reason:** strongest overall structured-external evidence, while maintaining high internal validation.

---

# Experiments discussed but deliberately not completed

## P1 — Full base-paper optimizer grid
Adam/SGD/AdaGrad/RMSprop × five backbones was not executed. RMSprop only.

## P2 — DenseNet121
A past conversational statement suggested DenseNet121, but no verified standardized artifact bundle supports it. It is not part of the final project.

## P3 — Full duplicate-group-aware retraining
Scientifically valuable future remediation but not required for the completed mini-project.

## P4 — More Grad-CAM layers / more occlusion variants
Not pursued after the XAI protocol was frozen; additional visualization tuning risked turning qualitative exploration into post-hoc cherry-picking.

## P5 — Per-image consensus analysis over the 600 web predictions
Discussed after the web test, judged optional and intentionally skipped because it would not change the main scientific conclusion.

## P6 — Clinical deployment / DICOM pipeline
Outside project scope.

---

# Raw evidence locations in this package

- Phase2 full JSON archive: `raw_packages/phase2_full_analysis_package.zip`
- Phase2 training 40-run table: `results/phase2_training_40run_summary.csv`
- all 80 Phase2 downstream state rows: `results/phase2_full_hybrid_80_state_evaluations.csv`
- BDNeuro full/balanced 15-model tables: `results/bdneuro_*_external_comparison_15.csv`
- web full master JSON: `results/web_stress_external_comparison_15.json`
