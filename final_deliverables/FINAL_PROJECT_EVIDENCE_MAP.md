# FINAL PROJECT EVIDENCE MAP

Each major claim in FINAL_PROJECT_REPORT.pdf and DL_TUMOR_CLASSIFICATION_FINAL_REVIEW.pptx with its repository source.

Status: **VERIFIED** = recomputed/inspected from current repository files during preparation; **HISTORICAL** = from project records/transfer documents without a preserved machine-readable output; **INTERPRETIVE** = reasoning drawn from verified results.

Paths are relative to `~/brain_tumor_project`; `<bb>` = backbone directory name.

| # | Claim | Repository source | JSON/CSV field or source lines | Status |
|---|---|---|---|---|
| 1 | Dataset: 7,023 images; Training 5,712 / Testing 1,311; per-class counts | `phase2/external_validation/bdneuro_overlap_audit/dataset_counts.json` | internal_total; internal_split_class_counts | VERIFIED |
| 2 | Split 4,569 / 1,143 / 1,311 (seed 42, stratified) | `experiments/<bb>/results/final_results.json; data/splits/*.npy` | deep_features.train_samples/val_samples/test_samples | VERIFIED |
| 3 | Validation notumor = 319/1,143 (27.9%) — explains ~28% VGG16 collapse value | `experiments/resnet152/pca/val_labels.npy` | bincount | VERIFIED (collapse value itself HISTORICAL) |
| 4 | Augmentation: H/V flip p=0.5; ColorJitter(0.1, 0.2, 0.2, 0.1); training only | `src/data_pipeline_torch.py` | lines 65–78 | VERIFIED |
| 5 | Five supported backbones; channels 512/2048/1536; ImageNet weights V1/V2 | `model_cnn_lstm.py` | SUPPORTED_BACKBONES; output_channels; *_Weights | VERIFIED |
| 6 | VGG backbones frozen (requires_grad=False) | `actual_training_scripts/train_backbone_30epoch.py` | lines 360–372 | VERIFIED |
| 7 | RMSprop optimizer | `actual_training_scripts/train_backbone*.py` | torch.optim.RMSprop (train_backbone.py:399, _30epoch.py:408, _hyperparams.py:429) | VERIFIED |
| 8 | Phase 1 early stopping patience 3, default max 10 epochs | `actual_training_scripts/train_backbone.py` | argparse --epochs default 10, --patience default 3 | VERIFIED |
| 9 | Phase 1 epochs run/max and best epochs | `experiments/<bb>/results/training_results.json` | epochs_completed, epochs_requested, best_epoch | VERIFIED |
| 10 | Phase 1 neural-head metrics | `experiments/<bb>/results/neural_head_results.json` | validation.accuracy, test.accuracy, test.macro_f1 | VERIFIED |
| 11 | Phase 1 PCA+SVM metrics and deltas | `experiments/<bb>/results/final_results.json; results/neural_vs_pca_svm_comparison.csv` | validation.*, test.*; *_delta_pp | VERIFIED |
| 12 | PCA explained variance 99.18–99.70% | `final_results.json (Phase 1 and Phase 2A)` | pca.explained_variance | VERIFIED |
| 13 | SVM probability=False (standardized); True only for legacy prototype | `experiments/<bb>/models/svm.joblib; models/resnet152_lstm_pca_svm.joblib` | SVC.probability | VERIFIED |
| 14 | Duplicate audit status and counts (464/98/562/481) | `final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json` | status; cross_split_exact_or_pixel_duplicate_pairs; high_confidence_near_duplicate_pairs; material_pairs; near_duplicate_review_pairs | VERIFIED |
| 15 | Unique implicated images train 354 / val 122 / test 250; test by class 0/109/136/5; 0 label mismatches | `duplicate_leakage_findings.csv; material_clean_excluded_test_images.csv` | material=True rows; test_path class folder | VERIFIED |
| 16 | Leakage policies exact 1,095 / material 1,061 / conservative 926 and accuracies | `final_project_docs/leakage_corrected_evaluation/leakage_corrected_summary.csv` | policy, test_n, excluded_n, accuracy_pct, macro_f1_pct | VERIFIED |
| 17 | material_clean does not repair training contamination | `—` | methodological interpretation | INTERPRETIVE |
| 18 | FINAL_QC_BLOCKER 'NOT locked' is a historical state | `final_project_docs/FINAL_QC_BLOCKER.md` | text | HISTORICAL |
| 19 | Phase 2A 30/30 epochs; best epochs 30/30/30/10/27 | `phase2/experiments_30epoch/<bb>/results/training_results.json` | epochs_completed; best_epoch | VERIFIED |
| 20 | Phase 2A SVM val / test per state | `phase2/experiments_30epoch/<bb>/evaluations/<state>/results/final_results.json` | validation.accuracy, validation.macro_f1, test.accuracy | VERIFIED |
| 21 | VGG16/VGG19/ResNet50 representative = best_val_within_30 @ epoch 30; checkpoint files not byte-identical | `frozen_15_selection.csv; models/*.pth` | checkpoint_state, selected_epoch; SHA-256 of both .pth | VERIFIED |
| 22 | Phase 2A material_clean (ResNet152 ep10: 98.869% / F1 98.788%) | `phase2/results/leakage_corrected_30epoch/PHASE2A_PRIMARY_material_clean_comparison.csv` | accuracy_pct, macro_f1_pct | VERIFIED |
| 23 | Phase 2B 35 runs × 30 epochs; 70 downstream evals; no test metrics | `phase2/hyperparameter_dl/<bb>/<run>/…` | epochs_completed; evaluation files; absence of test keys | VERIFIED |
| 24 | Phase 2B sensitivity ranges 4.62 / 0.58 / 0.42 pp | `phase2/**/training_results.json` | best_val_accuracy (range per factor family incl. 2A baseline) | VERIFIED (derived) |
| 25 | Best-state vs epoch30 downstream 20 / 8 / 7 | `phase2/hyperparameter_dl/**/final_results.json` | validation.accuracy per state | VERIFIED (derived) |
| 26 | Phase 2B representatives per backbone | `frozen_15_selection.csv; frozen_15_manifest.json` | stage=phase2b_tuned rows | VERIFIED |
| 27 | VGG16 lr_1e-5 run skipped by queue runner (pre-existing) | `transfer package docs/19` | text | HISTORICAL |
| 28 | 15 pipelines frozen; selection rule; 75 SHA-256 hashes match current artifacts | `phase2/external_validation/frozen_15_manifest.json` | selection_details.selection_rule; candidates[].paths/sha256 | VERIFIED |
| 29 | Freeze precedes external results (2026-10-02 16:50 vs 2026-10-03 07:54 UTC) | `file modification times` | mtime | VERIFIED (filesystem metadata) |
| 30 | BDNeuro raw 5,941; class/split counts | `bdneuro_overlap_audit/dataset_counts.json` | bdneuro_total; bdneuro_split_class_counts | VERIFIED |
| 31 | BDNeuro audit 3,421 / 3,519 pairs; 3,295 / 3,297 unique; pHash 9,981 / 6,664; within 0/0/1,588/65 | `bdneuro_overlap_audit/summary.json; cross_exact_*_duplicates.csv` | cross_dataset.*, within_bdneuro.*; unique bdneuro_path | VERIFIED |
| 32 | Exact-clean cohort 2,644 (1,159/472/412/601; splits 1,845/364/435) | `bdneuro_exact_clean_cohort/cohort_summary.json` | exact_pixel_clean_images_kept; kept_by_class; kept_by_original_bdneuro_split | VERIFIED |
| 33 | Balanced subset 1,648 = 412/class; SHA-256 ranking; no model outputs | `bdneuro_balanced_subset/balanced_subset_summary.json` | balanced_total; selection_method | VERIFIED |
| 34 | BDNeuro licence/citation unfilled; source 'Epic & CSCR Hospital' | `external_data/BDNeuro_V8_raw/Brain_Tumor_MRI_Dataset_Final/README.md` | Source/License/Citation lines | VERIFIED |
| 35 | BDNeuro full / balanced 15-pipeline results and stage means | `comparison_15_models_bdneuro_{full,balanced}/external_comparison_15.csv` | external_accuracy_pct; external_macro_f1_pct; external_minus_internal_accuracy_pp | VERIFIED |
| 36 | Per-class recall on BDNeuro (pituitary lowest in 12/15; balanced > full in 14/15) | `comparison_15_models_bdneuro_*/*/*/*.json` | external.confusion_matrix | VERIFIED (derived) |
| 37 | Web-40 audit (0 exact overlaps; pHash ≤4: 0; one pHash-6 candidate MEN_02 vs Tr-me_0633) | `web_stress_40_audit/summary.json; cross_phash_candidates.csv` | all fields | VERIFIED |
| 38 | NOR_01 replaced before inference (pHash 0 vs Te-no_0227) | `web_stress_40_audit/visual_review/NOR_01_comparison.jpg; transfer docs 11` | visual-review image; text | HISTORICAL (pre-replacement audit not retained as JSON) |
| 39 | Web cohort locked before inference | `web_stress_40_locked/lock_record.json` | status=LOCKED_BEFORE_INFERENCE; rule; mtime 10:37:22 vs results 10:38:09 | VERIFIED |
| 40 | Web source verification (40 rows with URLs) | `docs/…/manifests/Web_Stress_Test_40_Source_List.xlsx` | sheet1 rows | VERIFIED (repo template web_stress_40_manifest.csv unfilled: PENDING) |
| 41 | Web-40 15-pipeline results; EfficientNet-B3 P1 40.0%, P2A 42.5%; stage means | `comparison_15_models_web_stress_40/external_comparison_15.csv` | external_accuracy_pct; external_macro_f1_pct | VERIFIED |
| 42 | Web mean recall 12.7/30.7/90.7/7.3; predicted counts 2.1/8.3/28.3/1.3; 600 predictions | `comparison_15_models_web_stress_40/*/*/*.json` | external.confusion_matrix | VERIFIED (derived) |
| 43 | Final representative metrics (98.600 / 98.703 / 98.869 / 88.31 / 90.47 / 37.5) | `Phase 2A JSON; PHASE2A material CSV; three external CSVs` | as above | VERIFIED |
| 44 | Representative named by retrospective synthesis; web not used; no model altered | `transfer package final_selected_candidate.json; manifest timestamps` | selection_interpretation | INTERPRETIVE (procedure documented) |
| 45 | Grad-CAM ratings and agreement | `gradcam_evaluation/gradcam_evaluation_resnet152_seed42.csv` | layer3_rating, layer4_rating, fused_rating, best_map, classifiers_agree, svm_correct, neural_correct | VERIFIED |
| 46 | XAI used the Phase 1 ResNet152 via GUI registry | `gradcam_evaluation_app.py; occlusion_evaluation_app.py; gui/model_registry.py` | BACKBONE='resnet152'; registry → experiments/<bb> | VERIFIED |
| 47 | Occlusion 11 cases, 32/16/169, 58/1,859 flips, Te-pi_0078 38/169, 4 images with max drop <0.01 | `occlusion_evaluation/occlusion_comparison_resnet152_seed42.csv` | patch_size, stride, occlusion_count, prediction_flip_fraction, maximum_score_drop, ratings | VERIFIED |
| 48 | GUI tabs, upload types | `app.py` | lines 477–478; line 485 | VERIFIED |
| 49 | GUI serves Phase 1 pipelines | `gui/model_registry.py` | lines 51–60 | VERIFIED |
| 50 | TensorFlow INVALID_PTX / sm_120 failure; PyTorch migration | `transfer package docs 03/14; environment_summary.json` | text | HISTORICAL |
| 51 | Environment versions | `final_project_docs/environment_pip_freeze.txt; package results/environment_summary.json` | torch, torchvision, scikit-learn, numpy, streamlit | VERIFIED |
| 52 | Legacy ResNet152 prototype ~97.64% / 97.50% | `transfer package docs 03/06` | text (no JSON preserved) | HISTORICAL |
| 53 | Base-paper Table 2/3 values (RMSprop 94/96/99/99/98%) | `docs/claude_project_context/Hybrid CNN + LSTM + SVM … .pdf` | Table 2, Table 3 | VERIFIED (primary source) |
| 54 | Generalization gradient conclusion | `all of the above` | — | INTERPRETIVE |
| 55 | Domain-shift causes (acquisition, sequence, cropping, source cues) | `—` | no metadata | INTERPRETIVE (not causally established) |

The automated cross-check of every reported number against these sources is in FINAL_PROJECT_QC_REPORT.md.
