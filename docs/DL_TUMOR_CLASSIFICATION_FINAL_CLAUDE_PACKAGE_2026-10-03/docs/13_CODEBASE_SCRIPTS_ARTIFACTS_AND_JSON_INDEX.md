# CODEBASE, SCRIPTS, ARTIFACTS, AND JSON INDEX

This file is the practical code/artifact map for the final project. It distinguishes:
- files physically bundled in this transfer;
- project-machine files known to exist from outputs/logs but not physically available in this chat;
- historical/legacy artifacts.

## 1. Available scripts bundled here

| Script | Lines | Top-level classes | Representative functions |
| --- | --- | --- | --- |
| app.py | 613 | — | _model_manager, _open_image, _score_chart, _render_prediction, _render_explanation, _render_svm_occlusion… |
| audit_bdneuro_overlap.py | 680 | — | is_image, rel, save_json, write_csv, sha256_file, decoded_pixel_hash… |
| audit_web_stress_40.py | 256 | — | is_image, rel, sha256_file, pixel_hash, phash64, add_fingerprints… |
| build_bdneuro_balanced_subset.py | 511 | — | is_image, project_relative, sha256_file, selection_rank, save_json, main |
| build_bdneuro_exact_clean_cohort.py | 717 | — | is_image, project_relative, sha256_file, write_csv, save_json, collect_bdneuro_images… |
| compare_external_15.py | 1500 | — | load_json, save_json, sha256_file, relative_to_project, absolute_from_manifest, safe_float… |
| compare_external_15_PRE_RESULTS_NAME_VERSION.py | 1478 | — | load_json, save_json, sha256_file, relative_to_project, absolute_from_manifest, safe_float… |
| diagnose_vgg16_collapse.py | 100 | — | stats |
| evaluate_backbone.py | 1519 | — | extract_features, calculate_metrics |
| evaluate_backbone_30epoch.py | 1552 | — | extract_features, calculate_metrics |
| evaluate_final_test.py | 395 | — | — |
| extract_all_features.py | 606 | — | extract_features, print_class_counts |
| gradcam_evaluation_app.py | 414 | — | _test_directory, _class_directory, _evaluation_samples, _model_manager, _open_image, _load_records… |
| initialize_web_stress_test.py | 87 | — | main |
| model_cnn_lstm.py | 454 | CNNLSTM | — |
| model_resnet152_lstm.py | 113 | ResNet152LSTM | — |
| occlusion_evaluation_app.py | 413 | — | _frozen_gradcam_csv, _test_directory, _comparison_samples, _model_manager, _open_image, _analyze… |
| prepare_pca_features.py | 498 | — | — |
| run_all_dl_hyperparams.py | 543 | — | timestamp, append_summary, main |
| test_all_backbones.py | 337 | — | — |
| test_best_checkpoint_features.py | 433 | — | — |
| test_complete_model.py | 78 | — | — |
| test_lstm_section.py | 113 | — | — |
| test_post_lstm_layers.py | 113 | — | — |
| test_real_mri_batch.py | 220 | — | — |
| test_resnet152_features.py | 174 | — | — |
| test_training_step.py | 250 | — | — |
| train_backbone.py | 1259 | — | train_one_epoch, validate |
| train_backbone_30epoch.py | 1340 | — | train_one_epoch, validate |
| train_resnet152_lstm.py | 1045 | — | train_one_epoch, validate |
| train_svm.py | 596 | — | — |
| verify_gui_setup.py | 89 | — | main |

## 2. Core source roles

### `model_cnn_lstm.py`
Generic five-backbone architecture. Authoritative backbone scope:
- vgg16
- vgg19
- resnet50
- resnet152
- efficientnet_b3

Core path:
CNN feature map → spatial token sequence → LSTM → BatchNorm → Dropout → Dense128/ReLU → temporary 4-class head.

### `train_backbone.py`
Phase1 standardized trainer.
Important defaults/behavior:
- epochs default 10
- batch 8
- LR1e-4
- patience3
- seed42
- LSTM256
- Dense128
- dropout0.3
- VGG CNN layers frozen
- RMSprop
- CrossEntropyLoss
- `drop_last=True` in training loader.

### `evaluate_backbone.py`
Standardized hybrid evaluator:
- deterministic feature extraction;
- StandardScaler;
- PCA default 64;
- linear SVM default C=1;
- validation/test metrics and artifacts.

### `train_backbone_30epoch.py`
Phase2A fixed-horizon trainer:
- same baseline hyperparameters;
- fixed 30 epochs;
- patience deliberately set beyond horizon so there is no early termination;
- saves best-within30 and epoch30 model states.

### `evaluate_backbone_30epoch.py`
Phase2A retained-state downstream evaluator. The raw Phase2 full package contains both state evaluations for all five backbones.

### `run_all_dl_hyperparams.py`
Phase2B queue:
- five backbones;
- seven one-factor configs;
- 35 total runs;
- invokes project worker `actual_training_scripts/train_backbone_hyperparams.py`.

The worker itself is **not physically bundled** in this transfer, but its outputs are extensively preserved in `raw_packages/phase2_full_analysis_package.zip`.

### `compare_external_15.py`
This transfer uses the **repaired named-results version**, which supports `--results-name`.
Responsibilities:
- select 15 pipelines using internal validation;
- freeze SHA256 artifact manifest;
- verify hashes before external evaluation;
- require explicit `--confirm-audited`;
- enforce four-class ImageFolder mapping;
- extract external deep features;
- load saved scaler/PCA/SVM;
- never fit on external data;
- save per-model and master comparison outputs;
- avoid full-vs-balanced output collisions via named result directory.

`compare_external_15_PRE_RESULTS_NAME_VERSION.py` is preserved only to document the earlier version that lacked `--results-name`.

### External-data integrity scripts
- `audit_bdneuro_overlap.py`
- `build_bdneuro_exact_clean_cohort.py`
- `build_bdneuro_balanced_subset.py`
- `initialize_web_stress_test.py`
- `audit_web_stress_40.py`

### GUI/XAI
- `app.py`
- `gradcam_evaluation_app.py`
- `occlusion_evaluation_app.py`
- `verify_gui_setup.py`

The project-machine GUI also has `gui/*` modules that were not separately available as raw files in this transfer.

### Legacy/diagnostic scripts
- `model_resnet152_lstm.py`
- `train_resnet152_lstm.py`
- `extract_all_features.py`
- `prepare_pca_features.py`
- `train_svm.py`
- `evaluate_final_test.py`
- tensor-shape/training-step tests
- `diagnose_vgg16_collapse.py`

These are important for project history even where standardized generic scripts later superseded them.

---

## 3. Important scripts known to exist on the project machine but not physically bundled

These names are supported by queue/evaluation history:
- `src/data_pipeline_torch.py`
- `src/save_split.py`
- `src/check_dataset.py`
- `train_backbone_hyperparams.py`
- `evaluate_hyperparam_pca_svm.py`
- GUI modules under `gui/`
- the original duplicate/leakage audit worker used for Nickparvar
- any script that generated `material_clean` evaluation
- result/figure generation helpers not uploaded here

Do **not** infer their exact current source from prose if the real repository is available; inspect the repository.

---

## 4. Project artifact layout

### Phase1 standardized experiments
`experiments/<backbone>/`
- `models/best_model.pth`
- `models/svm.joblib`
- `pca/standard_scaler.joblib`
- `pca/pca_model.joblib`
- deep feature arrays
- PCA arrays
- `results/training_results.json`
- `results/final_results.json`

### Legacy first ResNet152
Root-level:
- `models/best_resnet152_lstm.pth`
- root `features/`
- root `pca_features/`
- legacy SVM bundle

Superseded for standardized comparison.

### Phase2A
`phase2/experiments_30epoch/<backbone>/`
- `models/best_val_within_30.pth`
- `models/epoch_30_model.pth`
- training JSON
- per-state evaluation directories
- per-state `final_results.json`

### Phase2B
`phase2/hyperparameter_dl/<backbone>/<variant>/`
- best-within30 checkpoint
- epoch30 checkpoint
- training JSON
- evaluation/best state `final_results.json`
- evaluation/epoch30 `final_results.json`

### External validation
`phase2/external_validation/`
- `frozen_15_manifest.json`
- `frozen_15_selection.csv`
- BDNeuro audit directories
- exact-clean cohort traceability
- balanced subset traceability
- full external comparison directory
- balanced external comparison directory
- web-stress audit
- web lock record
- web comparison directory

A text list of known paths is bundled at:
`manifests/KNOWN_PROJECT_MACHINE_ARTIFACT_PATHS.txt`.

---

## 5. Phase2 machine-readable JSON inventory

`raw_packages/phase2_full_analysis_package.zip` contains **116 JSON files**:
- **36 `training_results.json`**
  - 35 Phase2B changed-config training files
  - Phase2A aggregate/baseline package file(s) as stored in the archive
- **80 `final_results.json`**
  - 10 Phase2A retained-state evaluations
  - 70 Phase2B retained-state evaluations

Exact paths are listed in:
`manifests/PHASE2_JSON_INDEX.txt`.

Derived flat tables:
- `results/phase2_training_40run_summary.csv`
- `results/phase2_full_hybrid_80_state_evaluations.csv`
- `results/phase2_selected_states_by_backbone.csv`

---

## 6. Other bundled machine-readable evidence

### Development integrity
- `results/duplicate_leakage_audit.json`
- `results/material_clean_summary.json`

### Phase1
- `results/phase1_results_summary.json`

### BDNeuro
- `results/bdneuro_full_external_comparison_15.csv`
- `results/bdneuro_balanced_external_comparison_15.csv`
- `results/bdneuro_external_validation_reconstructed_summary.json`

Important: the reconstructed JSON is generated from the pasted master CSVs plus the audited project record. The **original evaluator master JSONs remain on the project machine** and are not physically reproduced here.

### Web stress
- `results/web_stress_audit_final_summary.json`
- `results/web_stress_external_comparison_15.csv`
- `results/web_stress_external_comparison_15.json`

The web master JSON is based on the complete pasted evaluator JSON and includes confusion matrices/classification reports for all 15 candidates.

### XAI
- `results/xai_summary.json`

The original Grad-CAM and occlusion CSVs existed in the project record but could not be raw-materialized into this package. Aggregate counts and key case identities are preserved.

### Final state
- `results/final_selected_candidate.json`
- `results/ppt_phase_status.json`
- `results/environment_summary.json`

---

## 7. Script/output compatibility rules

1. **Scaler before PCA** is the authoritative implemented order.
2. Do not assume all historical SVM joblib objects have identical `probability` settings merely from one current script. GUI uses `decision_function` and relative-score wording.
3. VGG checkpoint loading requires frozen-backbone architecture compatibility.
4. External evaluator must use the exact frozen artifacts; never re-fit scaler/PCA/SVM.
5. `ImageFolder` class mapping must remain:
   - glioma=0
   - meningioma=1
   - notumor=2
   - pituitary=3
6. Phase2B worker source is missing here; the result JSONs are the authoritative evidence of what ran.
7. The Review-I PPT is a plan/presentation snapshot, not current source-of-truth status.

---

## 8. Old repository manifest warning

`manifests/project_files.txt` is useful for the Phase1 repository snapshot, but it predates later Phase2/external/web-stress files. Do not use it as the complete final repository manifest.

The package-level manifest generated with this ZIP supersedes it for transfer-package contents.
