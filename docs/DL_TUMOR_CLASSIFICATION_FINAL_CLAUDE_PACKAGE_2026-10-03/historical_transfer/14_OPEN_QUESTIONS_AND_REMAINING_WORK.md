# Open Questions and Remaining Work

## Additional prerequisites exposed by hostile review

- Inspect every standardized `svm.joblib` and reconcile `.probability` with source and logs.
- Resolve the DenseNet121 contradiction across the complete repository, logs, and artifacts.
- Recover the exact historical data transform and distinguish it from the paper's broader augmentation recipe.
- Preserve zero-based saved versus one-based displayed epoch conventions.

## BLOCKING

### 1. Resolve duplicate groups and rebuild the split

- Parse all 562 material duplicate relations and adjudicate the 481 review candidates.
- Form transitive connected groups.
- Identify cross-label groups and quarantine or resolve them explicitly.
- Choose and document whether to preserve the dataset's original test-folder semantics or create a new group-aware holdout.
- Produce group-aware class-stratified train/validation/test manifests.
- Rerun the exact/pixel/perceptual audit and require zero declared material cross-split matches.

### 2. Retrain and reevaluate from scratch

- Do not initialize from contaminated-split checkpoints.
- Retrain all five neural models.
- Re-extract all 128-D features.
- Refit each StandardScaler, PCA, and SVM on new training data only.
- Evaluate the clean test once after development lock.
- Regenerate final JSON, predictions, confusion matrices, tables, curves, and explainability samples.

### 3. Correct all claims before submission

- Remove or qualify old “99%”/“untouched test”/“generalization” wording.
- Do not state that ResNet152 was the project winner; EfficientNet-B3 narrowly led the historical standardized test comparison.
- Mark all pre-audit metrics as contaminated-split historical results.
- Update abstract, report, PPT, poster, README, and GUI.

## NON-BLOCKING BUT IMPORTANT

- Regenerate `project_files.txt` after all later GUI and audit modules are present.
- Freeze exact package versions and hardware details.
- Add configuration files rather than relying on script defaults.
- Convert smoke scripts into automated tests with assertions and no expensive import-time side effects.
- Save per-sample predictions and sample IDs for paired statistical analysis.
- Add bootstrap confidence intervals and multiple seeds.
- Report parameter counts, training time, inference latency, peak VRAM, and artifact size.
- Review `Te-no_0216.jpg` with qualified expertise; do not self-relabel.
- Verify the exact augmentation parameters in `src/data_pipeline_torch.py`.
- Verify whether the GPU was RTX 5070 8 GB, 12 GB, or RTX 5070 Ti; project conversations contain inconsistent hardware descriptions.
- Verify title and final research objectives.

## FUTURE WORK

- External-dataset evaluation and domain-shift analysis.
- Dataset with patient/study IDs and patient-level split.
- Probability calibration and decision-curve analysis.
- Proper DICOM ingestion, orientation, windowing, and metadata handling.
- Expert lesion masks and quantitative localization metrics.
- Radiologist/inter-rater evaluation of explanations.
- Strong baselines: pretrained CNN with a simple linear/softmax head, CNN+SVM without LSTM, and LSTM ablation.
- Statistical ablation of PCA, scaler, SVM kernel, and LSTM ordering.
- Attention/transformer alternative, ensemble, or lightweight deployment only after valid baselines.
- Robustness tests for cropping, borders, skull removal, acquisition artifacts, and adversarial/spurious cues.

## UNKNOWN

- Exact mapping between Library filenames `final_results.json`, `(1)`…`(4)` and original repository paths is inferred from their internal `backbone` field, not filenames.
- Exact base-paper DOI and full citation fields were not verified in this transfer.
- Exact Kaggle dataset revision, license, and download timestamp.
- Whether duplicate pairs arise from literal dataset duplication, transformations, patient repeats, or all three.
- Whether every one of the 481 review candidates is material.
- Whether clean splitting will leave enough independent groups for stable four-class training and evaluation.
- Whether later source files not included in the attachment contain additional modifications.
- Whether report/PPT/abstract drafts have already been corrected elsewhere.

## Recommended immediate sequence

1. Archive current results under `historical_contaminated_split/` without deleting anything.
2. Recover the full current repository and regenerate its manifest.
3. Summarize duplicate groups by split pair, class pair, connected-component size, and label conflict.
4. Freeze the new split policy before training.
5. Execute clean reruns and audit them.
6. Only then update conclusions and lock the project.
