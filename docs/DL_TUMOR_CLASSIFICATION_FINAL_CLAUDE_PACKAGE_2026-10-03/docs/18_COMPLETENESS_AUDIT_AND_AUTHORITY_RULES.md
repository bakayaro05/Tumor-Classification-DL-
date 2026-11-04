# TRANSFER COMPLETENESS AUDIT AND AUTHORITY RULES

## Purpose
This file states what was actually captured in this ZIP, what is reconstructed, what remains only on the user's project machine, and which older statements were deliberately superseded.

## 1. Conversation/history coverage

Captured through:
- earlier adversarially reviewed transfer package built from project history;
- current Project conversation summaries;
- Review-I PPT;
- current scripts uploaded to the project;
- Phase2 machine-readable archives;
- final external-validation outputs supplied in chat;
- final web-stress master JSON;
- explicit user decisions through 3 October 2026.

The package preserves the earliest July classical direction, not only the later deep-learning project.

## 2. Methodology coverage
Captured:
- dataset/split;
- preprocessing;
- CNN-LSTM architecture;
- spatial-sequence interpretation;
- training settings;
- VGG freeze policy;
- neural head;
- scaler/PCA/SVM;
- Phase2 fixed horizon;
- hyperparameter variants;
- data-integrity audit;
- external freeze protocol;
- BDNeuro cleaning;
- balanced subset;
- web cohort selection/audit;
- XAI;
- GUI;
- final candidate.

## 3. Experiment coverage
Captured:
- successful experiments;
- TensorFlow failure;
- VGG collapse;
- legacy ResNet152;
- standardized five-backbone Phase1;
- neural-head evaluation;
- material_clean;
- Phase2A;
- Phase2B;
- 80 retained-state hybrid evaluations;
- XAI;
- BDNeuro external;
- web stress;
- final candidate.

## 4. Machine-readable coverage

Physically included:
- Phase2 training 40-run flat table;
- Phase2 80-state full-hybrid table;
- Phase2 selected-state table;
- three Phase2 result ZIPs;
- duplicate audit JSON;
- Phase1 summary JSON;
- material_clean summary JSON;
- BDNeuro full/balanced CSVs;
- reconstructed BDNeuro summary JSON;
- web full master JSON and CSV;
- web audit summary JSON;
- XAI aggregate summary JSON;
- final selected candidate JSON;
- phase-status JSON;
- environment summary JSON.

## 5. Raw files not physically captured in this ZIP

Known to remain on the project machine:
- actual model `.pth` checkpoints;
- scaler/PCA/SVM `.joblib` files;
- feature `.npy` arrays;
- original Phase1 `training_results.json` / `final_results.json` bundles;
- original BDNeuro evaluator master JSONs;
- original `frozen_15_manifest.json` including SHA256 hashes;
- original `frozen_15_selection.csv`;
- original BDNeuro audit CSVs/manifests;
- original web lock record;
- original Grad-CAM evaluation CSV;
- original occlusion comparison CSV;
- saved split indices;
- exact `src/data_pipeline_torch.py` source;
- Phase2B worker scripts not uploaded here;
- `gui/*` module files not separately uploaded.

Their paths and roles are documented. Numerical conclusions used in the final docs are supported by machine-readable outputs or the established project record.

## 6. Reconstructed files

`results/bdneuro_external_validation_reconstructed_summary.json` is explicitly reconstructed from:
- pasted full/balanced master CSVs;
- audited cohort counts;
- project decision record.

It is not claimed to be byte-identical to the original evaluator JSON.

Aggregate XAI JSON similarly summarizes the authoritative CSV findings; it is not the raw CSV.

## 7. Historical contradictions preserved and resolved

### DenseNet121
Historical mention exists. No verified standardized artifacts. Final scope = five backbones.

### SVM probability setting
Current scripts/GUI use decision margins and probability=False wording; some historical logs indicated probability-related behavior. Final reporting avoids claiming calibrated SVM probability.

### PCA/scaler order
Early diagrams sometimes differed. Implemented order = StandardScaler → PCA.

### Duplicate-group retraining status
Immediate post-audit docs: blocking.  
Final project policy: optional stronger future remediation after external evidence.

### Phase2B progress
An intermediate state said 1/35. Machine-readable final state = 35/35 + 70/70 downstream evaluations.

### External validation
Older docs say pending. Current final state = complete on BDNeuro full, balanced, and web stress.

### Final candidate
Earlier project preserved multiple architecture representatives. Final Phase3 synthesis now names ResNet152 Phase2A as the overall project representative.

## 8. What “literally everything” means in this transfer

This ZIP contains **all retrievable and materially relevant project history and evidence available in the Project/attached files at transfer time**, plus indexes for raw repository artifacts that were not physically attached.

It does not pretend to contain:
- invisible local files that were never uploaded;
- binary model weights not supplied;
- full verbatim ChatGPT transcripts when only prior transfer/history records were available;
- patient metadata that never existed.

Where raw evidence was unavailable, the package says so rather than fabricating it.

## 9. Authority rule for Claude
If a historical file conflicts with a final doc:
- final docs and current machine-readable result files win;
- historical docs remain evidence of how the project evolved.

## 10. Final completeness conclusion
For continuation, report writing, PPT generation, viva preparation and methodological understanding, this package is complete enough to reconstruct the project end-to-end without the original chat history.

For exact byte-level reproducibility, Claude should additionally receive the user's live repository/checkpoints/artifact folders from `~/brain_tumor_project`.
