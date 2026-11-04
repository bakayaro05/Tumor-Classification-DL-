# DL TUMOR CLASSIFICATION — FINAL CLAUDE TRANSFER PACKAGE

**Final consolidation date:** 3 October 2026  
**Project type:** Four-class brain MRI classification research/educational prototype  
**Base methodology:** 2026 Springer paper, *Hybrid CNN+LSTM+SVM model for advanced brain tumor classification*  
**Authoritative final scope:** VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3  
**Final selected project representative:** **ResNet152 Phase 2A standardized 30-epoch pipeline, best state at epoch 10**.

## Why this package exists

This package is designed so Claude or another ChatGPT session can reconstruct the project without relying on the original conversation history. It consolidates the project from the earliest classical-image-processing discussions in July 2026 through the completed CNN-LSTM-PCA-SVM implementation, five-backbone comparison, leakage audit, Phase 2A/2B experiments, GUI and explainability work, BDNeuro external validation, 40-image in-the-wild web stress test, and final model selection.

It was assembled from:
- the earlier adversarially reviewed transfer package;
- the actual Review-I PPT roadmap;
- the available project scripts;
- Phase 2 machine-readable JSON packages;
- the authoritative duplicate/leakage audit;
- BDNeuro and web-stress evaluation outputs supplied from the project machine;
- the final reports generated after external validation;
- the current project conversation and explicit user decisions.

## Authority order

When two files conflict, use this order:

1. `docs/01_MASTER_FINAL_PROJECT_CONTEXT.md`
2. `docs/02_PPT_PHASE_BY_PHASE_EXECUTION.md`
3. Current machine-readable files under `results/`
4. Current scripts under `scripts/`
5. `docs/03_FULL_PROJECT_TIMELINE.md` and `docs/04_DECISION_REGISTER_FINAL.md`
6. Other final docs in `docs/`
7. `historical_transfer/` only for history and superseded reasoning
8. Old PPT status markers only as historical Review-I presentation state

Never allow an older “pending” status to overwrite a later completed experiment.

## Final phase status

- Phase 1 — Base-Paper Implementation: **COMPLETE**
- Phase 2A — Fixed 30-Epoch Training: **COMPLETE**
- Phase 2B — Hyperparameter Sensitivity: **COMPLETE**
- Phase 2C — External Validation: **COMPLETE**
- Phase 2D — Data Integrity / Robustness: **COMPLETE**
- Phase 3 — Comparative Analysis & Final Candidate Selection: **COMPLETE**
- Phase 4 — Explainability: **COMPLETE**
- Phase 5 — Prototype Application: **COMPLETE**
- Phase 6 — Final Validation / Report / Demonstration: **COMPLETE for project experimentation and documentation**

The only major scientific extension left is optional: duplicate-group-aware re-splitting and full retraining of the original Nickparvar dataset if a stronger clean internal-generalization claim is required.

## Recommended Claude upload order

If context is limited, upload in this order:

1. `docs/01_MASTER_FINAL_PROJECT_CONTEXT.md`
2. `docs/02_PPT_PHASE_BY_PHASE_EXECUTION.md`
3. `docs/15_RESULTS_MASTER_FINAL_CANDIDATE_AND_CONCLUSIONS.md`
4. `docs/13_CODEBASE_SCRIPTS_ARTIFACTS_AND_JSON_INDEX.md`
5. `docs/16_LIMITATIONS_CLAIM_RULES_AND_FUTURE_WORK.md`
6. `results/final_selected_candidate.json`
7. The specific script/result file needed for the current task

If ample context is available, upload the whole ZIP as a Claude Project.

## Directory map

- `docs/` — authoritative final narrative and analysis.
- `results/` — machine-readable summaries, reconstructed external tables, and Phase 2 result indexes.
- `scripts/` — available source snapshots, including training/evaluation/audit/external-validation scripts.
- `references/` — Review-I PPT and final PDFs.
- `manifests/` — repository/file manifests and GUI/web-stress supporting files.
- `raw_packages/` — Phase 2A/2B machine-readable result archives.
- `historical_transfer/` — the earlier transfer package retained for chronology and hostile-review history.

## Critical terminology

Use:
- **“neural-head vs PCA+SVM downstream-classifier comparison”**, not “PCA-only ablation”.
- **“material_clean post-hoc test subset”**, not “clean independent test set”.
- **“BDNeuro V8 cross-dataset exact-overlap-clean external cohort”** for the 2,644-image primary external evaluation.
- **“40-image source-verified in-the-wild web MRI stress test”** for the exploratory web test.
- **“coarse contribution maps”** for Grad-CAM/occlusion, not segmentation.
- **“research/educational prototype”**, not medical device.

## Important evidence caveat

The original 7,023-image Nickparvar split contains material cross-split duplication. Original internal metrics remain valid recorded outputs but are not clean independent-generalization estimates. External validation and the web stress test were therefore added to examine generalization under increasingly strong domain shift.

## Conversation-history addendum
`docs/19_CONVERSATION_HISTORY_RECOVERY_NOTES.md` preserves additional chronological details recovered from prior project chats (classical-project origin, framework troubleshooting, staged architecture construction, VGG collapse, XAI freeze, audit interpretation changes, Phase2 completion, and external/web-test decisions).
