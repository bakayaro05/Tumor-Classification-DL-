# START HERE — DL TUMOR CLASSIFICATION FINAL TRANSFER

**Authoritative status: 3 October 2026**

This ZIP is the final consolidated project handoff for Claude/ChatGPT.

## Read in this order
1. `MASTER_FINAL_PROJECT_CONTEXT.md` — one-file start-to-end project context.
2. `docs/02_PPT_PHASE_BY_PHASE_EXECUTION.md` — exact mapping of the Review-I PPT roadmap to what was actually done.
3. `docs/15_RESULTS_MASTER_FINAL_CANDIDATE_AND_CONCLUSIONS.md` — final result tables, interpretation and selected candidate.
4. `docs/13_CODEBASE_SCRIPTS_ARTIFACTS_AND_JSON_INDEX.md` — script/artifact/output map.
5. `docs/16_LIMITATIONS_CLAIM_RULES_AND_FUTURE_WORK.md` — scientific wording and caveats.
6. `docs/18_COMPLETENESS_AUDIT_AND_AUTHORITY_RULES.md` — what is included, reconstructed, historical, or unavailable as raw bytes.

## Final selected project representative
**ResNet152 Phase 2A standardized 30-epoch pipeline, best retained checkpoint at epoch 10**
- RMSprop
- LR 1e-4
- LSTM 256
- dropout 0.3
- Dense128
- StandardScaler → PCA64 → linear SVM C=1

The selection is a project-level synthesis of the complete body of evidence; no model was retrained or altered using external results.

## Final PPT phase status
- Phase 1: COMPLETE
- Phase 2A: COMPLETE
- Phase 2B: COMPLETE
- Phase 2C: COMPLETE
- Phase 2D: COMPLETE
- Phase 3: COMPLETE
- Phase 4: COMPLETE
- Phase 5: COMPLETE
- Phase 6: COMPLETE for the mini-project's experimental validation/documentation

## Critical caution
The original Nickparvar split has material image-level cross-split duplication. Historical internal scores are therefore not clean independent-generalization estimates. `material_clean` is a post-hoc leakage-aware subset, not a newly independent test set. The structured BDNeuro exact-overlap-clean evaluation and the web stress test provide separate generalization evidence.

## Historical folder warning
`historical_transfer/` intentionally preserves older status documents. Some of those files say that external validation or group-aware retraining was still pending. Those statements are historical. Final files in `docs/` supersede them.
