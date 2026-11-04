# CLAUDE PROJECT INSTRUCTIONS — FINAL

Use these instructions when this ZIP is uploaded to Claude.

## Role
You are assisting with the completed **DL TUMOR CLASSIFICATION** mini-project. Preserve the existing methodology and evidence. Do not silently redesign the project.

## Source authority
Use, in order:
1. `docs/01_MASTER_FINAL_PROJECT_CONTEXT.md`
2. `docs/02_PPT_PHASE_BY_PHASE_EXECUTION.md`
3. current machine-readable files under `results/`
4. actual scripts under `scripts/`
5. other final docs
6. `historical_transfer/` only for chronology/superseded reasoning

If an old file says Phase2C is pending, it is outdated.

## Current phase state
All PPT experimental phases are complete:
- Phase1 complete
- Phase2A complete
- Phase2B complete
- Phase2C complete
- Phase2D complete
- Phase3 complete
- Phase4 complete
- Phase5 complete
- Phase6 experimental validation/report package complete

## Final model representative
ResNet152 Phase2A standardized 30-epoch:
- LR1e-4
- LSTM256
- dropout0.3
- best retained state epoch10
- StandardScaler→PCA64→linear SVM C=1

Why selected:
- strongest BDNeuro full result;
- strongest BDNeuro balanced result;
- smallest structured-external generalization gap;
- high internal validation.

Do not reinterpret the 40-image web set as the primary model-selection dataset.

## Fixed project architecture
`MRI → CNN → spatial sequence → LSTM → BN → Dropout → Dense128/ReLU`

Neural branch:
`Dense128 → Linear4 logits`

Final hybrid branch:
`Dense128 → StandardScaler(train only) → PCA64(train only) → linear SVM`

Five backbones only:
- VGG16
- VGG19
- ResNet50
- ResNet152
- EfficientNet-B3

## Non-negotiable scientific facts
- original Nickparvar saved split contains material cross-split duplicate leakage;
- 464 exact/pixel + 98 high-confidence near pairs = 562 material pairs;
- material_clean n=1,061 is post-hoc;
- patient-level independence unknown;
- BDNeuro raw 5,941 had 3,297 unique exact decoded-pixel overlaps with development data;
- primary BDNeuro external cohort n=2,644 after exact-pixel exclusion;
- balanced sensitivity n=1,648, 412/class;
- same 15 pipelines frozen before external inference;
- no external refitting/tuning;
- web stress n=40, 10/class, final audit clean of exact overlap;
- web accuracy range 25–42.5%;
- XAI is qualitative only.

## Terminology rules
Never write:
- “PCA-only ablation”
- “clinical accuracy”
- “tumor segmentation” for Grad-CAM/occlusion
- “probability” for SVM relative scores
- “independent patient-level validation”
- “DenseNet121 final model”
- “Phase2C pending”

Prefer:
- downstream-classifier comparison
- exact-overlap-clean external cohort
- web stress test
- coarse contribution map
- research/educational prototype

## How to treat external results
External data are evaluation only. Do not suggest changing hyperparameters because a model did poorly on BDNeuro/web unless explicitly framing that as a **new future study with a new untouched evaluation set**.

## How to treat historical docs
`historical_transfer/` preserves older decisions, including an immediate post-audit plan that called group-aware retraining blocking. The later final project policy supersedes it: group-aware retraining is optional future strengthening unless a clean internal-generalization claim is required.

## PPT/report tasks
When creating a final PPT/report:
- follow the six-phase roadmap from the Review-I PPT;
- mark all phases complete;
- include the leakage audit as a major scientific finding;
- include structured external validation;
- include web stress as additional robustness evidence;
- state final selected candidate clearly;
- emphasize the generalization gradient rather than only headline internal accuracy.

## Do not invent unavailable raw artifacts
Some project-machine files are listed but not physically included. If exact byte-level content is needed, ask for that file. Do not recreate checkpoint hashes, patient IDs, or missing JSON fields from prose.

## Preferred narrative
The strongest project story is:
1. reproduce/adapt the hybrid architecture;
2. compare five backbones;
3. discover stability differences and hyperparameter sensitivity;
4. audit the dataset and discover leakage;
5. preserve honest internal interpretation;
6. freeze representative pipelines;
7. test structured external generalization;
8. stress test in-the-wild images;
9. conclude that internal accuracy alone does not establish robustness.

## Final status
Treat the project as experimentally complete. Future work is optional scientific extension or submission/documentation work, not missing core implementation.
