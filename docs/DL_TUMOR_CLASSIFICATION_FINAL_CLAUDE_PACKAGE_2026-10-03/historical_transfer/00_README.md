# DL Tumor Classification — Transfer Package

## Purpose

This package transfers the technical state and development history of the four-class brain-MRI classification project commonly referred to as **DL TUMOR CLASSIFICATION**. It is intended to let another AI researcher continue the work without access to the original ChatGPT conversations.

The project implements and extends the 2026 paper **“Hybrid CNN+LSTM+SVM model for advanced brain tumor classification”** using five alternative ImageNet-pretrained CNN backbones, an LSTM over flattened spatial features, a 128-dimensional learned representation, training-only StandardScaler and PCA, and a linear SVM. A Streamlit GUI and two explainability paths—neural-head Grad-CAM and final-SVM occlusion sensitivity—were subsequently added.

## Critical status warning

The project is **not scientifically locked**. On 20 September 2026, an image-level cross-split audit reported `MATERIAL_DUPLICATION`:

- 7,023 images audited;
- 464 cross-split exact-file or pixel-duplicate pairs;
- 98 additional high-confidence near-duplicate pairs;
- 562 material pairs in total;
- 481 further perceptual candidates marked for review.

Consequently, all existing validation/test scores describe the historical contaminated split. They must not be presented as clean independent-generalization estimates. The blocking next step is duplicate-group-aware re-splitting followed by retraining and reevaluation of every backbone. Patient-level independence remains unknowable because reliable patient identifiers are unavailable.

## Authority order

When sources conflict, use this order:

1. The current codebase and machine-readable outputs actually present in the project.
2. `duplicate_leakage_audit.json` and its findings CSV for split-integrity status.
3. `07_CURRENT_PROJECT_STATE.md` for the latest interpreted state.
4. `08_FINAL_METHODOLOGY.md` for the intended current methodology.
5. Historical documents in this package for how and why the project evolved.
6. Conversation recollections, which may be incomplete.

Do not overwrite a current fact with an older chat statement. Do not convert a proposal into completed work.

## Files in this package

| File | Role |
|---|---|
| `00_README.md` | Index, authority rules, and reading order |
| `03_PROJECT_JOURNEY_TIMELINE.md` | Chronological development history |
| `04_DECISION_HISTORY.md` | Major decisions and their reasoning |
| `05_PROJECT_CHANGELOG.md` | Before/change/after record |
| `06_EXPERIMENT_HISTORY.md` | Successful, failed, partial, and proposed experiments |
| `07_CURRENT_PROJECT_STATE.md` | Authoritative current state and blocking issue |
| `08_FINAL_METHODOLOGY.md` | Methodology that should govern the next valid run |
| `08B_METHODOLOGY_EVOLUTION.md` | How the methodology changed |
| `09_CODEBASE_GUIDE.md` | Modules, artifacts, execution flows, and snapshot caveats |
| `10_LITERATURE_AND_REFERENCE_HISTORY.md` | Base paper and other literature discussions |
| `11_RESULTS_EVOLUTION.md` | Actual results and their changing interpretation |
| `12_PROBLEMS_DEBUGGING_AND_LESSONS.md` | Failures, diagnoses, fixes, and lessons |
| `13_REJECTED_APPROACHES.md` | Rejected, failed, removed, and unexecuted ideas |
| `14_OPEN_QUESTIONS_AND_REMAINING_WORK.md` | Blocking, non-blocking, future, and unknown work |
| `15_DOCUMENTATION_AND_PRESENTATION_CONTEXT.md` | Report, PPT, poster, and viva narrative |
| `16_CLAUDE_PROJECT_INSTRUCTIONS.md` | Paste-ready Claude Project instructions |
| `17_TRANSFER_COMPLETENESS_AUDIT.md` | Coverage audit and user-verification list |
| `18_HOSTILE_REVIEW_CORRECTIONS.md` | Corrections made after adversarial review of the first transfer |

## Recommended reading order

1. `00_README.md`
2. `07_CURRENT_PROJECT_STATE.md`
3. `08_FINAL_METHODOLOGY.md`
4. `09_CODEBASE_GUIDE.md`
5. `06_EXPERIMENT_HISTORY.md`
6. `04_DECISION_HISTORY.md`
7. `03_PROJECT_JOURNEY_TIMELINE.md`
8. `11_RESULTS_EVOLUTION.md`
9. Remaining documents as needed

## Companion material to upload

Upload the complete, current repository—not only the older `project_files.txt` snapshot—plus:

- all five `experiments/<backbone>/` directories;
- source files under `src/`, training/evaluation scripts, GUI modules, and tests;
- `duplicate_leakage_audit.json` and `duplicate_leakage_findings.csv`;
- `gradcam_evaluation_resnet152_seed42.csv`;
- `occlusion_comparison_resnet152_seed42.csv`;
- the selected base-paper PDF and any literature-survey papers;
- the saved split indices, dataset-version information, and environment lock files;
- figures, training curves, confusion matrices, checkpoints, scaler/PCA/SVM objects, and report/PPT drafts.

## Evidence labels used here

- **Verified output:** directly present in code or machine-readable result files.
- **Conversation-confirmed:** repeatedly recorded in project conversations but not independently rerun during this transfer.
- **Historical:** true of an earlier phase, not authoritative now.
- **Proposed:** discussed but not established as completed.
- **Unknown / requires verification:** evidence was unavailable or contradictory.

## Known internal contradictions preserved in this revision

- One pre-audit checkpoint said DenseNet121 had been implemented and evaluated, but the supplied `CNNLSTM` source, supported-backbone list, manifest, and standardized result files contain only VGG16, VGG19, ResNet50, ResNet152, and EfficientNet-B3. DenseNet121 is therefore **not part of the verified final comparison** unless separate code/results are recovered.
- Current `evaluate_backbone.py` and GUI documentation say `probability=False`, but preserved VGG19 and ResNet152 evaluation logs show scikit-learn's warning for `SVC(probability=True)`. The GUI actually uses `decision_function()` and labels the values as relative scores. The exact `probability` attribute of each saved SVM artifact must be inspected; do not assert one value for all artifacts without that check.
- Early conceptual documents sometimes show PCA before StandardScaler, whereas verified project code and GUI inference apply StandardScaler before PCA. The implementation order is authoritative.
