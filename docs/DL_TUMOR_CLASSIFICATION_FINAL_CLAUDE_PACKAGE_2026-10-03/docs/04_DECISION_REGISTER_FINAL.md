# FINAL DECISION REGISTER

This file records major project decisions, including decisions that were later superseded.

## D1 — Replace the early classical K-means/SVM direction with a recent deep-learning base paper
**Context:** recent-paper mini-project requirement.  
**Decision:** use the 2026 Springer CNN+LSTM+SVM paper.  
**Status:** authoritative.

## D2 — Four-class tumor-type classification, not benign/malignant
**Decision:** glioma, meningioma, notumor, pituitary.  
**Reason:** these are the actual dataset labels; tumor type is not equivalent to behavior/grade.  
**Status:** authoritative.

## D3 — Five backbones are alternatives, not an ensemble
**Decision:** train/evaluate five separate pipelines.  
**Status:** authoritative.

## D4 — Abandon TensorFlow for PyTorch
**Trigger:** RTX 5070/Blackwell `INVALID_PTX` / `sm_120` incompatibility.  
**Decision:** PyTorch under WSL2.  
**Status:** authoritative.

## D5 — Build the hybrid network incrementally
**Decision:** separate tensor-shape/CNN/LSTM/post-LSTM/training-step tests before full training.  
**Reason:** reduce hidden shape errors and help the user understand the model.  
**Status:** completed.

## D6 — LSTM consumes spatial tokens from one 2D image
**Decision:** `[B,C,H,W] → [B,H×W,C]`; use last LSTM output.  
**Rejected wording:** LSTM models temporal sequence or MRI slice order.  
**Status:** authoritative.

## D7 — Use training-only StandardScaler → PCA64 → linear SVM C=1
**Reason:** implement the hybrid paper concept without leakage through scaler/PCA fitting.  
**Status:** authoritative.

## D8 — Freeze VGG16/VGG19 after VGG collapse
**Trigger:** fully unfrozen VGG16 ~28% validation, near-constant learned features, PCA NaNs.  
**Decision:** freeze VGG CNN backbone; train recurrent/post-CNN layers.  
**Status:** authoritative historical configuration.

## D9 — Reuse one deterministic seed42 split across backbones
**Reason:** fair within-project comparison.  
**Later caveat:** split was found contaminated by duplicate groups.  
**Status:** historical but still the basis of completed experiments.

## D10 — Add Grad-CAM only after baseline completion
**Reason:** do not blur faithful implementation and extension.  
**Status:** completed.

## D11 — Grad-CAM explains neural head, not scikit-learn SVM
**Decision:** state target explicitly.  
**Status:** authoritative.

## D12 — Add final-SVM occlusion
**Reason:** perturb the complete deployed inference path.  
**Status:** completed.

## D13 — Do not call SVM relative scores calibrated probabilities
**Status:** authoritative GUI/reporting rule.

## D14 — Run duplicate/leakage audit before scientific lock
**Outcome:** `MATERIAL_DUPLICATION`.  
**Status:** completed and foundational.

## D15 — Withdraw pre-audit clean-generalization interpretation
**Decision:** old metrics remain recorded outputs but require leakage caveat.  
**Status:** authoritative.

## D16 — Do not add DenseNet121 without artifacts
A prior conversation checkpoint mentioned DenseNet121, but verified supported-backbone source/result structure contains only five backbones.  
**Status:** DenseNet121 not part of final verified project.

## D17 — Create material_clean post-hoc evaluation
**Decision:** exclude 250 materially implicated official-test images, leaving 1,061.  
**Reason:** quantify whether obvious cross-split overlap alone explained the high Phase1 test scores.  
**Caveat:** not equivalent to retraining on a group-separated split.  
**Status:** completed.

## D18 — Reclassify full duplicate-group-aware retraining as optional future strengthening
**History:** immediately after the audit, older transfer docs treated it as blocking.  
**Later reasoning:** the project could still transparently report contaminated internal results, add a narrow material_clean check, and obtain stronger out-of-source evidence by audited external validation.  
**Final decision:** group-aware retraining is optional unless claiming clean internal independence.  
**Status:** authoritative final planning decision.

## D19 — Run Phase2A fixed horizon before hyperparameter study
**Reason:** obtain comparable convergence/stability baselines.  
**Status:** completed.

## D20 — Phase2B uses one-factor-at-a-time sensitivity, not a black-box search
**Decision:** vary LR, LSTM width and dropout one family at a time.  
**Status:** completed.

## D21 — Evaluate both retained states downstream
**Decision:** for Phase2A/2B evaluate best-within30 and epoch30 through scaler/PCA/SVM.  
**Reason:** neural checkpoint quality and downstream SVM quality need not rank states identically.  
**Status:** 10/10 Phase2A + 70/70 Phase2B completed.

## D22 — External comparison uses 15 frozen pipelines
**Decision:** five Phase1 + five Phase2A + five Phase2B.  
**Reason:** architecture-wise comparison across project evolution.  
**Status:** completed.

## D23 — Freeze external candidates before seeing external results
**Implementation:** SHA256 manifest over checkpoint/scaler/PCA/SVM/internal result artifacts.  
**Rule:** do not regenerate/reselect after external results.  
**Status:** authoritative.

## D24 — Use BDNeuro V8 as structured external candidate
**Reason:** four-class compatibility and independent documented source context.  
**Status:** completed.

## D25 — Direct exact-pixel identity overrides provenance narrative
**Trigger:** 3,297 unique BDNeuro images exactly matched development images at decoded-pixel level.  
**Decision:** exclude confirmed overlaps.  
**Status:** authoritative.

## D26 — Do not discard non-identical MRIs solely because pHash is similar
**Reason:** anatomical similarity can produce small Hamming distances; pHash is screening evidence, not identity proof.  
**Status:** authoritative.

## D27 — Full exact-clean BDNeuro cohort is primary; balanced subset is secondary
**Primary:** n=2,644 natural distribution.  
**Sensitivity:** n=1,648, 412/class.  
**Status:** authoritative.

## D28 — No tuning/refitting on BDNeuro
**Status:** authoritative.

## D29 — Add a separate in-the-wild web stress test
**Reason:** answer practical behavior on heterogeneous source-verified web MRIs.  
**Status:** completed.

## D30 — Lock web cohort before any predictions
**Rule:** after predictions, do not replace difficult/wrong images.  
**Status:** followed.

## D31 — pHash0 web candidate requires visual review even when exact-pixel hash differs
**Outcome:** NOR_01 visibly same underlying Nickparvar slice; rejected and replaced before inference.  
**MEN_02:** pHash6 but clearly different scan; retained.  
**Status:** completed.

## D32 — Treat web test as exploratory stress test, not clinical validation
**Reason:** small convenience sample, heterogeneous publication images, n=40.  
**Status:** authoritative reporting rule.

## D33 — Final project candidate is ResNet152 Phase2A
**Configuration:** LR1e-4, LSTM256, dropout0.3, best state epoch10.  
**Reason:** strongest overall structured external evidence and smallest BDNeuro generalization gaps.  
**Caveat:** not best on the tiny web test; web n=40 does not override primary external validation.  
**Status:** final.

## D34 — Do not retrain based on external/web outcomes
**Reason:** would convert evaluation sets into tuning sets.  
**Status:** authoritative final lock rule.
