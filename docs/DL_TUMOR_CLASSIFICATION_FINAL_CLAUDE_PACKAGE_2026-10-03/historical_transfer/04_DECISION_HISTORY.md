# Decision History

## D1 — Use a recent hybrid paper as the official base

**Decision:** Implement the 2026 CNN+LSTM+SVM paper rather than the 2022 classical K-means/DWT/SVM paper.

**Context:** The project began on 16 July with four classical directions: Mandle et al. (K-means/DWT/PCA/kernel SVM), Rao/Karunakara (Python/advanced SVM), Saad et al. (detection/features/GUI), and Malathi/Sinthia (hybrid clustering/backpropagation). The academic requirement later called for a 2023-or-later base paper; the user also wanted a modern deep-learning project.

**Options considered:** 2022 classical segmentation/handcrafted-feature pipeline; 2023 Otsu/GLCM/DWT/SVM; 2026 hybrid CNN+LSTM+SVM.

**Reasoning:** The 2022 paper was too old to be the official base. The hybrid paper described five backbone alternatives and matched the intended deep-learning scope. Its full reference matrix was five backbones by four optimizers (20 runs), but the project only verified the RMSprop branch.

**Implementation:** Yes.

**Current status:** Authoritative selection, but “faithful/exact reproduction” is not supported: optimizer scope, batch size, epochs, augmentation, VGG freezing, tensor interpretation, preprocessing order, and split integrity differ or remain uncertain.

## D2 — Treat the task as four-class tumor-type classification

**Decision:** Classes are glioma, meningioma, no tumor, and pituitary. Do not relabel them as benign/malignant.

**Reasoning:** Tumor type is not equivalent to biological behavior or grade; the proposed shortcut would be medically inaccurate.

**Implementation:** Yes, consistently encoded in code and results.

**Current status:** Authoritative.

## D3 — Evaluate backbones independently, not as an ensemble

**Decision:** Train five separate CNN-LSTM-SVM pipelines.

**Context:** Multiple CNN names were initially easy to misread as an ensemble.

**Reasoning:** The base paper compares alternative feature extractors; it does not combine their predictions.

**Implementation:** Yes.

**Current status:** Authoritative. A future ensemble remains only a proposal.

## D4 — Abandon TensorFlow and use PyTorch

**Decision:** Use PyTorch under WSL2.

**Context:** TensorFlow encountered `INVALID_PTX` / `sm_120` compatibility failure on the RTX 5070-class GPU.

**Options:** Continue forcing TensorFlow/nightly builds; use CPU; migrate to PyTorch.

**Reasoning:** PyTorch with CUDA 12.8 worked reliably and avoided blocking the project.

**Implementation:** Yes.

**Current status:** Authoritative.

## D5 — Build and validate the architecture incrementally

**Decision:** Verify CNN shapes, sequence conversion, LSTM, post-LSTM layers, and a training step before full training.

**Reasoning:** The user wanted genuine understanding, and tensor-shape errors in a hybrid architecture are easy to hide in monolithic code.

**Implementation:** Yes, through dedicated test scripts.

**Current status:** Good engineering history; tests should be retained and converted into an automated suite where possible.

## D6 — Define the LSTM input as spatial positions

**Decision:** Convert `[B,C,H,W]` to `[B,H×W,C]` and use the last LSTM output.

**Reasoning:** This is the implementable interpretation of the paper for single 2D images. It avoids the false claim that the LSTM sees time or ordered MRI slices.

**Implementation:** Yes.

**Current status:** Authoritative for this codebase; still a methodological limitation because spatial raster order is artificial.

The paper sometimes described the LSTM as modeling dependencies across MRI slices. That language is not valid for this single-2D-image implementation and is a documented paper-to-project deviation.

## D7 — Use a train-only scaler and 64-component PCA before a linear SVM

**Decision:** Extract 128-D learned features, fit StandardScaler on training only, transform validation/test, fit PCA(64) on standardized training only, and train a linear SVM with `C=1`.

**Options:** Neural head only; kernel/nonlinear SVM; different PCA size; no PCA.

**Reasoning:** This matched the chosen hybrid design and supplied stable, compact features. Leakage through preprocessing was avoided by fitting only on training data.

**Implementation:** Yes.

**Current status:** The scaler→PCA order is verified in current code and differs from some early/paper diagrams that placed PCA first. Current `evaluate_backbone.py` specifies `probability=False`, but preserved VGG19 and ResNet152 logs contain scikit-learn's probability-parameter deprecation warning, indicating those runtime SVMs may have used `probability=True`. Inspect every saved `.joblib`; do not reconstruct artifact settings from current source alone.

## D8 — Freeze VGG backbones after collapse

**Decision:** Freeze VGG16 and VGG19 convolutional parameters; train only the recurrent/post-LSTM/head components.

**Context:** Fully unfrozen VGG16 collapsed around 27.9% validation accuracy. Learned features became effectively constant and PCA emitted NaNs, while direct pretrained activations were healthy, narrowing the failure to the trained configuration rather than the raw backbone.

**Options:** Tune lower learning rates, partial unfreezing, gradient clipping, optimizer changes, or full freezing.

**Reasoning:** Freezing produced a stable, reproducible transfer-learning baseline within project time and hardware constraints.

**Implementation:** Yes.

**Current status:** Authoritative historical configuration. VGG16 was resumed with `--epochs 30 --resume`; it completed 19 epochs and peaked at stored epoch 16 (displayed epoch 17). The base paper's freeze policy was not specified. If clean reruns change this policy, the change must be preregistered and applied transparently.

## D9 — Keep deterministic saved splits for backbone comparison

**Decision:** Save seed-42 stratified indices and use the same split for all models.

**Reasoning:** It improves within-project comparability.

**Implementation:** Yes.

**Current status:** Superseded. The split is contaminated by cross-split duplicates and must be replaced with duplicate-group-aware splitting.

## D10 — Add Grad-CAM only after baseline completion

**Decision:** Treat Grad-CAM as an extension, not part of the reproduced base methodology.

**Reasoning:** First establish the paper-inspired baseline; do not blur implementation and novelty. Grad-CAM cannot directly explain an external scikit-learn SVM.

**Implementation:** Yes, targeting the neural head.

**Current status:** Authoritative interpretation.

## D11 — Add final-SVM occlusion sensitivity

**Decision:** Complement Grad-CAM with perturbation analysis that reruns the complete final classifier path.

**Reasoning:** This directly probes the SVM decision, although it is coarse and computationally slower.

**Implementation:** Yes.

**Current status:** Implemented extension; qualitative evidence only.

## D12 — Do not label SVM scores as probabilities

**Decision:** Display standardized SVM decision values as relative scores and disclose that they are not calibrated probabilities.

**Context:** GUI inference uses `decision_function` followed by softmax-like normalization. Current code says `probability=False`, while some historical run logs suggest `probability=True`; neither makes the displayed transformed margins calibrated probabilities.

**Implementation:** Yes in GUI documentation/code.

**Current status:** Authoritative.

## D13 — Run a duplicate/leakage audit before lock

**Decision:** Audit train↔validation, train↔test, and validation↔test for exact and near duplicates.

**Reasoning:** It was the last major threat to scientific validity not covered by prior evaluation.

**Implementation:** Yes.

**Evidence:** Audit status `MATERIAL_DUPLICATION`; 562 material cross-split pairs.

**Current status:** The result invalidates the planned lock and supersedes the old split decision.

## D14 — Rebuild the experimental split and rerun all models

**Decision:** The project cannot be locked until duplicate-group-aware splitting, retraining, and reevaluation are complete.

**Options:** Ignore duplicates; remove only exact duplicates from test; report old scores with a caveat; rebuild groups and rerun.

**Reasoning:** Only a group-aware split prevents the same or near-identical image content from crossing evaluation boundaries. Selective test cleaning after viewing results risks bias.

**Implementation:** Not yet completed.

**Current status:** Blocking and authoritative.

## D15 — Do not silently add DenseNet121 to the verified comparison

**Decision:** Treat DenseNet121 as an unresolved documentation contradiction, not a completed sixth experiment.

**Context:** A pre-audit checkpoint said DenseNet121 was implemented and evaluated. The supplied `model_cnn_lstm.py`, manifest, and five final result files support only VGG16, VGG19, ResNet50, ResNet152, and EfficientNet-B3.

**Current status:** Requires repository/result recovery. Do not call DenseNet rejected, completed, or final without artifacts.

## D16 — Supersede the pre-audit “locked” interpretation

**Decision:** The later leakage audit governs the scientific status.

**Context:** A pre-audit checkpoint described dataset-level accuracy as valid and the interpretation as locked. The 20 September audit subsequently found material cross-split duplication.

**Current status:** The older lock language is historical and explicitly superseded; recorded metrics remain real outputs but not independent-generalization estimates.
