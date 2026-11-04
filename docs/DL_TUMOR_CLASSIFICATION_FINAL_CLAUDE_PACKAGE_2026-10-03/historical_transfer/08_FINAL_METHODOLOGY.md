# Final Methodology File — Provisional Post-Audit Remediation Protocol

**Status: NOT FINAL OR LOCKED.** This document proposes the methodology for the next scientifically valid run. The split policy and duplicate-candidate adjudication remain undecided, and exact current augmentation/source details must be recovered. Existing metrics do not become clean merely by adopting this document; all models must be retrained.

## 1. Study design

Comparative supervised classification of single 2D MRI images into four classes. Five CNN backbones are evaluated independently within the same CNN-LSTM-feature/SVM pipeline. This is not an ensemble and not a segmentation study.

## 2. Dataset inventory and provenance

Use the same 7,023-image Kaggle dataset version only if its file inventory can be frozen and checksummed. Record source URL/version, download date if known, directory/file counts, class mapping, image modes/sizes, and cryptographic manifest. Do not assume the original Training/Testing folders are independent.

## 3. Duplicate grouping before splitting

1. Compute exact file hashes and normalized-pixel hashes for every image.
2. Generate perceptual candidates using the documented pHash/dHash thresholds.
3. Preserve the audit's high-confidence material relations and manually adjudicate the 481 review candidates where feasible.
4. Build connected components: if A matches B and B matches C, all three belong to one indivisible group even if A and C are not directly matched.
5. Check class-label consistency within every group. Conflicting labels require manual review and must not be silently resolved.
6. Split groups—not individual images—so no group crosses train, validation, or test.
7. Run the audit again and require zero material cross-split relations under the declared protocol.

If the original official test folder is preserved as a semantic holdout, all duplicate groups touching it must be removed from the training/validation candidate pool, and any internal duplicate groups within test must be documented. If a completely new split is used, state clearly that results are not directly the dataset's original folder evaluation.

## 4. Split policy

Use a fixed seed and a group-aware class-stratified procedure. Report image counts, group counts, class counts, discarded/quarantined images, and any deviation from target proportions. Freeze the split before training. Never tune using the final test set.

Patient-level independence cannot be claimed without patient identifiers. Use the narrow wording: **“No material image-level cross-split duplicates were detected by the implemented audit.”**

## 5. Image preprocessing

- Load supported dataset images and convert to RGB.
- Use a fixed pretrained-CNN input size consistent with the actual data pipeline (historically 224×224).
- Apply ImageNet normalization.
- Apply stochastic augmentation only to training data. Preserve the exact current `src/data_pipeline_torch.py` transform in the repository and report each parameter.
- Use one deterministic evaluation transform for validation, test, saved feature extraction, and GUI inference.

The paper described horizontal/vertical flips, brightness ±0.1, contrast and saturation 0.8–1.2, and hue ±0.1. These are paper settings, not proof of the exact project transform. Recover `src/data_pipeline_torch.py` before freezing the rerun protocol.

## 6. Neural feature learner

For each backbone separately:

1. Initialize ImageNet weights.
2. Remove the pretrained classifier/global pooling as appropriate to retain the final spatial feature map.
3. Obtain `[B,C,H,W]`.
4. Permute and reshape to `[B,H×W,C]`.
5. Process with a one-layer LSTM, hidden size 256, batch-first.
6. Select the final sequence position.
7. Apply BatchNorm1d(256), Dropout(0.3), Linear(256→128), and ReLU.
8. Apply a temporary Linear(128→4) head for neural training.

Backbone channels are VGG16/VGG19 512, ResNet50/ResNet152 2,048, and EfficientNet-B3 1,536. Historically VGG convolutional backbones were frozen for stability. Preserve this policy for the clean rerun unless a new tuning protocol is declared before seeing test results.

## 7. Neural training

- Loss: four-class CrossEntropyLoss.
- Optimizer: RMSprop.
- Initial learning rate: `1e-4`.
- Seed: 42 for the direct historical comparison; preferably add multiple preregistered seeds after the core rerun.
- Batch size: 8 unless hardware or a preregistered change requires otherwise.
- Monitor validation loss and accuracy; save only the best validation checkpoint.
- Use early stopping with documented patience.
- Record epochs, wall time, learning curves, parameter counts, frozen/trainable counts, software versions, and hardware.
- Never resume from contaminated-split weights for the valid rerun.

## 8. Deterministic deep-feature extraction

Load the selected best checkpoint, set evaluation mode, disable gradients, and run train/validation/test through the deterministic transform with no shuffling. Save 128-D features and labels with sample IDs in preserved order. Verify finiteness, shapes, counts, and class mapping.

## 9. Classical classification stage

1. Fit StandardScaler on training features only.
2. Transform training, validation, and test.
3. Fit PCA with 64 components on scaled training only.
4. Transform all splits.
5. Before rerunning, choose and freeze the SVM probability setting. `probability=False` is the current-source proposal and is sufficient because inference uses `decision_function`; historical logs conflict with current source, so first inspect every old `.joblib` rather than retroactively relabeling it.
6. Use validation for any declared development decisions.
7. Evaluate the frozen pipeline once on final test.

Persist checkpoint, scaler, PCA, SVM, feature arrays, split manifest, and results for each backbone.

## 10. Outcomes

Minimum reporting:

- accuracy;
- macro and weighted precision, recall, and F1;
- per-class precision, recall, F1, and support;
- confusion matrix;
- neural validation performance and best epoch;
- PCA explained variance;
- training/inference time and model size if available.

Sensitivity and specificity may be reported only if explicitly computed one-vs-rest with a stated averaging rule. ROC-AUC may be reported only if class scores/probabilities and the multiclass procedure are documented. Do not invent these metrics from accuracy.

Prefer bootstrap confidence intervals or repeated-seed distributions in the clean rerun. Statistical comparisons must account for paired predictions.

## 11. Explainability

- Grad-CAM may target ResNet layer 3/layer 4/fused neural-head predictions.
- Explicitly state that it does not directly explain the scikit-learn SVM.
- Occlusion sensitivity may target the final SVM decision score using documented patch size, stride, fill value, and target class.
- Evaluate explanations on the clean test split only after classifier lock.
- Use expert masks/ratings where possible; keep qualitative ratings separate from classification metrics.
- Never call saliency maps segmentation or proof of causal clinical reasoning.

## 12. GUI and deployment

The GUI must load only clean-rerun artifacts, reuse the deterministic transform, validate class order and artifact compatibility, show the final SVM output, label noncalibrated scores accurately, and retain the research-prototype disclaimer. DICOM support remains outside current scope.

## 13. Reproducibility and audit trail

Regenerate a complete manifest after the clean rerun. Save the split-group file, audit protocol/results, configuration, environment lock, random seeds, Git commit, result JSON, evaluation predictions, and figures. Keep historical contaminated results in a clearly named archive rather than deleting them.
