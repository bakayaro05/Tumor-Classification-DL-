# Hostile Review Corrections

## Material defects found in the first transfer

1. It began the journey too late and omitted the 16 July four-paper classical phase, its proposed handcrafted pipeline, arbitrary K-means cluster-ID warning, and rejection of tumor-type→benign/malignant shortcuts.
2. It overstated paper fidelity. The paper described five backbones × four optimizers, 30 epochs, batch 32, and broader augmentation; the verified project comparison used RMSprop only, batch 8, generally shorter early-stopped runs, and project-specific decisions.
3. It omitted the dataset inspection: 7,023/7,023 readable, 0 corrupt; 3,093 `L`, 3,926 `RGB`, 3 `RGBA`, 1 `P`; varying sizes; converted to RGB and historically 224×224.
4. It compressed VGG debugging to “collapse.” The history records constant learned features, PCA NaNs, healthy pretrained activations, freezing, resume behavior, and zero-based saved versus one-based displayed epochs.
5. It underspecified explainability limits: exact Grad-CAM ratings, head agreement/shared error, deliberate occlusion selection, class-flip counts, and independent heatmap normalization are now preserved.
6. It silently omitted DenseNet121 despite one pre-audit claim that it was evaluated. Supplied source, manifest, and five final-result files do not support that claim. The contradiction remains open.
7. It stated SVM `probability=False` too confidently. Current source says False, while preserved VGG19/ResNet152 logs suggest some runtime models used True. GUI inference uses `decision_function`, so displayed relative scores remain noncalibrated. Inspect each serialized artifact.
8. It treated the remediation method as final. Split policy, duplicate-candidate adjudication, exact transforms, and artifact/source reconciliation are not locked.
9. It did not clearly show that the later leakage audit superseded the pre-audit “valid/locked” interpretation.

## Intentionally unresolved

- Whether DenseNet121 artifacts exist outside the supplied snapshot.
- The `.probability` flag of each saved standardized SVM.
- Exact project augmentation in the missing data-pipeline source.
- Final duplicate-group adjudication and split policy.
- Complete repository/environment/dataset provenance and corrected final report/PPT/abstract.

No unresolved item is marked final, completed, rejected, or absent without supporting evidence.
