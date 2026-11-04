# CONVERSATION HISTORY RECOVERY NOTES

## Purpose
This addendum records project-history details recovered from prior ChatGPT project conversations in addition to the older transfer package and current attached artifacts. It exists to reduce the chance that a historically important decision disappears when the project is moved to Claude.

These notes are historical context. If they conflict with the final machine-readable outputs or `01_MASTER_FINAL_PROJECT_CONTEXT.md`, the final sources win.

## July 2026 — original classical-image-processing concept
The project initially explored a conventional brain-MRI segmentation/classification workflow built around K-Means segmentation, handcrafted texture/shape features, dimensionality reduction and an RBF-SVM. Early literature candidates included Mandle et al., Rao/Karunakara, Saad et al., and Malathi/Sinthia.

A critical label-integrity decision was made early: do not casually convert dataset labels such as HGG/LGG or tumor-type labels into “benign/malignant” unless the dataset and pathology source actually support that label mapping. This prevented a scientifically invalid framing.

## August 2026 — recent base-paper pivot
A 2026 Springer hybrid CNN+LSTM+SVM paper became the implementation anchor. The intended pipeline was progressively clarified as CNN feature map → spatial sequence → LSTM → learned dense features → StandardScaler → PCA → linear SVM. The project chose to implement the core hybrid architecture first and defer novelty/explainability until the baseline worked.

The paper's broader design included five backbones and four optimizers (20 combinations), 30 epochs, LR 1e-4 and batch 32. The project later fixed RMSprop and used practical implementation settings such as batch 8. LSTM hidden size 256 and Dense128 were implementation choices rather than values that should be misrepresented as fully specified by the paper.

## 28 August 2026 — TensorFlow GPU troubleshooting
Under WSL/Ubuntu, TensorFlow initially had GPU/toolchain problems on the RTX 50-series/Blackwell environment. Diagnostics included GPU detection inconsistencies and later an `INVALID_PTX` / `sm_120` incompatibility in realistic workloads. A nightly TensorFlow route was explored but abandoned as unreliable. PyTorch became the authoritative framework.

## Early September 2026 — staged ResNet152 construction
The model was intentionally constructed in small tests before full training:
- pretrained ResNet152 feature extraction;
- `[B,2048,7,7]` feature map verification;
- reshape to `[B,49,2048]`;
- LSTM output `[B,49,256]`;
- last spatial-token output;
- BN/Dropout/Dense128/ReLU;
- temporary 4-class neural head;
- forward/backward training-step check;
- real-MRI batch check.

The key conceptual clarification was that the LSTM is not modeling time or MRI slices; it processes a sequence of spatial positions from one 2D CNN feature map.

## 6–12 September 2026 — generic five-backbone framework and VGG failure
The working ResNet152 design was generalized to VGG16, VGG19, ResNet50, ResNet152 and EfficientNet-B3. During this period, fully unfrozen VGG training collapsed to near-chance validation performance and degenerate learned features/PCA behavior. Diagnostics showed the pretrained convolutional activations themselves were not dead. The practical fix was to freeze VGG convolutional backbones while training the LSTM/post-CNN layers. ResNets/EfficientNet remained fine-tuned.

This created a documented training-policy asymmetry that must remain visible in final reporting.

## 12–20 September 2026 — GUI and explainability
The Streamlit GUI was built around the actual saved hybrid artifacts. It distinguished the final SVM output from the neural-head prediction and explicitly avoided presenting SVM margins/softmax views as clinically calibrated probabilities.

Grad-CAM was deliberately frozen before formal evaluation. The user explicitly required it to be treated as a contribution map, not a segmentation tool. A fixed stratified 20-image ResNet152 evaluation then demonstrated frequent off-lesion activation despite high classification accuracy. Final-SVM occlusion was added because Grad-CAM explains the neural head, not the external SVM.

## 20 September 2026 — duplicate audit changes the interpretation
Immediately before project lock, duplicate/leakage auditing was identified as the one remaining scientifically meaningful verification. The audit returned `MATERIAL_DUPLICATION` rather than a pass. This superseded earlier language that had treated the historical validation/test numbers as ordinary independent generalization estimates.

An immediate post-audit plan treated duplicate-group-aware re-splitting/retraining as blocking. Later project decisions changed that status: the existing split results were preserved as historical contaminated-split outputs, a 1,061-image `material_clean` post-hoc evaluation was added, and structured external validation was performed. Group-aware full retraining became an optional stronger future remediation rather than a required step for closing the mini-project.

## Late September–2 October 2026 — Phase2 completion
Phase2A completed fixed 30-epoch baselines for all five backbones. Phase2B then executed seven one-factor variants per backbone: three learning rates, two alternative LSTM widths and two alternative dropout values. The final queue state was 35/35 changed configurations available; one VGG16 LR1e-5 result was already complete and skipped by the batch runner while 34 were newly executed.

The project then evaluated both retained states (best-within-30 and epoch30) through the full downstream scaler→PCA→SVM path, yielding 10 Phase2A and 70 Phase2B retained-state evaluations.

## 2–3 October 2026 — external validation and web stress test
The final external protocol intentionally froze 15 pipelines before external-result inspection: five Phase1 representatives, five Phase2A representatives and five Phase2B representatives. BDNeuro V8 was selected, audited against Nickparvar, exact decoded-pixel overlaps removed, and evaluated both as the full 2,644-image exact-overlap-clean cohort and as a nested 1,648-image balanced sensitivity subset.

The 40-image web stress test was subsequently designed to answer the guide's practical “what happens on a random MRI outside the training dataset?” question in a controlled way. Images were source-verified before inference, audited for overlap, frozen, and then passed through all 15 pipelines. One initial normal image was visually confirmed as the same underlying Nickparvar slice after pHash distance 0 and was replaced before any predictions were viewed. A meningioma pHash-distance-6 candidate was visually confirmed to be a different MRI and retained.

## Final interpretive endpoint
The project now has a deliberately layered evidence chain:
1. historical internal validation/test performance;
2. post-hoc `material_clean` internal test evaluation;
3. structured cross-dataset BDNeuro exact-overlap-clean validation;
4. class-balanced BDNeuro sensitivity analysis;
5. heterogeneous source-verified web stress testing.

The final project representative is ResNet152 Phase2A, best retained state at epoch 10. This does not erase architecture-specific findings: ResNet50 Phase2B transferred particularly well to BDNeuro, and the web stress test was harsh for all models. The final candidate is a project-level synthesis, not an externally tuned model.
