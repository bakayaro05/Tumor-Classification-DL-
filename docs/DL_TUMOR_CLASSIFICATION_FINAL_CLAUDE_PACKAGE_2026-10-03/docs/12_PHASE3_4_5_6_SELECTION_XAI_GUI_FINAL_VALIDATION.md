# PHASES 3, 4, 5 AND 6 — FINAL SYNTHESIS, XAI, GUI, AND FINAL VALIDATION

## Phase 3 — Comparative analysis and final candidate selection

### Comparative evidence used
- Phase1 neural head
- Phase1 PCA+SVM
- material_clean
- Phase2A fixed horizon
- Phase2B sensitivity
- 15-model BDNeuro full
- 15-model BDNeuro balanced
- 15-model web stress test

### Final selected candidate
**ResNet152 Phase2A standardized 30-epoch, best retained state epoch10.**

Configuration:
- RMSprop
- LR1e-4
- LSTM256
- dropout0.3
- Dense128
- StandardScaler
- PCA64
- linear SVM C=1

Evidence:
- internal SVM validation 98.60%
- internal macro F1 98.55%
- BDNeuro full 88.31% / macro F1 87.85%
- BDNeuro balanced 90.47% / macro F1 90.50%
- smallest BDNeuro generalization drop among frozen candidates

Web stress: 37.5%. This exposes severe domain-shift limits, but the small exploratory web set does not override larger structured external evidence.

Phase3 status: **COMPLETE**.

---

## Phase 4 — Explainability

### Grad-CAM
- ResNet152
- 20 images
- 5/class
- seed42
- Layer3, Layer4, fused
- 15 tumor cases rated

Layer3: 1 Good / 9 Partial / 5 Poor  
Layer4: 3 Good / 2 Partial / 10 Poor  
Fused: 2 Good / 6 Partial / 7 Poor

Layer3 chosen most relevant 9/15; Layer4 6/15; fused 0/15.

### Final-SVM occlusion
- 11 cases
- 9 tumor / 2 no-tumor
- patch32, stride16
- 169 occlusions/image

Tumor localization: Good2 / Partial2 / Poor5.

1,859 perturbations; 58 flips; pooled 3.12%.

### Scientific conclusion
Explainability is qualitatively inconsistent. Correct predictions can rely on off-lesion or peripheral image features. XAI does not validate tumor localization.

Phase4 status: **COMPLETE**.

---

## Phase 5 — Prototype application

Streamlit application includes:
- upload raster MRI;
- five-backbone model selection;
- final SVM prediction;
- neural-head comparison;
- agreement warning;
- relative class-score visualization;
- metrics and confusion matrices;
- model comparison;
- Grad-CAM;
- SVM occlusion;
- methodology view and readiness checks.

Constraints:
- no DICOM;
- no on-upload refitting;
- no medical-device claim;
- SVM relative scores are not calibrated probabilities.

Phase5 status: **COMPLETE**.

---

## Phase 6 — Final validation/report/demonstration

Validation hierarchy:
1. historical internal validation/test;
2. material_clean post-hoc internal test;
3. BDNeuro exact-overlap-clean external full;
4. BDNeuro balanced sensitivity;
5. web in-the-wild stress test.

Reports:
- comprehensive project analysis PDF;
- dedicated web stress-test PDF;
- this final structured transfer package.

Demonstration:
- Streamlit GUI.

Phase6 status: **COMPLETE for experimental project work**.

---

## Final end-state interpretation
The project began as a high-accuracy classifier implementation but matured into a generalization/reliability study. Its strongest contribution is not a single headline percentage; it is the layered evidence showing that performance degrades as evaluation moves from contaminated internal data to structured external data and then to severe in-the-wild domain shift.
