# MASTER RESULTS, FINAL CANDIDATE, AND PROJECT CONCLUSIONS

## 1. Result hierarchy

The project has four distinct evidence layers:

1. **Historical internal split results** — high performance, but contaminated by cross-split duplicate groups.
2. **material_clean post-hoc test** — removes directly implicated official-test images, but does not repair training contamination.
3. **BDNeuro exact-overlap-clean external validation** — primary structured external evidence.
4. **40-image web stress test** — exploratory severe-domain-shift evidence.

These must not be collapsed into a single “accuracy.”

---

## 2. Phase1 historical internal results

Neural-head test → PCA+SVM test:

| Backbone | Neural test | PCA+SVM test | Test change |
| --- | ---: | ---: | ---: |
| VGG16 | 93.67% | 94.97% | +1.30 pp |
| VGG19 | 93.14% | 93.52% | +0.38 pp |
| ResNet50 | 97.33% | 98.09% | +0.76 pp |
| ResNet152 | 98.17% | 98.40% | +0.23 pp |
| EfficientNet-B3 | 98.02% | 98.47% | +0.45 pp |

Interpretation: downstream classifier improved the historical test result in all five, but validation did not uniformly improve.

---

## 3. material_clean results

| Backbone | Accuracy | Macro F1 |
| --- | ---: | ---: |
| VGG16 | 94.816% | 94.302% |
| VGG19 | 93.025% | 92.415% |
| ResNet50 | 98.303% | 98.134% |
| ResNet152 | 98.398% | 98.315% |
| EfficientNet-B3 | 98.680% | 98.511% |

Interpretation: direct test-image overlap removal did not collapse performance, but this remains post-hoc.

---

## 4. Phase2 internal full-hybrid representatives

### Phase2A
- VGG16: 95.888% SVM val
- VGG19: 95.626%
- ResNet50: 98.688%
- ResNet152: 98.600%
- EfficientNet-B3: 99.300%

### Phase2B changed-config representatives
- VGG16 LR5e-4: 96.500%
- VGG19 dropout0.5 epoch30: 95.538%
- ResNet50 dropout0.2: 99.125%
- ResNet152 LSTM512: 99.300%
- EfficientNet-B3 LSTM512: 99.388%

Internal tuning gains were small and did not reliably predict external behavior.

---

## 5. BDNeuro primary full exact-clean external results

Top structured external result:
**ResNet152 Phase2A = 88.31% accuracy / 87.85% macro F1.**

Other notable:
- EfficientNet-B3 Phase2A 86.65%
- EfficientNet-B3 Phase2B 86.57%
- ResNet152 Phase2B 86.54%
- ResNet50 Phase2B 86.04%

Stage means:
- Phase1 79.77%
- Phase2A 83.21%
- Phase2B 82.34%

Phase2A generalized best on average.

---

## 6. BDNeuro balanced sensitivity results

Top:
**ResNet152 Phase2A = 90.47% accuracy / 90.50% macro F1.**

Stage means:
- Phase1 81.80%
- Phase2A 85.39%
- Phase2B 84.33%

The balanced subset generally raised accuracy because class prevalence no longer weighted the natural cohort.

---

## 7. Web stress test

Range: **25–42.5%**.

Highest accuracy:
- ResNet50 Phase2A 42.5%
- EfficientNet-B3 Phase2A 42.5%

Stage means:
- Phase1 33%
- Phase2A 38%
- Phase2B 35%

Mean class recall across all 15:
- glioma 12.7%
- meningioma 30.7%
- notumor 90.7%
- pituitary 7.3%

Dominant pattern: severe `notumor` overprediction under strong domain shift.

---

## 8. Cross-backbone structured-external view

| Backbone | Best BDNeuro full stage | BD full acc | Balanced acc same stage | Best web stage | Best web acc |
| --- | --- | --- | --- | --- | --- |
| VGG16 | Phase2A | 80.07% | 80.70% | Phase2A/2B | 30.0% |
| VGG19 | Phase2A | 79.27% | 81.13% | Phase2A | 37.5% |
| ResNet50 | Phase2B | 86.04% | 87.86% | Phase2A | 42.5% |
| ResNet152 | Phase2A | 88.31% | 90.47% | Phase2A | 37.5% |
| EfficientNet-B3 | Phase2A | 86.65% | 87.86% | Phase1/2A | 42.5% |

Note: web “best” values are descriptive only; n=40 means each image shifts accuracy by 2.5 pp.

---

## 9. Final selected candidate

# ResNet152 Phase2A standardized 30-epoch — epoch10 best state

Configuration:
- CNN backbone: ResNet152
- RMSprop
- LR1e-4
- LSTM256
- dropout0.3
- Dense128
- StandardScaler
- PCA64
- linear SVM C=1

Evidence:
- internal full-hybrid val: **98.60%**
- internal macro F1: **98.55%**
- BDNeuro full: **88.31%**
- BDNeuro full macro F1: **87.85%**
- BDNeuro balanced: **90.47%**
- BDNeuro balanced macro F1: **90.50%**
- full external gap: **-10.29 pp**
- balanced external gap: **-8.13 pp**
- web stress: **37.5%**

### Why this model
It gives the strongest combined structured-external evidence and the smallest BDNeuro generalization gaps. It is therefore the final project representative.

### Why not choose based on web 42.5%
The web set is a tiny, intentionally heterogeneous n=40 stress sample. It is useful for robustness diagnosis, not fine model selection.

### Important lock rule
This retrospective selection did not lead to any external-data retraining or refitting.

---

## 10. Final project conclusions

### Conclusion A — the hybrid architecture works internally
All five CNN-LSTM-PCA-SVM pipelines can achieve high internal performance.

### Conclusion B — PCA+SVM can help, but not universally across all splits
Phase1 test improved for all five, while validation behavior was mixed.

### Conclusion C — training sensitivity is architecture-dependent
Learning rate mattered far more than the tested LSTM/dropout range.

### Conclusion D — tuning gains do not automatically transfer
Phase2B improved internal validation but often failed to beat Phase2A externally.

### Conclusion E — data integrity materially changes how accuracy must be interpreted
The original split contains material duplicate leakage.

### Conclusion F — structured external generalization is substantially lower than internal performance
The strongest BDNeuro result is ~88–90%, not ~99%.

### Conclusion G — severe domain shift can break the classifier
Web stress performance falls to 25–42.5%, with a strong no-tumor bias.

### Conclusion H — XAI is not proof of lesion-centered reasoning
Correct predictions frequently coexist with poor/off-target saliency.

### Conclusion I — the system is a research prototype
The evidence does not support clinical deployment or safety claims.

---

## 11. One-sentence viva summary

> We implemented the five-backbone CNN-LSTM-PCA-SVM architecture, then tested not only internal accuracy but training sensitivity, duplicate leakage, explainability and external generalization; the model remained strong on an exact-overlap-clean structured external dataset but degraded sharply under heterogeneous web-domain shift, showing why internal accuracy alone is insufficient evidence of robustness.
