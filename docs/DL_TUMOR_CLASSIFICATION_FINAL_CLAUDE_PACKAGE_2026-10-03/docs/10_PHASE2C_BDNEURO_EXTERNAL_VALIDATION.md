# PHASE 2C — BDNEURO STRUCTURED EXTERNAL VALIDATION

## Objective
Measure how the frozen project pipelines transfer to an out-of-source four-class MRI cohort without external tuning.

## Candidate dataset
BDNeuro-MRI Version 8.

Raw MRI class cohort:
- glioma 2,191
- meningioma 1,478
- notumor 787
- pituitary 1,485
- total 5,941

Raw splits:
- train 4,160
- val 892
- test 889

Two root-level plot PNGs explained an initial naive recursive count of 5,943; they were not class MRIs.

## Cross-dataset audit
Reference: Nickparvar 7,023 images.

Results:
- exact file pairs: 3,421
- exact decoded-pixel pairs: 3,519
- unique BDNeuro images implicated by exact file: 3,295
- unique BDNeuro images implicated by exact pixel: **3,297**
- pHash<=6 candidate pairs: 9,981
- pHash<=4 strong pairs: 6,664
- unique BDNeuro pHash<=6 candidate images: 4,328
- unique BDNeuro pHash<=4 candidate images: 4,098

Within BDNeuro:
- exact file duplicate pairs: 0
- exact pixel duplicate pairs: 0
- pHash<=6 candidates: 1,588
- pHash<=4 strong: 65

## Why exact pixel overlap was removed despite different documented provenance
External validation requires previously unseen image content. Direct decoded-pixel identity proves image reuse at the image level. Source-provenance descriptions cannot override that empirical fact.

The project does **not** claim to know which public dataset lineage or metadata source caused redistribution/reuse.

## Why pHash candidates were retained
MRI images can look very similar because anatomy and standardized slice planes are highly repetitive. A small pHash distance is a screening signal, not proof of identity. Final rule:
- exact decoded-pixel match → exclude;
- non-identical pHash candidate → retain unless another objective reason establishes reuse.

## Exact-overlap-clean cohort
- retained: 2,644
- excluded: 3,297

Class:
- glioma 1,159
- meningioma 472
- notumor 412
- pituitary 601

Original split origin:
- train 1,845
- val 364
- test 435

## Balanced sensitivity subset
Smallest class = 412.
Deterministic SHA256-ranked sample:
- 412 glioma
- 412 meningioma
- 412 notumor
- 412 pituitary
- total 1,648

No model result was used for subset selection.

## Frozen model design
Exactly 15 pre-specified hybrid pipelines:
- 5 Phase1
- 5 Phase2A
- 5 Phase2B

Selection performed from internal validation only before external evaluation. Frozen manifest hashes checkpoint, scaler, PCA, SVM and internal result artifacts.

## Primary full-cohort results
| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 77.61% | 75.50% | -17.32 pp |
| vgg16 | Phase2A | 80.07% | 78.71% | -15.82 pp |
| vgg16 | Phase2B | 77.57% | 76.01% | -18.93 pp |
| vgg19 | Phase1 | 75.76% | 74.80% | -18.73 pp |
| vgg19 | Phase2A | 79.27% | 78.12% | -16.35 pp |
| vgg19 | Phase2B | 75.00% | 74.27% | -20.54 pp |
| resnet50 | Phase1 | 81.09% | 79.93% | -16.46 pp |
| resnet50 | Phase2A | 81.73% | 81.42% | -16.96 pp |
| resnet50 | Phase2B | 86.04% | 84.83% | -13.08 pp |
| resnet152 | Phase1 | 81.28% | 80.84% | -16.97 pp |
| resnet152 | Phase2A | 88.31% | 87.85% | -10.29 pp |
| resnet152 | Phase2B | 86.54% | 85.67% | -12.76 pp |
| efficientnet_b3 | Phase1 | 83.09% | 81.93% | -15.42 pp |
| efficientnet_b3 | Phase2A | 86.65% | 85.12% | -12.65 pp |
| efficientnet_b3 | Phase2B | 86.57% | 85.53% | -12.81 pp |

Stage average:
| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 79.77% | 78.60% |
| Phase2A | 83.21% | 82.25% |
| Phase2B | 82.34% | 81.26% |

## Balanced sensitivity results
| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 77.18% | 76.82% | -17.74 pp |
| vgg16 | Phase2A | 80.70% | 80.62% | -15.18 pp |
| vgg16 | Phase2B | 78.16% | 77.88% | -18.35 pp |
| vgg19 | Phase1 | 77.49% | 77.64% | -17.00 pp |
| vgg19 | Phase2A | 81.13% | 81.24% | -14.50 pp |
| vgg19 | Phase2B | 77.00% | 77.15% | -18.54 pp |
| resnet50 | Phase1 | 83.74% | 83.80% | -13.81 pp |
| resnet50 | Phase2A | 86.77% | 86.90% | -11.92 pp |
| resnet50 | Phase2B | 87.86% | 87.87% | -11.26 pp |
| resnet152 | Phase1 | 85.13% | 85.13% | -13.12 pp |
| resnet152 | Phase2A | 90.47% | 90.50% | -8.13 pp |
| resnet152 | Phase2B | 89.81% | 89.76% | -9.49 pp |
| efficientnet_b3 | Phase1 | 85.44% | 85.60% | -13.08 pp |
| efficientnet_b3 | Phase2A | 87.86% | 87.79% | -11.44 pp |
| efficientnet_b3 | Phase2B | 88.83% | 88.96% | -10.55 pp |

Stage average:
| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 81.80% | 81.80% |
| Phase2A | 85.39% | 85.41% |
| Phase2B | 84.33% | 84.33% |

## Main findings
1. Every pipeline dropped from internal validation.
2. Structured external accuracy remained meaningful, mostly mid-70s to high-80s.
3. Phase2A generalized better on average than Phase2B.
4. Hyperparameter tuning did not consistently transfer externally.
5. ResNet50 Phase2B dropout0.2 transferred strongly.
6. VGG16/VGG19 Phase2B representatives degraded relative to Phase2A.
7. ResNet152 Phase2A was strongest on both full and balanced structured external evaluation.
8. EfficientNet-B3 remained competitive and relatively stable.

## Final structured-external selected result
ResNet152 Phase2A:
- full: 88.31% accuracy, 87.85% macro F1
- balanced: 90.47% accuracy, 90.50% macro F1

## Why balanced accuracy equals macro recall in the balanced subset
Each class has equal support (412). Therefore ordinary accuracy is the mean of class recalls. Weighted F1 also equals macro F1 because supports are equal.

## Reporting language
Preferred:
**“BDNeuro V8 cross-dataset exact-overlap-clean external cohort.”**

Do not say:
- independent at patient level;
- original BDNeuro 5,941 was an independent external set;
- pHash-clean;
- clinically validated.

## Evidence files
- `results/bdneuro_external_validation_reconstructed_summary.json`
- `results/bdneuro_full_external_comparison_15.csv`
- `results/bdneuro_balanced_external_comparison_15.csv`
- `scripts/audit_bdneuro_overlap.py`
- `scripts/build_bdneuro_exact_clean_cohort.py`
- `scripts/build_bdneuro_balanced_subset.py`
- `scripts/compare_external_15.py`
