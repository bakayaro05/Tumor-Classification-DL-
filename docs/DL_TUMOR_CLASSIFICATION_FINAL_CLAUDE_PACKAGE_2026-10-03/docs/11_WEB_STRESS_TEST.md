# 40-IMAGE SOURCE-VERIFIED WEB MRI STRESS TEST

## Purpose
Answer a practical question the structured external dataset does not fully answer:

> What happens when a frozen model receives MRI images sourced from unrelated medical/publication pages rather than another curated dataset with similar organization?

This is a **stress test**, not formal clinical validation.

## Cohort
40 images:
- 10 glioma
- 10 meningioma
- 10 notumor/healthy
- 10 pituitary

The source list records page, figure/panel, source-reported diagnosis, sequence/orientation where known, and collection instructions.

## Source rules
- search engines could be used for discovery;
- original medical/source page establishes diagnosis;
- use only indicated MRI panel;
- crop to isolate the panel, without altering anatomy/contrast to help the model;
- reject CT, pathology, segmentation maps, ambiguous diagnoses and unusable heavily annotated figures;
- do not replace an image after seeing predictions.

## Initial overlap audit
Against:
- Nickparvar 7,023
- BDNeuro exact-clean 2,644

Initial:
- exact file duplicates within web: 0
- exact pixel duplicates within web: 0
- cross exact file: 0
- cross exact pixel: 0
- two pHash<=6 candidates

### NOR_01
pHash distance0 against Nickparvar `Testing/notumor/Te-no_0227.jpg`.

Visual comparison showed the exact same underlying axial T2 slice despite different file/pixel representation. It was rejected and replaced **before inference**.

### MEN_02
pHash distance6 against a Nickparvar meningioma image.

Visual comparison showed clearly different scan/orientation/lesion presentation. It was retained.

## Final audit after replacement
- 40 images
- 10/class
- within-web exact file pairs 0
- within-web exact pixel pairs 0
- cross exact file pairs 0
- cross exact pixel pairs 0
- pHash<=4 candidates 0
- one distance6 reviewed candidate

Cohort then locked.

## Evaluation
Same 15 frozen pipelines as BDNeuro.
No retraining/refitting/reselection.
600 total predictions.

## Results
| Backbone | Stage | Accuracy | Macro F1 | External - internal acc |
| --- | --- | --- | --- | --- |
| vgg16 | Phase1 | 25.00% | 14.58% | -69.93 pp |
| vgg16 | Phase2A | 30.00% | 23.23% | -65.89 pp |
| vgg16 | Phase2B | 30.00% | 21.59% | -66.50 pp |
| vgg19 | Phase1 | 35.00% | 28.89% | -59.49 pp |
| vgg19 | Phase2A | 37.50% | 27.78% | -58.13 pp |
| vgg19 | Phase2B | 35.00% | 25.45% | -60.54 pp |
| resnet50 | Phase1 | 30.00% | 26.49% | -67.55 pp |
| resnet50 | Phase2A | 42.50% | 38.18% | -56.19 pp |
| resnet50 | Phase2B | 40.00% | 33.91% | -59.13 pp |
| resnet152 | Phase1 | 35.00% | 28.08% | -63.25 pp |
| resnet152 | Phase2A | 37.50% | 31.66% | -61.10 pp |
| resnet152 | Phase2B | 35.00% | 25.74% | -64.30 pp |
| efficientnet_b3 | Phase1 | 40.00% | 34.78% | -58.51 pp |
| efficientnet_b3 | Phase2A | 42.50% | 31.97% | -56.80 pp |
| efficientnet_b3 | Phase2B | 35.00% | 28.07% | -64.39 pp |

Stage averages:
| Stage | Mean accuracy | Mean macro F1 |
| --- | --- | --- |
| Phase1 | 33.00% | 26.56% |
| Phase2A | 38.00% | 30.56% |
| Phase2B | 35.00% | 26.95% |

## Highest observed web accuracy
- ResNet50 Phase2A 42.5% = 17/40
- EfficientNet-B3 Phase2A 42.5% = 17/40

One image = 2.5 percentage points, so small differences are not robust rankings.

## Strong class bias
Mean class recall across 15 models:
- glioma 12.7%
- meningioma 30.7%
- notumor 90.7%
- pituitary 7.3%

Mean predicted counts per 40:
- glioma 2.1
- meningioma 8.3
- notumor 28.3
- pituitary 1.3

The models systematically shifted toward `notumor` under strong web-domain shift.

## Why performance collapsed
The web set contains much larger distribution changes:
- different scanners/field strengths;
- different MRI sequences;
- contrast/non-contrast;
- unusual tumor variants and lesion sizes;
- varied orientation/slice level;
- journal-figure cropping/presentation;
- healthy-volunteer research scans;
- harder pathology presentations.

This does not mean the BDNeuro evaluation was invalid. It means the two experiments test different levels of domain shift.

## Correct conclusion
> The frozen models retained meaningful accuracy on a structured exact-overlap-clean external dataset, but performance deteriorated sharply on a small heterogeneous in-the-wild web MRI stress set, demonstrating limited robustness to severe distribution shift.

## What not to claim
- do not call 42.5% clinical accuracy;
- do not use the web set to tune/retrain;
- do not replace failed images;
- do not call the 40 cases representative of clinical prevalence;
- do not infer deployment safety.

## Evidence
- `results/web_stress_audit_final_summary.json`
- `results/web_stress_external_comparison_15.csv`
- `results/web_stress_external_comparison_15.json`
- `scripts/initialize_web_stress_test.py`
- `scripts/audit_web_stress_40.py`
- `manifests/Web_Stress_Test_40_Source_List.xlsx`
- `references/DL_Tumor_Classification_Web_Stress_Test_Consolidated_2026-10-03.pdf`
