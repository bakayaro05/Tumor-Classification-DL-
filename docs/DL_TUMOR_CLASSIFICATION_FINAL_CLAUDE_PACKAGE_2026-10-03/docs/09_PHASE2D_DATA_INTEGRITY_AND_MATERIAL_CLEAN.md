# PHASE 2D — DATA INTEGRITY, LEAKAGE AUDIT, AND MATERIAL-CLEAN EVALUATION

## Why this phase mattered
The project had very high internal accuracy. Before lock, the key unresolved scientific question was whether identical or near-identical images crossed the saved train/validation/test boundaries.

## Audit scope
All 7,023 development images:
- train 4,569
- validation 1,143
- official test 1,311

Cross-split comparisons:
- train↔validation
- train↔test
- validation↔test

## Audit mechanisms
- exact file identity
- decoded-pixel identity
- pHash candidate generation
- dHash candidate generation
- grayscale MAE/correlation review

Recorded thresholds:
- pHash Hamming <=4 candidate
- dHash Hamming <=6 candidate
- review rule included grayscale MAE<=12 and correlation>=0.995

## Authoritative audit outcome
`MATERIAL_DUPLICATION`

- exact file/pixel cross-split duplicate pairs: **464**
- additional high-confidence near-duplicate pairs: **98**
- material pairs total: **562**
- additional review candidates: **481**
- unique official-test images implicated: **250**

Patient-level independence: **unknown** because reliable patient identifiers are unavailable.

## Scientific consequence
The saved image-level split is not independent. Very high validation/test numbers may be optimistic. The result superseded any earlier statement that the split was scientifically locked.

## Material-clean post-hoc evaluation
Official test n=1,311 → remove 250 implicated test images → n=1,061.

| Backbone | material_clean acc | material_clean macro F1 | Original test acc |
| --- | --- | --- | --- |
| vgg16 | 94.816% | 94.302% | 94.966% |
| vgg19 | 93.025% | 92.415% | 93.516% |
| resnet50 | 98.303% | 98.134% | 98.093% |
| resnet152 | 98.398% | 98.315% | 98.398% |
| efficientnet_b3 | 98.680% | 98.511% | 98.474% |

## What the material-clean result means
Supported:
- directly implicated test duplicates were not the sole reason for the strong Phase1 test metrics;
- performance stayed high on the remaining images.

Not supported:
- that training is now uncontaminated;
- that validation is clean;
- that patient-level independence is established;
- that `material_clean` is a replacement for full group-aware retraining.

## Evolution of remediation policy
Older post-audit transfer docs called full duplicate-group-aware splitting/retraining blocking. That was a legitimate immediate reaction to the audit.

Later project policy:
1. preserve internal numbers transparently as contaminated-split history;
2. quantify a narrow post-hoc material_clean test;
3. complete controlled Phase2 studies with caveat;
4. prioritize audited external validation;
5. treat duplicate-group-aware full retraining as optional stronger future work unless a clean internal-generalization claim is required.

This later policy is authoritative.

## External integrity extension
The same philosophy was applied to external data:
- BDNeuro was audited against Nickparvar before evaluation;
- 3,297 exact decoded-pixel overlaps were excluded;
- the web stress cohort was audited against both Nickparvar and BDNeuro;
- one pHash0 same-slice candidate was visually identified and replaced before inference.

## Evidence
- `results/duplicate_leakage_audit.json`
- `results/material_clean_summary.json`
- `scripts/audit_bdneuro_overlap.py`
- `scripts/audit_web_stress_40.py`
