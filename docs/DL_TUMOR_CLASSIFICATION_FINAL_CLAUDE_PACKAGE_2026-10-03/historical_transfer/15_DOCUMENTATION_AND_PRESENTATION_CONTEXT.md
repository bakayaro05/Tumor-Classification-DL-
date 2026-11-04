# Documentation and Presentation Context

> **Hostile-review wording correction:** Begin with the 16 July four-source classical phase. Describe the final system as paper-inspired, not a faithful reproduction. Optimizer scope, batch size, epochs, augmentation, VGG policy, preprocessing order, and LSTM interpretation differ or remain uncertain. Explainability samples were fixed/selected and occlusion maps independently normalized, so no prevalence or method-superiority inference is supported.

## Honest project story

The strongest narrative is not “we built a 99% model.” It is:

1. A recent hybrid paper motivated a five-backbone CNN-LSTM-SVM implementation.
2. The architecture was built incrementally and migrated from TensorFlow to PyTorch after GPU incompatibility.
3. One ResNet152 baseline was validated, then generalized to a standardized comparison.
4. VGG collapse required a frozen-backbone correction.
5. Historical results were high, with EfficientNet-B3 narrowly leading and ResNet152 close behind.
6. Explainability evaluation revealed that many correct predictions relied on peripheral or extracranial regions.
7. A final split-integrity audit then found extensive cross-split duplication.
8. Therefore, the scientifically responsible conclusion is that clean generalization performance is currently unknown and the next phase is a duplicate-group-aware rerun.

This story demonstrates technical competence and research integrity. Do not rewrite history to imply that the final method and concerns were known from the beginning.

## Problem

Automated four-class classification of 2D brain MRI images into glioma, meningioma, no tumor, and pituitary.

## Motivation

MRI interpretation is complex; computer-aided research may support consistent classification. Hybrid models can combine pretrained visual representations, relational modeling, and margin-based classification. However, medical-AI performance must be interpreted with attention to leakage, dataset shift, and explainability.

## Research gap

The base paper reported very high within-dataset performance but left open reproducibility details, external validity, patient independence, and explainability. The project extended the methodology with a GUI and explanations, then identified split integrity as the dominant threat.

## Objectives

### Historical objectives

- implement a paper-inspired version of the selected architecture;
- compare five alternative backbones;
- evaluate with accuracy, precision, recall, F1, and confusion matrices;
- create an accessible GUI;
- add Grad-CAM explainability.

### Current objective

Re-establish valid performance using duplicate-group-aware splitting and complete retraining, while retaining explainability as an audit rather than a clinical claim.

## Methodology diagram content

Use the following logical blocks:

`Dataset inventory and duplicate grouping → group-aware split → train-only augmentation → pretrained CNN → spatial sequence → LSTM(256) → 128-D dense features → train-only StandardScaler → PCA(64) → linear SVM → four-class output`

Show five backbones as alternatives, never parallel ensemble branches.

## System architecture diagram content

Separate training and inference:

- **Training:** clean split, CNN-LSTM neural training, checkpoint selection, deterministic feature extraction, scaler/PCA/SVM fitting.
- **Inference:** uploaded image, deterministic transform, selected CNN-LSTM, saved scaler/PCA/SVM, final class, optional neural Grad-CAM/final-SVM occlusion.

## Results section

Present two clearly separated panels:

### Historical contaminated-split outputs

Report the five-model table with a large caveat. State that EfficientNet-B3 was the numerical leader (98.47% test accuracy; 98.41% macro F1), closely followed by ResNet152 (98.40%; 98.32%). Explain VGG freezing.

### Integrity and explainability findings

- 562 material cross-split duplicate/high-confidence-near-duplicate pairs;
- historical scores cannot estimate independent generalization;
- fused Grad-CAM Good/Partial/Poor on tumor images: 2/6/7;
- final-SVM occlusion Good/Partial/Poor on evaluated tumor images: 2/2/5;
- frequent border/extracranial focus.

Do not present clean-rerun results until they exist.

## Conclusion wording

Defensible conclusion:

> The project successfully implemented a reusable five-backbone CNN-LSTM-PCA-SVM system and demonstrated high apparent performance on the original split. Qualitative explanation analysis exposed frequent off-target attention, and a subsequent image-integrity audit identified material cross-split duplication. Consequently, the recorded classification metrics are retained as historical outputs but cannot establish independent generalization. A duplicate-group-aware split and complete retraining are required before comparative performance can be finalized.

## Limitations slide/report section

- contaminated historical split;
- no patient IDs;
- no external validation;
- artificial spatial ordering for LSTM;
- different VGG versus residual/EfficientNet fine-tuning policies;
- single principal seed and no confidence intervals;
- noncalibrated SVM scores;
- small qualitative explanation sample;
- no expert lesion masks;
- raster image input, not clinical DICOM workflow.

## Future-work slide

Lead with the clean rerun, not speculative architecture novelty. Then list external/patient-level validation, ablations, calibration, quantitative explainability, robustness, DICOM, and only later attention/ensembles.

## Viva questions and accurate answers

### Why use an LSTM on images?

The LSTM receives a sequence constructed from spatial CNN feature-map locations. It does not process time or longitudinal MRI.

### Is this an ensemble?

No. Each backbone defines a separate pipeline. Results are compared; predictions are not combined.

### Why SVM after deep learning?

CNN-LSTM learns a compact representation. The saved linear SVM performs the final margin-based classification after training-only scaling and PCA.

### Why freeze VGG but not ResNet/EfficientNet?

Fully unfrozen VGG16 collapsed in the tested setup. Freezing stabilized it. This asymmetry is a limitation of the comparison.

### Does Grad-CAM explain the final prediction?

It explains the neural head. The final output is the SVM, which is instead probed with occlusion sensitivity.

### Are the displayed percentages probabilities?

Neural softmax values are uncalibrated confidence scores. Standardized SVM outputs are relative scores derived from decision margins, not calibrated probabilities.

### Can the model locate a tumor?

No validated localization claim is supported. Grad-CAM and occlusion are coarse explanations, and many reviewed maps were off-target.

### Why are the old results not final?

The audit detected 562 material cross-split duplicate or very-high-similarity pairs. This violates independent evaluation and may inflate metrics.

### Does the audit prove no patient leakage after cleaning?

No. It is an image-level audit. Patient-level independence requires reliable patient identifiers.

## Abstract warning

The previously submitted/working abstract included approximately 99% values and stated ResNet152 superiority based on the base paper. It must not be reused unchanged as the project's final abstract. Distinguish paper-reported results from project results and disclose that valid clean-split results are pending.

## Poster/PPT visual priorities

Use one architecture diagram, one historical comparison table with caveat, one leakage-audit callout, one paired example of plausible versus off-target explanations, and one next-steps flow. Avoid crowded screenshots and avoid promotional medical-device language.
