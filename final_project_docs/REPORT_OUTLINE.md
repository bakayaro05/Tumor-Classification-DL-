# Final Report Outline

## 1. Introduction
- Brain tumor MRI classification problem.
- Motivation for hybrid CNN–LSTM–SVM classification.
- Base-paper relationship and project objective.
- Research/educational scope; no clinical-deployment claim.

## 2. Dataset and Experimental Protocol
- Masoud Nickparvar Brain Tumor MRI dataset, 7,023 images.
- Original Training directory: 5,712; official held-out Testing directory: 1,311.
- Fixed stratified 80/20 split of original Training directory using seed 42: 4,569 train and 1,143 validation.
- Four classes: glioma, meningioma, no tumor, pituitary.
- Final duplicate/leakage audit and its result.

## 3. Methodology
- Deterministic evaluation preprocessing and training augmentation.
- Five pretrained CNN backbones.
- Model-specific transfer learning: VGG16/VGG19 convolutional backbones frozen; other finalized backbones follow their fine-tuning configuration.
- CNN feature map → spatial sequence → LSTM(256) → BatchNorm → Dropout → Dense(128).
- Neural head used for neural training/monitoring; final deployed classifier is PCA–SVM.
- StandardScaler fitted only on training features.
- PCA fitted only on training features, 64 components.
- Linear SVM C=1.0, probability=False in the standardized pipeline.

## 4. Standardized Five-Backbone Results
- Main comparison table from REPORT_SOURCE_OF_TRUTH.md.
- Training/validation curves.
- Confusion matrices and per-class performance.
- Note that EfficientNet-B3 achieved the highest observed test accuracy; do not overstate the one-image difference versus ResNet152.

## 5. Explainability Methodology
- Explainability experiment restricted to ResNet152.
- Grad-CAM: Layer 3, Layer 4, fused map; fixed 20-image protocol.
- Explain that Grad-CAM targets the neural head, not the external SVM.
- Final-SVM occlusion: fixed 11-case subset, 32×32 patch, stride 16, full pipeline rerun.
- Explain within-image heatmap normalization limitation.

## 6. Explainability Results
- Grad-CAM qualitative counts.
- Occlusion localization counts.
- Grad-CAM-versus-occlusion comparison.
- Prediction-flip analysis, including the misclassified pituitary case.

## 7. Discussion
- Strong classification performance versus inconsistent lesion-focused reasoning.
- Possible shortcut features: cropping, orientation, contrast, scanner/acquisition patterns, borders, skull/facial anatomy, and dataset-specific correlations.
- Why Grad-CAM and SVM occlusion can disagree: they explain different decision mechanisms.
- Classification success does not establish causal or clinically valid reasoning.

## 8. Limitations
- Single public dataset and dataset-specific correlations.
- Patient-level independence cannot be verified without reliable patient IDs.
- Explainability assessment is qualitative and deliberately limited in sample size.
- Grad-CAM explains the neural head; occlusion is computationally coarse.
- Per-image normalization can make negligible absolute effects appear visually strong.
- No segmentation ground truth; maps are not tumor boundaries.
- No external clinical validation or prospective evaluation.

## 9. Conclusion
- Five-backbone hybrid pipeline successfully implemented and standardized.
- High held-out classification performance.
- Explainability analysis reveals that high accuracy does not guarantee lesion-centred reasoning.
- System remains a research/educational prototype, not a diagnostic medical device.

## Appendices
- Exact software environment.
- Final artifact manifest.
- Duplicate/leakage audit report.
- Full per-class metrics and confusion matrices.
- Fixed Grad-CAM and occlusion case tables.
