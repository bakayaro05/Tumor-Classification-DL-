# FINAL METHODOLOGY AND BASE-PAPER DEVIATIONS

## 1. Study design
Five independent four-class CNN-LSTM hybrid pipelines are compared. The final classifier is a linear SVM operating on PCA-reduced learned features. The study includes internal evaluation, fixed-horizon and hyperparameter-sensitivity studies, data-integrity audit, structured external validation, explainability and web-domain stress testing.

## 2. Dataset
Development dataset: Nickparvar Brain Tumor MRI, 7,023 images.  
Saved historical split:
- train 4,569
- validation 1,143
- test 1,311

This split is retained as the historical development basis but is known to contain cross-split image duplication.

## 3. Input preprocessing
- ordinary raster MRI image
- RGB conversion
- 224×224 resize
- ImageNet normalization
- training augmentation only
- deterministic evaluation transform otherwise

## 4. CNN-LSTM architecture
Backbone ∈ {VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3}.

CNN output `[B,C,H,W]` is permuted/flattened into `[B,H×W,C]`. A one-layer LSTM reads the spatial-token sequence. The last recurrent output is passed through:
- BatchNorm
- Dropout
- Dense 256→128 (or corresponding hidden width→128)
- ReLU
- temporary Linear128→4 neural classifier.

Baseline LSTM hidden=256, Dense=128, dropout=0.3.

## 5. Backbone trainability
- VGG16/VGG19 pretrained CNN blocks frozen after unfrozen collapse.
- ResNet50/ResNet152/EfficientNet-B3 fine-tuned.

## 6. Neural training
Baseline:
- RMSprop
- LR1e-4
- batch8
- seed42
- CrossEntropyLoss on raw logits
- training loader `drop_last=True` because 4,569 mod 8 leaves a single sample incompatible with BatchNorm1d training statistics.

Phase1 used early stopping/best-validation checkpointing.

Phase2A used fixed 30 epochs with no early termination.

Phase2B changed one factor at a time.

## 7. Deep-feature extraction
After neural training:
- model eval mode;
- deterministic preprocessing;
- no random augmentation;
- extract 128-D representation for train/validation/test.

## 8. Hybrid downstream classifier
Fit only on training features:
1. StandardScaler
2. PCA 64
3. linear SVM, C=1

Validation and test/external features are only transformed by already-fitted objects.

## 9. Metrics
- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1 where used
- confusion matrix
- per-class precision/recall/F1

Macro metrics are emphasized for external cohorts, especially imbalanced BDNeuro full cohort.

## 10. Data-integrity methodology
Development audit:
- SHA/file and decoded-pixel exact identity;
- perceptual hash screening;
- grayscale similarity/correlation review for near-duplicate candidates.

External BDNeuro audit:
- SHA256 bytes;
- exact decoded RGB pixel hash;
- 64-bit pHash.

Final BDNeuro exclusion rule: exact decoded-pixel overlap only.

Web audit:
- exact file/pixel duplicate checks;
- pHash screen against Nickparvar and BDNeuro;
- visual review of strong candidates before lock.

## 11. External-validation methodology
Pre-select/freeze 15 candidates using internal validation only. Freeze exact checkpoint/scaler/PCA/SVM artifacts with hashes. Evaluate without training, fitting or calibration.

Primary external cohort: BDNeuro exact-overlap-clean n=2,644.  
Secondary nested sensitivity: balanced n=1,648.  
Exploratory in-the-wild stress cohort: 40 source-verified images, 10/class.

## 12. Explainability
Grad-CAM:
- ResNet152 neural head;
- Layer3, Layer4, equal fusion;
- qualitative ratings.

Occlusion:
- target final SVM decision;
- patch32/stride16;
- recompute complete pipeline after each occlusion.

Neither method is segmentation.

## 13. GUI
Inference:
`uploaded image → eval_transform → CNN-LSTM → 128D → saved scaler → saved PCA → saved SVM`.

No per-image refitting. Neural head displayed for comparison. SVM margins transformed only for relative visualization, not clinical probability.

## 14. Base-paper deviations

| Base-paper concept | Project execution |
| --- | --- |
| Five CNN backbones | Same five verified backbones |
| 4 optimizers × 5 backbones | Only RMSprop branch executed |
| 30 epochs | Phase1 often early-stopped; Phase2A/B explicitly 30 |
| Batch 32 | Main project training batch 8 |
| LSTM details incompletely specified | Implementation choice: hidden 256 baseline, one layer, last spatial token |
| Dense feature width incompletely specified | Implementation choice: 128 |
| Freeze policy unspecified | VGG frozen after empirical collapse; others fine-tuned |
| Scaler/PCA/SVM hybrid | Implemented; current verified order is StandardScaler then PCA |
| External validation not central | Added as major project extension |
| Leakage audit absent | Added and changed interpretation |
| XAI absent/limited | Added Grad-CAM and final-SVM occlusion |
| GUI not required | Added research prototype |

## 15. What “final methodology” means here
It is the methodology actually executed and used to generate the project evidence. It does **not** pretend the contaminated development split became clean. Optional future group-aware retraining would constitute a new remediation experiment, not a retroactive description of the completed project.
