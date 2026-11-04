# PHASE 1 — BASELINE IMPLEMENTATION AND DOWNSTREAM-CLASSIFIER COMPARISON

## Goal
Establish the paper-inspired CNN-LSTM-PCA-SVM pipeline on the Nickparvar four-class dataset and compare five alternative pretrained CNN backbones.

## Implementation sequence
1. dataset inspection and saved split;
2. ResNet152 feature-map verification;
3. spatial-sequence reshape;
4. LSTM section;
5. post-LSTM layers;
6. neural training step;
7. full ResNet152 training;
8. deterministic deep-feature extraction;
9. StandardScaler;
10. PCA64;
11. linear SVM;
12. generic five-backbone implementation;
13. five-backbone training/evaluation;
14. deterministic neural-head evaluation.

## Final baseline architecture
`MRI → CNN → spatial sequence → LSTM256 → BN → Dropout0.3 → Dense128/ReLU`

Neural training branch:
`Dense128 → Linear4 logits → CrossEntropyLoss`

Hybrid evaluation branch:
`Dense128 → StandardScaler(train only) → PCA64(train only) → Linear SVM C=1`

## Backbone policy
- VGG16/VGG19: frozen pretrained CNN backbone after VGG16 collapse.
- ResNet50/ResNet152/EfficientNet-B3: fine-tuned family.

## Results
| Backbone | Best epoch | Neural test acc | Neural macro F1 | SVM val acc | PCA+SVM test acc | PCA+SVM macro F1 | Test acc change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vgg16 | 16 | 93.67% | 93.32% | 94.93% | 94.97% | 94.68% | +1.30 pp |
| vgg19 | 18 | 93.14% | 92.72% | 94.49% | 93.52% | 93.10% | +0.38 pp |
| resnet50 | 7 | 97.33% | 97.19% | 97.55% | 98.09% | 98.01% | +0.76 pp |
| resnet152 | 8 | 98.17% | 98.06% | 98.25% | 98.40% | 98.32% | +0.23 pp |
| efficientnet_b3 | 6 | 98.02% | 97.94% | 98.51% | 98.47% | 98.41% | +0.45 pp |

## Interpretation
- downstream PCA+SVM test metrics rose for all five versus neural head;
- gains were largest for VGG16 and smallest for ResNet152;
- validation did not improve uniformly, so “PCA+SVM always improves performance” is not supported;
- EfficientNet-B3 and ResNet152 were extremely close on historical Phase1 test;
- all original internal results must carry the later leakage caveat.

## Phase1 artifact family
Per backbone:
- `experiments/<backbone>/models/best_model.pth`
- `experiments/<backbone>/models/svm.joblib`
- `experiments/<backbone>/pca/standard_scaler.joblib`
- `experiments/<backbone>/pca/pca_model.joblib`
- deep-feature `.npy` files
- PCA-transformed `.npy` files
- `results/training_results.json`
- `results/final_results.json`

## Legacy root-level ResNet152 artifacts
The first working ResNet152 pipeline used root-level files such as:
- `models/best_resnet152_lstm.pth`
- root `features/`
- `pca_features/`
- root SVM bundle

These proved the pipeline, but standardized `experiments/resnet152/` artifacts superseded them.

## Scientific status
Phase1 is complete. Its internal results are historical contaminated-split evidence, not independent generalization.
