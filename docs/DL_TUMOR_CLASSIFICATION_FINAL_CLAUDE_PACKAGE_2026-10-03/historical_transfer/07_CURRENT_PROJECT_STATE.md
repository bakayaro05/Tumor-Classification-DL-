# Current Project State

## CURRENT AUTHORITATIVE STATE

The project has a complete historical five-backbone CNN-LSTM-PCA-SVM implementation, standardized experiment artifacts, a Streamlit GUI, Grad-CAM, and final-SVM occlusion sensitivity. However, it is **not ready to lock, publish, or claim independent test performance** because the 20 September 2026 audit found extensive material image duplication across train/validation/test splits.

The primary scientific blocker is a duplicate-group-aware split and full rerun. Prerequisites still requiring resolution include the split policy, candidate adjudication/group construction, complete source recovery, exact transform recovery, and inspection of saved SVM metadata. All existing metrics must be labeled **historical contaminated-split results** until that rerun is complete.

## Project identity

- **Working name:** DL Tumor Classification
- **Candidate presentation title:** *NeuroVision: An Intelligent Framework for MRI-Based Brain Tumor Analysis*
- **Formal alternative:** *An Intelligent Framework for Automated Brain Tumor Classification from MRI Images*
- **Title lock status:** Not confirmed.

## Research problem

Classify a single 2D brain MRI image into glioma, meningioma, no tumor, or pituitary using a hybrid learned-feature and classical-classifier pipeline.

## Motivation and gap

The selected base paper combines pretrained CNN features, an LSTM over a reshaped feature map, and an SVM. The project first aimed to reproduce this design, compare five backbones, and later add explainability. The practical gaps now are more important than novelty: independent evaluation, external generalization, patient-level separation, reliable localization, and calibrated confidence are not established.

## Dataset and current split

- Kaggle Brain Tumor MRI Dataset, 7,023-image version.
- Original directories: Training 5,712; Testing 1,311.
- Historical internal split: train 4,569; validation 1,143; test 1,311.
- Classes: glioma, meningioma, notumor, pituitary.
- Historical test supports: 300, 306, 405, and 300 respectively.
- Split integrity: **failed** at image level. The audit found 562 material cross-split pairs.
- Patient independence: unknown; reliable patient IDs are unavailable.
- Readability/modes: 7,023 readable, 0 corrupt; 3,093 `L`, 3,926 `RGB`, 3 `RGBA`, 1 `P`; dimensions varied. The pipeline converted to RGB and historically used 224×224.

## Current implemented architecture

For each backbone independently:

`RGB MRI → deterministic/augmented transform → pretrained CNN feature map → [B,H×W,C] sequence → 1-layer LSTM(256) → last timestep → BatchNorm → Dropout(0.3) → Dense(128) → ReLU`

During neural training, a four-class linear head and CrossEntropyLoss are used. After best-checkpoint selection, deterministic 128-D features are extracted. StandardScaler and PCA(64) are fitted on training only. A linear SVM (`C=1`) is the final classifier.

Backbones:

- VGG16, 512 channels, frozen convolutional backbone;
- VGG19, 512 channels, frozen convolutional backbone;
- ResNet50, 2,048 channels, fine-tuned;
- ResNet152, 2,048 channels, fine-tuned;
- EfficientNet-B3, 1,536 channels, fine-tuned.

## Training/evaluation configuration

- Seed: 42.
- Optimizer: RMSprop.
- Learning rate: `1e-4` default.
- Loss: cross entropy.
- Typical batch size: 8 in the comparative trainer.
- Early stopping/best validation checkpoint; patience configurable, default 3.
- Training loader used `drop_last=True`.
- Validation/test and feature extraction use deterministic evaluation preprocessing.
- The test set was evaluated after model/SVM selection, but its independence is now disproven by the image-level audit.
- The base paper described 20 optimizer/backbone combinations, 30 epochs, batch 32, and broader augmentation. The verified project comparison is RMSprop-only, usually batch 8 and 10 epochs/early stopping, with longer VGG targets; it is not an exact reproduction.

## Historical standardized results — not clean generalization estimates

| Backbone | NN val acc. | SVM val acc. | Test acc. | Test macro F1 |
|---|---:|---:|---:|---:|
| VGG16 | 94.58% | 94.93% | 94.97% | 94.68% |
| VGG19 | 94.58% | 94.49% | 93.52% | 93.10% |
| ResNet50 | 97.73% | 97.55% | 98.09% | 98.01% |
| ResNet152 | 98.25% | 98.25% | 98.40% | 98.32% |
| EfficientNet-B3 | 98.43% | 98.51% | 98.47% | 98.41% |

These outputs are real records, but their scientific interpretation is restricted by leakage.

## GUI state

The Streamlit GUI supports common raster images (PNG, JPEG/JPG, BMP, TIFF), five-model selection, final SVM prediction, neural-head comparison, metrics, confusion matrices, model comparison, training curves, Grad-CAM, and optional occlusion sensitivity. It does not support DICOM and is not a medical device.

The GUI correctly distinguishes:

- final SVM output from the neural-head output;
- relative SVM decision scores from calibrated probabilities;
- coarse explanations from tumor segmentation.

Current evaluation source specifies `probability=False`, but VGG19 and ResNet152 logs contain evidence consistent with runtime `probability=True`. The GUI uses `decision_function`, so its relative-score disclaimer remains correct; the saved artifacts' flags must be inspected individually.

## Explainability state

- Grad-CAM explains the differentiable neural head, not the external SVM.
- Occlusion sensitivity probes the final SVM decision by repeated end-to-end perturbation.
- Small qualitative evaluations found frequent border/extracranial attention and inconsistent lesion localization.
- Grad-CAM used a fixed 20-image set; occlusion used 11 deliberately selected cases and independently normalized each map. Neither sample supports prevalence or method-superiority claims.
- Explainability is useful for auditing spurious behavior, not for claiming clinical localization.

## Current codebase status

The attached source snapshot includes training/evaluation scripts, `CNNLSTM`, the main Streamlit app, tests, and evaluation apps. The main app imports modular GUI code such as `gui.inference_engine`, `gui.model_registry`, `gui.results_loader`, `gui.gradcam`, and `gui.occlusion`; some later modules were recorded separately and are not all present in the 25-file attachment. The old `project_files.txt` predates later GUI/audit additions and must be regenerated from the actual repository.

A pre-audit checkpoint says DenseNet121 was implemented/evaluated, but the supplied model registry/source, manifest, and five final result files do not support that claim. DenseNet is unresolved—not a verified result and not a rejected approach.

## Current research conclusions

1. The hybrid pipeline is technically operational.
2. On the historical split, residual and EfficientNet backbones outperformed frozen VGG backbones.
3. EfficientNet-B3 marginally led historical test accuracy; ResNet152 was close and remained the paper-centered model.
4. Saliency quality was often weak despite correct classification, suggesting reliance on peripheral or dataset-specific cues.
5. The leakage audit prevents claiming that the historical scores measure independent generalization.

## Known limitations

- Material image-level cross-split duplication.
- No patient IDs or patient-group split.
- Single public dataset and no external validation.
- Artificial raster ordering for LSTM timesteps.
- Different fine-tuning policies across backbone families.
- One principal seed/split and no confidence intervals.
- No probability calibration.
- Qualitative, small-sample explainability evaluation without ground-truth masks.
- Possible dataset-label anomaly identified in one reviewed no-tumor image.
- Raster-image input only; no DICOM/clinical workflow.

## Blocking remaining task

Construct duplicate groups from all material exact/near-duplicate relations, decide and document the handling of the original official test folder, create a group-aware stratified split with no group crossing boundaries, rerun the leakage audit, retrain all five neural models from scratch, regenerate features/scalers/PCA/SVMs, evaluate once on the clean holdout, and regenerate all results, figures, GUI artifacts, manifest, report, abstract, and PPT.
