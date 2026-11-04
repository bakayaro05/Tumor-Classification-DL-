# Project Changelog

## Research direction

**Before:** From 16 July, four classical paper directions were considered. The working pipeline was resize/grayscale→median filtering/contrast enhancement→skull removal→K-means→morphology→shape/GLCM→standardization/PCA→RBF-SVM. K-means cluster IDs were recognized as arbitrary, and HGG/LGG could not be inferred as malignant/benign from these labels.  
**Change:** Selected the 2026 hybrid CNN+LSTM+SVM paper.  
**After:** Five pretrained CNN backbones compared within a common hybrid pipeline.  
**Reason:** Recent-paper requirement and preference for a deep-learning implementation.

## Paper protocol versus executed protocol

**Paper reference:** Five backbones × Adam/SGD/AdaGrad/RMSprop (20 runs), 30 epochs, batch 32, learning rate `1e-4`, with horizontal/vertical flips plus brightness, contrast, saturation, and hue perturbations.  
**Executed project:** Only the RMSprop branch is verified; comparative training used batch 8, generally 10 epochs/early stopping, with VGG targets extended to 30. The exact project augmentation must be recovered from `src/data_pipeline_torch.py`.  
**Consequence:** The project is paper-inspired, not an exact reproduction.

## Dataset inspection

**Change:** All 7,023 files were checked as readable: 3,093 `L`, 3,926 `RGB`, 3 `RGBA`, and 1 `P`, with varying dimensions.  
**After:** Inputs were converted to RGB and resized to 224×224 in the implemented path.  
**Reason:** Normalize heterogeneous raster inputs for pretrained CNNs.

## Scope and novelty

**Before:** Potential immediate novelty such as attention, Grad-CAM, ensemble learning, or benign/malignant relabeling.  
**Change:** Lock the base implementation first.  
**After:** Explainability added only after baseline completion; medically invalid benign/malignant mapping rejected.  
**Reason:** Preserve reproducibility and conceptual clarity.

## Framework

**Before:** TensorFlow implementation under WSL2.  
**Change:** TensorFlow abandoned after Blackwell/PTX compatibility failures.  
**After:** PyTorch + torchvision + scikit-learn.  
**Reason:** Reliable CUDA execution.

## Data split

**Before:** Kaggle Training folder split 80/20 with stratification and seed 42; official Testing folder left untouched.  
**Change:** Exact/perceptual duplicate audit performed.  
**After:** Existing split designated contaminated; group-aware resplitting and full rerun required.  
**Reason:** 562 material cross-split duplicate or high-confidence near-duplicate pairs.

## Preprocessing

**Before:** General paper discussion included resize, rotation, shifting, zooming, and multiple possible augmentations.  
**Change:** Actual PyTorch path standardized around pretrained-CNN input, ImageNet normalization, and a narrower training augmentation set; deterministic evaluation transform reused for feature extraction and GUI inference.  
**After:** Training augmentation is separated from validation/test/inference preprocessing.  
**Reason:** Deterministic evaluation and consistency with transfer learning.

## Architecture

**Before:** Conceptual CNN→LSTM→SVM, with ambiguity about the LSTM timestep.  
**Change:** Explicitly flatten spatial feature positions: `[B,C,H,W]→[B,H×W,C]`.  
**After:** LSTM(256)→last timestep→BatchNorm→Dropout(0.3)→Dense(128)→ReLU; temporary neural head used for end-to-end training.  
**Reason:** Concrete implementation for single 2D images.

## Backbone training policy

**Before:** All backbones fine-tuned end to end.  
**Change:** VGG convolutional blocks frozen after VGG16 collapsed near chance.  
**After:** VGG16/VGG19 frozen; ResNet50/ResNet152/EfficientNet-B3 fine-tuned.  
**Reason:** Stable transfer learning.

## Artifact structure

**Before:** Root-level ResNet152 checkpoint, features, PCA, and SVM artifacts.  
**Change:** Standardized experiment directories per backbone.  
**After:** `experiments/<backbone>/{models,features,pca,results}` is the intended source for comparison and GUI. Root-level artifacts are legacy.  
**Reason:** Reproducible five-model comparison and simpler loading.

## SVM probability setting

**Before:** Early root-level `train_svm.py` used `SVC(..., probability=True)`.  
**Current source:** `evaluate_backbone.py` specifies `probability=False`.  
**Contradictory evidence:** Preserved VGG19 and ResNet152 evaluation logs show scikit-learn's probability-parameter deprecation warning, suggesting those runtime models were fitted with `probability=True` before source later changed.  
**After:** GUI uses `decision_function` and labels transformed margins as relative scores. Inspect each saved SVM's `.probability` before claiming its training setting.

## Results

**Before:** Early ResNet152 test accuracy 97.64%, macro F1 97.50%.  
**Change:** Standardized rerun and common artifact layout.  
**After:** Historical standardized ResNet152 test accuracy 98.40%, macro F1 98.32%; EfficientNet-B3 test accuracy 98.47%, macro F1 98.41%.  
**Reason:** Standardization and likely training/evaluation differences. Exact cause was not isolated.

## GUI

**Before:** No user interface.  
**Change:** Streamlit app and modular loaders/registry/inference/explainability code added.  
**After:** Upload, model selection, prediction, metrics, model comparison, Grad-CAM, and optional occlusion sensitivity.  
**Reason:** Demonstration and project presentation requirements.

## Explainability

**Before:** Grad-CAM proposed as future novelty.  
**Change:** Implemented layer 3/layer 4/fused Grad-CAM and final-SVM occlusion.  
**After:** Qualitatively evaluated; frequent border/extracranial attention documented.  
**Reason:** Determine whether high classification accuracy corresponded to plausible image focus.

The Grad-CAM review used seed 42, five images per class, Layer 3 (14×14), Layer 4 (7×7), and 50/50 fusion at opacity 0.55. Occlusion used 11 deliberately selected earlier-review cases—not a random representative sample—with 32×32 patches, stride 16, and independently normalized heatmaps. Those selection and normalization limits were added only after hostile review.

## Scientific conclusion

**Before:** Project appeared ready to lock after final housekeeping.  
**Change:** Leakage audit returned `MATERIAL_DUPLICATION`.  
**After:** Old metrics are historical contaminated-split results, not clean generalization evidence.  
**Reason:** Image-level duplicate leakage across splits.

## Documentation/PPT

**Before:** Submitted/working abstract repeated the base paper's approximately 99% claim and implied ResNet152 superiority.  
**Change required:** Replace those claims with actual project results, state that EfficientNet-B3 led the historical comparison, and prominently disclose the failed split-integrity audit.  
**After:** Not yet verified as completed.  
**Reason:** Academic accuracy.
