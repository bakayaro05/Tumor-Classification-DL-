# Project Journey Timeline

## Earliest classical-project phase — 16 July 2026

### What we were trying to accomplish

Define a workable MRI tumor project before the later CNN-LSTM-SVM paper was selected.

### What was discussed

Four candidate papers/directions were recorded: Mandle et al. (K-means plus DWT/PCA and kernel SVM), Rao/Karunakara (Python/advanced SVM), Saad et al. (detection/features/GUI), and Malathi/Sinthia (hybrid clustering/backpropagation). A concrete classical pipeline was proposed: resize/grayscale, median filtering/contrast enhancement, skull removal, K-means, morphology, shape and GLCM features, standardization/PCA, and RBF-SVM.

### Corrections made during discussion

K-means cluster numbers were recognized as arbitrary; a “tumor cluster” would require intensity, location, and connected-component reasoning. BraTS HGG/LGG labels were explicitly not to be relabeled automatically as malignant/benign.

### Consequence

This phase established the project's early segmentation/handcrafted-feature logic and medical-labeling caution. It was later superseded, not retroactively erased.

## Recent-paper selection phase — 17 August 2026

### What we were trying to accomplish

Select a recent base paper for a mini-project and build a four-class MRI classifier while learning deep learning from first principles.

### What was discussed

The user knew Python and programming but described limited deep-learning and medical-image-processing experience. Novelty was not initially required; faithful implementation of a 2023-or-later paper was the priority. A 2022 classical pipeline using median filtering, skull stripping, K-means, DWT, PCA, and kernel SVM was considered but could not serve as the official recent base paper. A 2023 Otsu/GLCM/DWT/SVM paper was also discussed as a possible classical baseline.

### What changed and why

The project selected the 2026 Springer paper **“Hybrid CNN+LSTM+SVM model for advanced brain tumor classification.”** It offered five comparable pretrained backbones and a hybrid deep/classical pipeline. The paper described a 20-combination design (five backbones × Adam/SGD/AdaGrad/RMSprop), 30 epochs, batch size 32, learning rate `1e-4`, and photometric augmentation including saturation and hue. The project ultimately reproduced the RMSprop branch only and used different practical training settings. The task was explicitly corrected to tumor-type classification—not benign-versus-malignant classification.

### Result and consequence

The implementation target became VGG16, VGG19, ResNet50, ResNet152, and EfficientNet-B3 as alternative backbones, not a simultaneous ensemble. Explainability was deferred until the base pipeline worked.

## Environment setup and framework failure — 28 August 2026

### What we were trying to accomplish

Configure GPU training under Windows 11/WSL2 on an NVIDIA RTX 5070-class laptop GPU.

### Problems encountered

TensorFlow detected the GPU and could execute a simple matrix multiplication but failed during real work with an `INVALID_PTX` / unsupported `sm_120` toolchain error. A nightly TensorFlow build was tried, but the compatibility path remained unreliable.

### Solution and decision

The project abandoned TensorFlow and moved to PyTorch. Recorded environment: Ubuntu under WSL2, Python 3.12.14 in `~/brain_tumor_project/venv`, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA available.

### Consequence

All authoritative implementation and artifact formats became PyTorch/scikit-learn based.

## Data-pipeline and split phase — late August 2026

### What was implemented

The Masoud Nickparvar/Kaggle Brain Tumor MRI dataset version with 7,023 images was used: 5,712 in the supplied Training directory and 1,311 in Testing. The training directory was split once into 4,569 training and 1,143 validation images using seed 42 and stratification; indices were saved. The official 1,311-image Testing directory was retained as test.

Images were converted to the pretrained-CNN input convention around 224×224 and ImageNet normalization. Training augmentation included horizontal flip, vertical flip, brightness perturbation (`max_delta` approximately 0.1 in the paper-faithful discussion), and contrast perturbation (approximately 0.8–1.2). Deterministic evaluation transforms were used for validation, test, and saved feature extraction.

Dataset inspection found all 7,023 files readable and zero corrupt images. Inputs were heterogeneous: 3,093 grayscale (`L`), 3,926 RGB, 3 RGBA, and 1 palette (`P`) image, with varied spatial dimensions. Conversion to RGB was therefore functionally necessary. Saturation/hue were paper-specified augmentations, but their presence in the actual saved PyTorch pipeline is not established by the attached source snapshot.

### Decision

Saved indices and a deterministic feature-extraction path were used to keep model comparisons consistent.

## Incremental architecture construction — late August to early September 2026

### What was implemented

Small tests verified each section before full training:

- ResNet152 convolutional output `[B, 2048, 7, 7]`;
- spatial flattening to `[B, 49, 2048]`;
- one-layer LSTM with hidden size 256;
- last LSTM timestep;
- BatchNorm, Dropout(0.3), Dense(128), ReLU;
- temporary four-class neural head;
- end-to-end forward/backward training step.

### What was clarified

The LSTM does not process time or MRI slices. It processes a sequence created by flattening spatial positions in a single 2D CNN feature map.

### Result

The first paper-faithful ResNet152 pipeline was ready for full training.

## First ResNet152 baseline — early September 2026

### What was implemented

ResNet152+LSTM was trained with RMSprop at learning rate `1e-4`, CrossEntropyLoss, batch size around 8, seed 42, and best-validation checkpointing. The best early checkpoint was originally recorded at epoch 8 with neural validation accuracy 97.99% and validation loss 0.0677.

The 128-D dense features were extracted deterministically, fitted with StandardScaler on training only, reduced to 64 PCA components (historically recorded as approximately 99.47% retained variance in the early root-level pipeline), and classified using a linear SVM (`C=1.0`). The early SVM validation accuracy was 98.08%; early untouched-test accuracy was 97.64%, macro F1 97.50%, weighted F1 97.63%.

### Consequence

The hybrid pipeline was proven operational. These root-level artifacts later became legacy artifacts after standardized five-backbone experiments were produced.

## Comparative five-backbone framework — 6–12 September 2026

### What was trying to be accomplished

Generalize the working ResNet152 implementation into a consistent comparison across five backbones.

### What was implemented

`CNNLSTM` selected one ImageNet-pretrained backbone, flattened its spatial feature map, applied LSTM(256), BatchNorm, Dropout(0.3), Dense(128), ReLU, and a temporary neural classifier. Standardized experiment bundles were written under `experiments/<backbone>/` with checkpoint, raw deep features, scaler, PCA, SVM, training results, and final results.

The project did **not** reproduce the base paper's complete optimizer comparison. All verified project runs used RMSprop. The comparative trainer defaulted to batch size 8 and 10 epochs with early stopping. VGG16 was first run for 10 frozen epochs and then resumed toward 30, stopping after 19 total epochs; VGG19 was launched with a 30-epoch target and stopped after 21. Thus “paper-faithful” refers to the core hybrid architecture and selected optimizer, not exact reproduction of all paper experiments.

### Problem and change

VGG16 collapsed to approximately 27.9% validation accuracy when the full convolutional backbone was unfrozen. The failure included nearly constant learned features and PCA NaNs, while diagnostic checks found the pretrained convolutional activations themselves healthy. This localized the problem to the attempted end-to-end optimization/learned-feature path rather than corrupt inputs or dead pretrained features, although a single definitive mechanism was not proved. VGG16 and then VGG19 were changed to frozen convolutional backbones; only LSTM/post-LSTM/classifier layers were trained. ResNets and EfficientNet remained trainable.

One later conversation checkpoint claimed DenseNet121 was also implemented and evaluated. This is contradicted by the supplied generic model, supported-backbone list, manifest, and standardized result artifacts, which contain no DenseNet121 experiment bundle. The claim is retained as unresolved history, not accepted as a completed experiment.

### Result

The five standardized historical test accuracies were:

| Backbone | Test accuracy | Test macro F1 |
|---|---:|---:|
| VGG16 | 94.97% | 94.68% |
| VGG19 | 93.52% | 93.10% |
| ResNet50 | 98.09% | 98.01% |
| ResNet152 | 98.40% | 98.32% |
| EfficientNet-B3 | 98.47% | 98.41% |

EfficientNet-B3 had the highest historical test accuracy, while ResNet152 remained the base-paper-centered model and explainability target.

## GUI construction — from 12 September 2026

### What was implemented

A Streamlit GUI was added with model selection, image upload, final SVM prediction, neural-head comparison, score charts, saved metrics, confusion matrices, per-class results, training curves, and artifact-readiness checks. It reused the project evaluation transform and loaded saved scaler/PCA/SVM objects without refitting.

### Important safeguards

The final displayed class is the SVM output. The GUI computes a softmax visualization from `decision_function()` values and labels it as relative—not calibrated—scores; neural softmax values are also not clinically calibrated. The GUI text/current evaluation script say `probability=False`, but historical evaluation logs show `probability=True` was used in at least some runs. This artifact/code mismatch does not change the GUI's use of decision margins, but the saved SVM flags require inspection. DICOM was excluded. The application is explicitly a research/educational prototype, not a medical device.

## Explainability implementation and evaluation — September 2026

### What was discussed and implemented

Grad-CAM was added as an extension beyond the base paper. Because an external scikit-learn SVM is not differentiable through the neural network, Grad-CAM targets the CNN-LSTM neural head, not the final SVM. Earlier ResNet layer 3 and layer 4 maps plus a fused map were evaluated to test whether higher resolution improved localization.

Final-SVM occlusion sensitivity was added as a complementary explanation. It masks transformed input patches and reruns the complete CNN→LSTM→StandardScaler→PCA→SVM path, measuring changes in the SVM decision score.

### Actual findings

In a frozen seed-42 sample of 20 ResNet152 test images, Layer 3 (14×14), Layer 4 (7×7), and a 50:50 fused map were reviewed at overlay opacity 0.55. The SVM and neural head agreed on all 20; both were correct on 19. The shared error was pituitary image `Te-pi_0078.jpg`, predicted as glioma. Many maps concentrated on skull borders, face, or extracranial regions. Among 15 tumor images, Layer 3 was Good/Partial/Poor in 1/9/5, Layer 4 in 3/2/10, and fusion in 2/6/7. Layer 3 was chosen as most relevant in 9/15 and Layer 4 in 6/15; fusion was never selected as best. One no-tumor image was flagged as a possible label anomaly/non-tumor abnormality.

In the later 11-case comparison, 9 had tumors: occlusion was rated Good/Partial/Poor in 2/2/5; occlusion was more anatomically relevant in 3 cases, Grad-CAM in 4, and both were predominantly off-lesion in 2. The subset was deliberately selected from the earlier review, not a representative random sample. Across 1,859 occlusions, 58 changed the predicted class (3.12% pooled); median per-image flip rate was 0%, and 5/11 images had any flip. The misclassified `Te-pi_0078.jpg` accounted for 38 flips (22.5%). Per-image heatmaps were independently normalized, so a bright hotspot could correspond to a negligible absolute score change; three reviewed examples had maximum decreases of only about 0.0016–0.0051.

### Decision

Explainability must be presented as qualitative, coarse, and unreliable for segmentation—not proof that the classifier localizes tumors correctly.

## Consolidation and pre-lock review — 20 September 2026

### What was discussed

The project appeared nearly complete. Housekeeping identified the absence of an exact/perceptual cross-split duplicate audit as the only scientifically meaningful check still pending, alongside refreshing the manifest and documentation.

### Consequence

The duplicate audit was run before lock.

## Leakage audit and current state — 20 September 2026

### Problem discovered

The audit status was `MATERIAL_DUPLICATION`, not a pass. Across 7,023 images it reported 464 exact-file or pixel-duplicate cross-split pairs, 98 additional high-confidence near-duplicate pairs, 562 material pairs total, and 481 review candidates.

### Interpretation

The prior split was not independent at image level. The impressive validation and test metrics may be inflated. The audit cannot establish patient-level separation.

This superseded an immediately earlier “locked scientific interpretation” that still called the classification accuracy valid at dataset level. That statement was defensible only before the duplicate audit; it is not current authority.

### Current decision

Do not lock or publish the old scores as clean generalization. Preserve them as historical contaminated-split results. Build duplicate groups, create a group-aware train/validation/test split, document how the original official test folder is altered or retained, retrain all five models from scratch, refit every scaler/PCA/SVM on the new training split only, and reevaluate once.
