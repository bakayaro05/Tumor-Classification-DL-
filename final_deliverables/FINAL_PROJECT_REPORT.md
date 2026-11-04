# Hybrid CNN–LSTM–PCA–SVM Model for Four-Class Brain Tumor MRI Classification

*Five-backbone implementation, leakage audit, frozen-pipeline external validation, web-domain stress testing and explainability*

**FINAL PROJECT REPORT — DL TUMOR CLASSIFICATION**

Submitted by: 127018005 Akash A; 127018017 Harish R; 127018045 Pranav S  
Guide: M. Rajakumaran / SoC  
SASTRA Deemed to be University · October 2026

> Research and educational prototype — not a clinically validated diagnostic system.

> Traceability copy of FINAL_PROJECT_REPORT.pdf, generated from the same content source. Figures referenced here are embedded in the PDF/DOCX.

## Abstract

Deep-learning classifiers often report near-perfect accuracy on public brain-MRI datasets, but a high score on a public split does not by itself show that a model generalizes. This project implemented, adapted and then systematically stress-tested the hybrid CNN–LSTM–SVM architecture proposed by Bérété et al. (2026) for four-class brain-tumour MRI classification (glioma, meningioma, no tumour, pituitary). Five ImageNet-pretrained backbones — VGG16, VGG19, ResNet50, ResNet152 and EfficientNet-B3 — were each used as an independent pipeline: the CNN feature map is flattened into a sequence of spatial positions, read by a single-layer LSTM, compressed to a 128-dimensional representation, and classified either by a temporary neural head or by a StandardScaler → PCA (64) → linear SVM classifier fitted on training data only. The work was carried out in PyTorch after a TensorFlow implementation failed on the RTX 5070 (Blackwell, sm_120) GPU. On the original Nickparvar split the hybrid pipelines reached 93.52%–98.47% test accuracy (Phase 1). A cross-split duplicate audit of all 7,023 images then found 562 material cross-split duplicate pairs (250 of 1,311 test images implicated), so these internal scores are reported as historical results on a contaminated split; a post-hoc material_clean test (n = 1,061) kept accuracy at 93.025%–98.680%. A fixed 30-epoch protocol (Phase 2A) and a one-factor hyperparameter study (Phase 2B; 35 training runs, 70 downstream evaluations) showed that learning rate was by far the strongest tested sensitivity (mean range 4.62 pp versus 0.58 pp for LSTM width and 0.42 pp for dropout). Fifteen pipelines were frozen and SHA-256-hashed before any external data were seen. On an exact-overlap-clean BDNeuro-MRI cohort (2,644 images, after removing 3,297 images that were pixel-identical to the development data) accuracy fell to 75.00%–88.31%, and on a source-verified 40-image web stress test it fell to 25.0%–42.5%, with a strong bias toward the no-tumour class. The final project representative, ResNet152 Phase 2A (epoch 10), achieved 98.60% internal validation, 98.869% on material_clean, 88.31% on BDNeuro and 37.5% on the web stress test. The central finding is that high public-dataset accuracy does not guarantee robust out-of-distribution performance. The system is a research and educational prototype, not a clinical diagnostic tool.

## 1 Introduction

### 1.1 Background

Magnetic resonance imaging (MRI) is a principal imaging modality for suspected brain tumours. It gives high soft-tissue contrast without ionizing radiation, and different acquisition sequences (for example T1-weighted, contrast-enhanced T1, T2 and FLAIR) emphasize different tissue properties. Interpreting these images requires specialist expertise, and reading large numbers of scans is time-consuming and subject to inter-observer variability. These are the usual motivations for computer-aided classification.

Over the last decade, convolutional neural networks (CNNs) pretrained on natural-image collections such as ImageNet have become the default starting point for medical-image classification. A CNN learns a hierarchy of filters: early layers respond to edges and textures, deeper layers to larger structures. Transfer learning reuses those filters and adapts them to a new task. Combining a CNN with a recurrent network or with a classical classifier such as a support vector machine (SVM) is a common way to build "hybrid" models, and such hybrids frequently report accuracies above 98% on public brain-MRI benchmarks.

### 1.2 Brain tumour MRI classification

The task studied here is single-image, four-class classification of two-dimensional brain MRI slices into **glioma**, **meningioma**, **pituitary** tumour, or **no tumour** (written `notumor` in the dataset and in the code). These are tumour-type labels supplied by the public dataset. They are not grades and must not be re-interpreted as benign or malignant. The input is a single ordinary raster image (JPEG or PNG), not a DICOM volume, and the output is one of the four classes.

### 1.3 Motivation

The Review-I presentation framed the motivation as follows: deep-learning models already report very high accuracy on public brain-MRI datasets, so the scientifically interesting question is not only *how accurate* a model is, but *how trustworthy that accuracy is*. Three threats make a headline number unreliable. First, **data leakage**: identical or near-identical images may appear in both the training and the test partitions. Second, **dataset shift**: a model may perform well only on images that look like its training data. Third, **model opacity**: a correct prediction may rest on image regions that have nothing to do with the lesion. This project set out to examine all three within one reproducible framework.

### 1.4 Research problem

*To build and rigorously evaluate a hybrid CNN–LSTM–PCA–SVM framework for four-class brain MRI classification (glioma, meningioma, no tumour, pituitary), examining the effect of CNN backbone choice, training behaviour, data integrity and generalization — as a research framework, not a clinical diagnostic device.* (Problem statement as defined in the Review-I presentation.)

As the project matured, the problem sharpened into a single question that runs through the rest of this report: **can a hybrid CNN–LSTM–PCA–SVM brain-MRI classifier achieve strong four-class performance, and how trustworthy are those results once training behaviour, duplicate leakage, external generalization and explanation behaviour have been examined?**

### 1.5 Research gap

The base paper and the related studies reviewed in Chapter 2 share several limitations. The literature survey in the Review-I presentation identified them, and this project was designed to address them:

- High accuracies are reported without any duplicate or leakage check on the dataset partitions [1].
- Evaluation uses a single public dataset, with external validation left as future work [1, 9, 12].
- Short or early-stopped training hides convergence and late-epoch behaviour.
- Hyperparameters are chosen by search procedures that make the effect of each factor hard to attribute [14].
- Ensembles or single-backbone studies hide architecture-specific behaviour [11, 13, 15].
- Grad-CAM cannot explain an external, non-differentiable SVM classifier [10], and heatmaps are often over-read as tumour localization [12].

### 1.6 Objectives

1. Reconstruct and adapt the base-paper hybrid pipeline, documenting every deviation.
2. Compare five CNN backbones under one common protocol.
3. Analyse training behaviour (fixed 30-epoch horizon) and deep-learning hyperparameter sensitivity.
4. Verify data integrity and report leakage-aware internal evaluation.
5. Assess generalization on an overlap-audited external dataset (image-level), with all models frozen in advance.
6. Examine model behaviour through explainability analysis (Grad-CAM and final-SVM occlusion).
7. Develop a research and educational prototype application.

No accuracy target was set. The aim was a defensible evaluation, not a headline number.

### 1.7 Scope

**In scope:** the five backbones listed above; the RMSprop optimizer; the Nickparvar Kaggle dataset as the development dataset; the BDNeuro-MRI Version 8 dataset and a 40-image web collection as evaluation-only external data; Grad-CAM and occlusion explanations; and a Streamlit prototype.

**Out of scope:** the base paper's full 5 × 4 optimizer grid; any sixth backbone (a DenseNet121 run mentioned once in conversation history has no verified artifacts and is excluded); tumour segmentation; DICOM or volumetric input; patient-level analysis (no patient identifiers exist); and any clinical deployment or safety claim.

**How to read this report.** Chapters 2–4 describe the base paper, the system design and the data. Chapter 5 tells the implementation story, including the failures. Chapters 6–14 follow the six-phase roadmap of the Review-I presentation, in the order the evidence was produced: Phase 1 (Chapter 6), the data-integrity audit that changed the interpretation (Phase 2D, Chapter 7), Phase 2A (Chapter 8), Phase 2B (Chapter 9), external validation (Phase 2C, Chapters 10–11), comparative analysis and final selection (Phase 3, Chapter 12), explainability (Phase 4, Chapter 13) and the prototype (Phase 5, Chapter 14). Chapters 15–18 discuss the findings, state the limitations, conclude and list future work. Throughout, each result table is followed by an **Interpretation** box that separates what was measured from what it means.

**Figure 1.** Project timeline from the initial concept (July 2026) to the final synthesis (3 October 2026). Dates before 20 September come from project notes and are approximate; later dates are taken from timestamps stored in the repository's CSV/JSON outputs. *(figure file: timeline.png)*

## 2 Base Paper and Literature Review

### 2.1 Base paper

The methodological anchor is: M. Bérété, A. Echtioui, L. Sellami and A. A. Ben Hmida, "Hybrid CNN+LSTM+SVM model for advanced brain tumor classification," *Network Modeling Analysis in Health Informatics and Bioinformatics*, vol. 15, art. 43, Springer, 2026, DOI 10.1007/s13721-025-00718-8 [1]. The paper was chosen in August 2026 because it is recent, Scopus-indexed, uses the same public four-class dataset, and defines five comparable backbones within one hybrid design.

The paper builds five separate hybrid models — VGG16, VGG19, ResNet50, ResNet152 and EfficientNetB3, each followed by an LSTM and an SVM. In each model a pretrained CNN produces a feature map, the feature map is reshaped into a sequence for an LSTM, and the LSTM output passes through batch normalization, dropout and dense layers. The resulting features are standardized, reduced by PCA to 64 components and classified by a linear SVM. The reported protocol (paper Table 2) is: optimizers Adam, SGD, AdaGrad and RMSprop (5 × 4 = 20 configurations); learning rate 1e-4; categorical cross-entropy; at most 30 epochs; batch size 32; dropout 0.3; and augmentation with horizontal/vertical flips, brightness, contrast, saturation and hue.

**Table 1.** Base-paper results under the RMSprop optimizer, as reported by the authors [1, Table 3]. These are the paper's numbers, not this project's.

| Model (RMSprop) | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| VGG16 + LSTM + SVM | 94% | 94% | 93.5% | 93.5% |
| VGG19 + LSTM + SVM | 96% | 95.5% | 95.25% | 95.25% |
| ResNet50 + LSTM + SVM | 99% | 99% | 98.75% | 98.75% |
| ResNet152 + LSTM + SVM | 99% | 99% | 99% | 99% |
| EfficientNetB3 + LSTM + SVM | 98% | 98% | 97.75% | 98% |

> **Interpretation.** The paper's headline is ResNet152 + LSTM + SVM with RMSprop at 99% accuracy, precision, recall and F1. The paper reports no duplicate check on the dataset partitions, no external dataset and no implemented explainability method; Grad-CAM and Score-CAM are mentioned only as future work. These omissions are the starting point for this project's extensions. Section 15.7 compares the project's results with the paper's.

### 2.2 CNN backbones

A **convolutional neural network** applies learned filters across an image and pools the responses, producing a spatial grid of feature vectors (a *feature map*). The five backbones represent three design families:

- **VGG16 / VGG19** [2] stack small 3 × 3 convolutions in 16 or 19 weight layers. The last convolutional block produces 512 channels.
- **ResNet50 / ResNet152** [3] add residual (skip) connections that let very deep networks train stably. The final stage produces 2,048 channels.
- **EfficientNet-B3** [4] scales depth, width and input resolution jointly. Its final stage produces 1,536 channels; it was designed for a native input larger than the 224 × 224 used here.

All backbones load ImageNet weights from torchvision (VGG and EfficientNet: IMAGENET1K_V1; ResNets: IMAGENET1K_V2; `model_cnn_lstm.py`). At a 224 × 224 input, every backbone yields a 7 × 7 spatial grid.

### 2.3 LSTM rationale

A **long short-term memory (LSTM)** network [5] is a recurrent network with gates that control what is remembered and forgotten as it reads a sequence. In this project the "sequence" is *not* time and *not* a stack of MRI slices. It is the 49 spatial positions of the CNN's 7 × 7 feature map, read in raster order (row by row). The LSTM therefore aggregates information across positions within one 2-D image. This follows the base paper's description of capturing "sequential dependencies across the feature maps". The raster order is arbitrary, and the project did not test whether the LSTM adds anything beyond simple pooling (Chapter 16).

### 2.4 SVM and PCA rationale

A **support vector machine** [6] finds the boundary that separates classes with the widest margin. A linear SVM with regularization C = 1 is a simple, well-understood classifier for compact feature vectors. Before the SVM, a **StandardScaler** rescales each of the 128 learned features to zero mean and unit variance, so that no feature dominates because of its scale. **Principal component analysis (PCA)** then projects the standardized features onto the 64 orthogonal directions of greatest variance, keeping a compact representation (in this project the 64 components retained 99.18–99.70% of the training-feature variance). All three objects are fitted on training data only and merely applied to validation, test and external data. This prevents information from evaluation data leaking into the classifier.

### 2.5 Explainable AI

**Grad-CAM** [7] computes the gradient of a class score with respect to a convolutional layer's activations, weights each activation map by its average gradient, and keeps the positive part. The result is a coarse heatmap of image regions that increased that class score. It needs a differentiable path from the score to the CNN, so here it can explain only the neural head, not the scikit-learn SVM.

**Occlusion sensitivity** [8] hides one patch of the input at a time and measures how much the model's output changes. Because it treats the model as a black box, it can probe the complete deployed pipeline (CNN → LSTM → scaler → PCA → SVM).

### 2.6 Limitations identified in the literature

**Table 2.** Literature survey summary (Review-I): description, drawback and inference for this project. Drawbacks marked (obs.) are this project's reading, not statements by the authors.

| Paper | Description | Drawback | Inference for this project |
|---|---|---|---|
| Bérété et al. [1], 2026 | Hybrid CNN–LSTM–SVM; 5 backbones × 4 optimizers; 4-class Kaggle MRI; 99% reported | Single public dataset; no duplicate/leakage analysis; XAI and broader validation left as future work | Add data-integrity checks, external validation and explainability |
| Başarslan [9], 2025 | Multi-CNN + multi-BiLSTM; 99.33% on Br35H | Binary task; authors note generalizability remains open | Apply CNN–LSTM to 4-class task; test externally |
| Panigrahi et al. [13], 2025 | Seven fine-tuned CNNs; majority-voting ensemble | Voting hides each backbone's behaviour (obs.) | Compare backbones individually |
| Hosny et al. [11], 2025 | DenseNet121 + InceptionV3 soft-voting with Grad-CAM; 99.02% | Computational cost; no feature interaction; propose hybrid DL + ML | Supports deep features + classical classifier |
| Özbay & Özbay [10], 2023 | Grad-CAM-processed features; mRMR; SVM/KNN | Grad-CAM builds features, does not explain final classifier (obs.) | Explain the final SVM directly via occlusion |
| Musthafa et al. [12], 2024 | ResNet50 + Grad-CAM; 98.52% | Single dataset; qualitative heatmaps (obs.) | Treat Grad-CAM as coarse evidence; add external validation |
| Sahoo et al. [14], 2024 | Metaheuristic hyperparameter optimization; 97.41% | Search-based tuning hides each factor's effect (obs.) | One-factor-at-a-time sensitivity study |
| Kumar et al. [15], 2021 | ResNet-50 + GAP; 3-class; 97.48% | Single backbone (obs.) | Standardized five-backbone comparison |

### 2.7 How this project addressed them

**Table 3.** Mapping from identified limitations to the project's response and the roadmap phase where it was carried out.

| Limitation identified | Project response | Phase | Status |
|---|---|---|---|
| Accuracy reported without leakage checks | Exact/pixel/perceptual cross-split audit; three post-hoc leakage-aware test policies | 2D | Complete |
| Single-dataset evaluation | 15 pipelines frozen and hashed, then evaluated on an overlap-audited external cohort with no tuning | 2C | Complete |
| Early stopping hides convergence | Fixed 30-epoch protocol; best-within-30 and epoch-30 states both evaluated | 2A | Complete |
| Search-based hyperparameter choice | Controlled one-factor study: learning rate, LSTM size, dropout | 2B | Complete |
| Ensembles / single backbones | Five independent backbones under one protocol | 1, 3 | Complete |
| Grad-CAM cannot explain an SVM | Grad-CAM for the neural head + occlusion for the final SVM | 4 | Complete |
| Heatmaps over-read as localization | Maps rated and reported as coarse contribution maps; no segmentation claim | 4 | Complete |
| (added) Severe domain shift untested | Source-verified 40-image web stress test, locked before inference | 2C+ | Complete |

## 3 System Design

### 3.1 Overall architecture

Each backbone defines one complete, independent pipeline (Figure 2). The five pipelines are alternatives compared under a common protocol. They are **not** an ensemble, and their predictions are never combined. A preprocessed image passes through a pretrained CNN whose classification layers have been removed. The resulting feature map is rearranged into a sequence and read by an LSTM, then compressed into a 128-dimensional deep feature. Two branches read this feature: a temporary neural head used for training (and for comparison and Grad-CAM), and the final hybrid classifier (StandardScaler → PCA 64 → linear SVM), which produces the system's official prediction.

**Figure 2.** Hybrid CNN–LSTM–PCA–SVM architecture. Green boxes are trained or adapted neural components; peach boxes are the two downstream branches. Channel count C is 512 (VGG), 2,048 (ResNet) or 1,536 (EfficientNet-B3). *(figure file: architecture.png)*

**Table 4.** Tensor shapes along the pipeline for ResNet152 (batch size B), as verified by the step-wise test scripts (Section 5.5).

| Stage | Shape | Notes |
|---|---|---|
| Input image | [B, 3, 224, 224] | RGB, resized, ImageNet-normalized |
| CNN feature map | [B, 2048, 7, 7] | 512 channels for VGG, 1,536 for EfficientNet-B3 |
| Spatial sequence | [B, 49, 2048] | each of the 7 × 7 cells becomes one sequence element |
| LSTM output | [B, 49, 256] | one layer, hidden size 256, batch-first |
| Last LSTM step | [B, 256] | output after the 49th position |
| Deep feature | [B, 128] | BatchNorm1d → Dropout(0.3) → Linear(256→128) → ReLU |
| Neural head | [B, 4] | Linear(128→4) logits; CrossEntropyLoss during training |
| After scaler / PCA | [B, 128] / [B, 64] | both fitted on training features only |

### 3.2 CNN feature extraction

`model_cnn_lstm.py` defines the class `CNNLSTM` and the list `SUPPORTED_BACKBONES`, which contains exactly the five backbones. For each backbone the ImageNet-pretrained network is truncated before global pooling, so that it outputs a spatial feature map [B, C, 7, 7]. For VGG16 and VGG19 the convolutional backbone is frozen (`requires_grad = False`; `train_backbone_30epoch.py`, lines 360–372). For ResNet50, ResNet152 and EfficientNet-B3 it is fine-tuned together with the rest of the network. Section 5.8 explains why.

### 3.3 Spatial sequence construction

The feature map is permuted and reshaped from [B, C, H, W] to [B, H·W, C]. Each of the 49 grid cells becomes a C-dimensional token. In this ordering a cell's horizontal neighbour is one step away in the sequence, but its vertical neighbour is seven steps away. This is a property of the design and is acknowledged as a limitation.

### 3.4 LSTM

A single-layer, batch-first LSTM with hidden size 256 (the baseline; Phase 2B also tests 128 and 512) reads the 49 tokens. Only the output at the last position is kept. It summarizes the whole sequence as a 256-dimensional vector.

### 3.5 Dense feature representation

The last LSTM output passes through BatchNorm1d (to stabilize training), Dropout (probability 0.3 in the baseline; to reduce overfitting) and a Linear(256 → 128) layer with ReLU. The resulting **128-dimensional deep feature** is the learned representation used by both branches. After training, features are extracted deterministically: model in evaluation mode, no augmentation, no shuffling.

### 3.6 Neural head

A Linear(128 → 4) layer produces four logits. During training, CrossEntropyLoss is applied directly to these *raw logits*, because CrossEntropyLoss already includes the log-softmax. Softmax is used only at inference time, to turn logits into displayable class scores. The neural head exists so that the network can be trained end to end, and it provides the differentiable target used by Grad-CAM. It is also evaluated separately (Section 6.2) as the "before PCA/SVM" baseline.

### 3.7 StandardScaler

The scaler subtracts the per-feature training mean and divides by the per-feature training standard deviation. Fitting it on training features only and applying it unchanged elsewhere prevents *preprocessing leakage*. This is a different and much narrower issue than the duplicate-image leakage discussed in Chapter 7, and preventing it does not address the latter.

### 3.8 PCA

PCA with 64 components is fitted on the standardized training features. Across the five Phase 1 pipelines the retained training variance was 99.18–99.45%, and across the Phase 2A retained states 99.35–99.70% (`final_results.json → pca.explained_variance`). The 64 components therefore preserve almost all of the variance of the 128 learned features.

### 3.9 Linear SVM

The final classifier is `SVC(kernel="linear", C=1.0, probability=False)` trained on the 64-D training projections (`evaluate_backbone.py` line 948 ff.; `evaluate_backbone_30epoch.py` line 981 ff.; `evaluate_hyperparam_pca_svm.py` line 686 ff.). Inspection of the saved artifacts confirmed `probability=False` for all five Phase 1 SVMs and the Phase 2 SVMs. Only the superseded root-level prototype (`models/resnet152_lstm_pca_svm.joblib`, created by `train_svm.py`) used `probability=True`. The SVM therefore produces decision values, not probabilities, and the prototype application labels a softmax of those values as *relative class scores*. Neither the PCA size nor C was tuned. This differs from the base paper, which enables SVM probability estimation; the decision boundary is the same either way.

## 4 Dataset and Preprocessing

### 4.1 Nickparvar dataset

The development dataset is the public **Brain Tumor MRI Dataset** by Masoud Nickparvar on Kaggle [16]. It is the same four-class dataset used by the base paper. It contains 7,023 two-dimensional MRI images supplied in a *Training* folder (5,712 images) and a *Testing* folder (1,311 images).

### 4.2 Class distribution

**Table 5.** Class distribution of the Nickparvar dataset (repository: phase2/external_validation/bdneuro_overlap_audit/dataset_counts.json → internal_split_class_counts).

| Class | Training folder | Testing folder | Total |
|---|---|---|---|
| glioma | 1,321 | 300 | 1,621 |
| meningioma | 1,339 | 306 | 1,645 |
| notumor | 1,595 | 405 | 2,000 |
| pituitary | 1,457 | 300 | 1,757 |
| Total | 5,712 | 1,311 | 7,023 |

> **Interpretation.** The classes are reasonably balanced (largest-to-smallest ratio about 1.23 : 1), with notumor the largest class. The project nevertheless reports macro-averaged precision, recall and F1 alongside accuracy, because macro metrics weight each class equally and so expose weak minority-class performance. This becomes important on the imbalanced external cohort in Chapter 10.

### 4.3 Dataset inspection

`src/check_dataset.py` opened every file. All 7,023 images were readable, but colour modes (single-channel and RGB) and spatial dimensions varied. This finding directly determined the next two preprocessing steps.

### 4.4 RGB conversion

All backbones expect three-channel input because they were pretrained on colour ImageNet images. Every image is therefore converted to RGB (`img.convert("RGB")`), which replicates a single grey channel three times.

### 4.5 Resize and normalization

Images are resized directly to 224 × 224 and normalized with the ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225), matching the statistics the pretrained weights were learned with. The direct resize distorts non-square images. No skull stripping, cropping, segmentation or MRI-specific intensity normalization is applied. The model can therefore see the skull, background and image borders, which matters for the explainability findings in Chapter 13.

### 4.6 Augmentation

Augmentation is applied **only during training**. The current source code (`src/data_pipeline_torch.py`, lines 65–78) defines:

- `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.5)`;
- `ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)`.

torchvision samples each jitter factor from [1 − x, 1 + x] (hue: [−x, x]). This reproduces the base paper's Algorithm 1 ranges: brightness ±0.1, contrast and saturation 0.8–1.2, hue ±0.1. Validation, test, feature extraction, external evaluation and GUI uploads all use one deterministic evaluation transform (RGB → resize → tensor → normalize). This keeps the features fed to PCA and the SVM repeatable.

### 4.7 Train / validation / test split

The dataset supplies no validation set. `src/save_split.py` created one by a stratified 80/20 split of the Training folder with seed 42, giving **4,569 training** and **1,143 validation** images. The indices are saved to `data/splits/train_indices.npy` and `val_indices.npy`, so every backbone and every experiment uses exactly the same images. The 1,311-image Testing folder is kept as the dataset-provided test set (Figure 3).

**Figure 3.** Roles of the three partitions. The test set was never used to fit any parameter or to select any checkpoint, configuration or state; Phase 2B deliberately did not evaluate it at all. *(figure file: data_roles.png)*

The training partition fits everything that is fitted: CNN–LSTM weights, the scaler, PCA and the SVM. The validation partition selects checkpoints, drives Phase 1 early stopping, chooses between retained states in Phase 2, and chooses representative configurations in Phase 2B. The test partition is for final internal evaluation only. Chapter 7 shows that the supplied Training and Testing folders already shared duplicated content. This split is therefore *not* image-level independent, despite this careful design.

### 4.8 Reproducibility

- Seed 42 for Python, NumPy, PyTorch and CUDA; seeded training data loader.
- Saved split indices; saved checkpoints (best and, in Phase 2, epoch-30); saved scaler, PCA and SVM per experiment; per-experiment JSON results.
- Training loader uses `drop_last=True`: 4,569 = 571 × 8 + 1, and BatchNorm1d cannot compute training statistics from a one-image batch. Shuffling means a different image is dropped each epoch.
- Environment (`final_project_docs/environment_pip_freeze.txt`): Ubuntu under WSL2, Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, scikit-learn 1.9.0, NumPy 2.5.2, Streamlit 1.63.0; NVIDIA GeForce RTX 5070 Laptop GPU (8 GB).
- Single seed and single split throughout. No repeated runs or confidence intervals exist (Chapter 16).

## 5 Implementation Journey

This chapter records how the system was actually built, including the failures. The reasoning behind each decision matters as much as the final configuration, and several later results only make sense in the light of these early choices.

### 5.1 Environment

Development took place on Windows 11 with Ubuntu under WSL2 and an NVIDIA GeForce RTX 5070 Laptop GPU with 8 GB of memory. Before the deep-learning direction, the project had briefly (16 July 2026) considered a classical pipeline: median filtering, skull removal, K-means segmentation, handcrafted shape/GLCM features and an RBF-SVM. Two conceptual risks were corrected at that stage. A K-means cluster index is arbitrary and cannot automatically be called "tumour", and dataset labels must not be relabelled benign or malignant. That direction was superseded on 17 August 2026, when a recent deep-learning base paper became a requirement.

### 5.2 TensorFlow attempt

The base paper's Algorithm 1 uses TensorFlow image-transformation functions, so the first implementation route (28 August 2026) was TensorFlow under WSL2. TensorFlow detected the GPU and could run simple operations.

### 5.3 RTX 5070 / INVALID_PTX problem

Realistic training workloads failed with an `INVALID_PTX` error. The RTX 5070 uses NVIDIA's Blackwell architecture (compute capability `sm_120`), which the TensorFlow build in use did not support. A TensorFlow-nightly route was tried and abandoned as unreliable. The lesson recorded at the time was that *hardware detection is not proof that the full framework and toolchain are compatible*. This was an environmental problem, not a scientific judgement about TensorFlow.

### 5.4 Migration to PyTorch

The project moved to PyTorch 2.11.0 built for CUDA 12.8 (`+cu128`), which supports `sm_120`. CUDA availability and real GPU workloads were verified before any model code was written. All authoritative project artifacts are therefore PyTorch checkpoints and scikit-learn objects.

### 5.5 Architecture verification scripts

Instead of writing the whole model at once, the team assembled and checked it one stage at a time, so that shape errors could not hide inside a monolithic model:

**Table 6.** Step-wise verification scripts (repository root and actual_training_scripts/). These are executable checks that print shapes and statistics, not an assertion-based test suite.

| Script | What it verified |
|---|---|
| test_resnet152_features.py | ResNet152 output [B, 3, 224, 224] → [B, 2048, 7, 7] and the spatial-sequence reshape |
| test_lstm_section.py | [B, 2048, 7, 7] → [B, 49, 2048] → LSTM(256) → [B, 49, 256] |
| test_post_lstm_layers.py | last step → BatchNorm → Dropout(0.3) → Dense 128 → ReLU |
| test_complete_model.py | complete forward pass in its different modes |
| test_real_mri_batch.py | real dataset batches and class-index mapping |
| test_training_step.py | loss, gradients and one optimizer step without full training |
| actual_training_scripts/test_all_backbones.py | all five backbones instantiate and run |
| actual_training_scripts/test_best_checkpoint_features.py | best-checkpoint loading and deterministic feature extraction |
| actual_training_scripts/diagnose_vgg16_collapse.py | pre-training activation statistics of pretrained VGG16 (Section 5.8) |
| verify_gui_setup.py | GUI packages, source files and per-backbone artifacts |

This stage also settled the key conceptual point that the LSTM reads flattened spatial positions of one image, not time or MRI slices. The Review-I presentation showed these verified shapes as completed work.

### 5.6 First ResNet152 implementation

The first end-to-end hybrid was a root-level ResNet152 pipeline (`model_resnet152_lstm.py`, `train_resnet152_lstm.py`, `extract_all_features.py`, `prepare_pca_features.py`, `train_svm.py`, `evaluate_final_test.py`; artifacts in `models/`, `features/`, `pca_features/`). Trained with RMSprop at learning rate 1e-4, its best checkpoint was at epoch 8. According to project notes its SVM reached about 97.64% test accuracy and 97.50% macro F1. No JSON result file for this run is preserved, so these figures are historical. The prototype proved that the complete CNN → LSTM → scaler → PCA → SVM path worked. It was then superseded by the standardized bundles, and its artifacts must not be mixed with them.

### 5.7 Generic five-backbone implementation

Between 6 and 12 September 2026 the ResNet-specific code was replaced by the generic `CNNLSTM` class, with standardized experiment bundles under `experiments/<backbone>/` (`models/`, `features/`, `pca/`, `results/`). `actual_training_scripts/train_backbone.py` trains one backbone (RMSprop, learning rate 1e-4, batch 8, seed 42; best-validation checkpointing; early stopping with patience 3; default maximum 10 epochs). `evaluate_backbone.py` extracts deterministic features and fits the scaler, PCA and SVM.

### 5.8 VGG collapse and fix

**What happened.** When VGG16 was fully unfrozen and fine-tuned like the residual networks, training collapsed. Validation accuracy fell to about 28%, the learned 128-D features became almost constant, and PCA then produced invalid (NaN) values. A useful check comes from the dataset itself: notumor contributes 319 of the 1,143 validation images (27.9%). That is exactly the accuracy of a model that always predicts the largest class, so the network had stopped discriminating altogether.

**How it was diagnosed.** `diagnose_vgg16_collapse.py` reported activation statistics (mean, standard deviation, range, non-zero fraction) of the pretrained VGG16 before training. They were healthy. The fault was therefore in the end-to-end training configuration, not in the data, the class mapping or the pretrained weights. A single root cause was never isolated, and lower learning rates or partial unfreezing were not studied.

**Decision.** The pretrained VGG16 convolutional backbone was frozen, and only the LSTM, BatchNorm, dense layer and neural head were trained. VGG16 then trained stably, and VGG19 was given the same policy. ResNet50, ResNet152 and EfficientNet-B3 remained fine-tuned.

> **Interpretation.** The consequence must be kept in view throughout the report: the five pipelines do not share identical optimization dynamics. The VGG networks act as fixed feature extractors with a trained read-out, whereas the others adapt their convolutional weights. Cross-backbone differences are therefore differences between practical pipelines, not a pure architecture comparison. The collapse figure (≈28%) comes from the project record; no JSON output of the failed run is preserved.

## 6 Phase 1 — Baseline System

**Roadmap status: COMPLETE.** Phase 1 had to establish that the complete hybrid pipeline works for all five backbones before any robustness or explainability extension was added.

### 6.1 Methodology

Each backbone was trained with `train_backbone.py` (RMSprop, learning rate 1e-4, batch 8, seed 42, LSTM 256, dense 128, dropout 0.3, CrossEntropyLoss on raw logits, training-only augmentation). The checkpoint with the highest validation accuracy was kept (`experiments/<backbone>/models/best_model.pth`), and training stopped after three epochs without improvement. The maximum epoch budget was not equal across backbones (`training_results.json → epochs_requested`): VGG16 and VGG19 had a 30-epoch budget and stopped early at 19 and 21 epochs, while ResNet50, ResNet152 and EfficientNet-B3 used the trainer's default 10-epoch budget. ResNet50 and ResNet152 ran all 10 epochs; EfficientNet-B3 stopped at 9. Best epochs are stored 1-based (verified: `history.val_accuracy[best_epoch − 1]` equals `best_val_accuracy`). This unequal, early-stopped horizon is exactly what Phase 2A was later designed to remove.

After training, `evaluate_backbone.py` reloaded the best checkpoint, extracted deterministic 128-D features for all three partitions, fitted the scaler and PCA (64) on training features, trained the linear SVM, and evaluated validation and test. `evaluate_neural_head.py` separately evaluated the saved neural head on the same checkpoints in evaluation mode (`neural_head_results.json`; `test_used_for_checkpoint_selection = false`).

### 6.2 Neural-head results

Table Table 7 reports both classifiers on the same checkpoints. The neural-head columns form the "before PCA/SVM" baseline first shown at Review-I.

**Table 7.** Phase 1 results on the original split: neural head versus PCA + linear SVM (sources: experiments/<backbone>/results/neural_head_results.json, final_results.json, training_results.json; results/neural_vs_pca_svm_comparison.csv). Δ = PCA+SVM test accuracy − neural-head test accuracy.

| Backbone | Best ep. | Epochs run/max | NH val acc | NH test acc | NH test F1 | SVM val acc | SVM test acc | SVM test F1 | Δ test |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | 16 | 19/30 | 94.58% | 93.67% | 93.32% | 94.93% | 94.97% | 94.68% | +1.30 pp |
| VGG19 | 18 | 21/30 | 94.58% | 93.14% | 92.72% | 94.49% | 93.52% | 93.10% | +0.38 pp |
| ResNet50 | 7 | 10/10 | 97.73% | 97.33% | 97.19% | 97.55% | 98.09% | 98.01% | +0.76 pp |
| ResNet152 | 8 | 10/10 | 98.25% | 98.17% | 98.06% | 98.25% | 98.40% | 98.32% | +0.23 pp |
| EfficientNet-B3 | 6 | 9/10 | 98.43% | 98.02% | 97.94% | 98.51% | 98.47% | 98.41% | +0.46 pp |

**Figure 4.** Phase 1 test accuracy of the neural head and of the PCA + linear SVM classifier for each backbone (original 1,311-image test set). *(figure file: p1_neural_vs_svm.png)*

### 6.3 PCA / SVM results

With the hybrid classifier, test accuracy was 94.97% (VGG16), 93.52% (VGG19), 98.09% (ResNet50), 98.40% (ResNet152) and 98.47% (EfficientNet-B3). The notumor class was the easiest for every backbone. The largest confusions were between glioma and meningioma: for example, ResNet152 classified 7 of 300 test gliomas as meningioma (`final_results.json → test.confusion_matrix`).

### 6.4 Downstream-classifier comparison

Replacing the neural head by StandardScaler → PCA → SVM raised test accuracy for all five backbones, by +0.23 pp to +1.30 pp. On validation, however, the change was mixed: +0.35 pp for VGG16, -0.09 pp for VGG19, -0.17 pp for ResNet50, +0.00 pp for ResNet152 and +0.09 pp for EfficientNet-B3. At the level of individual test images, the SVM corrected more neural-head errors than it introduced (for ResNet152: 11 images neural-wrong/SVM-right versus 8 the reverse; `results/neural_vs_pca_svm_comparison.csv`).

> **Interpretation.** The downstream classifier improved Phase 1 *test* accuracy and macro-F1 for every backbone, but it did not uniformly improve *validation* accuracy. A universal PCA benefit is therefore not claimed. Two things change at once (dimensionality reduction *and* the classifier family), so this is a **downstream-classifier comparison**, not a PCA-only ablation. The residual and EfficientNet backbones are about 3–5 pp above the frozen VGG backbones. That gap reflects architecture, the frozen-versus-fine-tuned policy and the unequal epoch budgets together.

### 6.5 Interpretation

Phase 1 answered the question "does the hybrid pipeline work?" in the affirmative: all five pipelines run end to end, and the three fine-tuned backbones reach about 98% on the supplied test set. These numbers were initially read as strong generalization. Chapter 7 shows why that reading had to be withdrawn: they are real recorded outputs, but on a split that is not image-level independent. They are reported from here on as **historical results on the contaminated original split**.

## 7 Phase 2D — Data Integrity Audit

**Roadmap status: COMPLETE.** The Review-I roadmap placed data integrity in Phase 2D and applied it across all Phase 2 evaluations. In practice the audit was run on 20 September 2026, before Phase 2A and 2B. Its result changed how every later number had to be interpreted, which is why it is presented here, immediately after Phase 1.

### 7.1 Why the audit was required

By mid-September the project appeared ready to lock: five working pipelines, about 98% test accuracy, a GUI and explainability. A pre-lock review identified one remaining unverified assumption that every result silently relied on: that the training, validation and test images were independent. In public image collections the same image can appear more than once, sometimes re-encoded or slightly altered. If copies sit on both sides of a split, the test set partly measures memorization. This is **duplicate leakage**.

### 7.2 Exact hashing

`final_project_qc.py` computed a SHA-256 hash of every file's bytes. Two files with the same hash are byte-identical copies.

### 7.3 Decoded pixel comparison

A second SHA-256 was computed over the decoded pixel array. This catches images that are visually and numerically identical but stored with different file encodings or metadata, which file hashing alone would miss.

### 7.4 Perceptual hashing

Perceptual hashes summarize an image's coarse visual structure, so that slightly altered copies (resized, recompressed, brightness-shifted) still produce similar 64-bit codes. The audit used pHash (candidate threshold: Hamming distance ≤ 4) and dHash (≤ 6). It then confirmed candidates with grayscale mean absolute error and correlation. Strict criteria made a pair a *high-confidence near-duplicate*; looser criteria (MAE ≤ 12, correlation ≥ 0.995) made it a *review-only* candidate, with contact sheets generated for visual inspection. All three cross-split pairings were compared: train↔validation, train↔test and validation↔test.

### 7.5 Findings

The audit (`final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json`, created 2026-09-20 17:53 UTC) hashed all 7,023 images and returned the status **MATERIAL_DUPLICATION**:

**Table 8.** Cross-split duplicate audit of the original Nickparvar split (duplicate_leakage_audit.json; duplicate_leakage_findings.csv).

| Finding | Count |
|---|---|
| Exact file or exact decoded-pixel cross-split pairs | 464  (304 file + 160 pixel-only) |
| Additional high-confidence near-duplicate pairs | 98 |
| Material cross-split pairs (sum of the above) | 562 |
| Additional review-only perceptual candidates | 481 |
| Material pairs by split pairing | train–test 344; train–validation 146; validation–test 72 |
| Unique images in a material pair | train 354/4,569 (7.75%); validation 122/1,143 (10.67%); test 250/1,311 (19.07%) |
| Implicated test images by class | glioma 0; meningioma 109; notumor 136; pituitary 5 |
| Material pairs whose two images carry different class labels | 0 |

**Figure 5.** Composition of the duplicate audit: pair types (left) and the share of each partition's images involved in at least one material pair (right). *(figure file: audit_composition.png)*

> **Interpretation.** One image can appear in several pairs, so pair counts overstate affected images. The unique-image counts are the better measure: about one test image in five (250 of 1,311) has a material counterpart in the training or validation partition. Most of this overlap was already present between the dataset's separately supplied Training and Testing folders; the project's own split did not create it. Train–validation overlap arises because duplicates already existed inside the Training folder and the random split placed copies on both sides. Every material pair carries the same class label on both sides, so this is duplication, not label noise. No glioma test image was implicated; the overlap is concentrated in meningioma and notumor.

### 7.6 Impact on interpretation

The audit did not invalidate the models, but it invalidated one *interpretation* of them. The earlier statement that ~98% test accuracy measured independent generalization was withdrawn. The Phase 1 numbers (and the Phase 2 internal numbers, which use the same split) remain real recorded outputs, reported as historical results on a contaminated split. Patient-level independence could not be assessed in any case, because the dataset has no patient identifiers.

The immediate post-audit plan (recorded in `final_project_docs/FINAL_QC_BLOCKER.md` and the historical transfer documents) treated full duplicate-group-aware re-splitting and retraining as a blocking requirement, and recorded the project as "NOT locked". That record is preserved unchanged as the historical state at that moment. Its leakage warning remains valid for the original split. Project review then refined the plan into the authoritative policy followed for the rest of the project:

1. preserve the original metrics transparently, labelled as historical contaminated-split outputs;
2. quantify a narrow post-hoc leakage-aware internal test (Sections 7.7–7.9);
3. continue the controlled Phase 2 studies with the leakage caveat stated;
4. obtain stronger generalization evidence from an independently audited external dataset (Chapters 10–11);
5. keep duplicate-group-aware retraining as a stronger, optional future remediation, needed only for a clean internal-generalization claim.

The post-hoc evaluation (`evaluate_leakage_clean_test.py`) loaded each experiment's saved test PCA features, labels and SVM, removed flagged test images under three pre-defined policies, and re-scored. Nothing was retrained and nothing was refitted.

### 7.7 exact_clean

Removes test images with an exact file or exact decoded-pixel duplicate across splits: **216 excluded, n = 1,095** (class supports 300 / 200 / 300 / 295). This is the most lenient policy.

### 7.8 conservative_clean

Removes all material pairs *and* all review-only candidates: **385 excluded, n = 926** (supports 300 / 197 / 134 / 295). This policy removes many notumor images (405 → 134) and so changes the class balance strongly. It is a robustness check rather than a replacement estimate.

### 7.9 material_clean

Removes test images in any material pair (exact or high-confidence near-duplicate): **250 excluded, n = 1,061** (supports 300 / 197 / 269 / 295). It was designated the **primary** leakage-aware internal evaluation in advance, because it corresponds directly to the audit's pre-specified definition of material duplication.

**Table 9.** Phase 1 PCA + SVM test accuracy under the post-hoc leakage policies (final_project_docs/leakage_corrected_evaluation/leakage_corrected_summary.csv and PRIMARY_material_clean_comparison.csv). Δ = material_clean − original accuracy.

| Backbone | Original (1,311) | exact_clean (1,095) | material_clean acc (1,061) | material_clean macro F1 | Δ acc | conservative (926) |
|---|---|---|---|---|---|---|
| VGG16 | 94.966% | 94.977% | 94.816% | 94.302% | -0.149 pp | 94.060% |
| VGG19 | 93.516% | 93.242% | 93.025% | 92.415% | -0.491 pp | 92.009% |
| ResNet50 | 98.093% | 98.356% | 98.303% | 98.134% | +0.210 pp | 98.056% |
| ResNet152 | 98.398% | 98.447% | 98.398% | 98.315% | -0.000 pp | 98.164% |
| EfficientNet-B3 | 98.474% | 98.630% | 98.680% | 98.511% | +0.206 pp | 98.488% |

**Figure 6.** Phase 1 PCA + SVM test accuracy on the original test set and under the three post-hoc leakage policies. *(figure file: leakage_policies.png)*

> **Interpretation.** Removing the directly implicated test images changed accuracy by less than 0.5 pp for every backbone. Two improved slightly, one was essentially unchanged and two declined slightly. The ordering (three fine-tuned backbones well above the two frozen VGG backbones) was preserved. This shows that the strong Phase 1 test scores were **not driven solely by the identified duplicated test images**. It does **not** show that the models are leakage-free. The networks were trained, and their checkpoints selected, on a train/validation split that itself contains duplicated content, so validation accuracy and best-epoch choices may be optimistic. material_clean removes contaminated *test* images; it does not repair contaminated *training*.

### 7.10 Limitations of the audit

- Image-level only: without patient identifiers, two different slices of the same patient cannot be detected, so **patient-level independence cannot be established**.
- Near-duplicates below the perceptual thresholds may remain undetected.
- Duplicates lying entirely inside one partition were not enumerated (they do not cause cross-split leakage).
- The leakage-aware evaluation is post-hoc and changes the class composition of the test set, so before/after differences partly reflect composition.
- Two of the explainability images (Te-me_0012, an exact-pixel duplicate of a training image, and Te-no_0060, an exact-file duplicate of a validation image) are in the material set. The original XAI ratings are preserved unchanged (Section 13.8).

## 8 Phase 2A — Standardized 30-Epoch Training

**Roadmap status: COMPLETE.**

### 8.1 Rationale

In Phase 1 the five backbones trained for different numbers of epochs (9–21), with early stopping deciding when to stop. Early stopping is efficient, but it confounds comparison. A backbone that stops at epoch 9 has never been observed at epoch 25, so one cannot tell whether it would have improved, plateaued or deteriorated. A best-checkpoint summary can also hide late-epoch instability. Phase 2A therefore removed the training horizon as a variable.

### 8.2 Fixed 30-epoch protocol

`actual_training_scripts/train_backbone_30epoch.py` trained all five baseline configurations (RMSprop, LR 1e-4, LSTM 256, dropout 0.3, dense 128, batch 8, seed 42) for exactly 30 epochs with no early termination. All five completed 30/30 epochs (`phase2/experiments_30epoch/<backbone>/results/training_results.json`).

### 8.3 Retained states

Two checkpoints were saved per backbone: `best_val_within_30.pth` (highest neural validation accuracy within the 30 epochs) and `epoch_30_model.pth` (the state after the final epoch). `evaluate_backbone_30epoch.py` then passed *both* through the full StandardScaler → PCA → SVM path, giving 10 full-hybrid evaluations (`evaluations/{best_val_within_30, epoch_30}/results/final_results.json`). Both were evaluated because the checkpoint with the best neural-head validation accuracy need not give the best downstream SVM.

**An important technical detail.** For VGG16, VGG19 and ResNet50 the best neural validation accuracy occurred at epoch 30 itself (`best_epoch = 30`). For these three backbones the two retained states represent the same training endpoint, and their downstream metrics are identical. Only 7 of the 10 evaluations are therefore distinct results. The checkpoint *files* are not byte-identical, however (their SHA-256 hashes differ), and the representative frozen for external validation is formally `best_val_within_30` at epoch 30, not `epoch_30_model.pth` (`frozen_15_selection.csv → checkpoint_state`). This report uses that exact label.

### 8.4 Results

**Table 10.** Phase 2A results. NN = neural-head validation accuracy from training history; SVM columns = full-hybrid PCA + SVM. The representative state is chosen by full-hybrid validation accuracy (ties → macro-F1 → prefer best_val_within_30). Internal test accuracy is on the contaminated original split and was not used for selection.

| Backbone | Best NN ep. | NN best val | NN val ep 30 | Representative state | SVM val acc | SVM val F1 | SVM val (epoch_30) | Internal test acc* | material_clean acc* |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | 30 | 95.45% | 95.45% | best_val_within_30 (ep 30) | 95.888% | 95.746% | 95.888% | 96.186% | 95.853% |
| VGG19 | 30 | 95.19% | 95.19% | best_val_within_30 (ep 30) | 95.626% | 95.462% | 95.626% | 95.271% | 95.099% |
| ResNet50 | 30 | 98.43% | 98.43% | best_val_within_30 (ep 30) | 98.688% | 98.644% | 98.688% | 98.932% | 98.963% |
| ResNet152 | 10 | 98.78% | 98.69% | best_val_within_30 (ep 10) | 98.600% | 98.545% | 98.513% | 98.703% | 98.869% |
| EfficientNet-B3 | 27 | 99.30% | 98.69% | best_val_within_30 (ep 27) | 99.300% | 99.270% | 99.213% | 99.314% | 99.246% |

**Figure 7.** Phase 2A neural-head validation accuracy per epoch for all five backbones under the fixed 30-epoch protocol; dots mark the best epoch within 30. *(figure file: p2a_curves.png)*

> **Interpretation.** Under a common horizon the ranking by full-hybrid validation was EfficientNet-B3 (99.300%), ResNet50 (98.688%), ResNet152 (98.600%), VGG16 (95.888%) and VGG19 (95.626%). Every backbone improved on its Phase 1 full-hybrid validation accuracy. For the frozen VGG backbones (best epoch 30, versus 16–18 in Phase 1) this suggests that Phase 1's early stopping halted them before convergence. Differences among the top three are fractions of a percentage point on a single seed and a contaminated validation split, and are not claimed to be significant.

### 8.5 Convergence interpretation

- **VGG16 / VGG19** kept improving until the last epoch (best epoch 30). With a frozen backbone only the recurrent and dense layers learn, and they learn slowly.
- **ResNet152** peaked early (epoch 10, neural validation 98.78%) and then stayed close to that level (98.69% at epoch 30).
- **EfficientNet-B3** peaked late (epoch 27, 99.30%) and fell slightly by epoch 30 (98.69%).
- **ResNet50** reached its best at epoch 30 (98.43%).

These trajectories, invisible under Phase 1's early stopping, are why fixed-horizon curves were kept for every later run.

### 8.6 Leakage-aware Phase 2A results

The material_clean policy was also applied to the Phase 2A retained states (`evaluate_leakage_clean_test_phase2.py`; `phase2/results/leakage_corrected_30epoch/PHASE2A_PRIMARY_material_clean_comparison.csv`). Seven distinct states were evaluated, because for VGG16, VGG19 and ResNet50 the best and epoch-30 states coincide.

**Table 11.** Phase 2A material_clean results (n = 1,061; 250 excluded). Full = original 1,311-image test accuracy of the same state.

| Experiment | Backbone | State | Epoch | Full test acc | material_clean acc | material_clean F1 | Δ acc |
|---|---|---|---|---|---|---|---|
| vgg16 epoch30 | VGG16 | best_val_within_30 | 30 | 96.186% | 95.853% | 95.442% | -0.333 pp |
| vgg19 epoch30 | VGG19 | best_val_within_30 | 30 | 95.271% | 95.099% | 94.655% | -0.172 pp |
| resnet50 epoch30 | ResNet50 | best_val_within_30 | 30 | 98.932% | 98.963% | 98.848% | +0.031 pp |
| resnet152 best epoch10 | ResNet152 | best_val_within_30 | 10 | 98.703% | 98.869% | 98.788% | +0.166 pp |
| resnet152 epoch30 | ResNet152 | epoch_30 | 30 | 98.627% | 98.680% | 98.598% | +0.053 pp |
| efficientnet b3 best epoch27 | EfficientNet-B3 | best_val_within_30 | 27 | 99.314% | 99.246% | 99.148% | -0.068 pp |
| efficientnet b3 epoch30 | EfficientNet-B3 | epoch_30 | 30 | 99.161% | 99.246% | 99.151% | +0.085 pp |

> **Interpretation.** As in Phase 1, removing the implicated test images changed accuracy by less than 0.35 pp. The final project representative (ResNet152 Phase 2A, epoch 10) scored 98.869% accuracy and 98.788% macro-F1 on material_clean, slightly above its 98.703% on the full test set. The same caveat applies: this is a post-hoc test-side check on a contaminated split, not a clean internal estimate.

## 9 Phase 2B — Hyperparameter Sensitivity

**Roadmap status: COMPLETE.**

### 9.1 Design

Phase 2B asked which of three deep-learning hyperparameters — learning rate, LSTM hidden size and dropout — most changes performance under an otherwise fixed protocol. It used a **one-factor-at-a-time** design around the Phase 2A baseline (LR 1e-4, LSTM 256, dropout 0.3): exactly one factor was changed per run, and every other setting, including the 30-epoch horizon and the downstream classifier, stayed fixed. The literature review motivated this choice (Section 2.6). A metaheuristic or grid search would find a good configuration but make the effect of each factor hard to attribute. The aim here was interpretable sensitivity, not maximum accuracy.

**Table 12.** Phase 2B one-factor configurations (changed value in each row; all other settings as the Phase 2A baseline).

| Run name | Learning rate | LSTM hidden | Dropout |
|---|---|---|---|
| lr_1e-5 | 1e-5 | 256 | 0.3 |
| lr_5e-5 | 5e-5 | 256 | 0.3 |
| lr_5e-4 | 5e-4 | 256 | 0.3 |
| lstm_128 | 1e-4 | 128 | 0.3 |
| lstm_512 | 1e-4 | 512 | 0.3 |
| dropout_0.2 | 1e-4 | 256 | 0.2 |
| dropout_0.5 | 1e-4 | 256 | 0.5 |

Five backbones × seven configurations = **35 training runs**, all completed for 30/30 epochs (`phase2/hyperparameter_dl/<backbone>/<run>/results/training_results.json`; verified count 35). Both retained states of every run were evaluated through the full hybrid path: 35 × 2 = **70 downstream PCA + SVM evaluations** (verified count 70). The runner was `phase2/run_all_dl_hyperparams.py`, the training worker `train_backbone_hyperparams.py` and the downstream worker `evaluate_hyperparam_pca_svm.py`. According to the project record, one run (VGG16 lr_1e-5) had already completed and was skipped by the batch runner, while the other 34 were executed in the queue.

**Why the test set was deliberately not evaluated.** Phase 2B compares 70 states and then picks the best one per backbone. If the test set were scored for every state, picking a winner would implicitly use test results, and the test set would become a tuning set. Phase 2B therefore extracted only training and validation features (`evaluations/*/features/` holds train and validation arrays only), and none of the 70 result files contains a test metric (verified: 0 test keys). All Phase 2B selection is validation-based. This is intentional, not missing data.

### 9.2 Learning-rate study

The learning rate had by far the largest effect. LR 1e-5 underfitted badly for the frozen VGG backbones (VGG16 peak neural validation 90.20%, VGG19 89.94%, versus about 95% at baseline). For them LR 5e-4 was tolerated well (VGG16 95.98%). For the fine-tuned ResNets the opposite held: LR 5e-4 was damaging (ResNet50 94.75%, ResNet152 93.96%). EfficientNet-B3 was best at the baseline LR among the learning-rate settings. **Learning-rate preference is therefore architecture-dependent**, which is consistent with frozen backbones needing larger updates to their few trainable layers.

The clearest late-epoch instability was ResNet50 at LR 5e-4: best neural validation 94.75% at epoch 25, then 90.99% at epoch 30 (-3.76 pp).

### 9.3 LSTM-size study

Replacing LSTM 256 by LSTM 512 raised peak neural validation for all five backbones, but only modestly (+0.09 pp to +0.70 pp). Several of these peaks were not maintained to epoch 30 (for example EfficientNet-B3 LSTM 512: 99.39% at epoch 29 versus 98.43% at epoch 30). LSTM 128 was mostly close to baseline. The evidence supports a *modest peak-improvement association*, not universal superiority of a wider LSTM.

### 9.4 Dropout study

Dropout 0.2 and 0.5 produced the smallest and least consistent changes. ResNet50 with dropout 0.2 reached the highest ResNet50 peak (99.13%), while for other backbones the direction of the effect varied.

### 9.5 All 35 runs

Table Table 13 lists every Phase 2B run. Figure Figure 8 shows the peak neural validation accuracy of each configuration alongside the Phase 2A baseline.

**Figure 8.** Peak neural validation accuracy (%) for the Phase 2A baseline and each of the seven Phase 2B configurations, per backbone. *(figure file: p2b_heatmap.png)*

**Table 13.** All 35 Phase 2B runs: neural peak and epoch-30 validation accuracy, late-epoch drift, and the full-hybrid SVM validation accuracy of both retained states (sources: phase2/hyperparameter_dl/<backbone>/<run>/results/training_results.json and evaluations/*/results/final_results.json).

| Backbone | Run | Best ep. | NN peak val | NN val ep 30 | Drift | SVM val (best state) | SVM val (epoch_30) |
|---|---|---|---|---|---|---|---|
| VGG16 | lr_1e-5 | 30 | 90.20% | 90.20% | +0.00 pp | 91.514% | 91.514% |
| VGG16 | lr_5e-5 | 30 | 95.28% | 95.28% | +0.00 pp | 95.188% | 95.188% |
| VGG16 | lr_5e-4 | 27 | 95.98% | 95.98% | +0.00 pp | 96.500% | 95.976% |
| VGG16 | lstm_128 | 30 | 95.80% | 95.80% | +0.00 pp | 95.626% | 95.626% |
| VGG16 | lstm_512 | 19 | 95.89% | 95.63% | -0.26 pp | 95.363% | 95.538% |
| VGG16 | dropout_0.2 | 27 | 95.63% | 95.54% | -0.09 pp | 95.538% | 96.413% |
| VGG16 | dropout_0.5 | 30 | 95.54% | 95.54% | +0.00 pp | 95.801% | 95.801% |
| VGG19 | lr_1e-5 | 30 | 89.94% | 89.94% | +0.00 pp | 89.764% | 89.764% |
| VGG19 | lr_5e-5 | 28 | 94.66% | 94.05% | -0.61 pp | 94.226% | 94.401% |
| VGG19 | lr_5e-4 | 27 | 95.54% | 95.01% | -0.52 pp | 95.451% | 95.188% |
| VGG19 | lstm_128 | 28 | 94.66% | 94.31% | -0.35 pp | 95.276% | 94.401% |
| VGG19 | lstm_512 | 30 | 95.89% | 95.89% | +0.00 pp | 95.276% | 95.276% |
| VGG19 | dropout_0.2 | 29 | 94.84% | 94.75% | -0.09 pp | 94.751% | 95.101% |
| VGG19 | dropout_0.5 | 28 | 95.36% | 95.19% | -0.17 pp | 95.451% | 95.538% |
| ResNet50 | lr_1e-5 | 27 | 96.94% | 96.76% | -0.17 pp | 97.638% | 96.850% |
| ResNet50 | lr_5e-5 | 23 | 98.60% | 98.43% | -0.17 pp | 98.600% | 98.688% |
| ResNet50 | lr_5e-4 | 25 | 94.75% | 90.99% | -3.76 pp | 95.888% | 94.838% |
| ResNet50 | lstm_128 | 28 | 98.43% | 96.68% | -1.75 pp | 98.425% | 97.900% |
| ResNet50 | lstm_512 | 29 | 99.04% | 98.25% | -0.79 pp | 98.863% | 98.688% |
| ResNet50 | dropout_0.2 | 28 | 99.13% | 98.51% | -0.61 pp | 99.125% | 98.513% |
| ResNet50 | dropout_0.5 | 26 | 98.34% | 97.46% | -0.87 pp | 98.688% | 97.988% |
| ResNet152 | lr_1e-5 | 29 | 97.03% | 96.94% | -0.09 pp | 97.113% | 97.375% |
| ResNet152 | lr_5e-5 | 17 | 98.95% | 98.51% | -0.44 pp | 99.213% | 98.600% |
| ResNet152 | lr_5e-4 | 20 | 93.96% | 93.61% | -0.35 pp | 95.101% | 94.488% |
| ResNet152 | lstm_128 | 29 | 98.95% | 98.69% | -0.26 pp | 98.950% | 98.688% |
| ResNet152 | lstm_512 | 26 | 99.21% | 98.78% | -0.44 pp | 99.300% | 98.688% |
| ResNet152 | dropout_0.2 | 27 | 98.60% | 98.08% | -0.52 pp | 99.125% | 98.425% |
| ResNet152 | dropout_0.5 | 29 | 98.86% | 98.43% | -0.44 pp | 98.775% | 98.513% |
| EfficientNet-B3 | lr_1e-5 | 28 | 96.41% | 96.33% | -0.09 pp | 96.938% | 97.375% |
| EfficientNet-B3 | lr_5e-5 | 14 | 98.95% | 98.95% | +0.00 pp | 98.950% | 98.600% |
| EfficientNet-B3 | lr_5e-4 | 27 | 98.43% | 97.64% | -0.79 pp | 98.600% | 98.425% |
| EfficientNet-B3 | lstm_128 | 30 | 99.21% | 99.21% | +0.00 pp | 99.038% | 99.038% |
| EfficientNet-B3 | lstm_512 | 29 | 99.39% | 98.43% | -0.96 pp | 99.388% | 98.863% |
| EfficientNet-B3 | dropout_0.2 | 15 | 98.95% | 98.69% | -0.26 pp | 98.950% | 98.863% |
| EfficientNet-B3 | dropout_0.5 | 26 | 99.13% | 98.69% | -0.44 pp | 99.125% | 98.600% |

### 9.6 All 70 downstream evaluations

Across the 35 changed configurations, the best-neural state gave the higher SVM validation accuracy in **20** cases, the epoch-30 state in **8**, and the two were identical in **7**. The ties are exactly the runs whose best neural epoch was 30. The best neural checkpoint and the best downstream SVM state are therefore usually, but not always, the same. Only two retained states per run were evaluated downstream, so the absolute best SVM epoch among all 30 was not searched.

**Table 14.** Strongest changed configuration per backbone (highest full-hybrid validation accuracy among its 14 Phase 2B states; ties → macro-F1 → prefer best_val_within_30). These are the Phase 2B representatives later frozen for external validation.

| Backbone | Configuration | State | Epoch | SVM val acc | SVM val F1 | Phase 2A val (ref.) |
|---|---|---|---|---|---|---|
| VGG16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% | 95.888% |
| VGG19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% | 95.626% |
| ResNet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% | 98.688% |
| ResNet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% | 98.600% |
| EfficientNet-B3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% | 99.300% |

### 9.7 Sensitivity analysis

Sensitivity was summarized as the within-backbone range (maximum − minimum) of peak neural validation accuracy across the settings of each factor family, including the Phase 2A baseline, using the 40 matched training histories (5 baselines + 35 variants).

**Table 15.** Within-backbone sensitivity ranges (pp) of peak neural validation accuracy, and the LSTM 512 − LSTM 256 peak difference.

| Backbone | Learning-rate range | LSTM-width range | Dropout range | LSTM512 − baseline |
|---|---|---|---|---|
| VGG16 | 5.77 pp | 0.44 pp | 0.17 pp | +0.44 pp |
| VGG19 | 5.60 pp | 1.22 pp | 0.52 pp | +0.70 pp |
| ResNet50 | 3.85 pp | 0.61 pp | 0.79 pp | +0.61 pp |
| ResNet152 | 4.99 pp | 0.44 pp | 0.26 pp | +0.44 pp |
| EfficientNet-B3 | 2.89 pp | 0.17 pp | 0.35 pp | +0.09 pp |
| Mean | 4.62 pp | 0.58 pp | 0.42 pp |  |

**Figure 9.** Within-backbone sensitivity ranges by factor family. *(figure file: p2b_ranges.png)*

> **Interpretation.** Learning rate dominated: its mean range (4.62 pp) was about eight times the LSTM-width range (0.58 pp) and eleven times the dropout range (0.42 pp). Within the tested values, the architecture choice and learning rate matter far more than LSTM width or dropout. The Phase 2B representatives improved internal validation over Phase 2A for four of five backbones, by 0.09–0.70 pp (VGG19 being the exception, −0.09 pp). With one seed and one contaminated validation split, these are controlled sensitivity findings, **not statistically established superiority claims**. The one-factor design also does not test interactions between factors.

### 9.8 Interpretation

Phase 2B succeeded in its stated aim of attributing sensitivity to individual factors. It also produced a set of slightly better internally-validated configurations. Whether those small internal gains are real improvements, or partly adaptation to one validation split, cannot be answered internally. That question motivated freezing *both* Phase 2A and Phase 2B representatives before external validation (Chapter 10).

## 10 Phase 2C — External Validation

**Roadmap status: COMPLETE.** The Review-I roadmap specified Phase 2C as: use an independent external brain-MRI dataset; document provenance and class mapping; audit overlap; freeze the CNN–LSTM, scaler, PCA and SVM; do not tune externally; and report accuracy, macro metrics, per-class results, confusion matrices and the internal → external shift.

### 10.1 Why external validation

After the audit, internal numbers could no longer stand as generalization evidence on their own. A model can only be shown to generalize by evaluating it on images it has never seen, from a different source, with nothing adjusted afterwards. External validation was therefore the most important remaining experiment. Duplicate-group-aware retraining would have produced cleaner *internal* estimates, but it would still not test transfer to a different source.

### 10.2 Frozen 15-pipeline protocol

Rather than exposing only one model, the project compared architectures across the whole project evolution. Before any external image was evaluated, exactly 15 full hybrid pipelines were selected and frozen: for each of the five backbones, the Phase 1 early-stopping pipeline, the Phase 2A representative and the Phase 2B representative. Selection used internal evidence only. The manifest records the rule: *"Use INTERNAL full-hybrid validation accuracy; break ties by macro-F1; on an exact tie prefer best_val_within_30 over epoch_30."* (Figure 10).

**Figure 10.** Pre-external freeze protocol. The frozen manifest predates every external result file; all 75 artifact hashes were re-verified during preparation of this report. *(figure file: frozen_protocol.png)*

**Table 16.** The 15 frozen pipelines (phase2/external_validation/frozen_15_selection.csv; frozen_15_manifest.json). Validation figures are internal full-hybrid validation on the original split.

| Stage | Backbone | Variant | Checkpoint state | Epoch | Int. val acc | Int. val F1 |
|---|---|---|---|---|---|---|
| Phase 1 | VGG16 | early_stopping | best_model | 16 | 94.926% | 94.718% |
| Phase 2A | VGG16 | lr1e-4_lstm256_dropout0.3 | best_val_within_30 | 30 | 95.888% | 95.746% |
| Phase 2B | VGG16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% |
| Phase 1 | VGG19 | early_stopping | best_model | 18 | 94.488% | 94.309% |
| Phase 2A | VGG19 | lr1e-4_lstm256_dropout0.3 | best_val_within_30 | 30 | 95.626% | 95.462% |
| Phase 2B | VGG19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% |
| Phase 1 | ResNet50 | early_stopping | best_model | 7 | 97.550% | 97.473% |
| Phase 2A | ResNet50 | lr1e-4_lstm256_dropout0.3 | best_val_within_30 | 30 | 98.688% | 98.644% |
| Phase 2B | ResNet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% |
| Phase 1 | ResNet152 | early_stopping | best_model | 8 | 98.250% | 98.203% |
| Phase 2A | ResNet152 | lr1e-4_lstm256_dropout0.3 | best_val_within_30 | 10 | 98.600% | 98.545% |
| Phase 2B | ResNet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% |
| Phase 1 | EfficientNet-B3 | early_stopping | best_model | 6 | 98.513% | 98.486% |
| Phase 2A | EfficientNet-B3 | lr1e-4_lstm256_dropout0.3 | best_val_within_30 | 27 | 99.300% | 99.270% |
| Phase 2B | EfficientNet-B3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% |

### 10.3 Hash manifest

`phase2/external_validation/frozen_15_manifest.json` stores, for every candidate, the SHA-256 hashes of five artifacts: checkpoint, scaler, PCA, SVM and internal results file. That is 15 × 5 = **75 hashes**. The manifest file is time-stamped 2026-10-02 16:50 UTC, before the first external result file (2026-10-03 07:54 UTC). While preparing this report every hash was recomputed from the current files: **75 of 75 match, 0 mismatch**. The externally evaluated pipelines are byte-for-byte the frozen ones, and no model was retrained, refitted, recalibrated or re-selected afterwards. (An earlier transfer document contained only a *reconstructed* selection record without hashes. The original manifest in the repository supersedes it.)

### 10.4 BDNeuro

The external dataset is **BDNeuro-MRI Version 8** (`external_data/BDNeuro_V8_raw/`), a four-class brain-MRI collection with the same class scheme (glioma, meningioma, no_tumor → notumor, pituitary) [17]. It was chosen because it was structurally compatible with the task, documented a different source context, and had not been used anywhere in development. Its raw class cohort contains **5,941** MRIs: glioma 2,191, meningioma 1,478, notumor 787, pituitary 1,485, distributed over train 4,160 / val 892 / test 889 (`dataset_counts.json`). A naive recursive file count gives 5,943, because two root-level plot PNGs are not class MRIs.

### 10.5 Provenance investigation

The dataset's README describes its source as "Epic & CSCR Hospital". It states that exact and near-duplicates were removed *within* the dataset and that its splits have zero internal overlap. Its licence, citation and full attribution fields are unfilled placeholders, and this report records that as a citation limitation. The README also warns that its de-duplication is image-level, so patient-level leakage may remain. Documented provenance is a claim; the project tested it directly.

### 10.6 Overlap audit

`audit_bdneuro_overlap.py` compared all 5,941 BDNeuro images with all 7,023 Nickparvar development images using SHA-256 of the file bytes, SHA-256 of the decoded RGB pixels, and 64-bit pHash.

**Table 17.** BDNeuro ↔ Nickparvar overlap audit (phase2/external_validation/bdneuro_overlap_audit/summary.json; cross_exact_*_duplicates.csv).

| Measure | Cross-dataset | Within BDNeuro |
|---|---|---|
| Exact file duplicate pairs | 3,421 | 0 |
| Exact decoded-pixel duplicate pairs | 3,519 | 0 |
| Unique BDNeuro images with exact-file overlap | 3,295 | — |
| Unique BDNeuro images with exact-pixel overlap | 3,297 | — |
| Exact-pixel pairs with differing class label | 0 | — |
| pHash ≤ 6 candidate pairs | 9,981 | 1,588 |
| pHash ≤ 4 strong candidate pairs | 6,664 | 65 |

> **Interpretation.** More than half of the raw BDNeuro cohort (3,297 of 5,941 images, 55.5%) is pixel-identical to images in the development dataset, and every one of those pairs has the same class label in both datasets. Whatever the documented provenance, these images are reused content, and evaluating on them would partly measure memorization. The project does not speculate about which dataset lineage caused the reuse. The objective fact of decoded-pixel identity outranks the provenance narrative.

### 10.7 Exact-pixel exclusion

Final rule: **exclude exact decoded-pixel cross-dataset matches only**; do not automatically remove non-identical pHash candidates. Brain MRI is anatomically repetitive. Two different patients' axial slices at the same level can have very similar perceptual hashes, and with thousands of pHash candidates, pHash-based exclusion would have removed large numbers of genuinely different scans. pHash similarity was therefore treated as screening evidence, not as proof of identity.

### 10.8 The 2,644-image full cohort

`build_bdneuro_exact_clean_cohort.py` produced the primary external cohort: 5,941 − 3,297 = **2,644** images (`cohort_summary.json`): glioma 1159, meningioma 472, notumor 412, pituitary 601. The retained images came from the original BDNeuro splits train 1845, val 364 and test 435. They were merged into one evaluation cohort, because no BDNeuro image was used for training or selection. The raw dataset was preserved unchanged.

**Figure 11.** From the raw BDNeuro cohort to the exact-overlap-clean full cohort and the nested balanced subset. *(figure file: bd_funnel.png)*

### 10.9 The 1,648-image balanced subset

The natural clean cohort is imbalanced (glioma 43.8%, notumor 15.6%). Accuracy on it is therefore dominated by glioma. `build_bdneuro_balanced_subset.py` drew a deterministic **secondary sensitivity subset** of 412 images per class (the smallest class count), **1,648** in total, by SHA-256 ranking within each class (seed string `bdneuro-balanced-v1-seed42`), using no model outputs. The balanced subset is **nested inside** the full cohort. It is not a second independent dataset; it reweights the same images. On it, accuracy equals the mean per-class recall.

### 10.10 15-model results

All 15 frozen pipelines were evaluated with the deterministic evaluation transform by the repaired `compare_external_15.py` (one fixed results directory per dataset via `--results-name`). Tables Table 18 and Table 19 list every pipeline.

**Table 18.** BDNeuro exact-overlap-clean full cohort (n = 2,644): all 15 frozen pipelines (comparison_15_models_bdneuro_full/external_comparison_15.csv). Gap = external accuracy − internal full-hybrid validation accuracy.

| Backbone | Stage | Variant | Ep. | Int. val | Ext. acc | Macro F1 | Macro P | Macro R | Gap |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | Phase 1 | early_stopping | 16 | 94.93% | 77.61% | 75.50% | 77.66% | 77.62% | -17.32 pp |
| VGG16 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 95.89% | 80.07% | 78.71% | 80.87% | 80.89% | -15.82 pp |
| VGG16 | Phase 2B | lr_5e-4 | 27 | 96.50% | 77.57% | 76.01% | 78.16% | 78.80% | -18.93 pp |
| VGG19 | Phase 1 | early_stopping | 18 | 94.49% | 75.76% | 74.80% | 76.39% | 78.04% | -18.73 pp |
| VGG19 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 95.63% | 79.27% | 78.12% | 78.79% | 80.86% | -16.35 pp |
| VGG19 | Phase 2B | dropout_0.5 | 30 | 95.54% | 75.00% | 74.27% | 76.32% | 77.20% | -20.54 pp |
| ResNet50 | Phase 1 | early_stopping | 7 | 97.55% | 81.09% | 79.93% | 80.56% | 83.92% | -16.46 pp |
| ResNet50 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 98.69% | 81.73% | 81.42% | 82.50% | 85.46% | -16.96 pp |
| ResNet50 | Phase 2B | dropout_0.2 | 28 | 99.13% | 86.04% | 84.83% | 84.27% | 87.96% | -13.08 pp |
| ResNet152 | Phase 1 | early_stopping | 8 | 98.25% | 81.28% | 80.84% | 81.08% | 85.18% | -16.97 pp |
| ResNet152 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 10 | 98.60% | 88.31% | 87.85% | 86.43% | 90.16% | -10.29 pp |
| ResNet152 | Phase 2B | lstm_512 | 26 | 99.30% | 86.54% | 85.67% | 84.59% | 88.98% | -12.76 pp |
| EfficientNet-B3 | Phase 1 | early_stopping | 6 | 98.51% | 83.09% | 81.93% | 82.55% | 85.40% | -15.42 pp |
| EfficientNet-B3 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 27 | 99.30% | 86.65% | 85.12% | 85.12% | 87.87% | -12.65 pp |
| EfficientNet-B3 | Phase 2B | lstm_512 | 29 | 99.39% | 86.57% | 85.53% | 85.62% | 88.76% | -12.81 pp |

**Table 19.** BDNeuro balanced sensitivity subset (n = 1,648, 412/class): all 15 frozen pipelines (comparison_15_models_bdneuro_balanced/external_comparison_15.csv).

| Backbone | Stage | Variant | Ep. | Int. val | Ext. acc | Macro F1 | Macro P | Macro R | Gap |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | Phase 1 | early_stopping | 16 | 94.93% | 77.18% | 76.82% | 79.41% | 77.18% | -17.74 pp |
| VGG16 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 95.89% | 80.70% | 80.62% | 83.31% | 80.70% | -15.18 pp |
| VGG16 | Phase 2B | lr_5e-4 | 27 | 96.50% | 78.16% | 77.88% | 80.90% | 78.16% | -18.35 pp |
| VGG19 | Phase 1 | early_stopping | 18 | 94.49% | 77.49% | 77.64% | 80.19% | 77.49% | -17.00 pp |
| VGG19 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 95.63% | 81.13% | 81.24% | 82.76% | 81.13% | -14.50 pp |
| VGG19 | Phase 2B | dropout_0.5 | 30 | 95.54% | 77.00% | 77.15% | 79.72% | 77.00% | -18.54 pp |
| ResNet50 | Phase 1 | early_stopping | 7 | 97.55% | 83.74% | 83.80% | 86.73% | 83.74% | -13.81 pp |
| ResNet50 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30 | 98.69% | 86.77% | 86.90% | 89.12% | 86.77% | -11.92 pp |
| ResNet50 | Phase 2B | dropout_0.2 | 28 | 99.13% | 87.86% | 87.87% | 89.21% | 87.86% | -11.26 pp |
| ResNet152 | Phase 1 | early_stopping | 8 | 98.25% | 85.13% | 85.13% | 87.42% | 85.13% | -13.12 pp |
| ResNet152 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 10 | 98.60% | 90.47% | 90.50% | 90.85% | 90.47% | -8.13 pp |
| ResNet152 | Phase 2B | lstm_512 | 26 | 99.30% | 89.81% | 89.76% | 90.58% | 89.81% | -9.49 pp |
| EfficientNet-B3 | Phase 1 | early_stopping | 6 | 98.51% | 85.44% | 85.60% | 88.03% | 85.44% | -13.08 pp |
| EfficientNet-B3 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 27 | 99.30% | 87.86% | 87.79% | 89.19% | 87.86% | -11.44 pp |
| EfficientNet-B3 | Phase 2B | lstm_512 | 29 | 99.39% | 88.83% | 88.96% | 91.02% | 88.83% | -10.55 pp |

**Table 20.** Stage means over the five backbones.

| Stage | Full: mean acc | Full: mean macro F1 | Balanced: mean acc | Balanced: mean macro F1 |
|---|---|---|---|---|
| Phase 1 | 79.77% | 78.60% | 81.80% | 81.80% |
| Phase 2A | 83.21% | 82.25% | 85.39% | 85.41% |
| Phase 2B | 82.34% | 81.26% | 84.33% | 84.33% |

**Figure 12.** BDNeuro full-cohort accuracy of the 15 frozen pipelines, grouped by backbone and coloured by stage. *(figure file: bd_full_bars.png)*

**Figure 13.** BDNeuro balanced-subset accuracy of the 15 frozen pipelines. *(figure file: bd_bal_bars.png)*

> **Interpretation.** Every frozen pipeline lost accuracy relative to its internal validation, by -10.29 pp to -20.54 pp on the full cohort. Structured external accuracy nevertheless remained meaningful: 75.00%–88.31% on the full cohort and 77.00%–90.47% on the balanced subset. **ResNet152 Phase 2A** was the strongest pipeline on both (88.31% / macro-F1 87.85% full; 90.47% / 90.50% balanced) and had the smallest internal → external gap (-10.29 pp and -8.13 pp). On average Phase 2A generalized better than Phase 2B (83.21% versus 82.34% full; 85.39% versus 84.33% balanced), and both exceeded Phase 1 (79.77%; 81.80%).

### 10.11 Generalization gaps

**Figure 14.** Internal full-hybrid validation accuracy (grey) versus BDNeuro full-cohort accuracy (coloured by stage) for every frozen pipeline, sorted by gap. *(figure file: bd_gap_dumbbell.png)*

Three patterns stand out in Figure 14:

- **Phase 2B tuning did not transfer consistently.** Phase 2B beat Phase 2A externally only for ResNet50 (dropout 0.2: 86.04% vs 81.73% full) and, on the balanced subset, for EfficientNet-B3. The VGG Phase 2B representatives fell below their Phase 2A counterparts (VGG19: 75.00% vs 79.27%).
- **Higher internal validation did not predict higher external accuracy.** EfficientNet-B3 Phase 2B had the highest internal validation of all 15 (99.39%) but was not the best externally. ResNet152 Phase 2A had the *lowest* internal validation (98.60%) of the six ResNet/EfficientNet Phase 2A/2B candidates, yet it was the best externally.
- **The frozen-VGG gap persisted externally.** VGG pipelines were the weakest on both cohorts, consistent with their weaker internal performance.

Per-class recall on the full cohort (from the per-pipeline confusion matrices) shows where the transfer losses occur. Pituitary had the lowest recall in 12 of the 15 pipelines and glioma in the other 3, while notumor and meningioma recall were usually the highest. For the final representative, glioma recall was lowest (968 / 1,159 = 83.5%), against 94.9% for meningioma, 93.2% for notumor and 89.0% for pituitary (Figure 18). Glioma makes up 43.8% of the natural cohort but only 25% of the balanced subset, and its recall was below the pipeline's mean class recall in 12 of the 15 pipelines. Equal class weighting therefore raised overall accuracy on the balanced subset in 14 of the 15 pipelines; the exception was VGG16 Phase 1 (77.61% full, 77.18% balanced).

## 11 In-the-Wild Web Stress Test

**Status: COMPLETE (additional work beyond the original roadmap).**

### 11.1 Motivation

The project guide asked a practical question that a curated external dataset does not fully answer: *what happens if a user uploads an MRI image that is not from the development dataset — for example, an image found on the web?* BDNeuro tests structured transfer to another curated four-class collection. The web stress test probes a much harsher condition: heterogeneous images from medical publications and open-access pages, with different scanners, sequences, orientations, crops and presentation.

It is a **source-verified exploratory stress test**, not a clinical external validation. Its purpose is to observe failure modes under severe distribution shift, not to estimate real-world accuracy.

### 11.2 Source verification

Search engines could be used to *find* candidate images, but the diagnosis had to come from the *original source page* (case reports, open-access articles, medical reference sites). For each image the source list records the page URL, figure/panel, source-reported diagnosis, MRI sequence and orientation where known, and source type (`docs/DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx`; 40 rows, each with a source URL). The rules were: use only the indicated MRI panel; crop to isolate it without altering anatomy or contrast to help the model; reject CT, histology, segmentation maps, ambiguous diagnoses and heavily annotated figures; and never replace an image after seeing predictions. (The repository-side template `phase2/external_validation/web_stress_40/web_stress_40_manifest.csv` was never filled in; all 40 rows still read `PENDING`. The authoritative provenance record is the source-list workbook.)

### 11.3 Balanced 40-image design

Exactly 40 images, 10 per class (glioma, meningioma, notumor/healthy, pituitary), stored in `external_data/web_stress_40_raw/`. The cohort is balanced, so accuracy equals mean per-class recall, and each image is worth 2.5 percentage points.

### 11.4 Pre-inference audit

`audit_web_stress_40.py` checked the web images against each other, against all 7,023 Nickparvar images and against the 2,644 BDNeuro exact-clean images, using exact file hashes, decoded-pixel hashes and pHash. The initial audit found no exact file or pixel duplicates within the web set and no exact overlaps with either dataset, but two pHash candidates (distance ≤ 6) required visual review.

### 11.5 NOR_01 decision

The original NOR_01 had pHash distance **0** to Nickparvar `Testing/notumor/Te-no_0227.jpg`, although its pixels were not identical. Side-by-side visual review showed the same underlying axial T2 slice in a different file and pixel representation. Because it was transformed reuse of a development image, NOR_01 was **rejected and replaced before any inference**. This is the one case where perceptual evidence led to exclusion. It was justified by visual confirmation, not by the hash distance alone. (Visual-review evidence: `phase2/external_validation/web_stress_40_audit/visual_review/NOR_01_comparison.jpg`.)

### 11.6 MEN_02 decision

MEN_02 had pHash distance **6** to Nickparvar `Training/meningioma/Tr-me_0633.jpg` (`cross_phash_candidates.csv`). Visual review showed a clearly different scan (different orientation and lesion presentation), so it was **retained**. This is the complementary lesson: a small pHash distance alone is not enough to exclude an image.

### 11.7 Cohort lock

After the replacement the audit was rerun (`web_stress_40_audit/summary.json`): 40 images, 10 per class; within-web exact file and pixel duplicate pairs 0 and 0; cross-dataset exact file and pixel pairs 0 and 0; images with pHash ≤ 4 0; one remaining pHash-6 candidate (MEN_02, already reviewed). The cohort was then locked (`web_stress_40_locked/lock_record.json`: status `LOCKED_BEFORE_INFERENCE`, cohort SHA-256 fingerprint recorded, rule: *"No image may be replaced, removed, recropped, edited or substituted after model predictions are viewed."*). The lock record is time-stamped 10:37:22 UTC on 3 October 2026, and the 15-model web comparison output 10:38:09 UTC: the lock preceded inference.

### 11.8 600 predictions

The same 15 frozen pipelines were applied, with no retraining, refitting or re-selection: 15 × 40 = **600 predictions** (`comparison_15_models_web_stress_40/`; per-image predictions CSV per pipeline).

### 11.9 Results

**Table 21.** Web-40 stress test: all 15 frozen pipelines (comparison_15_models_web_stress_40/external_comparison_15.csv). Gap = web accuracy − internal full-hybrid validation accuracy.

| Backbone | Stage | Variant | Accuracy | Correct | Macro F1 | Gap |
|---|---|---|---|---|---|---|
| VGG16 | Phase 1 | early_stopping | 25.0% | 10/40 | 14.58% | -69.93 pp |
| VGG16 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 30.0% | 12/40 | 23.23% | -65.89 pp |
| VGG16 | Phase 2B | lr_5e-4 | 30.0% | 12/40 | 21.59% | -66.50 pp |
| VGG19 | Phase 1 | early_stopping | 35.0% | 14/40 | 28.89% | -59.49 pp |
| VGG19 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 37.5% | 15/40 | 27.78% | -58.13 pp |
| VGG19 | Phase 2B | dropout_0.5 | 35.0% | 14/40 | 25.45% | -60.54 pp |
| ResNet50 | Phase 1 | early_stopping | 30.0% | 12/40 | 26.49% | -67.55 pp |
| ResNet50 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 42.5% | 17/40 | 38.18% | -56.19 pp |
| ResNet50 | Phase 2B | dropout_0.2 | 40.0% | 16/40 | 33.91% | -59.13 pp |
| ResNet152 | Phase 1 | early_stopping | 35.0% | 14/40 | 28.08% | -63.25 pp |
| ResNet152 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 37.5% | 15/40 | 31.66% | -61.10 pp |
| ResNet152 | Phase 2B | lstm_512 | 35.0% | 14/40 | 25.74% | -64.30 pp |
| EfficientNet-B3 | Phase 1 | early_stopping | 40.0% | 16/40 | 34.78% | -58.51 pp |
| EfficientNet-B3 | Phase 2A | lr1e-4_lstm256_dropout0.3 | 42.5% | 17/40 | 31.97% | -56.80 pp |
| EfficientNet-B3 | Phase 2B | lstm_512 | 35.0% | 14/40 | 28.07% | -64.39 pp |

**Table 22.** Web-40 stage means.

| Stage | Mean accuracy | Mean macro F1 |
|---|---|---|
| Phase 1 | 33.00% | 26.56% |
| Phase 2A | 38.00% | 30.56% |
| Phase 2B | 35.00% | 26.95% |

**Figure 15.** Web-40 accuracy of the 15 frozen pipelines. *(figure file: web_bars.png)*

> **Interpretation.** Accuracy fell to 25.0%–42.5%. The highest observed values were ResNet50 Phase 2A and EfficientNet-B3 Phase 2A, both 42.5% (17/40), with macro-F1 38.18% and 31.97%. EfficientNet-B3 Phase 1 reached 40.0% (16/40), not 42.5%. Because one image is 2.5 pp, differences of one or two images cannot support a ranking. Phase 2A again had the highest stage mean, but on n = 40 this is descriptive only.

### 11.10 Collapse toward notumor

**Table 23.** Mean per-class recall and mean predicted count per 40 images, averaged over all 15 pipelines (computed from the per-pipeline confusion matrices).

| Class | Mean recall | Mean predicted count (true count 10) |
|---|---|---|
| glioma | 12.7% | 2.1 |
| meningioma | 30.7% | 8.3 |
| notumor | 90.7% | 28.3 |
| pituitary | 7.3% | 1.3 |

**Figure 16.** Web-40: mean per-class recall and mean predicted class counts across the 15 frozen pipelines. *(figure file: web_bias.png)*

> **Interpretation.** The failure was not random. On average the models predicted `notumor` for about 28 of the 40 images, although only 10 were normal. Glioma and pituitary tumours were mostly called normal (mean recall 12.7% and 7.3%). Under strong domain shift the classifiers systematically defaulted to the no-tumour class. In any screening use, such false negatives (missed tumours) are generally regarded as the more serious type of error.

### 11.11 Interpretation

The web images differ from the training data in many ways at once: scanner and field strength, MRI sequence (contrast-enhanced T1, T2, FLAIR), slice level and orientation, lesion size and variant, journal-figure cropping and resolution, and healthy-volunteer research scans. The models were trained on one curated collection with its own characteristic appearance. Combined with the explainability findings (Chapter 13), the results are consistent with the models relying partly on *dataset-specific appearance cues* rather than on lesion morphology alone. Without acquisition metadata for the web images, however, the drop cannot be attributed to any single cause.

This result does not contradict the BDNeuro result. The two experiments test different levels of shift. The frozen models kept meaningful accuracy on a structured, exact-overlap-clean external dataset, but deteriorated sharply on a small heterogeneous in-the-wild set.

### 11.12 Limitations

- n = 40 convenience sample; one image = 2.5 pp; no confidence intervals. Results are not a population estimate.
- Publication figures are selected for teaching or illustration and over-represent striking or unusual cases.
- Source diagnoses are taken from the publishing page and were not independently re-read by a radiologist.
- Cropping from figures was done manually (anatomy and contrast not altered).
- Not a clinical evaluation. 42.5% is not "clinical accuracy", and the result does not prove clinical failure or success.

## 12 Phase 3 — Comparative Analysis and Final Model

**Roadmap status: COMPLETE.** Phase 3 brings the evidence of Phases 1, 2A, 2B, 2C and 2D together across all five backbones, compares stability, sensitivity and generalization, and names a final project representative.

### 12.1 Architecture comparison

**Table 24.** Cross-backbone summary across the evidence layers (best stage per backbone shown for external cohorts). Internal values are on the contaminated original split.

| Backbone | CNN policy | P1 test acc | P1 material_clean | P2A val (repr.) | Best BDNeuro full | Best BDNeuro bal. | Best web-40 |
|---|---|---|---|---|---|---|---|
| VGG16 | frozen | 94.97% | 94.816% | 95.888% | Phase 2A 80.07% | Phase 2A 80.70% | 2A/2B 30.0% |
| VGG19 | frozen | 93.52% | 93.025% | 95.626% | Phase 2A 79.27% | Phase 2A 81.13% | 2A 37.5% |
| ResNet50 | fine-tuned | 98.09% | 98.303% | 98.688% | Phase 2B 86.04% | Phase 2B 87.86% | 2A 42.5% |
| ResNet152 | fine-tuned | 98.40% | 98.398% | 98.600% | Phase 2A 88.31% | Phase 2A 90.47% | 2A 37.5% |
| EfficientNet-B3 | fine-tuned | 98.47% | 98.680% | 99.300% | Phase 2A 86.65% | Phase 2B 88.83% | 2A 42.5% |

> **Interpretation.** The three fine-tuned backbones (ResNet50, ResNet152, EfficientNet-B3) were consistently stronger than the two frozen VGG backbones, internally and on BDNeuro. Among the fine-tuned three, internal differences were fractions of a point. Externally they were larger: ResNet152 Phase 2A, ResNet50 Phase 2B and EfficientNet-B3 Phase 2A/2B all exceeded 86% on the full BDNeuro cohort (as did ResNet152 Phase 2B), while the best VGG pipeline reached 80.07%. No backbone was best on every layer of evidence. On the web set the best values came from ResNet50 and EfficientNet-B3 (42.5%), within one image of several others.

### 12.2 Phase 1 vs Phase 2A vs Phase 2B

**Table 25.** Stage means across the five backbones on the three external evaluations.

| Stage | BDNeuro full | BDNeuro balanced | Web-40 acc | Web-40 macro F1 |
|---|---|---|---|---|
| Phase 1 | 79.77% | 81.80% | 33.00% | 26.56% |
| Phase 2A | 83.21% | 85.39% | 38.00% | 30.56% |
| Phase 2B | 82.34% | 84.33% | 35.00% | 26.95% |

> **Interpretation.** The ordering Phase 2A > Phase 2B > Phase 1 held on all three external evaluations. Moving from early stopping (Phase 1) to a fixed 30-epoch horizon (Phase 2A) helped both internally and externally. The further one-factor tuning of Phase 2B raised internal validation slightly, but on average transferred *worse* than Phase 2A. The stage differences are modest (Phase 2A − Phase 2B: 0.86 pp full, 1.06 pp balanced, 3.0 pp on the web set, i.e. 1.2 images) and rest on five backbone pairs from a single seed, so they describe a consistent tendency, not an established effect.

### 12.3 Internal vs external behaviour

**Figure 17.** Generalization gradient. The line follows the final representative across evaluation layers; shaded bars show the range across all 15 frozen pipelines where all were evaluated. *(figure file: gradient.png)*

Figure 17 summarizes the central scientific result. As the evaluation moves from the training distribution (internal validation and test, roughly 93–99%), through a structured but different source (BDNeuro, 75.0–88.3% full), to heterogeneous web images (25.0–42.5%), performance falls step by step. The fall is large and consistent across all 15 pipelines, not a property of one unlucky model.

### 12.4 Why Phase 2B gains did not universally transfer

The Phase 2B representatives were chosen by their maximum validation accuracy among 14 states each, on a single validation split that itself overlaps with the training data. Choosing the maximum of many noisy estimates favours configurations that happen to fit that split well. This is the familiar *winner's curse*, or selection-induced optimism. The internal gains were small: 0.09–0.70 pp, i.e. 1–8 of the 1,143 validation images (one image = 0.087 pp). No repeated-seed runs exist to show that gains of this size exceed run-to-run variation. Configurations such as LSTM 512, or VGG16 with LR 5e-4, may also have fitted dataset-specific cues more closely. The data cannot separate these explanations (one seed, no repeated runs). The practical conclusion is clear, however: **small internal tuning gains on a single public split are not reliable evidence of better generalization**. ResNet50 with dropout 0.2 is the exception that transferred well, which shows that tuning *can* help but cannot be assumed to.

### 12.5 Final ResNet152 Phase 2A selection

**Final project representative: ResNet152 — Phase 2A standardized 30-epoch pipeline, checkpoint `best_val_within_30`, epoch 10.** Configuration: RMSprop, LR 1e-4, batch 8, LSTM 256, dropout 0.3, Dense 128, StandardScaler → PCA 64 → linear SVM (C = 1). Artifacts: `phase2/experiments_30epoch/resnet152/models/best_val_within_30.pth` and `…/evaluations/best_val_within_30/{pca,models,results}/`.

**How the choice was made.** Two steps must be kept apart. (1) The 15 candidates, including this pipeline, were selected and frozen using *internal validation evidence only*, before any external image was evaluated (Section 10.2). (2) Naming one of the 15 as the overall project representative was a retrospective, project-level synthesis of the whole body of evidence. It weighed high internal validation together with the structured evidence of the larger external cohort, where this pipeline had the strongest BDNeuro accuracy and the smallest internal → external gap on both the full and the balanced cohort. **The web-40 stress test was not used for this choice.** On the web set this pipeline scored 37.5%, below ResNet50 Phase 2A and EfficientNet-B3 Phase 2A (42.5%), and an n = 40 exploratory set does not outweigh the 2,644-image structured evidence. No model was retrained, refitted or altered after any external result.

**Table 26.** Evidence chain for the final project representative (ResNet152 Phase 2A, best_val_within_30, epoch 10).

| Evaluation layer | Accuracy (macro F1) | Status / caveat |
|---|---|---|
| Internal full-hybrid validation (original split) | 98.600% | Contaminated split; used for candidate selection |
| Internal test, original 1,311 images | 98.703% | Contaminated split; never used for selection |
| material_clean post-hoc test (n = 1,061) | 98.869% (F1 98.788%) | Post-hoc; does not repair training contamination |
| BDNeuro exact-overlap-clean full (n = 2,644) | 88.31% (F1 87.85%) | Primary external evidence; gap -10.29 pp |
| BDNeuro balanced subset (n = 1,648) | 90.47% (F1 90.50%) | Nested sensitivity subset; gap -8.13 pp |
| Web-40 stress test (n = 40) | 37.5% (F1 31.66%) | Exploratory; not used for selection |

**Figure 18.** Confusion matrices of the final representative on the internal test set, the BDNeuro full cohort and the web-40 set (rows: true class; columns: predicted). *(figure file: final_cms.png)*

> **Interpretation.** The representative is named to show the *trajectory* of a strong pipeline, not to claim uniformly high performance: about 98.6–98.9% internally, 88.3% / 90.5% on BDNeuro and 37.5% on the web set. Because BDNeuro results informed the naming in step (2), the representative's BDNeuro figures carry some selection optimism if read as the performance of "the chosen model". The complete 15-pipeline tables (Tables Table 18, Table 19 and Table 21) are the unbiased record. ResNet152 is not claimed to be universally superior.

### 12.6 Evidence chain

The project's conclusions rest on a chain in which each link answers a weakness of the previous one: (i) Phase 1 shows that the pipeline works; (ii) the audit shows that the internal split is contaminated; (iii) material_clean shows that the internal result is not solely an artefact of duplicated test images; (iv) Phases 2A/2B characterize training behaviour and sensitivity; (v) the freeze makes the external evaluation tamper-evident; (vi) BDNeuro measures structured transfer after objective overlap removal; (vii) the web test measures robustness to severe shift; and (viii) explainability (Chapter 13) tests whether correct predictions rest on plausible evidence.

## 13 Phase 4 — Explainability

**Roadmap status: COMPLETE.** Both formal explainability studies used the **Phase 1 ResNet152 pipeline** (`experiments/resnet152/`, best epoch 8), loaded through the same inference engine as the prototype (`gradcam_evaluation_app.py` and `occlusion_evaluation_app.py` set `BACKBONE = "resnet152"` and use the GUI artifact registry). They were completed on 20 September 2026, before Phase 2. They therefore explain the Phase 1 ResNet152 model, not the final Phase 2A representative, and their findings should not be transferred to other backbones or checkpoints without new analysis.

### 13.1 Grad-CAM method

Grad-CAM (`gui/gradcam.py`) back-propagates the neural head's score for the predicted class to a convolutional layer, averages the gradients per channel to obtain channel weights, forms the weighted sum of activation maps, keeps the positive part, upsamples it to 224 × 224 and rescales it to [0, 1]. cuDNN's LSTM kernels cannot run backward in evaluation mode, so during the explanation pass the LSTM alone is switched to training mode with cuDNN disabled, while BatchNorm and Dropout remain in evaluation mode, keeping the result deterministic. The protocol was frozen before rating: ResNet152, sample seed 42, 5 test images per class (20 images; 15 tumour, 5 no-tumour), overlay opacity 0.55, and manual ratings of Good / Partial / Poor localization for tumour images.

### 13.2 Layer 3

`layer3[-1]` produces a 14 × 14 map with finer spatial detail but less class-specific semantics.

### 13.3 Layer 4

`layer4[-1]` produces a 7 × 7 map, the last convolutional stage that feeds the LSTM. It is more semantic but coarser.

### 13.4 Fused map

The fused map is the equal (50 : 50) average of the normalized layer 3 and layer 4 maps.

### 13.5 Results

**Table 27.** Grad-CAM localization ratings for the 15 tumour images (gradcam_evaluation/gradcam_evaluation_resnet152_seed42.csv).

| Map | Good | Partial | Poor | Judged most relevant |
|---|---|---|---|---|
| Layer 3 (14 × 14) | 1 | 9 | 5 | 9/15 |
| Layer 4 (7 × 7) | 3 | 2 | 10 | 6/15 |
| Fused (50 : 50) | 2 | 6 | 7 | 0/15 |

**Figure 19.** Grad-CAM localization ratings by map (left) and the map judged most anatomically relevant per tumour image (right). *(figure file: gradcam_ratings.png)*

**Figure 20.** Example Grad-CAM output from the formal protocol (gradcam_evaluation/figures/): a correctly classified glioma whose layer 4 and fused maps are dominated by off-lesion border activation. Coarse contribution maps, not segmentations.

The neural head and the SVM agreed on all 20 images, and both were correct on 19. The one shared error was `Te-pi_0078.jpg` (true pituitary, predicted glioma), for which all three maps were rated Poor.

> **Interpretation.** Correct classification frequently coexisted with poor localization. Maps were often dominated by the skull margin, orbital or facial anatomy, extracranial regions or image borders, and only occasionally overlapped the visible lesion. Layer 3 was most often the most relevant map (9/15), layer 4 in 6/15, and the fused map was never chosen. This does not prove that the model uses shortcuts, but it means high accuracy cannot be taken as evidence of lesion-centred reasoning. One notumor image (Te-no_0216) showed a conspicuous rounded bright region and was flagged for expert review as a possible label anomaly; it was not relabelled.

### 13.6 Final-SVM occlusion

Grad-CAM explains the neural head, but the deployed decision is made by the scikit-learn SVM, through which gradients cannot flow. Occlusion (`gui/occlusion.py`) therefore slides a 32 × 32 patch over the 224 × 224 input with stride 16 (169 occlusions per image). Each patch is set to zero in normalized space (the ImageNet channel mean), and the complete pipeline is rerun: CNN → LSTM → dense → scaler → PCA → SVM. For each occlusion the drop in the SVM decision value of the originally predicted class is recorded and prediction changes are counted. Eleven cases (9 tumour, 2 no-tumour) were chosen from the Grad-CAM set *before* occlusion results were viewed: 11 × 169 = 1,859 occlusions in total.

**Table 28.** Final-SVM occlusion per case (occlusion_evaluation/occlusion_comparison_resnet152_seed42.csv). Flips = occlusions that changed the predicted class.

| Image | True class | Occlusion rating | Comparison with Grad-CAM | Flips | Max score drop |
|---|---|---|---|---|---|
| glioma/Te-glTr_0002.jpg | glioma | Poor | Both predominantly off-lesion | 0/169 | 0.0051 |
| glioma/Te-gl_0030.jpg | glioma | Partial | Occlusion more anatomically relevant | 6/169 | 1.4187 |
| glioma/Te-gl_0282.jpg | glioma | Good | Occlusion more anatomically relevant | 5/169 | 1.5125 |
| meningioma/Te-me_0012.jpg | meningioma | Poor | Grad-CAM more anatomically relevant | 0/169 | 0.0016 |
| meningioma/Te-me_0118.jpg | meningioma | Poor | Grad-CAM more anatomically relevant | 8/169 | 1.0422 |
| meningioma/Te-me_0137.jpg | meningioma | Partial | Grad-CAM more anatomically relevant | 0/169 | 0.0045 |
| notumor/Te-no_0060.jpg | notumor | Not applicable | Not applicable / no tumor | 0/169 | 0.0190 |
| notumor/Te-no_0352.jpg | notumor | Not applicable | Not applicable / no tumor | 0/169 | 0.0061 |
| pituitary/Te-piTr_0008.jpg | pituitary | Poor | Grad-CAM more anatomically relevant | 0/169 | 0.0149 |
| pituitary/Te-pi_0078.jpg | pituitary | Poor | Both predominantly off-lesion | 38/169 | 2.5324 |
| pituitary/Te-pi_0267.jpg | pituitary | Good | Occlusion more anatomically relevant | 1/169 | 1.0409 |

**Figure 21.** Number of class-changing occlusions per case (out of 169). *(figure file: occlusion_flips.png)*

### 13.7 Comparison

For the nine tumour cases, occlusion localization was rated Good 2, Partial 2 and Poor 5. Grad-CAM was judged more anatomically relevant in 4 cases, occlusion in 3, and both were predominantly off-lesion in 2. In no case were both clearly lesion-focused. Across all 1,859 occlusions, 58 changed the predicted class (3.12% pooled). The median per-image flip rate was 0%, and 5 of 11 images had any flip. The misclassified Te-pi_0078 was unusually unstable: 38 of its 169 occlusions (22.5%) changed the prediction, against 20 of 1,690 (1.18%) for the ten correctly classified cases.

> **Interpretation.** Occlusion directly probes the deployed SVM decision, but it did not solve the interpretability problem: it was no more consistently lesion-focused than Grad-CAM. Correct predictions were mostly stable under single-patch occlusion, while the one misclassified case was fragile. Every occlusion map is normalized to its own maximum. For four images, no single patch changed the decision value by more than 0.01, yet their maps look as vivid as those with much larger effects. Maps must therefore always be read with their numerical effect sizes.

### 13.8 Limitations

- Both methods produce **coarse contribution / sensitivity maps**. They are not tumour segmentation and give no lesion boundary.
- Small qualitative studies (20 and 11 cases), single-rater subjective ratings, no radiologist review and no expert lesion masks.
- Phase 1 ResNet152 only; not repeated for the final Phase 2A representative or other backbones.
- Two rated images (Te-me_0012, Te-no_0060) were later found to be material duplicates. The original CSVs are preserved unchanged. Removing them leaves 18 Grad-CAM cases (14 tumour) and 9 comparison cases, and the core interpretation is unchanged.
- Per-image normalization of maps prevents visual comparison across images.

## 14 Phase 5 — Prototype Application

**Roadmap status: COMPLETE.**

### 14.1 Streamlit architecture

The prototype is a Streamlit [23] application (`app.py`) supported by the `gui/` package: `model_registry.py` (artifact paths), `model_factory.py` (rebuilds the trained architecture without downloading ImageNet weights), `inference_engine.py` (single-image inference and bundle validation), `results_loader.py` (metrics display), `gradcam.py` and `occlusion.py`. `verify_gui_setup.py` checks packages, source files and per-backbone artifacts. Only one backbone bundle is held in memory per session.

**Figure 22.** Inference path of the Streamlit prototype. *(figure file: gui_pipeline.png)*

### 14.2 Functionality

The application has four tabs (`app.py`, line 477–478): **Predict MRI**, **Model performance**, **Compare models** and **Methodology**.

- Upload a PNG, JPG/JPEG, BMP or TIFF image (`app.py` line 485); DICOM is not supported.
- Choose any of the five backbones.
- Final SVM prediction, with relative class scores (a softmax of the SVM decision values, explicitly *not* probabilities).
- Neural-head prediction and softmax scores for comparison, with an agreement / disagreement notice.
- Model performance: validation and test metrics, confusion matrices, per-class results and training curves.
- Model comparison table and chart across the five backbones.
- Grad-CAM overlay (layers 3 and 4 plus fusion for ResNets; last layer for VGG/EfficientNet) and optional final-SVM occlusion.
- Methodology view with the inference path and artifact-readiness checks.

### 14.3 Artifact loading

For each backbone, `gui/model_registry.py` (lines 51–60) resolves:

```
experiment = project_root / "experiments" / backbone
checkpoint = experiment / "models" / "best_model.pth"
svm        = experiment / "models" / "svm.joblib"
scaler     = experiment / "pca" / "standard_scaler.joblib"
pca        = experiment / "pca" / "pca_model.joblib"
final_results    = experiment / "results" / "final_results.json"
```

Before use, the loaded bundle is validated: the class mapping, the SVM class order and the scaler/PCA dimensions must agree.

### 14.4 Inference

An uploaded image goes through the same deterministic evaluation transform used for feature extraction, then one forward pass through the CNN–LSTM produces both the neural-head logits and the 128-D feature. The saved scaler, PCA and SVM are applied; **nothing is refitted on uploaded images**. The SVM's class is the system's answer.

### 14.5 Explainability

Grad-CAM requires one extra backward pass and is fast. Occlusion runs 169 masked copies through the whole pipeline in small batches, so it is slower and optional. Both are labelled in the interface as coarse contribution maps.

### 14.6 Current GUI pipeline limitation

> **Verified clarification.** The GUI loads `experiments/<backbone>/`, so it serves the **Phase 1 standardized pipelines**. For ResNet152 this is the Phase 1 model (best epoch 8; 98.40% original test accuracy), **not** the final ResNet152 Phase 2A epoch-10 representative (`phase2/experiments_30epoch/resnet152/…/best_val_within_30`). The "Model performance" and "Compare models" tabs likewise display the pre-audit Phase 1 results on the original 1,311-image test set. Pointing the registry at the Phase 2A artifacts would be a small, separate software change. It was deliberately not made during report preparation, which was read-only with respect to the repository.

### 14.7 Research-use disclaimer

The application is a **research and educational prototype**. It is not a medical device, it has not been clinically validated, and it must not be used for diagnosis or treatment decisions. Its scores are relative, not calibrated probabilities. Chapter 11 shows that its accuracy on heterogeneous web images can be very low, with a strong tendency to call tumours normal.

## 15 Discussion

### 15.1 What worked

- The hybrid CNN–LSTM–PCA–SVM architecture was implemented faithfully in its core design for all five backbones and runs end to end.
- The incremental, test-first construction caught shape and conceptual errors early; saving every artifact made later re-analysis (leakage-aware re-scoring, external evaluation) possible without retraining.
- Freezing the VGG backbones rescued training after the collapse.
- The duplicate audit found a real, quantifiable problem in a widely used public dataset, and the post-hoc policies measured its effect honestly.
- The freeze-and-hash protocol made the external evaluation tamper-evident: every evaluated artifact matches its pre-registered hash.
- Exact-pixel auditing of the external dataset prevented a badly inflated external result. Without it, more than half of BDNeuro would have been reused development images.

### 15.2 What did not

- TensorFlow could not run on the Blackwell GPU, which cost time and forced a framework change.
- Fully fine-tuned VGG16 collapsed, and the root cause was never isolated.
- Explainability did not show consistently lesion-centred evidence, either for the neural head (Grad-CAM) or for the deployed SVM (occlusion).
- Internal hyperparameter gains (Phase 2B) did not reliably transfer externally.
- Robustness to heterogeneous web images was poor, with a strong no-tumour bias.

### 15.3 Domain shift

**Domain shift** means that the distribution of test images differs from that of training images. BDNeuro represents moderate shift: a different source, but a curated four-class dataset of similar format. The web set represents severe shift. The step-wise drop from about 98–99% to 75–88% to 25–42.5% is consistent with models that learned the appearance of their training collection (intensity profile, crop style, slice selection, sequence mix) as well as tumour morphology. The no-tumour bias suggests that unfamiliar images lack the cues the models associate with tumour classes. The Grad-CAM observation of frequent peripheral, off-lesion attention points in the same direction. Acquisition metadata are unavailable for most of these images, so these explanations remain plausible hypotheses, not demonstrated causes.

### 15.4 Leakage implications

About 19% of the official test images and nearly 8% of the training images take part in a material cross-split pair in the original split. Any accuracy reported on this split, including the base paper's 99%, is therefore partly a within-collection estimate that may be optimistic. Removing implicated test images barely changed accuracy here, which is reassuring, but it cannot correct for duplicated content shaping training and checkpoint selection. A second, larger leakage risk appeared at the external stage: 3,297 BDNeuro images were exact pixel copies of development images. Public medical-image datasets are evidently redistributed and re-packaged, and **any external validation on public data must begin with an overlap audit**.

### 15.5 Hyperparameter sensitivity

Learning rate was the dominant tested factor (mean range 4.62 pp), and its effect depended on the training policy: frozen VGG backbones needed larger learning rates, while fine-tuned ResNets were damaged by them. LSTM width and dropout mattered much less. Fixed-horizon training exposed late-epoch drift (up to −3.76 pp for ResNet50 at LR 5e-4), which a best-checkpoint summary would hide. Evaluating two retained states showed that the best neural checkpoint is usually, but not always, the best input for the SVM.

### 15.6 Explainability findings

The explainability results are the qualitative counterpart of the domain-shift results. A model that frequently attends to skull margins and image borders is a model whose accuracy may depend on how images are framed and cropped. The finding is limited by its scale and subjectivity (Chapter 13), but its direction agrees with the external evidence.

### 15.7 Comparison with the base paper

**Table 29.** This project versus the base paper (RMSprop column of the paper's Table 3). Project internal numbers are on the same, contaminated, public split; the paper reports no leakage check.

| Backbone | Paper acc [1] | P1 test acc | P1 material_clean | P2A repr. test acc | Best BDNeuro full |
|---|---|---|---|---|---|
| VGG16 | 94% | 94.97% | 94.816% | 96.186% | 80.07% |
| VGG19 | 96% | 93.52% | 93.025% | 95.271% | 79.27% |
| ResNet50 | 99% | 98.09% | 98.303% | 98.932% | 86.04% |
| ResNet152 | 99% | 98.40% | 98.398% | 98.703% | 88.31% |
| EfficientNet-B3 | 98% | 98.47% | 98.680% | 99.314% | 86.65% |

> **Interpretation.** Internally the project lands close to the paper for VGG16 and EfficientNet-B3 (within about 1 pp) and about 0.3–0.9 pp below the paper's 99% for the ResNets. VGG19 is the largest gap (2.5 pp below in Phase 1, 3.0 pp on material_clean), consistent with the VGG family being the most fragile in this project. The paper's 99% headline is **not reproduced** at the same number. Plausible contributors are batch size 8 versus 32, early-stopped versus fixed 30-epoch training in Phase 1, the frozen VGG policy, and RMSprop-only training. More importantly, the project shows what the paper does not measure: the same split contains material duplicate leakage, and structured external accuracy for the same architecture family is about 10–20 pp lower.

**Table 30.** Deviations from the base paper.

| Base-paper element | Paper | This project |
|---|---|---|
| Framework | TensorFlow | PyTorch (sm_120 incompatibility) |
| Backbones | 5 | Same 5; independent pipelines |
| Optimizers | Adam, SGD, AdaGrad, RMSprop (20 runs) | RMSprop only |
| Batch size | 32 | 8 (BatchNorm1d; drop_last) |
| Epochs | 30 | Phase 1: early stopping (max 10 or 30); Phase 2A/2B: fixed 30 |
| LSTM hidden / dense width | not fully specified | 256 (128/512 in 2B) / 128 — implementation choices |
| Backbone freezing | not specified | VGG frozen after collapse; others fine-tuned |
| Scaler → PCA 64 → linear SVM (C = 1) | yes | same, fitted on training only |
| SVM probability | enabled | probability=False; relative scores |
| Augmentation | flips, brightness, contrast, saturation, hue | same effective ranges (ColorJitter) |
| Leakage audit | none | added; MATERIAL_DUPLICATION |
| External validation | future work | added: BDNeuro (frozen 15) + web-40 |
| Explainability | future work | added: Grad-CAM + final-SVM occlusion |
| GUI | none | added: Streamlit prototype |

## 16 Limitations

The following limitations are stated explicitly. Each one bounds what the results can be used to claim.

1. **Contaminated development split.** The original Nickparvar train/validation/test split contains 562 material cross-split duplicate pairs. All internal numbers (Phase 1, 2A, 2B) are historical results on this split.
2. **material_clean is post-hoc.** It removes implicated test images only. It does not repair contaminated training or validation, and it changes test-set class composition.
3. **No patient-level independence.** No reliable patient identifiers exist for Nickparvar, BDNeuro or the web images. Patient-level independence cannot be established anywhere in this project.
4. **Single seed, single split.** Phase 1, 2A and 2B use seed 42 only. No repeated runs, confidence intervals or significance tests exist, and small differences (including all Phase 2B gains) are not statistically established.
5. **Unequal training policy and budgets.** VGG backbones were frozen while the others were fine-tuned, and Phase 1 epoch budgets differed (30 vs 10). Cross-backbone comparisons are pipeline comparisons.
6. **Limited hyperparameter coverage.** Only learning rate, LSTM width and dropout were varied, one at a time; interactions were not tested. Only RMSprop was used.
7. **Untested architectural components.** No CNN-only, no-LSTM or no-PCA ablation and no alternative spatial ordering were run, so the contribution of the LSTM and of PCA is not isolated.
8. **Simple preprocessing.** Direct resize without skull stripping or cropping lets the model use skull, background and border cues.
9. **External cohort limits.** BDNeuro exact-overlap cleaning improves image-level independence but does not establish patient-level independence. Non-identical near-duplicates (pHash candidates) were retained by design. The dataset's licence and citation fields are unfilled in its README.
10. **Selection optimism for the named representative.** The final representative was named after BDNeuro results were known (no model was changed), so its BDNeuro figures are slightly optimistic as an estimate for "the chosen model". The 15-pipeline tables are the unbiased record.
11. **Web stress test scale.** n = 40 convenience images; each image is worth 2.5 pp; publication images are not representative of clinical prevalence; diagnoses come from the source pages.
12. **Explainability scale.** Small, qualitative, single-rater studies on the Phase 1 ResNet152 model only; no expert masks.
13. **Prototype scope.** Raster images only (no DICOM), uncalibrated relative scores, and the GUI currently serves the Phase 1 pipelines rather than the final representative.
14. **Historical figures without machine-readable records.** The VGG16 collapse (~28%) and the legacy ResNet152 prototype result (~97.64%) come from the project record; no JSON output of those runs is preserved.
15. **No clinical validation.** No claim of diagnostic accuracy, safety or deployment readiness is supported.

## 17 Conclusion

This project implemented the hybrid CNN–LSTM–PCA–SVM architecture of Bérété et al. (2026) for four-class brain-tumour MRI classification with five independent backbones. It then followed the six-phase roadmap of the Review-I presentation to completion, adding a web stress test beyond the original plan. Every phase is complete: Phase 1, 2A, 2B, 2C, 2D, 3, 4, 5 and 6.

On the public Nickparvar split the pipelines reached 93.5–98.5% test accuracy, and the downstream PCA + SVM classifier improved test accuracy over the neural head for every backbone. A duplicate audit then showed that this split contains material cross-split leakage (562 pairs; 250 test images implicated). The internal results were reinterpreted as historical, and a post-hoc material_clean test confirmed that they were not driven solely by duplicated test images. A fixed 30-epoch protocol improved every backbone, and a 35-run one-factor study identified learning rate as the dominant sensitivity (4.62 pp mean range).

Fifteen pipelines were frozen and hashed before external evaluation. On an exact-overlap-clean BDNeuro cohort, from which 3,297 reused images had first been removed, accuracy fell to 75.0–88.3%. On a locked, source-verified 40-image web set it fell to 25–42.5%, with most tumours classified as normal. The final project representative, ResNet152 Phase 2A (best_val_within_30, epoch 10), illustrates the whole trajectory: 98.60% internal validation, 98.70% internal test, 98.869% material_clean, 88.31% BDNeuro full, 90.47% BDNeuro balanced and 37.5% on the web stress test.

**The central conclusion is that high public-dataset accuracy does not guarantee robust out-of-distribution performance.** Accuracy around 99% on a public split coexisted with duplicate leakage, with 10–20 pp lower accuracy on a structured external dataset, with severe failure under heterogeneous web-domain shift, and with explanation maps that were often off-lesion. The project's strongest contribution is therefore not a headline percentage but layered, auditable evidence of how and where performance degrades. The system remains a research and educational prototype.

## 18 Future Work

The following items would strengthen the methodology. They are optional extensions, not unfinished requirements of the completed project. Any future change motivated by the BDNeuro or web results must be evaluated on a *new*, untouched evaluation set, because those two sets have now been seen.

- **Duplicate-group-aware retraining:** group exact and near-duplicate images, split whole groups, retrain all five backbones from scratch, refit scaler/PCA/SVM, and repeat internal evaluation. Even then, patient-level independence should not be claimed without patient identifiers.
- **Patient-identified, multi-site data:** evaluate on datasets with patient or study identifiers and, ideally, prospective multi-site collection.
- **Statistical rigour:** repeated seeds and splits, confidence intervals and paired tests for backbone and configuration comparisons.
- **Ablations:** CNN-only, no-LSTM, no-PCA and alternative spatial orderings, to measure what each component contributes.
- **Robustness methods:** MRI-specific intensity normalization, skull stripping or brain cropping, and domain-adaptation techniques, evaluated on new external data.
- **Calibration:** calibrated probability estimates and reliability analysis.
- **Quantitative explainability:** expert lesion masks with localization metrics; repeating XAI on the final representative.
- **Sequence-aware and volumetric modelling:** using multi-sequence or 3-D MRI where available.
- **Prototype:** point the GUI registry at the final Phase 2A representative, add DICOM ingestion, and conduct a usability study.
- **Base-paper completion:** the remaining optimizers (Adam, SGD, AdaGrad) and batch size 32, if a full reproduction is desired.

## References

References [1]–[15] are the references verified for the Review-I presentation. [16]–[17] are dataset sources. [18]–[23] are standard method and software references used in the project documentation.

[1] M. Bérété, A. Echtioui, L. Sellami and A. A. Ben Hmida, "Hybrid CNN+LSTM+SVM model for advanced brain tumor classification," *Network Modeling Analysis in Health Informatics and Bioinformatics*, vol. 15, art. 43, 2026. doi:10.1007/s13721-025-00718-8.  
[2] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015. arXiv:1409.1556.  
[3] K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778. doi:10.1109/CVPR.2016.90.  
[4] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, PMLR 97, 2019, pp. 6105–6114.  
[5] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. doi:10.1162/neco.1997.9.8.1735.  
[6] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273–297, 1995. doi:10.1007/BF00994018.  
[7] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *International Journal of Computer Vision*, vol. 128, no. 2, pp. 336–359, 2020. doi:10.1007/s11263-019-01228-7.  
[8] M. D. Zeiler and R. Fergus, "Visualizing and understanding convolutional networks," in *Proc. ECCV*, 2014, pp. 818–833. doi:10.1007/978-3-319-10590-1_53.  
[9] M. S. Başarslan, "M-C&M-BL: a novel classification model for brain tumor classification: multi-CNN and multi-BiLSTM," *Journal of Supercomputing*, vol. 81, art. 502, 2025. doi:10.1007/s11227-025-06964-x.  
[10] F. A. Özbay and E. Özbay, "Brain tumor detection with mRMR-based multimodal fusion of deep learning from MR images using Grad-CAM," *Iran Journal of Computer Science*, vol. 6, pp. 245–259, 2023. doi:10.1007/s42044-023-00137-w.  
[11] K. M. Hosny, M. A. Mohammed, R. A. Salama and A. M. Elshewey, "Explainable ensemble deep learning-based model for brain tumor detection and classification," *Neural Computing and Applications*, vol. 37, pp. 1289–1306, 2025. doi:10.1007/s00521-024-10401-0.  
[12] M. M. Musthafa, T. R. Mahesh, V. Vinoth Kumar and S. Guluwadi, "Enhancing brain tumor detection in MRI images through explainable AI using Grad-CAM with Resnet 50," *BMC Medical Imaging*, vol. 24, art. 107, 2024. doi:10.1186/s12880-024-01292-7.  
[13] S. Panigrahi, D. R. D. Adhikary and B. K. Pattanayak, "Brain tumor classification: a blend of ensemble learning and fine-tuned pre-trained models," *Discover Applied Sciences*, 2025. doi:10.1007/s42452-025-06695-x.  
[14] S. Sahoo, S. Mishra, B. Brahma, P. Barsocchi and A. K. Bhoi, "SSO-CCNN: a correlation-based optimized deep CNN for brain tumor classification using sampled PGGAN," *International Journal of Computational Intelligence Systems*, vol. 17, 2024. doi:10.1007/s44196-024-00574-w.  
[15] R. L. Kumar, J. Kakarla, B. V. Isunuri and M. Singh, "Multi-class brain tumor classification using residual network and global average pooling," *Multimedia Tools and Applications*, vol. 80, no. 9, pp. 13429–13438, 2021. doi:10.1007/s11042-020-10335-4.  
[16] M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle. Available: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. (Dataset version, licence and access date to be confirmed by the team.)  
[17] "Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)" — BDNeuro-MRI, Version 8 (project copy: external_data/BDNeuro_V8_raw/Brain_Tumor_MRI_Dataset_Final/README.md; source stated as "Epic & CSCR Hospital"). The distributed README leaves the licence and citation fields unfilled; a formal citation must be added by the team before submission.  
[18] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255.  
[19] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. ICML*, 2015, pp. 448–456.  
[20] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.  
[21] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.  
[22] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.  
[23] Streamlit open-source app framework. Available: https://streamlit.io  

## Appendix A — Scripts and Artifact Index

**Table 31.** Principal scripts and their roles.

| Script(s) | Role |
|---|---|
| src/data_pipeline_torch.py | Transforms (training augmentation; deterministic evaluation), saved-split loading, datasets/loaders |
| src/save_split.py · src/check_dataset.py | Stratified 80/20 split (seed 42) and saved indices · dataset readability/mode inspection |
| model_cnn_lstm.py | Generic CNNLSTM model; SUPPORTED_BACKBONES (five) |
| model_resnet152_lstm.py, train_resnet152_lstm.py, extract_all_features.py, prepare_pca_features.py, train_svm.py, evaluate_final_test.py | Legacy root-level ResNet152 prototype (superseded) |
| actual_training_scripts/train_backbone.py · evaluate_backbone.py | Phase 1 training (early stopping) · features + scaler + PCA + SVM + metrics |
| actual_training_scripts/evaluate_neural_head.py · summarize_neural_head_results.py · compare_neural_vs_pca_svm.py | Phase 1 neural-head evaluation and downstream-classifier comparison |
| final_project_qc.py | Cross-split duplicate/leakage audit |
| evaluate_leakage_clean_test.py · evaluate_leakage_clean_test_phase2.py | Post-hoc leakage-aware evaluation (Phase 1 · Phase 2A) |
| actual_training_scripts/train_backbone_30epoch.py · evaluate_backbone_30epoch.py | Phase 2A fixed 30-epoch training · both retained states downstream |
| phase2/run_all_dl_hyperparams.py · actual_training_scripts/train_backbone_hyperparams.py | Phase 2B queue runner · training worker |
| phase2/run_all_hyperparam_pca_svm.py · actual_training_scripts/evaluate_hyperparam_pca_svm.py | Phase 2B downstream runner · worker (validation only) |
| actual_training_scripts/audit_bdneuro_overlap.py · build_bdneuro_exact_clean_cohort.py · build_bdneuro_balanced_subset.py | BDNeuro overlap audit · exact-clean cohort · balanced subset |
| actual_training_scripts/compare_external_15.py | Freeze/hash 15 pipelines; external evaluation (--results-name) |
| actual_training_scripts/initialize_web_stress_test.py · audit_web_stress_40.py | Web-40 initialization (manifest template) · overlap audit |
| actual_training_scripts/diagnose_vgg16_collapse.py · test_all_backbones.py · test_best_checkpoint_features.py | Diagnostics and backbone/feature checks |
| app.py · gui/*.py · verify_gui_setup.py | Streamlit prototype and setup check |
| gradcam_evaluation_app.py · occlusion_evaluation_app.py | Formal XAI rating tools (fixed protocols, CSV export) |

## Appendix B — Experiment Directory Map

**Table 32.** Repository directories holding experiment evidence.

| Directory | Contents |
|---|---|
| data/brain_tumor_mri/{Training,Testing}/ · data/splits/ | Nickparvar images (5,712 + 1,311) · saved split indices |
| experiments/<backbone>/{models,features,pca,results}/ | Phase 1 bundles: best_model.pth, svm.joblib, scaler/PCA, final/training/neural-head JSON |
| phase2/experiments_30epoch/<backbone>/ | Phase 2A: best_val_within_30.pth, epoch_30_model.pth, evaluations/<state>/ (features incl. test, pca, svm, final_results.json) |
| phase2/hyperparameter_dl/<backbone>/<run>/ | Phase 2B: 35 runs; evaluations/<state>/ with train/val features only |
| phase2/results/leakage_corrected_30epoch/ | Phase 2A material_clean / exact / conservative results |
| phase2/external_validation/ | frozen_15_manifest.json, frozen_15_selection.csv, BDNeuro audit/cohort/subset, 3 comparison folders, web_stress_40{,_audit,_locked} |
| external_data/ | BDNeuro_V8_raw, BDNeuro_V8_exact_clean (2,644), BDNeuro_V8_exact_clean_balanced (1,648), web_stress_40_raw (40) |
| final_project_docs/ | duplicate_leakage_audit/, leakage_corrected_evaluation/, environment_pip_freeze.txt, historical QC blocker |
| gradcam_evaluation/ · occlusion_evaluation/ | XAI rating CSVs and figures |
| models/ · features/ · pca_features/ | Legacy root-level ResNet152 prototype artifacts (superseded; not used by the GUI) |

## Appendix C — Major JSON/CSV Evidence Map

A claim-by-claim map with exact fields and verification status is provided separately in `final_deliverables/FINAL_PROJECT_EVIDENCE_MAP.md`.

**Table 33.** Evidence sources by result family.

| Result family | Repository source |
|---|---|
| Phase 1 metrics | experiments/<backbone>/results/final_results.json → validation.*, test.*; neural_head_results.json; results/neural_vs_pca_svm_comparison.csv |
| Leakage audit | final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json; duplicate_leakage_findings.csv |
| Leakage policies | final_project_docs/leakage_corrected_evaluation/leakage_corrected_summary.csv; PRIMARY_material_clean_comparison.csv; *_excluded_test_images.csv |
| Phase 2A | phase2/experiments_30epoch/<backbone>/results/training_results.json; evaluations/<state>/results/final_results.json; phase2/results/leakage_corrected_30epoch/*.csv |
| Phase 2B | phase2/hyperparameter_dl/<backbone>/<run>/results/training_results.json; evaluations/<state>/results/final_results.json |
| Freeze | phase2/external_validation/frozen_15_manifest.json → selection_details, candidates[].sha256; frozen_15_selection.csv |
| BDNeuro | bdneuro_overlap_audit/summary.json, dataset_counts.json, cross_exact_*.csv; bdneuro_exact_clean_cohort/cohort_summary.json; bdneuro_balanced_subset/balanced_subset_summary.json |
| External results | comparison_15_models_{bdneuro_full,bdneuro_balanced,web_stress_40}/external_comparison_15.csv; per-model JSON (confusion matrices) and predictions CSV |
| Web-40 audit/lock | web_stress_40_audit/summary.json, cross_phash_candidates.csv; web_stress_40_locked/lock_record.json, locked_40_manifest.csv |
| XAI | gradcam_evaluation/gradcam_evaluation_resnet152_seed42.csv; occlusion_evaluation/occlusion_comparison_resnet152_seed42.csv |
| GUI paths / augmentation | gui/model_registry.py lines 51–60; src/data_pipeline_torch.py lines 65–78 |

## Appendix D — Hyperparameter Configuration

**Table 34.** Training and downstream configuration by phase.

| Setting | Phase 1 | Phase 2A | Phase 2B (changed factor) |
|---|---|---|---|
| Optimizer | RMSprop | RMSprop | RMSprop |
| Learning rate | 1e-4 | 1e-4 | 1e-5, 5e-5, 5e-4 (one at a time) |
| Batch size | 8 (drop_last) | 8 | 8 |
| Epochs | max 30 (VGG) / 10 (others), patience 3 | exactly 30 | exactly 30 |
| LSTM hidden | 256 | 256 | 128, 512 |
| Dropout | 0.3 | 0.3 | 0.2, 0.5 |
| Dense features | 128 | 128 | 128 |
| Seed | 42 | 42 | 42 |
| Loss | CrossEntropy (raw logits) | same | same |
| Retained states | best_model | best_val_within_30, epoch_30 | best_val_within_30, epoch_30 |
| Downstream | Scaler → PCA 64 → SVM C=1 | same | same (validation only) |
| Test set evaluated | yes | yes (not used for selection) | no (deliberately) |

## Appendix E — Per-Class Recall of the 15 Frozen Pipelines

**Table 35.** BDNeuro full cohort: per-class recall (%) computed from each pipeline's confusion matrix.

| Backbone | Stage | Glioma | Meningioma | Notumor | Pituitary |
|---|---|---|---|---|---|
| VGG16 | Phase 1 | 82.7 | 75.0 | 94.7 | 58.1 |
| VGG16 | Phase 2A | 82.5 | 85.8 | 93.7 | 61.6 |
| VGG16 | Phase 2B | 79.0 | 81.8 | 94.7 | 59.7 |
| VGG19 | Phase 1 | 72.3 | 82.4 | 90.0 | 67.4 |
| VGG19 | Phase 2A | 77.4 | 85.0 | 90.0 | 71.0 |
| VGG19 | Phase 2B | 72.2 | 82.2 | 89.8 | 64.6 |
| ResNet50 | Phase 1 | 76.9 | 90.0 | 98.5 | 70.2 |
| ResNet50 | Phase 2A | 73.9 | 92.2 | 98.8 | 77.0 |
| ResNet50 | Phase 2B | 83.4 | 92.8 | 98.1 | 77.5 |
| ResNet152 | Phase 1 | 72.8 | 93.2 | 97.8 | 76.9 |
| ResNet152 | Phase 2A | 83.5 | 94.9 | 93.2 | 89.0 |
| ResNet152 | Phase 2B | 82.0 | 96.2 | 97.6 | 80.2 |
| EfficientNet-B3 | Phase 1 | 79.7 | 89.6 | 98.1 | 74.2 |
| EfficientNet-B3 | Phase 2A | 86.6 | 94.3 | 96.8 | 73.7 |
| EfficientNet-B3 | Phase 2B | 83.3 | 95.1 | 99.0 | 77.5 |

**Table 36.** Web-40: per-class recall (%) per pipeline (10 images per class; each image = 10 pp of class recall).

| Backbone | Stage | Glioma | Meningioma | Notumor | Pituitary |
|---|---|---|---|---|---|
| VGG16 | Phase 1 | 10.0 | 0.0 | 90.0 | 0.0 |
| VGG16 | Phase 2A | 20.0 | 0.0 | 90.0 | 10.0 |
| VGG16 | Phase 2B | 10.0 | 20.0 | 90.0 | 0.0 |
| VGG19 | Phase 1 | 10.0 | 30.0 | 90.0 | 10.0 |
| VGG19 | Phase 2A | 10.0 | 40.0 | 100.0 | 0.0 |
| VGG19 | Phase 2B | 10.0 | 30.0 | 100.0 | 0.0 |
| ResNet50 | Phase 1 | 10.0 | 20.0 | 70.0 | 20.0 |
| ResNet50 | Phase 2A | 20.0 | 80.0 | 60.0 | 10.0 |
| ResNet50 | Phase 2B | 20.0 | 30.0 | 100.0 | 10.0 |
| ResNet152 | Phase 1 | 20.0 | 10.0 | 100.0 | 10.0 |
| ResNet152 | Phase 2A | 10.0 | 40.0 | 90.0 | 10.0 |
| ResNet152 | Phase 2B | 0.0 | 40.0 | 90.0 | 10.0 |
| EfficientNet-B3 | Phase 1 | 10.0 | 30.0 | 100.0 | 20.0 |
| EfficientNet-B3 | Phase 2A | 10.0 | 60.0 | 100.0 | 0.0 |
| EfficientNet-B3 | Phase 2B | 20.0 | 30.0 | 90.0 | 0.0 |

## Appendix F — Grad-CAM Per-Case Ratings

**Table 37.** Grad-CAM ratings for the 20-image protocol (ResNet152 Phase 1, seed 42). N/A = no-tumour image.

| # | Image | True | SVM pred. | Layer 3 | Layer 4 | Fused | Best map |
|---|---|---|---|---|---|---|---|
| 1 | Te-glTr_0002 | glioma | glioma | Partial | Poor | Poor | Layer 3 |
| 2 | Te-gl_0030 | glioma | glioma | Poor | Poor | Poor | Layer 4 |
| 3 | Te-gl_0085 | glioma | glioma | Poor | Poor | Poor | Layer 3 |
| 4 | Te-gl_0148 | glioma | glioma | Partial | Poor | Poor | Layer 3 |
| 5 | Te-gl_0282 | glioma | glioma | Poor | Partial | Partial | Layer 4 |
| 6 | Te-me_0012 | meningioma | meningioma | Good | Good | Good | Layer 4 |
| 7 | Te-me_0118 | meningioma | meningioma | Partial | Poor | Poor | Layer 3 |
| 8 | Te-me_0137 | meningioma | meningioma | Partial | Good | Good | Layer 4 |
| 9 | Te-me_0155 | meningioma | meningioma | Partial | Partial | Partial | Layer 4 |
| 10 | Te-me_0204 | meningioma | meningioma | Partial | Good | Partial | Layer 4 |
| 11 | Te-no_0060 | notumor | notumor | Not applicable | Not applicable | Not applicable | None / not applicable |
| 12 | Te-no_0157 | notumor | notumor | Not applicable | Not applicable | Not applicable | None / not applicable |
| 13 | Te-no_0191 | notumor | notumor | Not applicable | Not applicable | Not applicable | None / not applicable |
| 14 | Te-no_0216 | notumor | notumor | Not applicable | Not applicable | Not applicable | None / not applicable |
| 15 | Te-no_0352 | notumor | notumor | Not applicable | Not applicable | Not applicable | None / not applicable |
| 16 | Te-piTr_0008 | pituitary | pituitary | Partial | Poor | Partial | Layer 3 |
| 17 | Te-pi_0010 | pituitary | pituitary | Poor | Poor | Poor | Layer 3 |
| 18 | Te-pi_0078 | pituitary | glioma | Poor | Poor | Poor | Layer 3 |
| 19 | Te-pi_0267 | pituitary | pituitary | Partial | Poor | Partial | Layer 3 |
| 20 | Te-pi_0282 | pituitary | pituitary | Partial | Poor | Partial | Layer 3 |

## Appendix G — Glossary

**Table 38.** Glossary of key terms.

| Term | Meaning |
|---|---|
| Accuracy | Fraction of images classified correctly. |
| Macro F1 | Unweighted mean over classes of the harmonic mean of precision and recall; treats every class equally. |
| CNN | Convolutional neural network: learned image filters producing a spatial feature map. |
| LSTM | Long short-term memory: gated recurrent network; here it reads the 49 spatial positions of one feature map. |
| StandardScaler | Rescales each feature to zero mean and unit variance using training statistics. |
| PCA | Principal component analysis: projection onto the directions of greatest variance (64 here). |
| SVM | Support vector machine: maximum-margin classifier; linear kernel, C = 1. |
| Neural head | Temporary Linear(128→4) classifier used to train the network; also evaluated for comparison. |
| Grad-CAM | Gradient-weighted class activation map; coarse heatmap of regions that increased a class score. |
| Occlusion | Masks input patches and measures the change in the model output; probes the full pipeline. |
| Duplicate leakage | Identical or near-identical images present in both training and evaluation partitions. |
| material_clean | Test subset with all images in a material cross-split pair removed (n = 1,061); post-hoc. |
| pHash | Perceptual hash: compact code of coarse image structure; small Hamming distance = visually similar. |
| Domain shift | Difference between the distribution of evaluation images and training images. |
| best_val_within_30 | Checkpoint with the highest neural validation accuracy within the 30-epoch run. |
| epoch_30 | Checkpoint after the final (30th) epoch. |
| Exact-overlap-clean | External cohort after removing images pixel-identical to development images. |
| Frozen pipeline | Checkpoint + scaler + PCA + SVM fixed and hashed before external evaluation. |

## Appendix H — Reproducibility Notes

- All numbers in this report were recomputed from the repository's JSON/CSV files by a read-only script during report preparation; no training, refitting or re-evaluation was run.
- The 75 SHA-256 hashes in frozen_15_manifest.json were recomputed from the current artifacts: 75/75 match.
- Environment: Ubuntu (WSL2), Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, scikit-learn 1.9.0, NumPy 2.5.2, Streamlit 1.63.0, RTX 5070 Laptop GPU 8 GB.
- Best epochs in all current JSON files are 1-based. An older note describing them as zero-based refers to superseded artifacts.
- Historical status documents (final_project_docs/FINAL_QC_BLOCKER.md, REPORT_SOURCE_OF_TRUTH.md, and the transfer package's historical_transfer/ folder) are preserved unchanged as records of earlier project states; their statements that external validation was pending or that group-aware retraining was blocking are superseded by the later evidence described here.
- Much of the Phase 2 and external-validation evidence is not yet committed to git (it is present on disk). Committing or archiving it is recommended before submission.
