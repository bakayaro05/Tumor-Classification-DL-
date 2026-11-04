# Literature and Reference History

## Hostile-review additions

- The 16 July discussion considered Mandle et al., Rao/Karunakara, Saad et al., and Malathi/Sinthia as four distinct classical directions. Exact bibliography requires verification.
- The base paper described five backbones × Adam/SGD/AdaGrad/RMSprop, 30 epochs, batch 32, learning rate `1e-4`, and horizontal/vertical flips with brightness ±0.1, contrast/saturation 0.8–1.2, and hue ±0.1. Verified project evidence is RMSprop-only, batch 8, generally shorter early-stopped training, with exact project transforms still to recover.
- Paper language sometimes describes MRI-slice dependencies; this project uses one 2D image and rasterized spatial tokens, not inter-slice modeling.

This document records only literature that was meaningfully discussed in the available project history. It is not a complete bibliography. Bibliographic details not verified from the source files are marked accordingly.

## Primary base paper

### Bérété, Echtioui, Sellami, and Ben Hmida — “Hybrid CNN+LSTM+SVM model for advanced brain tumor classification”

- **Entry into project:** Selected during the initial paper-selection phase in August 2026.
- **Publication:** 2026, *Network Modeling Analysis in Health Informatics and Bioinformatics* (Springer). The source PDF exists in the project Library.
- **Why selected:** It satisfied the recent-paper requirement and defined a modern hybrid comparison across VGG16, VGG19, ResNet50, ResNet152, and EfficientNet-B3.
- **Contribution to project:** Four-class task, five alternative backbones, CNN→LSTM→SVM concept, RMSprop-centered training, PCA reduction, and the approximately 99% reference result.
- **Implemented:** The project implemented five independent pipelines, spatial feature-map sequencing, LSTM, 128-D learned features, StandardScaler, PCA(64), and linear SVM.
- **Differences/uncertainties:** The project used an explicit PyTorch architecture and saved split whose exact equivalence to the paper cannot be assumed. The order of scaler and PCA was corrected in the actual implementation to scaler then PCA, even though some early diagrams/text placed PCA before StandardScaler. VGG freezing was a project-specific stabilization. Explainability and GUI were project extensions. The historical split was later found contaminated.
- **Remaining relevance:** Still the methodological anchor, but should not be used to justify claiming paper-level performance or clinical validity.

## Earlier classical paper — 2022 K-means/DWT/PCA/kernel-SVM approach

- **Entry:** Considered before choosing the official base paper.
- **Pipeline discussed:** Median filtering→skull stripping→K-means segmentation→DWT features→PCA→kernel SVM.
- **Reported figure in conversation:** 98.75% accuracy. This was a paper-reported value, not a project result.
- **Contribution:** Helped explain handcrafted features, segmentation, DWT, PCA, and SVM; suggested a clean experiment replacing DWT with CNN features while holding other stages constant.
- **Implemented:** No.
- **Why not selected:** It did not satisfy the 2023-or-later base-paper requirement.
- **Status:** Rejected as official base; potentially useful as background only.

## 2023 Otsu/GLCM/DWT/SVM paper

- **Title recorded in conversation:** “Distinctive approach in brain tumor detection and feature extraction using biologically inspired DWT method and SVM.”
- **Contribution:** Introduced a recent classical alternative using noise removal, Otsu thresholding, morphology, GLCM, DWT, PCA, and kernel SVM.
- **Implemented:** No.
- **Status:** Considered but not selected. Exact bibliographic metadata/DOI must be verified before citation.

## Panigrahi (2025) ensemble paper (“Paper 3” in discussion)

- **Entry:** Literature-survey comparison.
- **Contribution:** Clarified the difference between an actual ensemble of four pretrained CNNs and the base paper's one-backbone-at-a-time comparison.
- **Implemented:** No ensemble was built.
- **Status:** Contextual comparison. Exact title, venue, authorship details, and DOI require source verification before citation.

## Grad-CAM literature

- **Entry:** Discussed as a possible novelty/extension after the baseline.
- **Contribution:** Motivated visual inspection of class-discriminative image regions and helped identify spurious peripheral focus.
- **Implemented:** Yes, adapted to the CNN-LSTM neural head at multiple ResNet stages.
- **Difference from casual claims:** It does not directly explain the separate scikit-learn SVM and does not produce a tumor boundary.
- **Citation status:** The canonical Grad-CAM paper should be added and verified in the final bibliography; the exact citation was not preserved in the available transfer evidence.

## Occlusion sensitivity literature

- **Entry:** Added after recognizing the Grad-CAM/SVM explanation mismatch.
- **Contribution:** Provides a model-agnostic perturbation test of the actual final SVM decision score.
- **Implemented:** Yes.
- **Limitations discussed:** Coarse, fill-value dependent, computationally expensive, and not necessarily anatomically faithful.
- **Citation status:** Add and verify a primary occlusion/sensitivity reference before publication.

## Masoud Nickparvar Kaggle Brain Tumor MRI Dataset

- **Entry:** Dataset selected to match the base paper's 7,023-image version.
- **Contribution:** Four labeled classes and commonly used Training/Testing directory structure.
- **Critical new finding:** Those folders and the derived validation split are not image-independent under the audit. Dataset reuse and provenance likely contribute to duplicate risk.
- **Citation/provenance task:** Record exact Kaggle URL, version, license, download date, and file manifest. Do not cite only “Kaggle” generically in the report.

## Literature conclusions that remain defensible

- Transfer-learned CNNs can yield strong within-dataset classification performance.
- A hybrid CNN/LSTM/SVM design is technically feasible.
- An LSTM over flattened spatial features is not temporal MRI modeling.
- High reported accuracy does not establish external clinical validity.
- Explainability maps may reveal spurious cues but do not validate medical reasoning.
- Duplicate and patient-level leakage must be investigated before accepting high public-dataset scores.

## Citation work still required

The final report must verify the base paper's DOI and bibliographic fields from the PDF, the dataset citation/version, canonical Grad-CAM and occlusion references, the 2022 and 2023 classical papers, and the Panigrahi ensemble paper. Never use the conversational shorthand “Paper 3” in a publication.
