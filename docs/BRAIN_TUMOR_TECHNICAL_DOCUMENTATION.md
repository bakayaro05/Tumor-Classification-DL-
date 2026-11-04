# Technical Documentation and Research Archive Guide

**Hybrid CNN–LSTM–PCA–SVM for Four-Class Brain Tumor MRI Classification**  
**GitHub repository:** https://github.com/bakayaro05/Tumor-Classification-DL-  
**Repository audited:** `main`, 4 November 2026; Git tree `b881e99829afac69de34ccc21350010c697ab7c1`  
**Project type:** University college mini-project  
**Publication purpose:** Explain the implemented system, methodological journey, experiments, numerical evidence, artifact locations, reproducibility constraints and public repository contents.  
**Use restriction:** Research/education only; no radiological diagnosis, segmentation or treatment advice.

> **Authority rule.** This document is based on the uploaded *revised 60-page final report*, the updated GitHub repository, the original project decision/history records, and primary dataset/publisher sources. Where an older README, analysis snapshot or early project plan conflicts with the revised final report or later machine-readable evidence, the latter controls. Earlier files are still described as historical records, not silently erased.

## Executive technical summary

The project implemented a trainable **CNN–LSTM** feature extractor with five ImageNet-pretrained alternative CNN backbones and a final **StandardScaler → PCA64 → linear SVM** decision stage. It used the four supplied labels glioma, meningioma, pituitary and no-tumour; no grading, three-dimensional MRI or patient-level diagnosis was attempted. Its implementation differs from the 2026 Springer base paper in recorded ways (PyTorch rather than TensorFlow, RMSprop-only experimental focus, batch size eight, and frozen VGG feature extractors after collapse).

The completed research has five overlapping strands. **Model development** demonstrated end-to-end operation and compared five backbones and neural-versus-SVM classification. **Integrity analysis** audited duplicates and revised the interpretation of original accuracy. **Training analysis** replaced unequal early stopping with 30 epochs and tested 35 one-factor configurations. **External validation** froze 15 models, independently checked dataset overlaps, and compared structured BDNeuro and a difficult Web-40 set. **Explainability and software** completed a fixed formal Grad-CAM/occlusion study and turned the application into an interactive, seven-tab research dashboard.

The principal scientific conclusion is **methodological**: high internally measured public-dataset accuracy is insufficient evidence of robust transfer or lesion-centred reasoning. Reporting how accuracy drops under increasingly independent and unfamiliar evaluation conditions is more defensible than selecting a single maximum benchmark score. See the [complete anonymized GitHub research-report edition](RESEARCH_REPORT_GITHUB_EDITION.pdf) for the authoritative tables, graphical results, application screenshots, literature review and appendices.

## 1. Architecture and the training decision path

**Classification target.** One two-dimensional raster MRI image is mapped to one of four classes. These labels are from the dataset and must not be reinterpreted as benign/malignant, tumour grade, molecular diagnosis or patient-level disease status. All five backbones are independently selectable; the project has **no five-backbone ensemble vote**.

**Transform.** A source MRI image is opened as RGB, directly resized to **224 × 224**, converted to a tensor and normalized by ImageNet means **(0.485, 0.456, 0.406)** and standard deviations **(0.229, 0.224, 0.225)**. There is no skull stripping, direct tumour crop, denoising, histogram harmonization or DICOM study parsing. During *training only*, each image may be horizontally/vertically flipped (probability 0.5 each) and jittered for brightness 0.1, contrast 0.2, saturation 0.2 and hue 0.1. Validation, feature extraction, internal/external tests and GUI uploads use the same deterministic evaluation transform.

**CNN.** [`model_cnn_lstm.py`](../model_cnn_lstm.py) defines VGG16/VGG19 (512 output channels), ResNet50/ResNet152 (2,048), EfficientNet-B3 (1,536) without the classifier/global pooling head. The final feature grid is nominally **7×7** for a 224×224 input. Fine-tuning collapsed the VGG16 hybrid in an early experiment, yielding near-majority-class validation, near-constant embeddings and NaNs during PCA. The diagnostic located the problem in the overall fine-tuning setup rather than the pretrained convolutional output alone; a single causal fault was not isolated. **Both VGG CNN extractors were frozen**, and their LSTM/normalization/dense/head modules trained; the two ResNets and EfficientNet-B3 remained fine-tuned. This introduces a training-policy difference into comparisons.

**Spatial-to-sequence.** The grid `[B,C,7,7]` is permuted/reshaped to `[B,49,C]`; raster-order **positions within a single image** become the sequence steps. The LSTM has **one recurrent layer**, `batch_first=True`, baseline hidden size **256** (test variants 128/512). It is not a temporal or multi-slice MRI model. Only the **last** LSTM position representation is passed to `BatchNorm1d → Dropout(0.3) → Linear(256,128) → ReLU`. This creates a **128-dimensional** learned feature for each MRI image.

**Two classifier paths.** A temporary neural `Linear(128,4)` head is used to train the CNN–LSTM with cross-entropy on logits and supports direct neural prediction and neural-head Grad-CAM. For the *official final* four-class decision, deterministic **train-feature-derived** StandardScaler and 64-component PCA are fitted; an `SVC(kernel="linear", C=1.0, probability=False)` is trained on the resulting training features. It receives **128-D → 64-D** scaled/PCA feature vectors and outputs the predicted class. No validation/test/external/upload features are used to fit these transformations. The GUI displays SVM **relative decision scores**, not calibrated probabilities, even when it displays a softmax of score values for convenience.

**Training.** Baseline optimizer **RMSprop**, LR **1e-4**, batch size **8**, seed **42**, dropout **0.3**, dense features **128**, LSTM **256**. The training loader uses `drop_last=True` because 4,569 training examples leave a one-image remainder at batch size eight and BatchNorm cannot train on a batch of one. At the same time, random shuffling prevents systematically excluding that same image every epoch. Both saved split indices and recorded random seeds support repeatability but do not eliminate GPU nondeterminism or make one-seed evidence statistically conclusive.

**Architecture tensor table (ResNet152 baseline):**

| Operation | Shape |
|---|---|
| RGB MRI | `[B, 3, 224, 224]` |
| ResNet152 convolutional grid | `[B, 2048, 7, 7]` |
| Spatial sequence | `[B, 49, 2048]` |
| One-layer LSTM output | `[B, 49, 256]` |
| Last LSTM step | `[B, 256]` |
| Dense / ReLU representation | `[B, 128]` |
| Scaled and PCA features | `[B, 128] → [B, 64]` |
| Final SVM output | One class among four |

**Code entry points.** Core definition: `model_cnn_lstm.py`. Data transforms/loaders: `src/data_pipeline_torch.py`. Train/evaluation scripts: `actual_training_scripts/`. Saved artifact registry: `gui/model_registry.py`. Runtime inference: `gui/inference_engine.py` and final root `app.py`.

## 2. Datasets, splits and why the labels matter

### 2.1 Original development dataset

The project used the [Nickparvar Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset): **7,023** images, in `Training/` (5,712) and `Testing/` (1,311). Within the supplied `Training/` folder, a stratified **80/20** split with seed 42 and saved array indices generated **4,569 train / 1,143 validation**. Every model was fitted with the same training images, and checkpoints/configurations were selected on validation; the official 1,311-image Testing partition was not a tuning set.

The image count by class was glioma **1,321/300** (supplied train/test), meningioma **1,339/306**, no tumour **1,595/405**, pituitary **1,457/300**. The original dataset did **not** supply reliable patient identifiers, study identifiers, acquisition metadata or verified patient-level independence.

The later data-integrity audit showed the source's supplied Training and Testing folders already overlapped in image content. The split strategy was therefore methodologically consistent (test never used to train or select) but **not image-level independent**. This distinction is central to interpreting every original internal result.

### 2.2 Structured external dataset

The exact external release was [BDNeuro-MRI Version 8](https://data.mendeley.com/datasets/zwr4ntf94j/8), DOI [10.17632/zwr4ntf94j.8](https://doi.org/10.17632/zwr4ntf94j.8), published 8 September 2026. The project counted **5,941** class images (4,160 train, 892 validation, 889 test in source organization). Before using them in *evaluation*, it computed file/pixel and perceptual comparisons with all **7,023** Nickparvar images. It found **3,297 unique exact decoded-pixel matches** with Nickparvar and excluded those only. Thus, the primary full evaluation cohort comprised **2,644** images, by class **1,159 glioma / 472 meningioma / 412 no tumour / 601 pituitary**. Since no external images were used for fitting or selection of the frozen candidates, source-train/val/test were pooled into one external test cohort.

A deterministic **412-per-class balanced sensitivity subset** of **1,648** images was drawn from the same primary clean 2,644 images. It is a *nested reweighting* of that cohort and **not an independent second external dataset**. Non-identical pHash near matches were retained because anatomically similar brain slices need not be the same underlying scan. The supplier's claim of zero *within-dataset* duplication did not imply zero *cross-dataset* overlap with Nickparvar.

### 2.3 Web-40 stress-test set

The project assembled **40** source-verified 2-D MRI panels (10/class) from primary medical publishing/teaching pages; see the [source-list workbook](DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx) for **per-image URLs and figure provenance**. Search results were only candidate-discovery tools: a diagnosis came from the source page, and CT, histology, synthetic segmentation overlays or unsupported diagnoses were excluded. The initial audit identified two perceptually similar candidates. `NOR_01`, though not exact-pixel-identical, was visually the same underlying development MRI and **replaced before inference**; `MEN_02` was visibly a different scan and **retained**. The final 40-image set was fingerprinted and locked **before model predictions**. A total of **15 × 40 = 600** outputs was scored. This is a small convenience sample probing severe presentation differences, not a credible stand-alone population accuracy estimate.

**Full licensing/citation details:** [Datasets and references](DATASETS_AND_REFERENCES.md).

## 3. Experiment registry and the complete progression

### Phase 1 — five standardized, early-stopped pipelines

Objective: verify that the hybrid architecture works for all five backbones and compare the trained neural head with final PCA+SVM on the **same checkpoints**. The VGGs had a maximum 30-epoch budget and stopped at **19** (VGG16) and **21** (VGG19) epochs; ResNet50 and ResNet152 each ran **10**, EfficientNet-B3 ran **9**. Early stopping patience was three. Different horizons were later recognized as a confound, motivating Phase 2A.

| Backbone | Best epoch | Neural validation | SVM validation | Neural test* | SVM test* | SVM test macro-F1* |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | 16 | 94.58% | 94.93% | 93.67% | **94.97%** | 94.68% |
| VGG19 | 18 | 94.58% | 94.49% | 93.14% | **93.52%** | 93.10% |
| ResNet50 | 7 | 97.73% | 97.55% | 97.33% | **98.09%** | 98.01% |
| ResNet152 | 8 | 98.25% | 98.25% | 98.17% | **98.40%** | 98.32% |
| EfficientNet-B3 | 6 | 98.43% | 98.51% | 98.02% | **98.47%** | 98.41% |

`*` **Duplicate-contaminated original split** discovered later. The downstream scaler+PCA+SVM branch improved SVM test versus neural-head test by **0.23–1.30 pp** for the five backbones, but not consistently on validation. This is **not** a PCA-only ablation: PCA and classifier type both change.

**Canonical evidence:** `experiments/<backbone>/results/{training_results,neural_head_results,final_results}.json` and `results/neural_vs_pca_svm_comparison.csv`. Earlier root-level ResNet152 proof-of-concept results (~97.64% according to historical project notes) are not mixed into the standardized five-backbone comparisons.

### Phase 2D — data integrity, executed before other Phase 2 runs

The 20 September 2026 audit scanned all **7,023** original images, across **train–validation, train–test, validation–test**. It used file SHA-256, decoded-pixel SHA-256, pHash/dHash candidate search and strict pixel MAE/correlation confirmation. Results:

| Finding | Count |
|---|---:|
| Exact file or exact decoded-pixel cross-split pairs | **464** = 304 byte-identical + 160 pixel-only |
| Additional high-confidence near-duplicate pairs | **98** |
| **Material cross-split duplicate pairs** | **562** |
| Additional review-only perceptual candidates | **481**, not included in 562 |
| Unique implicated training images | **354 / 4,569** |
| Unique implicated validation images | **122 / 1,143** |
| Unique implicated official-test images | **250 / 1,311 (19.07%)** |
| Material pairs with contradictory class labels | **0** |

**Pairs are not images**. The 562 is a pair count, and the 250 excluded test images are unique affected test files. The primary, post-hoc `material_clean` test removed those **250** from the original **1,311**, leaving **1,061**. Other predefined test-only policies yielded `exact_clean` n=1,095 and `conservative_clean` n=926. Post-hoc re-scoring kept Phase 1 SVM accuracy **93.025–98.680%** across models, showing that identified duplicate test images were *not the sole explanation* for high internal scores. But training/validation overlaps remained, and no patient identifiers existed; **a new independent internal score was not established**.

Immediately after discovering leakage, an earlier status note marked the project `NOT locked` and treated duplicate-group retraining as blocking. The later, documented methodology preserved the contaminated-split metrics as historical, performed a test-only leakage-aware sensitivity analysis, froze models and sought a separately audited external cohort. **No duplicate-group-aware retraining was carried out**; it remains optional scientific strengthening after this completed mini-project. The old status file survives in `final_project_docs/FINAL_QC_BLOCKER.md` for traceability, not as the latest project status.

### Phase 2A — standardized fixed 30-epoch protocol

Each original baseline configuration was run for **exactly 30 epochs** without early stopping. Both `best_val_within_30` (highest **neural** validation within the 30 epochs) and the **epoch-30** checkpoint were saved and evaluated through train-only scaler, PCA and SVM. Selection of the retained state used the **final full-hybrid validation accuracy**, macro-F1 tie-break and a deterministic preference for the neural-best state on exact tie; no test-selection tuning.

| Backbone | Selected state and epoch | SVM validation* | SVM original test* | SVM post-hoc `material_clean` test |
|---|---|---:|---:|---:|
| VGG16 | best-within-30, epoch 30 | 95.888% | 96.186% | 95.853% |
| VGG19 | best-within-30, epoch 30 | 95.626% | 95.271% | 95.099% |
| ResNet50 | best-within-30, epoch 30 | 98.688% | 98.932% | 98.963% |
| **ResNet152** | **best-within-30, epoch 10** | **98.600%** | **98.703%** | **98.869%** |
| EfficientNet-B3 | best-within-30, epoch 27 | 99.300% | 99.314% | 99.246% |

`*` Contaminated original partition(s). These are **five** representative states; the full Phase 2A directory also has the second state per backbone (10 total downstream evaluations). In VGG16, VGG19 and ResNet50 the best neural epoch happened to be 30; the retained **state name** nevertheless remains `best_val_within_30`, not a silently substituted `epoch_30_model.pth`.

### Phase 2B — 35 controlled runs, 70 downstream validations

Each five-backbone baseline in Phase 2A was compared with seven one-factor changes: LR `{1e-5, 5e-5, 5e-4}` (baseline `1e-4`), LSTM size `{128,512}` (baseline 256), dropout `{0.2,0.5}` (baseline 0.3). Each variant completed **30 epochs**, with both retained checkpoints evaluated downstream. **The official test set was deliberately not evaluated for any Phase 2B variant**, preventing 70 downstream states from becoming test-based model selection candidates.

Mean across-backbone ranges of **peak neural validation** were **4.62 pp for learning rate**, **0.58 pp for LSTM width** and **0.42 pp for dropout**. Learning-rate effects differed by architecture and frozen-vs-fine-tuned optimization policy; 1e-5 underfit the VGGs while 5e-4 destabilized fine-tuned ResNets. Among 35 runs, the neural-best state yielded higher SVM validation in **20**, epoch-30 state in **8**, and **7** tied. Internal gains of fractions of a point do **not** establish reproducible or external superiority (one seed, one split, no factor interactions).

| Backbone | Phase 2B internal-validation representative | State / epoch | SVM validation* |
|---|---|---|---:|
| VGG16 | LR 5e-4 | best-within-30 / 27 | 96.500% |
| VGG19 | dropout 0.5 | epoch-30 / 30 | 95.538% |
| ResNet50 | dropout 0.2 | best-within-30 / 28 | 99.125% |
| ResNet152 | LSTM 512 | best-within-30 / 26 | 99.300% |
| EfficientNet-B3 | LSTM 512 | best-within-30 / 29 | 99.388% |

### Phase 2C — candidate freeze and structured external validation

For each backbone, Phase 1, Phase 2A and best-on-internal-validation Phase 2B candidates were frozen as **15** independently evaluable pipelines. The manifest lists a selection rule, variant, state and epoch; for every pipeline, **five** artifact hashes cover neural checkpoint, scaler, PCA, SVM and internal results, **75 SHA-256 values** in total. The manifest precedes the earliest external result (2 October 2026 freeze vs 3 October external runs). Original QC reported **75/75 matching local artifact hashes**. Since the public Git checkout contains **zero `.pth` files**, a new public-only audit cannot independently recompute the checkpoint portion; do not claim we have rerun it on GitHub.

The first external raw image set, BDNeuro V8, had **5,941** class images; the audited full cohort had **2,644** after **3,297** pixel-duplicate image exclusions. The balanced sensitivity subset was **1,648** nested images. ResNet152 Phase 2A achieved **88.31% full** and **90.47% balanced**; all 15 full-cohort scores ranged **75.00–88.31%**, meaning corresponding declines of **10.29–20.54 pp** from internal validation. The Phase 2A stage mean was **83.21%** full, compared with **82.34%** for Phase 2B and **79.77%** for Phase 1. These are descriptive comparisons, not a causal proof about the training budget or a repeated-run hypothesis test.

### Web-40 — strong presentation shift

After a source and duplicate audit, the 40-image set was locked. Evaluation of **15 frozen pipelines × 40 = 600 decisions** produced **25.0–42.5%** accuracy across candidates. The two highest observed results were **ResNet50 Phase 2A and EfficientNet-B3 Phase 2A, both 42.5% (17/40)**. Those maxima should not be overstated: one additional image is 2.5 percentage points.

Averaged over the 15 pipelines, mean per-class recall was **12.7% glioma, 30.7% meningioma, 90.7% no tumour, and 7.3% pituitary**. Mean predicted count for no-tumour was **28.3 per 40 images**, versus a true count of 10. These figures demonstrate a severe *failure mode for this particular exploratory set*, **not** measured sensitivity in a population of real-world patients. Neither the source images nor model crops were changed after performance was viewed.

### Phase 3 — retrospective representative selection

The final **research representative** was ResNet152 Phase 2A, neural-best within the 30-epoch run at **epoch 10**, LR1e-4/LSTM256/dropout0.3/Dense128/PCA64/linear-SVM(C=1). The representative combined high internal validation with the **strongest BDNeuro full and balanced cohort scores** and the smallest structured generalization gaps. **All 15 candidates were originally frozen based on internal validation**; however **the naming of one final representative used already observed BDNeuro outcomes**. The distinction is recorded to avoid pretending its BDNeuro results were an untouched evaluation of the selected winner. The exploratory Web-40 result **was not used** for representative naming, and ResNet152 was **not** the best pipeline on that small set.

**Full representative trajectory:** original validation **98.600%***; original test **98.703%***; post-hoc `material_clean` test **98.869%** (macro F1 **98.788%**); BDNeuro full **88.31%** (F1 **87.85%**); nested balanced **90.47%** (F1 **90.50%**); exploratory Web-40 **37.5%** (F1 **31.66%**). `*` Original partition has material duplicates.

### Phase 4 — formal XAI and the neural-versus-SVM distinction

The **formally rated** study was completed **20 September 2026** using **Phase 1 ResNet152**, before Phase 2 retraining. It **does not** formally rate the final Phase 2A representative. On 20 fixed official-test examples (seed 42, five per class), Grad-CAM measured the differentiable **neural head** from layer 3 (14×14), layer 4 (7×7) and 50:50 fusion (fixed overlay opacity 0.55). For 15 tumour cases the Good/Partial/Poor counts were layer 3 **1/9/5**, layer 4 **3/2/10**, fused **2/6/7**. Correct neural/SVM classification was often paired with poor apparent tumour-region focus, including skull edges, face, background and extracranial activation.

Because the **deployed final classifier** is a non-differentiable SVM, a separate final-SVM **occlusion-sensitivity** study reran the complete CNN→LSTM→Dense→Scaler→PCA→SVM chain after masking each **32×32 patch**, stride **16**, at **169** windows per image. Eleven predetermined cases yielded **1,859 occlusions**, **58** of which changed the SVM class; the single shared misclassified case `Te-pi_0078` accounted for **38** flips. Occlusion tumour-image ratings were Good **2**, Partial **2**, Poor **5**. It did **not** systematically correct Grad-CAM's off-lesion focus. Single-image maps are normalized and cannot be read as lesion boundaries; even vivid maps sometimes corresponded to minuscule absolute score changes.

Two formal-study images were later found in the material-duplicate set; their original CSV entries are preserved, and exclusion did not change the qualitative conclusion. These are single-rater observations without a radiologist or expert mask. The application may show *live qualitative* Grad-CAM/occlusion for Phase 2A/2B pipelines but these are **not** separate formally rated experiments.

### Phase 5 — final, seven-tab research prototype

The final root [`app.py`](../app.py) contains the complete dashboard, which **supersedes** the earlier [`README_GUI.md`](../README_GUI.md) and any old four-tab screenshots/descriptions. It opens on **Predict MRI** and exposes **five backbones × three stages** as 15 independent choices, including the final ResNet152 Phase 2A epoch-10 representative. The remaining six tabs are **Base-paper reproduction, Model card, Hyperparameter sensitivity, External evaluation, XAI study, Frozen 15 comparison**. Its purpose is to expose **what one selected pipeline predicts AND what the completed evidence says about that pipeline's limitations**.

An uploaded PNG/JPG/BMP/TIFF is transformed without any model fitting. Displayed information includes the SVM predicted class, temporary neural head class, agreement, relative class scores, inference time and optional contribution maps; the results tabs display saved JSON/CSV evidence rather than re-running training. For compatible selected pipelines the ResNets can show layer3/layer4/fused Grad-CAM, raw and overlay maps, and final-SVM patch occlusion. The XAI study tab deliberately distinguishes *formal Phase 1 XAI results* from new qualitative demonstrations. **No DICOM, segmentation or clinical decision-support functionality is implemented.**

## 4. Directory and artifact reference — authoritative public GitHub layout

The repository `main` tree was examined on 4 November 2026. It contained **12,020 tracked files (~388 MB of blob payloads)**, including 10,713 JPEG images, 240 `.joblib` objects, 405 JSON records, 100 CSV files, 342 `.npy` arrays, 85 `.py` files, and **0 tracked `.pth` checkpoints**. These are directory/index counts, **not verification that every experiment artifact is usable in isolation**. This section is organized by where a reader actually searches in the present public GitHub state.

```text
Tumor-Classification-DL-/
├── app.py                         # FINAL seven-tab research dashboard
├── model_cnn_lstm.py              # reusable five-backbone CNN-LSTM model
├── model_resnet152_lstm.py        # legacy initial ResNet152 model
├── gui/                           # registry, inference, model factory, XAI, metrics
├── src/                           # dataset validation/splits/loaders/transforms
├── data/splits/                   # saved train_indices.npy and val_indices.npy
├── actual_training_scripts/       # 23 primary training/evaluation/audit scripts
├── experiments/                   # Phase 1 evidence: vgg16, vgg19, resnet50,
│                                  # resnet152, efficientnet_b3
├── phase2/
│   ├── experiments_30epoch/        # five Phase 2A folders; two state evaluations each
│   ├── hyperparameter_dl/          # 35 Phase 2B run folders and downstream states
│   ├── results/                     # Phase 2A leakage-aware comparisons
│   ├── external_validation/        # freeze manifest; BDNeuro/Web-40 audits & results
│   ├── run_all_dl_hyperparams.py
│   └── run_all_hyperparam_pca_svm.py
├── external_data/                 # BDNeuro V8 copies, exact-clean, balanced, Web-40
├── final_project_docs/            # main duplicate audit, Phase1 clean test, historical notes
├── gradcam_evaluation/            # formal fixed study CSV
├── occlusion_evaluation/          # formal full-SVM patch study CSV
├── results/                       # cross-backbone Phase1 comparison tables
├── final_deliverables/            # earlier report, deck, evidence map and QC records
├── docs/                          # historical context + NEW public-facing docs
├── phase2_full_analysis_package/  # earlier consolidation snapshot
├── phase2a_baseline_results/       # snapshot of baseline summaries
├── phase2b_analysis_package/      # historical sensitivity analysis package
├── README_GUI.md                  # LEGACY early GUI overview
├── requirements-gui.txt          # GUI-only Python requirements
├── verify_gui_setup.py           # LEGACY Phase1-only bundle preflight
├── final_project_qc.py           # cross-split duplicate audit
└── evaluate_leakage_clean_test*.py
```

**Not included by normal tracked-path conventions:** `data/brain_tumor_mri/` original Nickparvar images, plus every `.pth` CNN checkpoint; in this observed public tree there were **zero** `.pth` files. The presence of `*.joblib` SVM/scaler/PCA artifacts or tensor features is **not sufficient** to reconstruct the learned neural CNN/LSTM weights. The full setup can be run from the machine holding valid frozen bundles; the public repository alone is a source-and-evidence archive.

### 4.1 Source code and training scripts

| File / location | Purpose |
|---|---|
| `src/check_dataset.py` | Readability and image-mode inspection for all development files |
| `src/save_split.py` | Deterministic stratified split creation and saved train/validation indices |
| `src/data_pipeline_torch.py` | Training/evaluation transforms, fixed split loading, dataset and DataLoader construction |
| `model_cnn_lstm.py` | Generic `CNNLSTM` and five ImageNet backbone implementations |
| `actual_training_scripts/train_backbone.py` | Phase 1 early-stopped training |
| `actual_training_scripts/evaluate_backbone.py` | Load checkpoint, extract deterministic representations, train-only PCA/SVM fit, metrics |
| `actual_training_scripts/evaluate_neural_head.py` | Evaluate the direct neural head on the same checkpoint |
| `actual_training_scripts/compare_neural_vs_pca_svm.py` | Downstream versus neural-head comparison |
| `actual_training_scripts/diagnose_vgg16_collapse.py` | Investigated degenerate VGG16 feature collapse |
| `actual_training_scripts/train_backbone_30epoch.py` | Phase 2A fixed-30 training, retains two states |
| `actual_training_scripts/evaluate_backbone_30epoch.py` | Both 2A checkpoint states → scaler/PCA/SVM/results |
| `phase2/run_all_dl_hyperparams.py` | Queue orchestration for 35 Phase 2B training runs |
| `actual_training_scripts/train_backbone_hyperparams.py` | One-factor training worker |
| `phase2/run_all_hyperparam_pca_svm.py` | Queue downstream processing for Phase 2B variants |
| `actual_training_scripts/evaluate_hyperparam_pca_svm.py` | Train/validation-only downstream runs; deliberately **no test** |
| `final_project_qc.py` | Development-set duplicate/near-duplicate audit |
| `evaluate_leakage_clean_test.py` / `_phase2.py` | Post-hoc exact/material/conservative clean test re-scoring |
| `actual_training_scripts/audit_bdneuro_overlap.py` | External-versus-development file/pixel/perceptual overlap audit |
| `actual_training_scripts/build_bdneuro_exact_clean_cohort.py` | Exact-pixel exclusion and primary 2,644-image cohort |
| `actual_training_scripts/build_bdneuro_balanced_subset.py` | Deterministic nested 1,648-image subset |
| `actual_training_scripts/compare_external_15.py` | Frozen-pipeline external evaluation; distinct named results roots |
| `actual_training_scripts/audit_web_stress_40.py` | Web-40 candidate and cross-dataset integrity checks |
| `app.py` | FINAL UI with its own 15-pipeline specifications and result views |
| `gui/model_registry.py`, `gui/inference_engine.py` | 15 frozen-pipeline registry and general inference tools |
| `gui/gradcam.py`, `gui/occlusion.py` | Differentiable neural-head Grad-CAM vs full-SVM occlusion |
| `gui/model_factory.py`, `gui/results_loader.py` | Checkpoint-compatible model reconstruction / saved-metric presentation |
| `gradcam_evaluation_app.py`, `occlusion_evaluation_app.py` | Reproducible formal Phase 1 XAI case rating apps |

**Historical detail:** An initial `compare_external_15.py` iteration overwrote output folders between BDNeuro full and balanced evaluations. The later script's separate `--results-name` parameter fixed this. One temporary zero-byte script accidentally passed Python bytecode compilation (because an empty file is syntactically valid), so command-line `--help` behavior was also checked. Neither debugging incident is itself evidence of model performance; the final separate result directories are the canonical records.

### 4.2 Exact layout of one artifact bundle

**Phase 1:**

```text
experiments/<backbone>/
    results/training_results.json
    results/neural_head_results.json
    results/final_results.json
    models/best_model.pth               # local trained checkpoint; NOT tracked in Git
    models/svm.joblib                   # downstream artifact (availability varies)
    pca/standard_scaler.joblib
    pca/pca_model.joblib
```

**Phase 2A:**

```text
phase2/experiments_30epoch/<backbone>/
    results/training_results.json
    models/best_val_within_30.pth      # local, NOT tracked
    models/epoch_30_model.pth          # local, NOT tracked
    evaluations/best_val_within_30/
        results/final_results.json
        models/svm.joblib
        pca/{standard_scaler,pca_model}.joblib
    evaluations/epoch_30/
        results/final_results.json
        models/svm.joblib
        pca/{standard_scaler,pca_model}.joblib
```

**Phase 2B (one of the 35 runs):**

```text
phase2/hyperparameter_dl/<backbone>/<variant>/
    results/training_results.json
    models/{best_val_within_30,epoch_30_model}.pth  # local, NOT tracked
    evaluations/best_val_within_30/
        results/final_results.json                # train/validation ONLY
        models/svm.joblib
        pca/{standard_scaler,pca_model}.joblib
    evaluations/epoch_30/
        results/final_results.json                # train/validation ONLY
        models/svm.joblib
        pca/{standard_scaler,pca_model}.joblib
```

`<backbone>` ∈ `{vgg16, vgg19, resnet50, resnet152, efficientnet_b3}`. `<variant>` ∈ `{lr_1e-5, lr_5e-5, lr_5e-4, lstm_128, lstm_512, dropout_0.2, dropout_0.5}`. The chosen Phase 2A/2B bundle must be combined with **its own** checkpoint, scaler, PCA and SVM. **Never mix** checkpoints, feature scalers or SVM files from different phases/epochs/backbones.

### 4.3 External-evidence and audit files

| Folder/path | What a researcher should inspect |
|---|---|
| `phase2/external_validation/frozen_15_selection.csv` | Phase/backbone/variant/retained state chosen before external inference |
| `phase2/external_validation/frozen_15_manifest.json` | Pre-external selection rule, 15 candidates and 75 SHA-256 hashes |
| `phase2/external_validation/bdneuro_overlap_audit/summary.json` | Raw counts, exact pixel overlap, candidate screens |
| `phase2/external_validation/bdneuro_exact_clean_cohort/cohort_summary.json` | Main `5,941 − 3,297 = 2,644` definition |
| `phase2/external_validation/bdneuro_balanced_subset/balanced_subset_summary.json` | Nested balanced 412/class subset |
| `phase2/external_validation/comparison_15_models_bdneuro_full/` | Full natural-distribution external results; model JSON & per-image CSVs |
| `phase2/external_validation/comparison_15_models_bdneuro_balanced/` | Nested balanced sensitivity analysis |
| `phase2/external_validation/comparison_15_models_web_stress_40/` | Web-40 exploratory performance and per-image outputs |
| `phase2/external_validation/web_stress_40_audit/` | Original web-set audit, pHash candidates and visual-review references |
| `phase2/external_validation/web_stress_40_locked/lock_record.json` | Pre-inference dataset fingerprint, 40 class-count record, no-replacement rule |
| `final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json` | Audit counts, pair types and status `MATERIAL_DUPLICATION` |
| `final_project_docs/leakage_corrected_evaluation/` | Phase 1 post-hoc material/exact/conservative subsets and metrics |
| `phase2/results/leakage_corrected_30epoch/` | Corresponding Phase 2A post-hoc evidence |
| `gradcam_evaluation/*.csv` and `occlusion_evaluation/*.csv` | Frozen Phase 1 ResNet152 qualitative ratings and numerical occlusion data |

### 4.4 Historical, generated and reference material

`docs/DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/docs/` houses detailed previously written histories of methodology, experiment register, debugging, rejected approaches and source-recovery notes. `docs/claude_project_context/` and the `phase2*_analysis_package/` folders are earlier snapshots used for project continuity. `final_deliverables/FINAL_PROJECT_REPORT.pdf` is the **earlier**, pre-revision full project deliverable; this GitHub edition uses the **later 60-page revised report supplied in the conversation**. `final_deliverables/FINAL_PROJECT_EVIDENCE_MAP.md` supplies claim-to-file traceability and `FINAL_PROJECT_QC_REPORT.md` records original local checks (**174/174 checks passed, 75/75 local hashes matched**). These are recorded QC claims, not newly executed full checks on the public `.pth`-free clone.

## 5. Use the repository: environment, running, and what remains unavailable

The recorded development environment was **Ubuntu under WSL2**, **Python 3.12.14**, **PyTorch 2.11.0+cu128**, torchvision 0.26.0+cu128, scikit-learn 1.9.0, NumPy 2.5.2, Streamlit 1.63.0 and **RTX 5070 Laptop GPU (8GB)**. A historical TensorFlow setup failed for the Blackwell GPU with an unsupported PTX path, so the project used PyTorch. These are **reproduction environment details**; the current exact version combination may not be generally available from every package index, and hardware compatibility should be checked with official PyTorch instructions.

```bash
git clone https://github.com/bakayaro05/Tumor-Classification-DL-.git
cd Tumor-Classification-DL-
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -U pip
# Install suitable PyTorch/torchvision for local CPU or CUDA first.
# See: https://pytorch.org/get-started/locally/
python -m pip install -r requirements-gui.txt
streamlit run app.py
```

**Main readiness limitation:** public `main` currently has **zero `.pth` files**, so live predictions require the authentic trained neural checkpoints. This is separate from code correctness and from whether research-result tabs can read committed CSV/JSON. Use `app.py` or the 15-pipeline registry to inspect exact required artifact paths; an incomplete bundle should be shown as missing, not silently reconstructed from different runs. The older `verify_gui_setup.py` checks Phase 1 only. Restoring `.pth` checkpoints is **not equivalent** to rerunning the research or refitting PCA/SVM; avoid loading untrusted checkpoint/joblib data.

**Original dataset is also absent from its development path.** To retrain on the original Nickparvar data, download the exact 7,023-image version, verify original folder names and ordering against saved indices, and confirm GPU availability. If the dataset version differs, recompute an appropriate split and redo the entire evaluation, rather than projecting fixed index arrays onto a different sample order.

**Safe order for an independent reviewer:** read source descriptions and result JSON/CSV → verify provenance and split counts → inspect frozen selection/hashes as records → review BDNeuro overlap exclusions → inspect 15-pipeline external results and Web-40 lock → inspect Grad-CAM/occlusion ratings → only then consider executable inference with trusted checkpoint bundles. Do not download public MRI images from this mirror and infer patient-level independence or permission to use them clinically.

## 6. Debugging and decisions that must remain visible

The long implementation journey matters because it prevents readers from mistaking rejected methods for actual experiments:

1. **Classical preprocessing/GLCM/K-means/RBF-SVM** was an initial idea only. The final network does not do K-means segmentation; K-means cluster indices cannot be interpreted as tumour categories. Tumour type cannot be trivially relabelled benign versus malignant.
2. **TensorFlow compatibility failure:** GPU detection alone did not guarantee real-world compatibility; actual `sm_120`/`INVALID_PTX` on RTX 5070 drove the switch to PyTorch. This is an implementation history, not a comparison of framework accuracy.
3. **Architecture-stage tests:** small scripts proved CNN grid size, reshaping, LSTM dimension, post-LSTM shapes, real batch and backward pass before final long training. Correctly describe the LSTM input as 49 spatial positions.
4. **VGG collapse:** about 28% VGG16 validation with near-constant features and PCA NaNs prompted **frozen VGG convolutional policy**; full fine-tuning was not quietly reported as successful.
5. **Legacy model:** an early root ResNet152 prototype established feasibility but is not the standardized reference; its approximated ~97.64% result is historical and was superseded.
6. **BN batch-one edge case:** `drop_last=True` avoids a training batch of size one; a script-level fix rather than dataset deletion.
7. **Scientific pre-lock audit:** the largest problem was not a training bug but the failure of a presumed image-independent test split. Duplicates and the withdrawal of an independent-generalization claim were documented, not hidden.
8. **Remediation policy evolved:** initial post-audit 'not locked' note became an explicit contaminated-split interpretation + `material_clean` sensitivity test + frozen external cohort. Group-aware retraining remains future work.
9. **External provenance surprise:** an apparently different dataset contained over 3,000 exact image overlaps with Nickparvar. Direct pixel evidence took precedence over a narrative of independent provenance, without assigning blame for dataset lineage.
10. **pHash needs human judgment:** Web `NOR_01` was transformed reuse and removed before inference, whereas `MEN_02` was genuinely different and retained. Automatic pHash exclusion is not scientific proof of duplication.
11. **Outputs must be isolated:** an evaluator initially collided full/balanced results; named outputs fixed it. A zero-length script can pass syntax compilation; functional CLI help and result paths matter too.
12. **Explainability must match the deployed output:** Grad-CAM is neural-head attribution, while the final SVM requires a separate perturbation approach. Neither is a radiological segmentation mask.
13. **No after-the-fact tuning:** after frozen external/Web-40 performance became known, no pipeline was retrained or adjusted. Any future improvement requires a *new* untouched test set.
14. **Not executed:** the base paper's entire four-optimizer 20-configuration grid, a sixth DenseNet121 backbone, an ensemble classifier, DICOM/3-D classification, new XAI variants chosen to beautify maps, and post-hoc web-image replacement are **not** part of the completed study.

## 7. Interpretation guardrails and reproducibility limits

**Internal scores:** historical on a known material-duplicate-contaminated split; not an estimate of independent patient-level performance. `material_clean` is a post-hoc **test-only** operation and cannot fix train↔validation duplication.

**External scores:** BDNeuro was audited to remove exact *cross-dataset* pixel overlaps, not guaranteed patient-level non-overlap; the balanced subset is nested; the 15 already-frozen pipelines are valid descriptive candidates, but the representative name used BDNeuro outcomes retrospectively.

**Web results:** convenience panels (n=40), no radiologist re-adjudication, one image = 2.5pp, multiple acquisition/presentation shifts at once. Strong notumor bias is a serious *research signal*, but the measured Web-40 percentages are not clinical sensitivity, screening performance, or a population prevalence estimate.

**Explainability:** small fixed, single-rater Phase 1 ResNet152 protocol; two evaluated cases had material duplicate status; maps are contributions/sensitivities not expert tumour masks. New UI-generated maps do not expand the formal study population.

**Uncertainty:** one original seed and saved split, no repeated training seeds, confidence intervals, paired tests, stratified patient-level split, verified acquisition metadata, or quantified ablations of LSTM/PCA presence. `PCA+SVM` versus neural head changes **two** downstream components. The observed ranking across practical frozen/fine-tuned backbones is not a pure architectural ranking.

**Medical scope:** research-only 2-D slice classification and visualization. No DICOM, no whole-scan integration, no calibrated clinical probabilities, no validated deployment or regulatory approval. The project cannot be used to diagnose individuals.

**Public artifacts and rights:** the GitHub repository currently includes **many external MRI images**, but the source's license, patient consent and individual web-panel permissions need verification. A current public commit is not proof of redistribution authorization. See [dataset provenance and licensing](DATASETS_AND_REFERENCES.md) and the [release audit](PUBLICATION_NOTES.md).

## 8. Reference trail

- [Primary base paper](https://doi.org/10.1007/s13721-025-00718-8) — Bérété et al. (2026), methodological starting point.
- [Original development dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Masoud Nickparvar, Kaggle.
- [Exact BDNeuro-MRI study release, Version 8](https://data.mendeley.com/datasets/zwr4ntf94j/8) — Mendeley Data, DOI [10.17632/zwr4ntf94j.8](https://doi.org/10.17632/zwr4ntf94j.8).
- [Web-40 per-image source list](DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx) — individual original medical sources rather than a single benchmark.
- [Completed research QC](../final_deliverables/FINAL_PROJECT_QC_REPORT.md), [evidence map](../final_deliverables/FINAL_PROJECT_EVIDENCE_MAP.md), [frozen manifest](../phase2/external_validation/frozen_15_manifest.json), [three cohort comparisons](../phase2/external_validation/), and [GitHub-edition full report](RESEARCH_REPORT_GITHUB_EDITION.pdf).

This document deliberately credits **source papers and datasets** but **does not identify project members or a project guide**. The research originated as a **university college mini-project**. Copyright and academic source attribution remain applicable to outside works.
