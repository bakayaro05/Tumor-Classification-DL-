# Hybrid CNN–LSTM–PCA–SVM for Four-Class Brain Tumor MRI Classification

> **University college mini-project · completed research study · PyTorch / scikit-learn / Streamlit**  
> **Research and educational use only. Not a clinical diagnostic system, medical device, or tumour segmentation tool.**

This repository documents a complete implementation, experimental comparison, data-integrity investigation, external-validation study, explainability evaluation, and interactive research dashboard for **four-class brain tumour classification from individual two-dimensional MRI images**. The four supplied dataset classes are **glioma, meningioma, pituitary, and no tumour** (`notumor`). They are classification labels, not tumour grades or benign/malignant diagnoses.

The work began as an implementation of the hybrid **CNN + LSTM + SVM** framework described by [Bérété et al. (2026)](https://doi.org/10.1007/s13721-025-00718-8) and developed into an investigation of a more consequential question: **Does near-99% accuracy on a familiar public MRI dataset survive duplicate-leakage auditing, changes in training protocol, independent external evaluation, and explanation analysis?**

**Main finding:** Not reliably. The original public split contained cross-partition image duplication; the best structured external result was substantially lower than internal validation, and a small, heterogeneous 40-image web stress test showed severe performance degradation and a tendency to predict *no tumour*. High classification accuracy also did not consistently correspond to lesion-centred explanation maps. The project therefore reports the full evidence chain rather than promoting a single headline score.

**Read next:** [Detailed repository documentation](docs/REPOSITORY_DOCUMENTATION.md) · [GitHub edition of the full research report](docs/RESEARCH_REPORT_GITHUB_EDITION.pdf) · [Dataset and literature references](docs/DATASETS_AND_REFERENCES.md) · [Public repository audit and reproducibility notes](docs/PUBLICATION_NOTES.md).

**Available model releases:** [`models-v1.0` — all 15 frozen inference pipelines](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) · [`research-checkpoints-v1.0` — 71 additional historical neural checkpoints](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/research-checkpoints-v1.0) · [Model download, restoration and 75-hash verification guide](docs/MODEL_WEIGHTS.md) · [Full reproduction guide](docs/REPRODUCE.md). These releases are **separate from Git-tracked files**.

## At a glance

| Item | Final project record |
|---|---|
| Task | Single **2-D raster MRI** image → glioma / meningioma / pituitary / no tumour |
| Backbones | **VGG16, VGG19, ResNet50, ResNet152, EfficientNet-B3** — five **independent** pipelines, **not** an ensemble |
| Architecture | ImageNet-pretrained CNN → 7×7 feature grid → 49-position spatial sequence → 1-layer LSTM → BatchNorm / Dropout → 128-D feature → StandardScaler → PCA(64) → linear SVM |
| Development data | [Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset): **7,023** images; 4,569 train / 1,143 validation / 1,311 dataset-supplied test |
| Model variants | Phase 1 early stopping; Phase 2A fixed 30 epochs; Phase 2B one-factor sensitivity study |
| Sensitivity study | **35** runs (5 × 7 variants); **70** downstream PCA/SVM validations |
| Frozen external comparison | **15** hybrid pipelines (5 backbones × 3 phases), selected on internal validation and SHA-256 recorded before external evaluation |
| Primary structured external cohort | [BDNeuro-MRI **Version 8**](https://data.mendeley.com/datasets/zwr4ntf94j/8), DOI [10.17632/zwr4ntf94j.8](https://doi.org/10.17632/zwr4ntf94j.8); **2,644** exact-overlap-clean images |
| Other evaluations | Nested, class-balanced BDNeuro subset (**1,648**); locked, source-verified exploratory **Web-40** set |
| Explainability | Formal Phase 1 ResNet152 Grad-CAM (20 images) and full-pipeline SVM occlusion (11 cases) |
| Application | Streamlit **seven-tab** dashboard exposing all **15 frozen pipelines**, predictions, model cards, sensitivity, external results, formal XAI evidence and comparisons |
| Final project representative | **ResNet152 · Phase 2A · `best_val_within_30` · epoch 10** (retrospective project-level selection; not universally best) |
| Platform | Ubuntu / WSL2; Python 3.12.14; PyTorch 2.11.0+cu128; RTX 5070 Laptop GPU (8 GB) in the recorded environment |

## The complete model

```text
One MRI image (RGB, 224 × 224)
       │
       ▼
ImageNet-pretrained CNN backbone
(VGG16 / VGG19 / ResNet50 / ResNet152 / EfficientNet-B3)
       │  7 × 7 × C feature grid
       ▼
Spatial reshape: 49 positions × C channels
       │  (positions within ONE image; NOT a temporal MRI sequence)
       ▼
One-layer LSTM, hidden size 256 (512 in selected Phase 2B variants)
       ▼
BatchNorm1d → Dropout (baseline 0.3) → Dense 128 → ReLU
       │
       ├── Temporary 4-logit neural classification head
       │     Trains CNN–LSTM using cross-entropy; also supports Grad-CAM
       │
       └── Final decision path (trained on TRAIN features only)
             StandardScaler → PCA(64) → Linear SVM (C = 1)
                                     │
                                     ▼
                         Predicted MRI image class
```

All five CNN–LSTM–PCA–SVM systems were trained and evaluated separately. **VGG16 and VGG19 used frozen convolutional backbones** after an unstable full-fine-tuning run; **ResNet50, ResNet152 and EfficientNet-B3 were fine-tuned**. Consequently, cross-backbone differences compare these *implemented training pipelines*, not architecture alone. The LSTM reads raster-ordered **spatial feature positions**, not a time series, image volume or patient scan history.

For `ResNet152`, the tensor path is `[B,3,224,224] → [B,2048,7,7] → [B,49,2048] → [B,256] → [B,128] → [B,64] → 4-class SVM decision`. The standardized SVM outputs **decision values** (`probability=False`); derived class bars in the GUI are **relative scores, not calibrated probabilities**.

## Research journey: why each phase happened

| When / phase | What was done | Why the direction changed |
|---|---|---|
| July–August 2026: concept | Initially considered median filtering, skull removal, K-means, handcrafted texture/GLCM and RBF-SVM. | Superseded by the requirement to implement a recent deep-learning base paper. K-means cluster IDs were not treated as tumour labels. |
| August 2026: environment | Built WSL2/Ubuntu environment. TensorFlow detected the RTX 5070 but failed on realistic workloads with unsupported `sm_120`/`INVALID_PTX`; attempted a nightly build. | Migrated to a working **PyTorch + CUDA** stack. |
| Late August–early September: proof of concept | Checked the CNN feature map, reshape, LSTM, post-LSTM layers, real-MRI forward pass and training step separately; trained an initial ResNet152 hybrid. | Established a correct executable architecture before extending to five backbones. The early root-level ResNet152 artifacts are **legacy**. |
| Phase 1: standardized baselines | Trained five CNN–LSTM pipelines, extracted 128-D features, fitted train-only scaler/PCA/SVM and compared the neural head with the downstream SVM. | Achieved 93.52–98.47% original-split SVM test accuracy, then asked whether this measured independent generalization. |
| **Phase 2D: integrity audit** (conducted before 2A/2B) | File hashes, decoded-pixel hashes and validated perceptual matching across train↔val, train↔test and val↔test. | Found **562 material cross-split duplicate pairs** involving **250 test images**. Original-split results were recategorized as **historical, duplicate-contaminated measurements**. |
| Phase 2A: convergence | Fixed **30 epochs for every backbone**; retained neural-best and epoch-30 states; evaluated both through PCA/SVM. | Replaced unequal early-stopped training horizons with a common observation period. |
| Phase 2B: sensitivity | Seven one-factor changes per backbone: learning rate, LSTM width, dropout; **35 runs / 70 downstream evaluations**. | Learning rate was the dominant tested sensitivity; small validation gains did not consistently transfer externally. |
| Phase 2C: external verification | Selected **15 pipelines internally**, froze checkpoint/scaler/PCA/SVM bundles, recorded **75 hashes**, audited BDNeuro overlap and evaluated an exact-overlap-clean cohort. | **3,297** of 5,941 BDNeuro images were pixel-identical to development images; excluded them before the primary external evaluation. |
| Web-40 stress test | Verified source pages for 40 web MRI images, audited candidates, replaced a visually confirmed reused image **before inference**, locked the set and made 600 frozen-pipeline predictions. | Accuracy fell to **25.0–42.5%** across pipelines; no-tumour overprediction emerged. |
| Phase 3: synthesis | Compared all 15 frozen pipelines and evaluated evidence at internal, BDNeuro and web-domain levels. | Named **ResNet152 Phase 2A epoch 10** as the overall representative using the stronger structured external evidence; not a claim of universal superiority. |
| Phase 4: explanation | Formal Phase 1 ResNet152 Grad-CAM and final-SVM occlusion, with fixed cases and qualitative ratings. | Correct predictions often had partial/poor tumour-region localization; Grad-CAM explains the neural head while occlusion probes the final SVM. |
| Phase 5: application | Expanded the early Streamlit prototype to a **seven-tab, 15-pipeline research dashboard**. | Converted the UI from a prediction demo to an evidence-delivery interface. |

**Historical decisions are not current scientific claims.** In particular: an immediate post-audit `FINAL_QC_BLOCKER.md` says the project was not locked; subsequent verified evaluations were completed with the contamination disclosed. Duplicate-group-aware *retraining* remains future work, not a completed experiment. `README_GUI.md` describes an earlier four-tab interface; **root `app.py` is the final seven-tab implementation**.

## Results and how to interpret them

### Phase 1 — historical original test set (1,311 images)

| Backbone | Neural-head test accuracy | PCA + linear SVM test accuracy | SVM macro-F1 |
|---|---:|---:|---:|
| VGG16, frozen CNN | 93.67% | 94.97% | 94.68% |
| VGG19, frozen CNN | 93.14% | 93.52% | 93.10% |
| ResNet50 | 97.33% | 98.09% | 98.01% |
| ResNet152 | 98.17% | 98.40% | 98.32% |
| EfficientNet-B3 | 98.02% | 98.47% | 98.41% |

**These are genuine recorded outputs, not independent generalization estimates.** A subsequent audit found **464 exact-byte/decoded-pixel duplicate pairs + 98 high-confidence near-duplicate pairs = 562 material cross-split pairs**, involving **250/1,311 (19.07%)** official test images. The post-hoc `material_clean` test excludes the implicated test images (**1,061 retained**) but **does not repair training/validation contamination or prove patient-level independence**. The 481 additional review-only perceptual pairs were not automatically classed as material duplicates.

### Phase 2 and frozen external study

| Evaluation layer | Final representative: ResNet152 Phase 2A, epoch 10 | Necessary qualification |
|---|---:|---|
| Original validation | **98.600%** | Duplicates exist across the original partitions; used for candidate selection |
| Original internal test, n=1,311 | **98.703%** | Historical duplicate-contaminated split; not used for selecting the pipeline |
| `material_clean` test, n=1,061 | **98.869%**, macro-F1 **98.788%** | Post-hoc exclusion of affected test images only |
| BDNeuro exact-overlap-clean full, n=2,644 | **88.31%**, macro-F1 **87.85%** | Primary structured external cohort; **−10.29 percentage-point** gap from internal validation |
| BDNeuro balanced, n=1,648 | **90.47%**, macro-F1 **90.50%** | **Nested subset of the same cohort**, not an additional independent dataset |
| Source-verified Web-40, n=40 | **37.5%**, macro-F1 **31.66%** | Exploratory stress test; one image = 2.5 percentage points; not used to select the representative |

Across **all 15 frozen pipelines**, exact-overlap-clean BDNeuro full-cohort accuracy was **75.00–88.31%**, whereas Web-40 accuracy was **25.0–42.5%**. The study found stronger average BDNeuro performance for Phase 2A than Phase 2B, despite some Phase 2B gains on the original validation split. On Web-40, the average pipeline predicted **no tumour for ~28.3/40 images**, although only 10 were actually no-tumour examples.

The **15 candidates were frozen before external performance was known**; however, **the single final representative was named retrospectively partly from BDNeuro results**, so its BDNeuro score should not be described as an untouched, post-selection estimate. The full 15-pipeline comparisons are the more transparent evidence. No frozen model was changed after external inference.

### Explainability, correctly scoped

The formal qualitative study used **Phase 1 ResNet152** (not the final Phase 2A representative): **20 Grad-CAM images** (15 tumour cases rated for localization) and **11 SVM-occlusion images × 169 patches = 1,859 occlusions**. Layer-3 Grad-CAM received 1 Good, 9 Partial and 5 Poor ratings on the 15 tumour cases; layer 4 received 3/2/10 and fused 2/6/7. Occlusion had **58 class-changing perturbations** in total. Coarse maps frequently emphasized skull margins, face or image borders. These observations do **not** establish the presence of a tumour boundary, prove shortcut learning, or replace expert anatomical assessment. Live Grad-CAM and occlusion for other frozen pipelines in the application are **qualitative demonstrations**, not additional formally rated studies.

## Repository map: where to find everything

The following paths describe the **observed GitHub `main` tree**, not an idealized layout. All links are relative to this repository.

| Directory / file | What it contains and when to use it |
|---|---|
| [`app.py`](app.py) | **Current final seven-tab Streamlit dashboard**; the primary application entry point. |
| [`gui/`](gui/) | 15-pipeline registry, model reconstruction/loading, inference, Grad-CAM, SVM occlusion, and result presentation. |
| [`model_cnn_lstm.py`](model_cnn_lstm.py) | Generic five-backbone CNN–LSTM architecture; [`model_resnet152_lstm.py`](model_resnet152_lstm.py) is the earlier ResNet-specific proof of concept. |
| [`src/`](src/) | Dataset inspection and fixed split, training/evaluation transforms, `create_dataloaders` and support functions. |
| [`data/splits/`](data/splits/) | **Tracked** seed-42 train/validation index arrays. Development MRI files are **not** tracked under `data/brain_tumor_mri/`. |
| [`actual_training_scripts/`](actual_training_scripts/) | Phase 1 trainer/evaluator; 30-epoch and hyperparameter workers; neural-head and downstream comparisons; BDNeuro/web audits and external-evaluation utilities. |
| [`experiments/`](experiments/) | Five Phase 1 backbone directories with `results/*.json` (training, final SVM, neural-head metrics, etc.). Runtime checkpoint and some fitted artifacts may be absent from Git. |
| [`phase2/experiments_30epoch/`](phase2/experiments_30epoch/) | Phase 2A, five × 30-epoch training histories, both retained-state evaluations and downstream saved outputs. |
| [`phase2/hyperparameter_dl/`](phase2/hyperparameter_dl/) | **35 Phase 2B configurations**, each with training results and two downstream SVM-validation result sets. |
| [`phase2/results/leakage_corrected_30epoch/`](phase2/results/leakage_corrected_30epoch/) | Phase 2A post-hoc exact/material/conservative clean policy comparisons. |
| [`final_project_docs/duplicate_leakage_audit/`](final_project_docs/duplicate_leakage_audit/) | Cross-split duplicate audit JSON/CSV and human-review candidate sheets. |
| [`final_project_docs/leakage_corrected_evaluation/`](final_project_docs/leakage_corrected_evaluation/) | Phase 1 post-hoc leakage-aware results (`material_clean`, `exact_clean`, `conservative_clean`). |
| [`phase2/external_validation/`](phase2/external_validation/) | **Frozen 15** selection/hash manifest, BDNeuro overlap audit, exact-clean cohort manifests, balanced subset, and three complete external-comparison folders. |
| [`external_data/`](external_data/) | Checked-in BDNeuro V8 raw, exact-clean and balanced image copies, plus the Web-40 images. **See data provenance/redistribution note below.** |
| [`gradcam_evaluation/`](gradcam_evaluation/) / [`occlusion_evaluation/`](occlusion_evaluation/) | Locked formal Phase 1 ResNet152 case-rating CSVs; [`gradcam_evaluation_app.py`](gradcam_evaluation_app.py) and [`occlusion_evaluation_app.py`](occlusion_evaluation_app.py) are study tools. |
| [`results/`](results/) | Summary comparisons across neural head and PCA/SVM outputs. |
| [`final_deliverables/`](final_deliverables/) | Earlier final report (PDF/DOCX/MD), presentation, evidence map and QC report. **The subsequently revised 60-page report supplied for this GitHub edition is a separate later source.** |
| [`docs/`](docs/) | Historical transfer packages, project decision history, analysis PDFs and supporting source/reference materials. New public-facing documentation is placed here. |
| [`phase2_full_analysis_package/`](phase2_full_analysis_package/), [`phase2a_baseline_results/`](phase2a_baseline_results/), [`phase2b_analysis_package/`](phase2b_analysis_package/) | Earlier analysis/consolidation snapshots; use the canonical `phase2/` tree for run-level evidence. |
| [`final_project_qc.py`](final_project_qc.py) | Primary cross-split data-integrity audit program. |
| [`requirements-gui.txt`](requirements-gui.txt) | GUI dependencies **only**. Does not install the CUDA-specific PyTorch toolchain. |
| [`verify_gui_setup.py`](verify_gui_setup.py) | **Legacy Phase 1-only** model-artifact preflight; not a complete check of the current 15-pipeline dashboard. |
| [`README_GUI.md`](README_GUI.md) | Historical early four-tab GUI instructions; **superseded by this README and `app.py`**. |

### Key evidence files, directly linked

- [Frozen 15 selection](phase2/external_validation/frozen_15_selection.csv) and [SHA-256 manifest](phase2/external_validation/frozen_15_manifest.json).
- [BDNeuro cross-dataset audit](phase2/external_validation/bdneuro_overlap_audit/summary.json), [exact-clean cohort summary](phase2/external_validation/bdneuro_exact_clean_cohort/cohort_summary.json), [balanced-subset summary](phase2/external_validation/bdneuro_balanced_subset/balanced_subset_summary.json).
- External metrics: [BDNeuro full](phase2/external_validation/comparison_15_models_bdneuro_full/external_comparison_15.csv), [BDNeuro balanced](phase2/external_validation/comparison_15_models_bdneuro_balanced/external_comparison_15.csv), [Web-40](phase2/external_validation/comparison_15_models_web_stress_40/external_comparison_15.csv).
- [Web-40 pre-inference lock record](phase2/external_validation/web_stress_40_locked/lock_record.json); [source-provenance workbook](docs/DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx).
- [Duplicate audit](final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json), [Phase 1 `material_clean` summary](final_project_docs/leakage_corrected_evaluation/leakage_corrected_summary.csv), [Phase 2A primary comparison](phase2/results/leakage_corrected_30epoch/PHASE2A_PRIMARY_material_clean_comparison.csv).
- [Formal Grad-CAM ratings](gradcam_evaluation/gradcam_evaluation_resnet152_seed42.csv) and [formal occlusion ratings](occlusion_evaluation/occlusion_comparison_resnet152_seed42.csv).
- [Previous deliverables' detailed evidence map](final_deliverables/FINAL_PROJECT_EVIDENCE_MAP.md) and [QC report](final_deliverables/FINAL_PROJECT_QC_REPORT.md): the **174/174 checks and 75/75 hashes were reported during original local preparation**, and cannot be reverified from GitHub's current checkpoint-free checkout.

## Running the final application

**A fresh Git checkout contains no tracked `.pth` neural weights, but the authentic saved bundles are now publicly available as separate GitHub Releases.** Download the **15 frozen five-part pipelines** from [`models-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0); the optional [`research-checkpoints-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/research-checkpoints-v1.0) preserves 71 additional historical neural states. Follow [`docs/MODEL_WEIGHTS.md`](docs/MODEL_WEIGHTS.md) to restore each archive's project-relative paths and check all **75 frozen constituent hashes** against the published manifest. Results tabs can be read without these downloads, but **live prediction and explainability need the exact matching checkpoint/scaler/PCA/SVM**. Do **not** interchange phases or epochs. A fresh-clone restore/inference replay is not yet independently certified.

```bash
# On a machine with Git and Python 3.12; Ubuntu / WSL2 matches the recorded setup.
git clone https://github.com/bakayaro05/Tumor-Classification-DL-.git
cd Tumor-Classification-DL-

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Install a compatible torch + torchvision build separately for your machine.
# For GPU setups, use the official PyTorch selector: https://pytorch.org/get-started/locally/
python -m pip install -r requirements-gui.txt

# BEFORE prediction: restore `models-v1.0` from the GitHub Release and
# verify 75 constituent SHA-256 hashes per docs/MODEL_WEIGHTS.md.
streamlit run app.py
```

Streamlit typically prints a local address such as `http://localhost:8501`. The development dataset is also needed to rerun train/test analyses; it is **not** present under `data/brain_tumor_mri/` in the public Git tree. To re-create the saved split faithfully, use the same Nickparvar version and folder ordering expected by [`src/data_pipeline_torch.py`](src/data_pipeline_torch.py), with the tracked `data/splits/*.npy` indices. Do not assume similarly named but differently ordered dataset copies map to the same indices.

The **seven current tabs**, in order, are **Predict MRI · Base-paper reproduction · Model card · Hyperparameter sensitivity · External evaluation · XAI study · Frozen 15 comparison**. On the first tab a stage and backbone select **one** frozen candidate. Uploads support ordinary PNG/JPG/BMP/TIFF; **DICOM and 3-D volume inference are not implemented**. `verify_gui_setup.py` is historical and may report only Phase 1 readiness; use the current dashboard's pipeline-specific artifact checks for the other stages.

### Reproducing the research without retraining

A reader can inspect the **committed JSON/CSV evidence directly**, including all five Phase 1 experiments, Phase 2A and 2B training histories, freeze manifest, both BDNeuro cohorts, Web-40 results and formal XAI ratings. The repository's Git tree intentionally omits `.pth` files; the original selected model artifacts can now be downloaded from [`models-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) and compared to the frozen manifest using the [release restoration guide](docs/MODEL_WEIGHTS.md). Access to the archives is not equivalent to a verified clean-clone performance reproduction. Original machine environment and preprocessing choices are documented in [`final_project_docs/environment_pip_freeze.txt`](final_project_docs/environment_pip_freeze.txt) and the [technical documentation](docs/REPOSITORY_DOCUMENTATION.md). Checkpoint loading should use only trusted model artifacts; untrusted PyTorch/joblib files can execute code during deserialization.

## Datasets, attribution and publication considerations

1. **Development:** M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. The **7,023-image version** and the original 5,712/1,311 supplied folders define the study; downloaded versions with different file counts need an explicit compatibility audit.
2. **Structured external:** M. I. K. Hira *et al.*, *BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification*, **Mendeley Data, Version 8 (8 September 2026)**, DOI **10.17632/zwr4ntf94j.8**, https://data.mendeley.com/datasets/zwr4ntf94j/8. The source currently lists **CC BY 4.0**, but the repository's **distributed V8 internal README contains blank attribution/licence placeholders**. The published dataset record, not those placeholders, is the appropriate citation; separately review the rights and source-image provenance before redistributing MRI files.
3. **Web-40:** Not a single independently curated public dataset. It comprises **40 source-verified images** selected from individual publication/reference pages. The **individual URLs and figure/panel provenance** are recorded in the [40-row source-list workbook](docs/DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx). An externally sourced figure may have **different redistribution rights** from the paper or page where it was found.

**Data-integrity distinction:** BDNeuro V8 describes de-duplication *inside BDNeuro*. This project found **3,297 exact decoded-pixel overlaps with Nickparvar**, a separate cross-dataset question. Removing these overlaps reduced 5,941 images to the **2,644-image primary external cohort**. The two statements are not contradictory because they test different relationships. Exact-image cleaning does **not** establish patient-level independence.

The current public Git tree already tracks substantial MRI imagery, including BDNeuro-derived copies and Web-40 panels. **Public availability is not itself evidence that every redistribution is authorized**; maintain source attribution, inspect dataset/version licences and web-image permissions, and consider linking to original dataset downloads rather than mirroring them. The project does not claim clinical approval or research ethics clearance not documented in the source record.

## Limitations that should travel with every result

- The original development **train/validation/test partitions overlap in content**; the material-clean subset is post-hoc, and no group-aware retraining or patient-level split was performed.
- There was **one fixed seed (42), one development split**, no repeated-seed uncertainty analysis, no confidence intervals and no statistical claim that narrow model differences are significant.
- **VGG was frozen; ResNet/EfficientNet fine-tuned**. Phase 1 epoch budgets were unequal. Five backbone outcomes are not a pure backbone ablation.
- The LSTM models spatial positions, **not longitudinal/temporal MRI sequences**. No separate no-LSTM or no-PCA ablation establishes the individual causal contribution of these components.
- **The balanced BDNeuro set is nested**, and Web-40 is a **small exploratory convenience sample**; it is not a general clinical accuracy estimate.
- **The final representative was named after seeing structured external evidence.** No model weights were changed; selection optimism remains relevant to its reported external performance.
- Grad-CAM targets the neural head; occlusion probes the final SVM. Neither method provides a segmentation mask or definitive anatomical validation.
- The GUI supports **single-image raster inference**, not DICOM, patient-level diagnosis, or calibrated probabilities; **false no-tumour predictions under domain shift** rule out clinical-use claims.

## Reading order and research references

For a quick tour, start with **this README**. For precise directory layouts, script roles, experiment-level paths, dataset constraints, implementation/debugging history and execution notes, read [the repository documentation](docs/REPOSITORY_DOCUMENTATION.md). For the full methods, result tables, plots, GUI screenshots and technical appendices, read [the GitHub edition of the revised research report](docs/RESEARCH_REPORT_GITHUB_EDITION.pdf). For sourced dataset identifiers, citations and licenses, read [the reference guide](docs/DATASETS_AND_REFERENCES.md).

**Base paper:** M. Bérété, A. Echtioui, L. Sellami and A. A. Ben Hmida, “Hybrid CNN + LSTM + SVM model for advanced brain tumor classification,” *Network Modeling Analysis in Health Informatics and Bioinformatics*, vol. 15, article 43 (2026). https://doi.org/10.1007/s13721-025-00718-8. Its reported scores and experiment settings are a **reference**, not figures measured by this project. The complete bibliography remains in the research report; dataset and direct-result citations are linked throughout this README.

**Status:** The research phases and final dashboard are complete as documented. This public repository is a **research evidence and code archive**; Git-tracked source alone does not provide `.pth` weights, but **15 frozen inference bundles are now downloadable from GitHub Releases** and must be restored and verified before inference. Any future retraining/tuning motivated by the published external results requires **new untouched evaluation data**.
