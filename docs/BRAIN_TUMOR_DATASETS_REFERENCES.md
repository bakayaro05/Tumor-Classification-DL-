# Datasets, scholarly references, provenance and licensing

**Project:** Hybrid CNN–LSTM–PCA–SVM for Four-Class Brain Tumor MRI Classification  
**Context:** University college mini-project, 2026  
**Evidence scope:** Final revised 60-page research report, the published repository's audit/result artifacts, and independently checked publisher/dataset landing pages (checked 4 November 2026).  
**No clinical validation is claimed.**

## D1. Nickparvar: development and original internal test set

**Official landing page:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  
**Credit:** Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle.  
**Study copy:** 7,023 raster brain MRI images, with the supplied `Training/` folder holding **5,712** images and the supplied `Testing/` folder **1,311**. The project further divided the Training folder into **4,569 train + 1,143 validation** using stratification, seed 42, and committed fixed index arrays.

| Class | Supplied Training | Supplied Testing | Total |
|---|---:|---:|---:|
| glioma | 1,321 | 300 | 1,621 |
| meningioma | 1,339 | 306 | 1,645 |
| notumor | 1,595 | 405 | 2,000 |
| pituitary | 1,457 | 300 | 1,757 |
| **Total** | **5,712** | **1,311** | **7,023** |

The original split **is not image-level independent**: the project found **562 material cross-split pairs** involving **250 supplied-test images**, using file/pixel hashes and validated high-confidence near-duplicate criteria. This is a property observed **in the project's version/split** and is not a blanket claim about every public release or alternate repackaging of the Kaggle dataset.

**Reproduction caution:** Download the version with the same 7,023 images and original folder structure. If file sets or image order differ, the saved `data/splits/train_indices.npy` and `val_indices.npy` cannot safely be assumed to reference the same images. No raw Nickparvar images are present under the repository's normally expected `data/brain_tumor_mri/` path.

**Evidence:** [`phase2/external_validation/bdneuro_overlap_audit/dataset_counts.json`](../phase2/external_validation/bdneuro_overlap_audit/dataset_counts.json); [`final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json`](../final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json); [`data/splits/`](../data/splits/).

**License note:** Consult the Kaggle dataset's current license/terms before redistribution. This document does **not** infer a right to mirror every MRI file from the fact that its URL is public.

## D2. BDNeuro-MRI: primary structured external validation

**Exact study release:** **BDNeuro-MRI Version 8**, published **8 September 2026**, not later Version 9.  
**Official dataset page:** https://data.mendeley.com/datasets/zwr4ntf94j/8  
**Version-specific DOI:** https://doi.org/10.17632/zwr4ntf94j.8  
**Dataset title:** *BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification*.  
**Contributors listed by publisher:** MD Irfanul Kabir Hira; MD Sohag Hossain; Mst Moriom Akter Bithee; Umme Sara Sara; MD Mahmudul Hasan; Abdullah Al Towsif; Md Kowsar Ahmed.  
**Publisher-listed license (V8):** Creative Commons Attribution 4.0 (**CC BY 4.0**). **The committed dataset's internal README still has empty license/attribution placeholders; use the authoritative Mendeley landing page for the version citation, and verify actual upstream source-image permissions before republication.**

**Recommended dataset citation:**

Hira, M. I. K., Hossain, M. S., Bithee, M. M. A., Sara, U. S., Hasan, M. M., Al Towsif, A., & Ahmed, M. K. (2026). *BDNeuro-MRI: A Bangladeshi clinical brain tumor MRI dataset for four-class deep learning classification* (Version 8) [Data set]. Mendeley Data. https://doi.org/10.17632/zwr4ntf94j.8

The source describes a 5,941-image, four-class collection with internal train 4,160 / validation 892 / test 889. **This project's own cross-dataset audit discovered 3,297 unique BDNeuro images with exact decoded-pixel overlap with the 7,023-image Nickparvar development collection**, despite BDNeuro's within-dataset de-duplication claim. The external cleaning rule was to remove only decoded-pixel-identical *cross-dataset* images, not every perceptually similar MRI (which could represent distinct anatomical slices). This produced the **2,644-image primary external cohort**, classes 1,159 glioma / 472 meningioma / 412 notumor / 601 pituitary. A **1,648-image balanced subset** (412/class) was drawn deterministically **from that same 2,644-image cohort**, not from an additional independent source.

**Evidence:** [`bdneuro_overlap_audit/summary.json`](../phase2/external_validation/bdneuro_overlap_audit/summary.json); [`bdneuro_exact_clean_cohort/cohort_summary.json`](../phase2/external_validation/bdneuro_exact_clean_cohort/cohort_summary.json); [`bdneuro_balanced_subset/balanced_subset_summary.json`](../phase2/external_validation/bdneuro_balanced_subset/balanced_subset_summary.json); [`comparison_15_models_bdneuro_full/external_comparison_15.csv`](../phase2/external_validation/comparison_15_models_bdneuro_full/external_comparison_15.csv).

**Crucial distinction:** The external cohort is *exact-overlap-clean relative to Nickparvar*, **not proven patient-independent**. Its publisher and the project also do not demonstrate that images from one patient could not be in both sources. The source's described provenance is reproduced as a dataset claim rather than independently corroborated clinical provenance.

## D3. Web-40: source-verified exploratory stress test

**Definition:** Exactly 40 selected 2-D MRI panels (10 per class), from **multiple external medical publications and reference pages**. This is **not** an officially distributed benchmark with one citation or a statistical sample of clinical use.

**Per-image primary citations and figure/panel provenance:** [`Web_Stress_Test_40_Source_List.xlsx`](../docs/DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx). Each record contains the source page URL and additional identity/diagnosis metadata. Cite the individual source page for any reproduced case image, not merely this project's workbook.

**Protocol:** source diagnosis and image panel verified; exclude CT/histology/segmentation and unsupported diagnoses; audit candidate set against the Nickparvar images, BDNeuro exact-clean images and internally; visually reject transformed reuse (`NOR_01`) and retain genuinely different `MEN_02`, both before model inference; lock 40-image fingerprint; evaluate all 15 already-frozen pipelines (**600 predictions**). The lock record was created before the first web predictions. The test is for **failure modes under severe presentation shift**, not clinical accuracy estimation. Note that every image equals **2.5 percentage points** of overall accuracy.

**Evidence:** [`web_stress_40_locked/lock_record.json`](../phase2/external_validation/web_stress_40_locked/lock_record.json); [`web_stress_40_audit/summary.json`](../phase2/external_validation/web_stress_40_audit/summary.json); [`comparison_15_models_web_stress_40/external_comparison_15.csv`](../phase2/external_validation/comparison_15_models_web_stress_40/external_comparison_15.csv).

**Rights caution:** A published open-access page, medical teaching site or article figure can impose different reuse conditions. Publication here requires a panel-by-panel license/permissions check, proper figure credit and any necessary removal/replacement of images where public redistribution is not authorized. It is unsafe to assign a single license to all Web-40 images.

## R1. The research paper implemented and investigated

Bérété, M., Echtioui, A., Sellami, L., & Ben Hmida, A. A. (2026). *Hybrid CNN + LSTM + SVM model for advanced brain tumor classification*. **Network Modeling Analysis in Health Informatics and Bioinformatics**, 15, article 43. https://doi.org/10.1007/s13721-025-00718-8

**Relationship to this project:** methodological starting point, **not** co-authorship, not an exact TensorFlow reproduction, and not the source of the project's Phase 2 integrity, external, web-stress or interpretability experiments. The project used PyTorch, RMSprop only, batch 8 rather than the paper's 32, and a frozen-VGG adaptation. The paper's performance metrics are not substituted for the project's verified outputs.

## Other foundational methods used

- Simonyan, K., & Zisserman, A. (2015). *Very deep convolutional networks for large-scale image recognition*. ICLR. https://arxiv.org/abs/1409.1556.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR. https://doi.org/10.1109/CVPR.2016.90.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. ICML. https://proceedings.mlr.press/v97/tan19a.html.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. **Neural Computation**, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.
- Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. **Machine Learning**, 20, 273–297. https://doi.org/10.1007/BF00994018.
- Selvaraju, R. R., et al. (2020). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. **International Journal of Computer Vision**, 128, 336–359. https://doi.org/10.1007/s11263-019-01228-7.
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and understanding convolutional networks*. ECCV. https://doi.org/10.1007/978-3-319-10590-1_53.
- Paszke, A., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library*. NeurIPS. https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. **JMLR**, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html.
- Streamlit: https://streamlit.io/; PyTorch installation documentation: https://pytorch.org/get-started/locally/.

For the complete literature survey and all numbered references, consult the [GitHub-edition full research report](RESEARCH_REPORT_GITHUB_EDITION.pdf) — **References**, originally research pages 37–38.

## Suggested BibTeX

```bibtex
@misc{nickparvar_brain_mri,
  author = {Nickparvar, Masoud},
  title  = {Brain Tumor MRI Dataset},
  howpublished = {Kaggle},
  url    = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset},
  note   = {Study used the 7,023-image release}
}

@dataset{hira_bdneuro_2026_v8,
  author = {Hira, MD Irfanul Kabir and Hossain, MD Sohag and Bithee, Mst Moriom Akter and Sara, Umme Sara and Hasan, MD Mahmudul and Towsif, Abdullah Al and Ahmed, Md Kowsar},
  title  = {BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification},
  year   = {2026},
  version = {8},
  publisher = {Mendeley Data},
  doi = {10.17632/zwr4ntf94j.8},
  url = {https://data.mendeley.com/datasets/zwr4ntf94j/8}
}

@article{berete2026hybrid,
  author = {Bérété, Moriba and Echtioui, Amira and Sellami, Lamia and Ben Hmida, Ahmed Almokhtar},
  title  = {Hybrid CNN + LSTM + SVM model for advanced brain tumor classification},
  journal= {Network Modeling Analysis in Health Informatics and Bioinformatics},
  year   = {2026},
  volume = {15},
  number = {43},
  doi    = {10.1007/s13721-025-00718-8}
}
```

**Citation scope:** Dataset contributor names and base-paper author names are necessary scholarly attribution; **project-member and project-guide names are deliberately omitted** from this public-facing documentation.
