# LIMITATIONS, CLAIM RULES, AND FUTURE WORK

## 1. Mandatory limitations

### Original split contamination
The development split contains exact and near-duplicate cross-split pairs.

Required wording:
- “historical contaminated-split result”
- not “independent internal test performance”

### material_clean is post-hoc
It removes directly implicated test images only.

Required wording:
- “post-hoc leakage-aware subset”
- not “new independent test set”

### Patient-level independence unknown
No reliable patient IDs.

Never claim:
- patient-independent split;
- no patient leakage.

### Single seed
Phase2A/2B use seed42 only.

Never claim:
- statistically significant hyperparameter superiority.

### VGG training asymmetry
VGG backbones frozen; residual/EfficientNet fine-tuned.

### Limited hyperparameter coverage
Only LR, LSTM width and dropout were varied one factor at a time.

### Explainability qualitative
Grad-CAM and occlusion are coarse contribution/perturbation maps.

Never claim:
- segmentation;
- precise tumor boundary;
- clinically valid localization;
- causal medical reasoning.

### External BDNeuro limits
Exact image overlap was removed, but public metadata cannot establish patient-level independence.

### Web stress limits
n=40 convenience sample, source-verified but heterogeneous.

Never claim:
- clinical accuracy;
- population prevalence estimate;
- general medical-device performance.

### Prototype scope
Raster images only; no DICOM pipeline; no prospective clinical validation.

---

## 2. Claims that are supported

Supported:
- five-backbone hybrid implementation completed;
- neural-head and downstream hybrid comparison completed;
- original split has material image-level duplication;
- Phase2 learning rate was the strongest tested sensitivity;
- LSTM512 produced small peak-neural gains, not universal external gains;
- Phase2A generalized better on average than Phase2B on BDNeuro and web stress;
- ResNet152 Phase2A gave the strongest structured external result;
- web-domain shift produced severe performance degradation and no-tumor bias;
- XAI often highlighted non-lesion regions;
- system is a research/educational prototype.

---

## 3. Claims that are not supported

Do not say:
- “99% clinically accurate.”
- “No leakage.”
- “Patient-independent.”
- “PCA alone caused the gain.”
- “LSTM512 is proven best.”
- “Dropout0.2 is the best universal setting.”
- “Grad-CAM shows the tumor.”
- “SVM scores are probabilities.”
- “BDNeuro raw 5,941 is independent.”
- “The 40 web images prove clinical failure.”
- “The final model is safe for diagnosis.”
- “DenseNet121 was part of the verified final comparison.”

---

## 4. Correct terminology

Use:
- **neural-head vs PCA+SVM downstream-classifier comparison**
- **material_clean post-hoc test subset**
- **BDNeuro V8 cross-dataset exact-overlap-clean external cohort**
- **balanced sensitivity subset**
- **source-verified in-the-wild web MRI stress test**
- **coarse contribution map**
- **research/educational prototype**
- **final project representative / selected candidate** for ResNet152 Phase2A

---

## 5. Optional future work

### Highest-value scientific extension
Duplicate-group-aware re-splitting/retraining:
1. build exact/near-duplicate groups;
2. assign whole groups to one partition;
3. retrain all five from scratch;
4. refit scaler/PCA/SVM on the new training partition;
5. repeat internal evaluation.

This is needed only if the project wants a stronger **clean internal-generalization** claim.

### Other valid future work
- patient-ID-aware datasets;
- multi-site prospective validation;
- repeated seeds/confidence intervals;
- calibration analysis;
- sequence-aware 3D/volumetric MRI modeling;
- explicit lesion segmentation when masks exist;
- domain adaptation / stain-style normalization equivalents for MRI intensity/scanner shifts;
- DICOM ingestion;
- prospective clinical usability study.

### Not recommended as immediate “future work” merely to chase numbers
- more random backbones;
- post-hoc external tuning;
- replacing failed web images;
- extra saliency variants selected by visual attractiveness;
- arbitrary ensembles without a preregistered evaluation plan.

---

## 6. What remains for the user operationally

The research experimentation is complete. Remaining actions are administrative:
- copy final package into Claude Project;
- update final submission PPT/report with the latest external/web findings;
- rehearse viva narrative;
- archive the local repository and exact model artifacts.

No additional experiment is required by the original PPT roadmap.
