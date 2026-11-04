# Rejected and Obsolete Approaches

> **Hostile-review classification warning:** Rao/Karunakara, Saad et al., and Malathi/Sinthia were superseded classical directions, not experimentally disproven. Adam/SGD/AdaGrad branches were unexecuted, not rejected. DenseNet121 is unresolved: one narrative claims evaluation, but supplied artifacts do not substantiate it.

## REJECTED — Benign/malignant relabeling from tumor type

Do not map glioma to malignant and meningioma/pituitary to benign. The dataset labels tumor categories, not grade or biological behavior. This would be medically invalid.

## REJECTED AS OFFICIAL BASE — 2022 classical K-means/DWT/SVM paper

Rejected because it did not meet the 2023-or-later paper requirement. It remains useful only as historical background.

## PROPOSED BUT NEVER TESTED — 2023 Otsu/GLCM/DWT/SVM implementation

Considered as a recent classical baseline but not implemented.

## TESTED AND FAILED — TensorFlow GPU implementation

Abandoned after PTX/`sm_120` failures. The failure was environmental/toolchain-related, not evidence that TensorFlow is methodologically inferior.

## TESTED AND FAILED — Fully unfrozen VGG16 configuration

Collapsed around 27.9% validation accuracy. Replaced by a frozen-backbone transfer-learning configuration. Full unfreezing should not be silently restored without a declared tuning study.

## IMPLEMENTED THEN SUPERSEDED — Root-level ResNet152 pipeline

The first working pipeline generated `models/best_resnet152_lstm.pth`, root `features/`, `pca_features/`, and a root SVM. It was superseded by standardized `experiments/resnet152/` artifacts. It should be preserved as history but not loaded by the current GUI.

## IMPLEMENTED THEN SUPERSEDED — Original image-level saved split

The 4,569/1,143/1,311 split was reproducible and used for all historical comparisons. It is now scientifically obsolete because of material cross-split duplication.

## REJECTED INTERPRETATION — LSTM learns MRI temporal information

The model processes one 2D image. The LSTM receives flattened spatial feature locations. Do not claim longitudinal, temporal, or slice-sequence modeling.

## REJECTED INTERPRETATION — Five backbones form an ensemble

They are separately trained alternatives. No voting, stacking, or feature fusion across backbones was implemented.

## REJECTED INTERPRETATION — Grad-CAM explains the final SVM

The implemented Grad-CAM explains the neural head. Final-SVM occlusion is a separate perturbation method. Never merge these claims.

## REJECTED INTERPRETATION — Heatmap equals tumor segmentation

Neither Grad-CAM nor occlusion provides a validated boundary. The maps are coarse contribution/sensitivity visualizations.

## REJECTED CLAIM — Historical test metrics prove generalization

The leakage audit disproved split independence. Preserve the numbers as outputs, not as clean generalization evidence.

## PROPOSED BUT NOT EXECUTED

- attention-enhanced LSTM;
- true multi-backbone ensemble;
- CNN-versus-DWT controlled experiment;
- benign/malignant classification with a suitable dataset;
- DICOM support;
- external hospital/dataset validation;
- model compression/lightweight deployment;
- formal radiologist localization study;
- patient-level split;
- multi-seed statistical comparison.

## NOT KNOWN TO HAVE BEEN REMOVED

No reliable evidence shows that major implemented architecture components were later deleted, apart from superseding legacy artifact paths/configurations. Do not claim code removal without repository history.
