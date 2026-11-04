# PROBLEMS, DEBUGGING, REJECTED APPROACHES, AND LESSONS

This document intentionally preserves failures and discarded ideas so a future model does not repeat them.

## 1. Early classical pipeline was superseded
Initial plan: K-means segmentation + handcrafted features + RBF-SVM.  
Why superseded: recent-paper requirement and move to deep hybrid architecture.  
Lesson: do not merge the obsolete segmentation pipeline into the final CNN-LSTM methodology.

## 2. Benign/malignant relabeling rejected
Tumor class labels do not justify benign/malignant inference.  
Lesson: preserve dataset labels exactly.

## 3. Five backbones are not an ensemble
The architecture compares five independent pipelines.  
Lesson: do not write “ensemble” unless a future project actually combines predictions.

## 4. TensorFlow/PTX failure
Problem:
- GPU could be detected;
- simple TensorFlow operations could run;
- real workload failed with `INVALID_PTX` / unsupported `sm_120`.

Attempts:
- standard TensorFlow CUDA route;
- nightly TensorFlow.

Fix:
- migrate to PyTorch 2.11.0+cu128.

Lesson:
- hardware detection is not proof that the full framework/toolchain is compatible.

## 5. LSTM conceptual ambiguity
Potential misleading interpretation: LSTM models temporal MRI sequence.  
Actual implementation: flattened spatial positions of one 2D feature map.  
Lesson: be explicit in reports/viva.

## 6. BatchNorm single-sample issue
Training set 4,569 with batch8 leaves one final sample. `BatchNorm1d` cannot compute training statistics from batch size 1.  
Fix: training loader `drop_last=True`; shuffle means the same image is not permanently omitted every epoch.

## 7. Fully unfrozen VGG16 collapse
Symptoms:
- ~28% validation;
- learned features near constant;
- PCA NaNs.

Diagnostics:
- pretrained backbone activation itself healthy.

Decision:
- freeze VGG16 CNN;
- apply same policy to VGG19.

Lesson:
- backbone optimization policy can dominate transfer-learning stability.

## 8. Legacy versus standardized artifacts
First ResNet152 root-level pipeline proved feasibility but was later superseded by `experiments/resnet152/`.  
Lesson: do not mix root-level legacy scaler/PCA/SVM with standardized model checkpoints.

## 9. SVM probability terminology
Some historical run logs may suggest `probability=True`, while current code/GUI uses `decision_function` and may specify `probability=False`.  
Final reporting rule: shown SVM values are **relative scores, not calibrated probabilities**.  
Lesson: inspect actual joblib object before making artifact-specific setting claims.

## 10. Grad-CAM target mismatch
Grad-CAM explains the neural head, not the scikit-learn SVM.  
Fix: add explicit GUI/report distinction and final-SVM occlusion.

## 11. XAI off-target saliency
Observed:
- skull margin;
- face;
- image border;
- extracranial hotspots;
- constant/blank layer in one case.

Lesson:
- correct classification does not prove lesion-centered reasoning.

## 12. Possible dataset-label anomaly
`Te-no_0216.jpg` was labelled no-tumor yet visually contained a conspicuous rounded hyperintense region.  
Decision: flag for expert review; do not relabel from model/visual inspection alone.

## 13. Cross-split duplicate leakage
This was the largest methodological issue:
- 464 exact/pixel pairs
- 98 high-confidence near pairs
- 562 material pairs

Lesson:
- always audit public image datasets before treating random image-level splits as independent evidence.

## 14. Patient-level independence remains unknowable
No reliable patient identifiers.  
Lesson:
- even perfect image-level deduplication would not prove patient-level separation.

## 15. Immediate post-audit “retrain everything now” decision evolved
Older transfer docs treated group-aware retraining as blocking.  
Later project policy prioritized transparent caveats + external validation and moved full retraining to optional future strengthening.  
Lesson:
- preserve chronology; do not read historical blocking language as current status.

## 16. DenseNet121 contradiction
One conversation checkpoint mentioned DenseNet121. Verified final source/artifacts do not support it.  
Decision: exclude from final comparison unless artifacts are recovered.

## 17. External BDNeuro provenance conflict
Documented BDNeuro provenance differed from Nickparvar, yet 3,297 unique exact decoded-pixel matches were found.  
Decision: direct pixel identity outranks provenance narrative; do not speculate who copied from whom.

## 18. pHash over-removal risk
Thousands of BDNeuro pHash candidates existed.  
Decision: do not equate MRI perceptual similarity with duplicate identity.

Lesson:
- use pHash as screening, not an automatic exclusion rule.

## 19. External evaluator filename/output collision
Original `compare_external_15.py` used a fixed results directory. Full and balanced external runs needed separate output roots.  
Fix:
- revised evaluator with `--results-name`;
- there was a temporary copy error that produced a 0-byte script;
- `python -m py_compile` misleadingly returned success because an empty Python file is syntactically valid;
- repair verified by checking `--help | grep results-name`.

Lesson:
- syntax success alone does not prove expected functionality; verify CLI behavior.

## 20. Web pHash0 case
`NOR_01` had no exact pixel match but pHash distance0 to a Nickparvar image. Side-by-side review showed the same underlying MRI slice with altered representation.  
Action: reject and replace before inference.

Lesson:
- exact hash is objective, but visual/perceptual review can still identify transformed reuse.

## 21. MEN_02 pHash6 false positive
Images were clearly different (including orientation).  
Action: retain.

Lesson:
- pHash distance alone is not enough to exclude.

## 22. Do not tune after external/web evaluation
No model, crop or image was changed in response to observed external prediction performance.  
Lesson:
- otherwise evaluation data become tuning data.

## 23. Rejected or intentionally unexecuted additions
- complete 20-run optimizer grid;
- DenseNet sixth backbone;
- post-hoc ensemble;
- extra Grad-CAM layer tuning after protocol freeze;
- additional occlusion variants just to improve visual appearance;
- web-image replacement after failure;
- clinical/DICOM deployment;
- per-image consensus analysis of 600 web predictions (optional, skipped).

## 24. Central lesson of the project
The most important debugging was not a code bug. It was a **scientific validity bug**: high accuracy on a public split did not guarantee independent generalization. The project became stronger when it treated data integrity and domain shift as first-class experiments.
