# Problems, Debugging, and Lessons

## Hostile-review corrections

- VGG learned features became effectively constant and PCA emitted NaNs, while untouched pretrained activations were healthy; root cause remains unproven.
- Current source says SVM `probability=False`, while preserved VGG19/ResNet152 logs suggest some runtime models used `True`. Inspect serialized artifacts.
- Occlusion maps were independently normalized and came from a deliberately selected subset; color is not comparable effect size or prevalence.
- Saved epochs are zero-based, displayed epochs one-based.
- DenseNet121 is an unresolved documentation/artifact contradiction.

## TensorFlow/PTX incompatibility

**Problem:** TensorFlow could see the new NVIDIA GPU but failed under real execution.  
**Symptom:** `INVALID_PTX`; PTX toolchain did not support target `sm_120`.  
**Investigation:** GPU enumeration and simple matrix multiplication; TensorFlow and nightly CUDA packaging.  
**Root cause:** GPU architecture/toolchain incompatibility in the tested TensorFlow stack.  
**Solution:** Migrate to PyTorch 2.11.0+cu128/torchvision 0.26.0+cu128 under WSL2.  
**Lesson:** A successful device listing or trivial kernel is not proof that the full training stack is compatible.

## LSTM conceptual ambiguity

**Problem:** “LSTM for MRI” can be incorrectly explained as temporal or slice-sequence modeling.  
**Investigation:** Explicit tensor-shape tests.  
**Root cause:** The paper-style hybrid description hides the constructed sequence.  
**Solution:** Define timesteps as flattened spatial feature-map locations: `[B,C,H,W]→[B,H×W,C]`.  
**Lesson:** Report what the tensor actually represents; do not borrow temporal language from standard LSTM use cases.

## VGG16 training collapse

**Problem:** Fully unfrozen VGG16 did not learn stably.  
**Symptom:** Validation accuracy around 27.9%.  
**Investigation:** Dedicated diagnostic script examined data/model behavior.  
**Root cause:** Not conclusively isolated in the available record; likely optimization/transfer instability rather than dataset class mapping, but this remains an inference.  
**Solution:** Freeze the pretrained VGG convolutional backbone and train the LSTM/post-LSTM/head layers.  
**Result:** Historical test accuracy rose to 94.97%.  
**Lesson:** Fine-tuning policies need not transfer uniformly across architectures; disclose unequal policies in comparisons.

## Deterministic feature extraction

**Problem:** Using stochastic training transforms while extracting features would make PCA/SVM inputs unstable and potentially duplicate augmented variants unpredictably.  
**Solution:** Reconstruct train/validation subsets using the saved indices but apply `eval_transform`; no shuffling during feature extraction.  
**Lesson:** Separate neural training augmentation from frozen downstream-feature generation.

## Legacy versus standardized artifacts

**Problem:** Early ResNet152 files lived at root-level paths, while later five-backbone files lived under `experiments/`. SVM probability settings also differed.  
**Solution:** GUI uses only standardized experiment artifacts.  
**Lesson:** Artifact provenance must include backbone, checkpoint, preprocessing, class order, and pipeline version.

## Misleading “probability” language

**Problem:** Current standardized source specifies `probability=False`, historical log evidence conflicts, and UI users expect percentages.  
**Solution:** Convert decision values only for a relative visualization and explicitly say they are not calibrated probabilities.  
**Lesson:** A softmax-shaped display does not convert margins into clinical probability.

## Grad-CAM target mismatch

**Problem:** Standard Grad-CAM requires gradients, while the final classifier is an external scikit-learn SVM after scaler/PCA.  
**Solution:** Grad-CAM targets the neural classifier head; GUI clearly discloses this. Add occlusion to probe the final SVM.  
**Lesson:** Explainability must match the claimed prediction target.

## Constant/off-target saliency

**Problem:** Some maps were constant, peripheral, or extracranial.  
**Symptom:** Skull-margin/facial hotspots and one blank constant map.  
**Solution:** Display a constant map as blank, compare layer 3/layer 4/fused maps, and evaluate a fixed sample.  
**Lesson:** Saliency is an audit tool, not automatic validation. Higher spatial resolution did not guarantee better lesion focus.

## Possible dataset-label anomaly

**Problem:** A reviewed `notumor` image (`Te-no_0216.jpg`) showed a conspicuous rounded hyperintense region.  
**Solution:** Flag for expert review without relabeling.  
**Lesson:** Dataset labels should not be treated as infallible; visual concern alone is insufficient for diagnosis.

## Cross-split duplicate leakage

**Problem:** The reproducible split was not independent.  
**Symptom:** Audit found 464 exact/pixel duplicate pairs plus 98 high-confidence near-duplicate pairs across splits.  
**Root cause:** Dataset duplication combined with individual-image splitting/original folder assumptions. Exact provenance of every duplicate is not yet established.  
**Solution required:** Connected-component duplicate grouping, group-aware split, zero-cross-group re-audit, and complete retraining/refitting.  
**Lesson:** Reproducibility is not the same as validity. A fixed contaminated split reliably reproduces biased estimates.

## Patient-level leakage remains unresolved

**Problem:** Multiple different images may come from the same patient.  
**Root cause:** No reliable patient identifiers.  
**Solution:** None available from the current dataset. State the limitation and seek a dataset with patient/study IDs for stronger validation.  
**Lesson:** Passing image-level duplicate checks cannot prove patient-level independence.

## Stale manifest and incomplete source snapshot

**Problem:** `project_files.txt` includes the standardized experiment bundles but predates later GUI modules and audit outputs. The attached transfer snapshot also omits some imported files.  
**Solution:** Generate a new manifest from the final repository and verify that every import and artifact path resolves.  
**Lesson:** A handover must transfer the repository state, not only selected files or an older file listing.
