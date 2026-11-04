# Methodology Evolution

## Evolution map

Classical handcrafted-feature idea  
→ recent-paper requirement  
→ hybrid CNN+LSTM+SVM base paper  
→ TensorFlow prototype attempt  
→ PyTorch migration  
→ incremental ResNet152 implementation  
→ standardized five-backbone framework  
→ VGG freezing after collapse  
→ deterministic 128-D features + train-only scaler/PCA + linear SVM  
→ Streamlit GUI  
→ neural-head Grad-CAM  
→ final-SVM occlusion sensitivity  
→ duplicate/leakage audit  
→ required group-aware resplit and complete clean rerun.

## Initial methodology

From 16 July, the project compared four classical paper directions and formed an explicit pipeline: resize/grayscale, median filtering/contrast enhancement, skull removal, K-means, morphology, shape/GLCM features, standardization/PCA, and RBF-SVM. Its pedagogical value was that every stage was visible. The discussion also corrected two tempting errors: K-means cluster numbers have no fixed semantic meaning, and HGG/LGG cannot be inferred from these four class labels.

## Selection of the hybrid methodology

The recent-paper requirement and project interest shifted the target to CNN feature extraction plus LSTM plus SVM. Five CNNs became alternatives rather than an ensemble. The paper's nominal experiment was 20 combinations (five backbones × four optimizers), but the project executed the RMSprop branch only and used different effective batch/epoch settings. Novelty was deferred.

## Clarification of the sequence model

Early language risked suggesting temporal MRI analysis. Implementation clarified the exact transformation from a single image's spatial feature map to a sequence of spatial vectors. This is now the only correct description of the LSTM stage.

## Framework change

TensorFlow was not rejected scientifically; it was abandoned because the available GPU/toolchain produced PTX failures. The method moved to PyTorch without changing the research question.

## First working baseline

ResNet152 was built component by component and trained first. A temporary neural classifier allowed end-to-end feature learning; the 128-D representation was then standardized, reduced to PCA(64), and classified with a linear SVM.

## Comparative generalization

The ResNet152-specific code was generalized into `CNNLSTM`. Per-backbone experiment directories made checkpoints, features, PCA/SVM objects, and results consistent. The old root-level ResNet artifacts became legacy.

## VGG stabilization

The original intention was to fine-tune all CNNs. VGG16 collapsed near 27.9% validation accuracy. Freezing the VGG feature extractor restored useful performance, and VGG19 followed the same policy. This created a methodological asymmetry—frozen VGG versus fine-tuned ResNet/EfficientNet—that must be disclosed.

The debugging record is stronger than “low accuracy”: trained VGG features were effectively constant and PCA produced NaNs, whereas untouched pretrained activations were healthy. VGG16 was later resumed toward a 30-epoch target and peaked at stored epoch 16/displayed epoch 17. Stored epochs are zero-based; console/displayed epochs are one-based.

## Evaluation maturation

Evaluation expanded from validation accuracy to final test accuracy, macro/weighted metrics, per-class reports, confusion matrices, and training curves. However, ROC-AUC, specificity, confidence intervals, multiple seeds, and external validation were not established.

## Explainability extension

Grad-CAM was introduced after baseline completion. Technical discussion correctly separated neural-head Grad-CAM from the external SVM. Occlusion sensitivity was then added to probe the final SVM directly. The qualitative evaluations weakened any simplistic claim that the model “looks at the tumor”; many explanations were peripheral or extracranial.

## Integrity correction

The most important methodological revision came last. The original saved stratified split was reproducible but not independent. Duplicate auditing found 562 material cross-split pairs. Therefore, the method's next version must split duplicate-connected groups before any training. This is not a minor housekeeping adjustment; it changes the evidentiary status of every reported metric.

This explicitly supersedes a pre-audit checkpoint that called dataset-level classification accuracy valid and the interpretation locked. That statement was true only in chronology, not in the current scientific state.
