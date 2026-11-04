# Reproduce the Brain Tumor MRI Classification Research Project

> **University college mini-project | CNN–LSTM–PCA–SVM | PyTorch | Research only**  
> Repository: [Tumor-Classification-DL-](https://github.com/bakayaro05/Tumor-Classification-DL-)  
> **Scope:** Reproduce *this project's adaptation and evaluation* of the Bérété et al. hybrid CNN + LSTM + SVM research framework. This is **not** an exact, exhaustive reimplementation of every experiment in the original paper, nor a clinical diagnostic system.

**Base paper:** Bérété et al. (2026), *Hybrid CNN + LSTM + SVM model for advanced brain tumor classification*, [DOI: 10.1007/s13721-025-00718-8](https://doi.org/10.1007/s13721-025-00718-8).  
**Data:** [Nickparvar Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset); [BDNeuro-MRI Version 8 (Mendeley Data)](https://doi.org/10.17632/zwr4ntf94j.8).  
**Expanded methods/results:** [Technical documentation](REPOSITORY_DOCUMENTATION.md) · [Research report](RESEARCH_REPORT_GITHUB_EDITION.pdf) · [Dataset citations and provenance](DATASETS_AND_REFERENCES.md).
**Frozen model downloads:** [15 selected pipelines (`models-v1.0`)](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) · [71 additional checkpoints (`research-checkpoints-v1.0`)](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/research-checkpoints-v1.0) · [verified restoration procedure](MODEL_WEIGHTS.md).

## Contents

1. [What can and cannot be reproduced](#1-what-can-and-cannot-be-reproduced)
2. [Environment and checkout](#2-environment-and-checkout)
3. [Development MRI dataset and saved split](#3-development-mri-dataset-and-saved-split)
4. [Data-integrity audit: run before trusting scores](#4-data-integrity-audit-run-before-trusting-scores)
5. [Phase 1: five early-stopping baselines](#5-phase-1-five-early-stopping-baselines)
6. [Phase 2A: standardized 30-epoch experiments](#6-phase-2a-standardized-30-epoch-experiments)
7. [Phase 2B: 35 sensitivity runs and 70 evaluations](#7-phase-2b-35-sensitivity-runs-and-70-evaluations)
8. [Post-hoc material-clean evaluation](#8-post-hoc-material-clean-evaluation)
9. [Freeze 15 pipelines and validate externally](#9-freeze-15-pipelines-and-validate-externally)
10. [Web-40 stress test](#10-web-40-stress-test)
11. [Explainability and the Streamlit dashboard](#11-explainability-and-the-streamlit-dashboard)
12. [Evidence, expected results, and interpretation](#12-evidence-expected-results-and-interpretation)
13. [Troubleshooting and reproducibility limits](#13-troubleshooting-and-reproducibility-limits)
14. [Final verification checklist](#14-final-verification-checklist)

## 1. What can and cannot be reproduced

There are **three different tasks**. Do not confuse them:

| Task | Public GitHub checkout | What else is needed |
|---|---|---|
| **Audit the documented results** | Yes: tracked JSON/CSV, split indices, evaluation protocols, selected-pipeline records, and written report | Inspect the referenced machine-readable evidence; no GPU training required |
| **Train a new implementation of the five-backbone pipeline** | Source scripts are present | The matching 7,023-image development dataset, Python environment, computation, and fresh training; new numerical results may differ |
| **Replay the original 15 frozen models** | **Not from Git tree alone**: the `.pth` weights are provided separately in [`models-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) | Download/restore all three phase archives, verify all 75 hashes against the frozen manifest, install the compatible environment, and use the originally selected images and file ordering; [instructions](MODEL_WEIGHTS.md) |

The project **does** contain 15 recorded frozen pipeline selections (five CNN backbones × Phase 1/2A/2B), 35 Phase 2B training histories, 70 downstream validation evaluations, the external-evaluation tables, and 75 artifact hashes in the [freeze manifest](../phase2/external_validation/frozen_15_manifest.json). The historical local QC reported 75/75 hashes matching. The original 15 bundles are now downloadable from a public GitHub Release, so those hashes **can be independently rechecked after restoration**; such clean-clone verification is still pending, not asserted here.

**Run potentially writing scripts in a disposable clone**, not inside the copy containing the published experimental artifacts. Training and evaluation scripts reuse the canonical output folders and may overwrite or skip data. Do not erase the published [frozen manifest](../phase2/external_validation/frozen_15_manifest.json), results, or reports to make a rerun pass.

The original development split was later shown to contain material cross-partition duplicates. Repeating it is useful for *historical reproduction*, **not** for claiming leakage-free generalization. A genuinely new independent/patient-grouped study requires a redesigned split and retraining, and is not the experiment documented here.

## 2. Environment and checkout

### 2.1 Recorded environment

The completed project ran under **Ubuntu / WSL2**, **Python 3.12.14**, **PyTorch 2.11.0+cu128**, **torchvision 0.26.0+cu128**, an **NVIDIA RTX 5070 Laptop GPU (8 GB VRAM)**, and the package versions recorded in [`final_project_docs/environment_pip_freeze.txt`](../final_project_docs/environment_pip_freeze.txt). Essential versions from that record:

| Package | Recorded version |
|---|---|
| `numpy` | `2.5.2` |
| `pandas` | `2.3.3` |
| `scipy` | `1.18.1` |
| `scikit-learn` | `1.9.0` |
| `joblib` | `1.5.3` |
| `Pillow` | `12.3.0` |
| `matplotlib` | `3.11.1` |
| `streamlit` | `1.63.0` |
| `altair` | `5.5.0` |

The first TensorFlow attempts did **not** produce the final training results: TensorFlow GPU operations failed on the available Blackwell `sm_120` hardware with `INVALID_PTX`. **Use the PyTorch implementation in this repository.** A different supported CPU/GPU/PyTorch stack may work, but bit-for-bit numerical equivalence is not guaranteed.

### 2.2 Clone into the path expected by the existing queue scripts

Two Phase 2B queue scripts, and the original Web-40 initialization utility, use a **hard-coded path**: `Path.home() / "brain_tumor_project"`. For the *unmodified* code, use exactly this directory on a machine where it does not already exist:

```bash
# New/reproduction machine only; do not replace an existing project directory.
git clone https://github.com/bakayaro05/Tumor-Classification-DL-.git "$HOME/brain_tumor_project"
cd "$HOME/brain_tumor_project"

git status --short
git rev-parse HEAD
```

If you clone to a different directory, **before Phase 2B** update `PROJECT_ROOT` in both [`phase2/run_all_dl_hyperparams.py`](../phase2/run_all_dl_hyperparams.py) and [`phase2/run_all_hyperparam_pca_svm.py`](../phase2/run_all_hyperparam_pca_svm.py) to `Path(__file__).resolve().parents[1]`. The original initializer [`actual_training_scripts/initialize_web_stress_test.py`](../actual_training_scripts/initialize_web_stress_test.py) also has a hard-coded project root; the curated locked Web-40 data are already in the repository, so you normally **should not rerun that initializer**.

### 2.3 Python, PyTorch, and other dependencies

```bash
# Python 3.12 must already be installed.
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching CPU/CUDA pair of **torch + torchvision** using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For the closest environment match, use the recorded `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` **where those builds are supported and available**. Never combine an incompatible CUDA-specific wheel and torchvision build. The project's [`requirements-gui.txt`](../requirements-gui.txt) does **not** install PyTorch itself.

Once PyTorch/torchvision are installed:

```bash
# Version-matched non-torch packages, as recorded on the training machine.
python -m pip install \
  'numpy==2.5.2' 'pandas==2.3.3' 'scipy==1.18.1' \
  'scikit-learn==1.9.0' 'joblib==1.5.3' \
  'Pillow==12.3.0' 'matplotlib==3.11.1' \
  'streamlit==1.63.0' 'altair==5.5.0'

python -m pip check
python - <<'PY'
import sys, torch, torchvision, sklearn, scipy, streamlit
print('Python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('scikit-learn:', sklearn.__version__)
print('scipy:', scipy.__version__)
print('Streamlit:', streamlit.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

Alternatively, install the GUI's declared **compatible ranges** with `python -m pip install -r requirements-gui.txt`; those ranges are not a training-environment lockfile. The historical [`environment_pip_freeze.txt`](../final_project_docs/environment_pip_freeze.txt) includes obsolete TensorFlow exploration packages; blindly installing its *entire* contents is not necessary to reproduce the final PyTorch pipeline.

On an 8-GB GPU, retain the code's baseline **training batch size of 8** and avoid concurrent model training. CPU-only runs may be exceptionally slow. Save the locally installed package versions (`python -m pip freeze > reproduction_pip_freeze.txt`) for your own experiment record; this is not a command to overwrite the published environment file.

## 3. Development MRI dataset and saved split

### 3.1 Download the correct development dataset

Obtain the [Masoud Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), using the **7,023-image** version represented in the project (5,712 supplied training-folder images, 1,311 supplied testing-folder images). A similarly named Kaggle copy with changed images or directory layout is **not automatically interchangeable**.

The local structure expected by [`src/data_pipeline_torch.py`](../src/data_pipeline_torch.py) is:

```text
brain_tumor_project/
└── data/
    ├── brain_tumor_mri/                 # download/extract locally; not tracked in Git
    │   ├── Training/
    │   │   ├── glioma/
    │   │   ├── meningioma/
    │   │   ├── notumor/
    │   │   └── pituitary/
    │   └── Testing/
    │       ├── glioma/
    │       ├── meningioma/
    │       ├── notumor/
    │       └── pituitary/
    └── splits/
        ├── train_indices.npy           # already tracked
        └── val_indices.npy             # already tracked
```

**Do not** substitute the BDNeuro folders under `external_data/` for this development dataset. They serve different roles. The original `notumor` directory name matters for the expected `ImageFolder` class mapping.

Expected source class counts:

| Class | `Training` | `Testing` |
|---|---:|---:|
| glioma | 1,321 | 300 |
| meningioma | 1,339 | 306 |
| notumor | 1,595 | 405 |
| pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

### 3.2 Verify the images and the saved indices

From the repository root, after extracting the dataset:

```bash
python src/check_dataset.py
python - <<'PY'
from pathlib import Path
import numpy as np
from torchvision.datasets import ImageFolder

root = Path('data/brain_tumor_mri')
training = ImageFolder(root / 'Training')
testing = ImageFolder(root / 'Testing')
train_idx = np.load('data/splits/train_indices.npy')
val_idx = np.load('data/splits/val_indices.npy')
expected = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

print('Class-to-index:', training.class_to_idx)
print('Training folder:', len(training))
print('Saved train/val:', len(train_idx), len(val_idx))
print('Testing folder:', len(testing))
assert training.class_to_idx == testing.class_to_idx == expected
assert (len(training), len(testing), len(train_idx), len(val_idx)) == (5712, 1311, 4569, 1143)
assert sorted(np.r_[train_idx, val_idx].tolist()) == list(range(len(training)))
print('Basic dataset and split checks passed.')
PY
```

**Crucial limitation:** the saved `.npy` arrays identify **positions in torchvision `ImageFolder`'s sorted file listing**, not stable per-image IDs. Matching image **counts alone does not prove the same images occupy the same indices**. Preserve the original folder/file names and sorted ordering; reconcile any change against the source manifests/audit before treating your run as an exact split reproduction. The public repository does not provide a complete canonical path-and-file-hash manifest for every Nickparvar image to independently guarantee identity.

The split was originally generated by [`src/save_split.py`](../src/save_split.py) with seed 42 and stratified 80/20 division of the supplied `Training` folder. **Do not rerun `save_split.py` just because the dataset was downloaded**: it rewrites the committed split files. Run it only in a deliberate, separately labeled new-split experiment after verifying the dataset content and order.

### 3.3 Image preprocessing and model definition

The code converts each input to **RGB**, resizes it to **224 × 224**, and normalizes using ImageNet means/standard deviations. Training-only augmentation uses horizontal/vertical flips (each `p=0.5`) and `ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)`. Validation, test, and external evaluation use the **deterministic** [`eval_transform`](../src/data_pipeline_torch.py).

[`model_cnn_lstm.py`](../model_cnn_lstm.py) implements one **independent** pipeline per backbone:

```text
Single 2-D MRI image
    -> ImageNet-pretrained VGG16 / VGG19 / ResNet50 / ResNet152 / EfficientNet-B3
    -> spatial feature map (nominally 7 x 7)
    -> reshape to 49 spatial positions x C channels
    -> one-layer LSTM (hidden 256 in baseline; optional 128 or 512 in Phase 2B)
    -> BatchNorm1d -> Dropout -> Dense(128) -> ReLU
    -> temporary four-class neural head for training
    -> extract 128-D features
    -> fit StandardScaler on TRAIN only
    -> fit PCA(64) on TRAIN only
    -> train linear SVM (C=1.0) on TRAIN only
    -> predict one of four classes
```

**The LSTM sees spatial positions of ONE MRI image, not time steps, 3-D slices, or longitudinal scans.** The neural head is used for training and Grad-CAM; the final classification path ends at the SVM. Neither its decision scores nor dashboard-derived class bars are calibrated probabilities.

## 4. Data-integrity audit: run before trusting scores

The original study's cross-partition audit, performed after the initial baselines were built, discovered **562 material cross-split image pairs**: **464** byte-identical or decoded-pixel-identical pairs plus **98** additional high-confidence near-duplicate pairs. **250 of the 1,311 official test images** were implicated. There were also **481 review-only perceptual candidates**. This contamination affects interpretation of **all original-split internal scores**.

The executable audit is [`final_project_qc.py`](../final_project_qc.py). Run it on a **disposable working copy**, directing outputs away from the published audit:

```bash
python final_project_qc.py \
  --project-root "$PWD" \
  --output-dir "$PWD/reproduction_outputs/duplicate_leakage_audit"
```

For the historical source version, `MATERIAL_DUPLICATION` and **exit code 2** are expected: **exit code 2 here is a scientific audit blocker, not a reason to delete images silently**. This script also writes companion QC/manifest/environment documents to the *parent of the specified output directory*, which is why a separate output root is used. It does **not** retrain models or repair the split. Compare its JSON and CSV with the retained [official audit](../final_project_docs/duplicate_leakage_audit/duplicate_leakage_audit.json). Perceptual matches require review rather than indiscriminate removal.

For scientifically cleaner **new** experiments, implement patient/group-aware deduplication **before training** and label the resulting metrics as a new protocol. Patient-level independence **cannot** be established from this project because reliable patient IDs are unavailable. Do not present such a redesign as an exact reproduction of the historical results.

## 5. Phase 1: five early-stopping baselines

### 5.1 Start with a single working backbone

Run from the repository root after the dataset preflight. The first command is the existing early-stopping CNN–LSTM trainer; the second extracts neural features and fits/evaluates the training-only StandardScaler, PCA, and linear SVM:

```bash
python actual_training_scripts/train_backbone.py \
  --backbone resnet152 --epochs 10 --batch-size 8 \
  --learning-rate 1e-4 --patience 3 --num-workers 4

python actual_training_scripts/evaluate_backbone.py \
  --backbone resnet152 --pca-components 64 --svm-c 1.0

python actual_training_scripts/evaluate_neural_head.py \
  --backbone resnet152
```

The baseline evaluation script computes **the original supplied Testing-folder metrics** as well as training/validation metrics. These numbers are **historical duplicate-contaminated metrics** in the project dataset. **Do not use the supplied test set to select hyperparameters or checkpoint states.**

### 5.2 Repeat for the other four backbones

The VGG baselines used a **30-epoch maximum budget with early stopping** (the observed training ended earlier), whereas ResNet50/ResNet152/EfficientNet-B3 originally used a 10-epoch maximum. Those Phase 1 horizons are not identical; **Phase 2A fixes this**. The training scripts freeze the pretrained convolutional part for **VGG16/VGG19** after an unstable fully fine-tuned VGG16 experiment; other CNNs remain fine-tuned.

```bash
# VGG16 and VGG19: longer maximum budget, early stopping still enabled.
for backbone in vgg16 vgg19; do
  python actual_training_scripts/train_backbone.py \
    --backbone "$backbone" --epochs 30 --batch-size 8 \
    --learning-rate 1e-4 --patience 3
  python actual_training_scripts/evaluate_backbone.py --backbone "$backbone"
  python actual_training_scripts/evaluate_neural_head.py --backbone "$backbone"
done

# ResNet50 and EfficientNet-B3; ResNet152 was handled above.
for backbone in resnet50 efficientnet_b3; do
  python actual_training_scripts/train_backbone.py \
    --backbone "$backbone" --epochs 10 --batch-size 8 \
    --learning-rate 1e-4 --patience 3
  python actual_training_scripts/evaluate_backbone.py --backbone "$backbone"
  python actual_training_scripts/evaluate_neural_head.py --backbone "$backbone"
done

python actual_training_scripts/compare_neural_vs_pca_svm.py
```

The historical Phase 1 final SVM results live in [`experiments/<backbone>/results/final_results.json`](../experiments/) and comparison tables in [`results/`](../results/). A trained Phase 1 runtime bundle normally contains:

```text
experiments/<backbone>/
  models/best_model.pth                 # neural checkpoint; ignored by Git
  models/svm.joblib                    # downstream final classifier
  pca/standard_scaler.joblib
  pca/pca_model.joblib
  results/training_results.json
  results/final_results.json
  results/neural_head_results.json
```

The checkout may include some compact result files but **no pretrained project-specific `.pth` weights**. A successful new training run will generate its own weights. **Do not mix its scaler/PCA/SVM files with the originally published run's results.**

## 6. Phase 2A: standardized 30-epoch experiments

[`train_backbone_30epoch.py`](../actual_training_scripts/train_backbone_30epoch.py) runs **all 30 epochs** for one backbone (early stopping disabled in this implementation), saving **both** the highest-neural-validation checkpoint and the literal epoch-30 checkpoint. [`evaluate_backbone_30epoch.py`](../actual_training_scripts/evaluate_backbone_30epoch.py) independently extracts downstream features/fits PCA+SVM for **each state**. The selected full-hybrid state was subsequently chosen by **internal SVM validation**, not by test accuracy.

```bash
for backbone in vgg16 vgg19 resnet50 resnet152 efficientnet_b3; do
  python actual_training_scripts/train_backbone_30epoch.py \
    --backbone "$backbone" --epochs 30 --batch-size 8 \
    --learning-rate 1e-4

  python actual_training_scripts/evaluate_backbone_30epoch.py \
    --backbone "$backbone" --checkpoint-type best

  python actual_training_scripts/evaluate_backbone_30epoch.py \
    --backbone "$backbone" --checkpoint-type epoch30
done
```

Each Phase 2A directory includes:

```text
phase2/experiments_30epoch/<backbone>/
  models/best_val_within_30.pth
  models/epoch_30_model.pth
  results/training_results.json
  evaluations/best_val_within_30/
    models/svm.joblib
    pca/{standard_scaler,pca_model}.joblib
    results/final_results.json
  evaluations/epoch_30/
    models/svm.joblib
    pca/{standard_scaler,pca_model}.joblib
    results/final_results.json
```

A `best_val_within_30` file at **epoch 30** is still a separately named *best-within-run state*. Do not automatically relabel it `epoch_30` or assume two checkpoint files are byte-identical. **Phase 2A evaluation includes the historical original test set**; these results retain the duplicate-split caveat.

## 7. Phase 2B: 35 sensitivity runs and 70 evaluations

This is a **one-factor-at-a-time study**, not a full factorial search. For **each of five** backbones, the unchanged Phase 2A baseline is compared with seven modified configurations:

| Factor | Baseline | Alternative runs |
|---|---|---|
| Learning rate | `1e-4` | `lr_1e-5`, `lr_5e-5`, `lr_5e-4` |
| LSTM hidden size | `256` | `lstm_128`, `lstm_512` |
| Dropout | `0.3` | `dropout_0.2`, `dropout_0.5` |

`5 backbones × 7 variants = 35` training runs; the **best-within-30** and **epoch-30** checkpoints of each run are separately evaluated via scaler → PCA → SVM, giving **70 downstream evaluations**. Phase 2B deliberately avoids official-test evaluation to prevent using the test labels for model selection.

**Path prerequisite:** Both Phase 2B queue runners expect `~/brain_tumor_project` unless modified as explained in §2.2. They also check for existing results/checkpoints; use a fresh reproduction working copy, not the authoritative report checkout.

```bash
# First: train 35 thirty-epoch variants (substantial GPU work).
python phase2/run_all_dl_hyperparams.py

# Next: 70 downstream scaler/PCA/linear-SVM evaluations.
# This runner requires both checkpoint states for every configuration.
python phase2/run_all_hyperparam_pca_svm.py
```

For a **single smoke-test variant** rather than the full 35-run queue:

```bash
python actual_training_scripts/train_backbone_hyperparams.py \
  --backbone resnet152 --run-name lstm_512 \
  --epochs 30 --batch-size 8 --learning-rate 1e-4 \
  --lstm-hidden-size 512 --dropout 0.3

python actual_training_scripts/evaluate_hyperparam_pca_svm.py \
  --backbone resnet152 --run-name lstm_512 --checkpoint-type best

python actual_training_scripts/evaluate_hyperparam_pca_svm.py \
  --backbone resnet152 --run-name lstm_512 --checkpoint-type epoch30
```

The canonical paths are [`phase2/hyperparameter_dl/<backbone>/<variant>/`](../phase2/hyperparameter_dl/) with `models/`, `results/`, and `evaluations/{best_val_within_30,epoch_30}/`. The original study's observed average neural-validation sensitivity ranges were **4.62 percentage points for learning rate, 0.58 for LSTM width, and 0.42 for dropout**. A new run can yield different ranges due to optimization/hardware effects.

## 8. Post-hoc material-clean evaluation

This **does not repair the trained models or replace the original split**. It removes duplicate-implicated *test images* from the already computed predictions and reports a **post-hoc subset analysis**.

The Phase 1 script [`evaluate_leakage_clean_test.py`](../evaluate_leakage_clean_test.py) needs the original saved **test PCA features**, labels, and the correctly matched SVM; Phase 2A [`evaluate_leakage_clean_test_phase2.py`](../evaluate_leakage_clean_test_phase2.py) likewise depends on its corresponding evaluation artifacts. They can fail on a Git-only clone missing runtime artifacts.

```bash
# Run only after producing the corresponding Phase 1 and 2A artifacts,
# and verifying exact dataset image ordering.
python evaluate_leakage_clean_test.py
python evaluate_leakage_clean_test_phase2.py
```

Three established test-image exclusion policies are reported: `exact_clean` (**n=1,095**), primary `material_clean` (**n=1,061**), and conservative sensitivity `conservative_clean` (**n=926**). Recorded outputs are in [`final_project_docs/leakage_corrected_evaluation/`](../final_project_docs/leakage_corrected_evaluation/) and [`phase2/results/leakage_corrected_30epoch/`](../phase2/results/leakage_corrected_30epoch/). **These scripts write into the canonical analysis directories**, so only execute them in the disposable clone. A cleaned test subset is **not** proof of clean training or patient independence.

## 9. Freeze 15 pipelines and validate externally

### 9.1 Select and freeze *before* viewing external results

The external evaluator [`compare_external_15.py`](../actual_training_scripts/compare_external_15.py) selects one Phase 1, one Phase 2A, and one Phase 2B candidate per backbone, using **internal full-hybrid validation accuracy** with macro-F1 and deterministic state tie-breaks. For each selected pipeline it records a SHA-256 of **five** items: neural checkpoint, scaler, PCA, SVM, and internal result JSON. That is 15 pipelines × 5 files = **75 hashes**.

The public checkout **already contains the original immutable published** `phase2/external_validation/frozen_15_manifest.json`. Therefore this command in an ordinary public clone **will NOT generate a new freeze manifest**:

```bash
python actual_training_scripts/compare_external_15.py --freeze-only
```

The implementation deliberately refuses to overwrite an existing freeze manifest. If you have independently retrained **all phases** and need to generate *your own* selection/hashes, do this **only in the disposable reproduction clone**, first preserving the published reference:

```bash
mkdir -p reproduction_outputs/published_reference
cp phase2/external_validation/frozen_15_manifest.json \
  reproduction_outputs/published_reference/frozen_15_manifest.published.json
cp phase2/external_validation/frozen_15_selection.csv \
  reproduction_outputs/published_reference/frozen_15_selection.published.csv

# In the disposable clone, move the existing manifest out of the tool's output path.
mv phase2/external_validation/frozen_15_manifest.json \
   reproduction_outputs/published_reference/frozen_15_manifest.from_checkout.json

# This creates a NEW manifest from the NEW train/validation runs.
python actual_training_scripts/compare_external_15.py --freeze-only
```

The original reference hashes will **not generally match newly trained checkpoint bytes**, even if broad accuracy results are similar. Never silently claim the new manifest is identical to the project's historical frozen record. **Do not select a new preferred pipeline based on the BDNeuro or Web-40 labels.**

### 9.2 Audit BDNeuro V8 for cross-dataset overlap

Use **BDNeuro-MRI Version 8** ([dataset version and citation](https://doi.org/10.17632/zwr4ntf94j.8)), not an unspecified newer edition. The public repository contains raw and previously generated BDNeuro-derived directories, subject to the source's attribution and redistribution terms. The source dataset has the form:

```text
external_data/BDNeuro_V8_raw/Brain_Tumor_MRI_Dataset_Final/
  train/{glioma,meningioma,no_tumor,pituitary}/
  val/{glioma,meningioma,no_tumor,pituitary}/
  test/{glioma,meningioma,no_tumor,pituitary}/
```

The following commands place new audit/derived outputs in **separate `reproduction_*` paths** so they do not overwrite existing Git-tracked evidence:

```bash
python actual_training_scripts/audit_bdneuro_overlap.py \
  --output-dir "$PWD/phase2/external_validation/reproduction_bdneuro_overlap_audit"

python actual_training_scripts/build_bdneuro_exact_clean_cohort.py \
  --audit-csv "$PWD/phase2/external_validation/reproduction_bdneuro_overlap_audit/cross_exact_pixel_duplicates.csv" \
  --output-dir "$PWD/external_data/BDNeuro_V8_exact_clean_REPRO" \
  --manifest-dir "$PWD/phase2/external_validation/reproduction_bdneuro_exact_clean_cohort"

python actual_training_scripts/build_bdneuro_balanced_subset.py \
  --source-dir "$PWD/external_data/BDNeuro_V8_exact_clean_REPRO" \
  --output-dir "$PWD/external_data/BDNeuro_V8_exact_clean_balanced_REPRO" \
  --manifest-dir "$PWD/phase2/external_validation/reproduction_bdneuro_balanced_subset" \
  --per-class 412
```

The source study recorded **5,941** BDNeuro images, **3,297** unique decoded-pixel overlaps removed against Nickparvar, and **2,644** exact-overlap-clean images (**1,159 glioma, 472 meningioma, 412 notumor, 601 pituitary**). The nested balanced subset contains **1,648 images (412/class)**. The original audit separately screened perceptual similarities but **did not automatically remove all perceptually similar images**. These counts are source-version-dependent; if your audit disagrees, stop and investigate file identity, class mapping, or versions rather than forcing the expected count.

The **2,644 and 1,648 cohorts are not independent of each other**. Image-level cross-dataset exact cleaning does not establish patient-level independence.

### 9.3 Evaluate the frozen 15 on both BDNeuro cohorts

Only after (a) obtaining complete **matched frozen bundles**, (b) generating/verifying your own manifest, and (c) finishing dataset provenance/overlap auditing:

```bash
python actual_training_scripts/compare_external_15.py \
  --external-dir "$PWD/external_data/BDNeuro_V8_exact_clean_REPRO" \
  --confirm-audited \
  --results-name reproduction_15_bdneuro_full

python actual_training_scripts/compare_external_15.py \
  --external-dir "$PWD/external_data/BDNeuro_V8_exact_clean_balanced_REPRO" \
  --confirm-audited \
  --results-name reproduction_15_bdneuro_balanced
```

Use **different `--results-name` values for the two cohorts**; otherwise outputs can collide. Each produces `external_comparison_15.csv` and `.json` under `phase2/external_validation/<results-name>/`. The evaluator checks the frozen artifact hashes and stops on missing/mismatched files. **Do not use `--rerun` to overwrite the published study's original result directories.**

The archived completed comparisons are independently viewable without checkpoints:

- [Primary BDNeuro full-cohort results](../phase2/external_validation/comparison_15_models_bdneuro_full/external_comparison_15.csv)
- [Nested balanced-cohort results](../phase2/external_validation/comparison_15_models_bdneuro_balanced/external_comparison_15.csv)
- [Original freeze selection](../phase2/external_validation/frozen_15_selection.csv) and [hash manifest](../phase2/external_validation/frozen_15_manifest.json)

## 10. Web-40 stress test

The exploratory Web-40 set contains **40 source-verified images**, **10 per class**, and was **locked before inference**. It is **not a clinical performance benchmark**. Provenance resides in the restored [Web-40 source workbook](DL_TUMOR_CLASSIFICATION_FINAL_CLAUDE_PACKAGE_2026-10-03/manifests/Web_Stress_Test_40_Source_List.xlsx) and the [locked case manifest](../phase2/external_validation/web_stress_40_locked/locked_40_manifest.csv). The older [`web_stress_40_manifest.csv`](../phase2/external_validation/web_stress_40/web_stress_40_manifest.csv) includes original blank/`PENDING` scaffolding rows: **it is not the completed provenance source**.

Do **not** rerun the original Web-40 initializer on the finalized case directory, recrop panels, or replace any image after inspecting predictions. Before an independent replay, confirm the image files still match the locked SHA-256 values:

```bash
python - <<'PY'
import csv, hashlib
from pathlib import Path

manifest = Path('phase2/external_validation/web_stress_40_locked/locked_40_manifest.csv')
with manifest.open(newline='', encoding='utf-8-sig') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 40, f'Expected 40 locked cases; got {len(rows)}'
for row in rows:
    path = Path(row['relative_path'])
    assert path.is_file(), f'Missing: {path}'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == row['sha256'], f'Hash changed: {path}'
print('Web-40: 40 files present and SHA-256 matched the locked manifest.')
PY
```

Run the existing image-level cross-dataset audit into a new output directory:

```bash
python actual_training_scripts/audit_web_stress_40.py \
  --web-root "$PWD/external_data/web_stress_40_raw" \
  --bdneuro-clean "$PWD/external_data/BDNeuro_V8_exact_clean_REPRO" \
  --output-dir "$PWD/phase2/external_validation/reproduction_web40_overlap_audit"
```

The original Web-40 protocol visually rejected a transformed duplicate (`NOR_01`), replaced it **before model predictions**, reviewed another candidate (`MEN_02`) as genuinely different, then locked the dataset. Review the historical [audit JSON](../phase2/external_validation/web_stress_40_audit/summary.json) and [lock record](../phase2/external_validation/web_stress_40_locked/lock_record.json); do not claim that a pHash distance alone proves identity.

**Only after** auditing and confirming the cohort and the 15 frozen bundles:

```bash
python actual_training_scripts/compare_external_15.py \
  --external-dir "$PWD/external_data/web_stress_40_raw" \
  --confirm-audited \
  --results-name reproduction_15_web40
```

This evaluates 15 pipelines × 40 images = **600 predictions**. The original recorded overall accuracies were **25.0%–42.5%**. The final retrospective ResNet152 Phase 2A representative obtained **37.5%** on the Web-40 cohort. One case changes overall accuracy by **2.5 percentage points**; do not generalize these figures to clinical populations. All source-image rights must be checked individually before redistributing the panels.

## 11. Explainability and the Streamlit dashboard

### 11.1 Formal XAI study versus live demonstrations

The **formally rated Grad-CAM and SVM occlusion studies** were performed on **Phase 1 ResNet152**, not the selected Phase 2A ResNet152 representative. The project retained:

- [Formal Grad-CAM rated results](../gradcam_evaluation/gradcam_evaluation_resnet152_seed42.csv): **20 MRI cases** (15 tumour cases for lesion-localization ratings).
- [Formal SVM occlusion rated results](../occlusion_evaluation/occlusion_comparison_resnet152_seed42.csv): **11 cases**, 169 patch positions each = **1,859 occlusions**; 58 changed the final SVM class.

Use these original research tools with a **complete Phase 1 ResNet152 runtime bundle**:

```bash
streamlit run gradcam_evaluation_app.py
# In another session, if needed:
streamlit run occlusion_evaluation_app.py
```

Grad-CAM explains the **neural classification head**. Occlusion probes the **full final CNN → LSTM → scaler → PCA → SVM** classifier. Their maps are not interchangeable, anatomical masks, or clinical localization validation. Reproducing the numerical/image processing does not automatically recreate the historical **human visual ratings**.

### 11.2 Current final dashboard — 15 pipelines, seven tabs

```bash
streamlit run app.py
```

The [final `app.py`](../app.py) presents **Predict MRI**, **Base-paper reproduction**, **Model card**, **Hyperparameter sensitivity**, **External evaluation**, **XAI study**, and **Frozen 15 comparison**. It supports PNG/JPG/JPEG/BMP/TIFF uploads, not DICOM. Stage/backbone selection identifies one of the **15 frozen configurations**. The selected pipeline must have its own authentic `.pth` checkpoint, SVM, scaler and PCA in the paths mapped by [`app.py`](../app.py) and [`gui/model_registry.py`](../gui/model_registry.py).

The Git tree has **zero tracked `.pth` checkpoints**, but the authentic 15-pipeline model archive is now available as [`models-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0). **Restore and hash-verify it first**; live prediction and XAI still cannot run from the unmodified checkout alone. Some report/metric tabs can still read committed JSON/CSV. The current dashboard itself reports missing pipeline artifacts; a reader should not borrow a Phase 1 checkpoint to make a Phase 2A pipeline appear ready. The existing [`verify_gui_setup.py`](../verify_gui_setup.py) and [`README_GUI.md`](../README_GUI.md) are **Phase 1/four-tab legacy guides**, not comprehensive preflights for this seven-tab application.

## 12. Evidence, expected results, and interpretation

The following are **recorded historical outputs**, not performance guarantees from a fresh rerun:

| Experiment | Recorded value | How it should be labeled |
|---|---:|---|
| Phase 1 VGG16 SVM internal test | 94.97% | Original duplicate-contaminated test |
| Phase 1 VGG19 SVM internal test | 93.52% | Original duplicate-contaminated test |
| Phase 1 ResNet50 SVM internal test | 98.09% | Original duplicate-contaminated test |
| Phase 1 ResNet152 SVM internal test | 98.40% | Original duplicate-contaminated test |
| Phase 1 EfficientNet-B3 SVM internal test | 98.47% | Original duplicate-contaminated test |
| Phase 2A ResNet152 epoch 10, internal validation | 98.600% | Original duplicate-contaminated split, used for internal checkpoint selection |
| Phase 2A ResNet152 epoch 10, internal test | 98.703% | Original duplicate-contaminated test |
| Phase 2A ResNet152 epoch 10, `material_clean` test, n=1,061 | 98.869%, macro-F1 98.788% | Post-hoc test-only exclusions; training leakage remains |
| Phase 2A ResNet152 epoch 10, BDNeuro exact-overlap-clean full, n=2,644 | 88.31%, macro-F1 87.85% | Primary structured external evaluation |
| Same model, BDNeuro balanced, n=1,648 | 90.47%, macro-F1 90.50% | Nested cohort, **not independent** of the full BDNeuro set |
| Same model, locked Web-40, n=40 | 37.5%, macro-F1 31.66% | Exploratory presentation-shift stress test; not clinical accuracy |

The project-level representative **ResNet152 Phase 2A epoch 10** was selected **retrospectively partly using the BDNeuro results**. All 15 inputs to that comparison were frozen first, but its BDNeuro score is not an untouched, post-selection estimate of the chosen winner. The Web-40 outcomes were not used to retune models. [Full 15-pipeline results](../phase2/external_validation/) should accompany any claim of best model.

**Reproduction does not mean equivalent scientific validity.** Even if you recreate the historical scores, their correct interpretation remains constrained by duplicate-contaminated original partitions, the post-hoc nature of the material-clean test, lack of verified patient independence, one fixed data split/seed, absence of confidence intervals/repeated-seed trials, and domain-shift failures. This is a **single-image 2-D research system**, not a clinical classifier or lesion segmenter.

## 13. Troubleshooting and reproducibility limits

| Symptom | Likely cause and appropriate response |
|---|---|
| `FileNotFoundError: data/brain_tumor_mri/Training` | The **original Nickparvar development images** were not downloaded/extracted to the expected path. The BDNeuro data already in the repo are not a replacement. |
| Correct counts, unexpected results | Check original dataset version, filenames, **ImageFolder sorted order**, preprocessing, package versions, random initialization and exact checkpoint state. Counts alone do not prove split identity. |
| Missing `best_model.pth` / `best_val_within_30.pth` | The Git tree excludes neural `.pth` weights. Restore the frozen states from the matching [`models-v1.0`](https://github.com/bakayaro05/Tumor-Classification-DL-/releases/tag/models-v1.0) TAR files, then verify frozen SHA-256 hashes per [MODEL_WEIGHTS](MODEL_WEIGHTS.md). Never substitute a different backbone/phase/epoch. |
| Training CUDA out-of-memory | Use the baseline batch 8; avoid competing GPU processes. Changes to batch size alter the experiment and should be reported. |
| VGG validation collapses to majority-class behavior | Confirm the implemented **frozen VGG convolutional backbone** policy. Fully fine-tuned VGG16 originally collapsed (~28% validation) and produced near-constant features/PCA failures. |
| `BatchNorm1d` error for batch of one | Phase 1 training set has 4,569 samples; with batch 8, the final remainder is one. The trainer uses `drop_last=True` for the training loader. Do not remove this safeguard without redesigning BatchNorm. |
| Phase 2B runner says trainer not found | Queue scripts assume `~/brain_tumor_project` by default. Use that checkout name or make the path-relative adjustment in §2.2. |
| Phase 2B queue skips runs but no weights exist | The **tracked JSON history is not equivalent to a complete runtime bundle**. Use a clean clone; verify both `.pth` checkpoints per configuration before downstream evaluation. |
| Existing BDNeuro cohort output prevents building | Builders intentionally refuse to overwrite existing directories unless `--rebuild` is specified. Prefer unique `*_REPRO` paths as shown; do not delete original published cohorts. |
| Freeze-only reports manifest already exists | This is intentional protection. Preserve the original and create a new freeze **only after retraining in a disposable clone** (§9.1). |
| Frozen hash mismatch during external evaluation | Missing/different neural weights or mismatched scaler/PCA/SVM/results. **Stop**; do not bypass verification or reselect after viewing external labels. |
| Grad-CAM and SVM occlusion disagree | They explain **different decisions** (neural head vs final hybrid SVM). Heatmaps are not tumour segmentation masks. |
| `final_project_qc.py` returns code 2 | The historical audit **detects material cross-split duplication**. This is expected and must remain disclosed. |
| Existing `FINAL_QC_BLOCKER.md` says "not locked" | It is an **older post-audit snapshot**. Subsequent external-validation experiments and final documentation were completed with material leakage caveats retained. Do not erase the historical record. |

**Safety and provenance:** Download source images only from their authorized providers; cite [Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [BDNeuro V8](https://doi.org/10.17632/zwr4ntf94j.8), the [base paper](https://doi.org/10.1007/s13721-025-00718-8), and individual Web-40 source figures. Review redistribution licenses before publishing images or trained weights. `torch.load`/`joblib.load` should only be used with **trusted artifact files** because deserialization can be unsafe. No clinical evaluation, approval, DICOM workflow, patient-level labels, or prospective validation is claimed.

## 14. Final verification checklist

After a **new reproduction run**, verify the following independently and retain the results under your own run ID:

- [ ] Repository revision (`git rev-parse HEAD`) and environment versions recorded; no original results overwritten.
- [ ] Exact **7,023-image** Nickparvar study version, folder names, and sorted file ordering verified; class mapping matches `{'glioma':0,'meningioma':1,'notumor':2,'pituitary':3}`.
- [ ] Saved **4,569/1,143** train/validation indices and **1,311** dataset-supplied test images verified.
- [ ] Duplicate audit run and **MATERIAL_DUPLICATION** disclosed; no claim that historical partitions are independent.
- [ ] All five Phase 1 checkpoints and their own scaler/PCA/SVM outputs created (or clearly marked unavailable).
- [ ] All five Phase 2A training runs completed **30 epochs**, and both checkpoint states evaluated.
- [ ] All **35 Phase 2B** training variants finished, both states evaluated (**70** downstream runs), with **no test-driven Phase 2B selection**.
- [ ] Fifteen candidates selected using internal validation only; new manifest frozen and all **75 hashes** verified from locally available files before external inference.
- [ ] BDNeuro V8 cross-dataset overlap audit inspected; primary **2,644-image** cohort distinguished from nested **1,648-image** balanced subset.
- [ ] Final locked 40 Web images and source citations verified before inference; no images modified after results were viewed.
- [ ] Original formal Phase 1 ResNet152 XAI study distinguished from live GUI XAI demos for other pipelines.
- [ ] Internal, material-clean, BDNeuro and Web-40 results reported **with their separate limitations**; new numeric results not silently substituted for published metrics.

---

**Bottom line:** The repository is an extensive, inspectable **source-and-evidence archive** and an executable *training workflow when its source datasets and compatible environment are supplied*. A public Git clone does **not** contain neural checkpoint bytes, but the **15 original frozen inference bundles are available via `models-v1.0`** for independent restoration and hash verification; the 71 extra research checkpoints are a separate optional archive. This guide deliberately separates *inspection*, *fresh retraining*, and *faithful frozen-artifact replay* so readers understand exactly what they can reproduce.
