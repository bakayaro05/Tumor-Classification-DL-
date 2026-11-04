# Codebase Guide

> **Hostile-review warning:** A pre-audit narrative names DenseNet121 as implemented/evaluated, but supplied source/manifests/five results do not. Current evaluation source uses `probability=False`, while preserved VGG19/ResNet152 logs suggest some runtime SVMs used `True`; inspect each `.joblib`. GUI inference uses `decision_function`. Saved epoch fields are zero-based while display text is one-based.

## Snapshot caveat

The supplied 25-file attachment is a substantial but incomplete snapshot. `app.py` imports a `gui/` package whose modules are not all present in that attachment. Later project records confirm at least `model_registry.py`, `model_factory.py`, `inference_engine.py`, `results_loader.py`, and `gradcam.py`; `app.py` also imports `gui.occlusion`. The old `project_files.txt` predates these additions and the leakage audit. Regenerate the manifest from the real repository.

## Intended repository shape

```text
brain_tumor_project/
├── app.py
├── model_cnn_lstm.py
├── model_resnet152_lstm.py                 # legacy/specific model
├── src/
│   ├── data_pipeline_torch.py
│   ├── check_dataset.py
│   └── save_split.py
├── actual_training_scripts/
│   ├── train_backbone.py
│   ├── evaluate_backbone.py
│   ├── train_resnet152_lstm.py             # legacy first baseline
│   ├── extract_all_features.py             # legacy first baseline
│   ├── prepare_pca_features.py             # legacy first baseline
│   ├── train_svm.py                        # legacy first baseline
│   ├── evaluate_final_test.py              # legacy first baseline
│   └── diagnose_vgg16_collapse.py
├── gui/
│   ├── model_registry.py
│   ├── model_factory.py
│   ├── inference_engine.py
│   ├── results_loader.py
│   ├── gradcam.py
│   └── occlusion.py
├── experiments/<backbone>/
│   ├── models/{best_model.pth,svm.joblib}
│   ├── features/{train,val,test}_{features,labels}.npy
│   ├── pca/{standard_scaler.joblib,pca_model.joblib,...}
│   └── results/{training_results.json,final_results.json}
├── gradcam_evaluation_app.py
├── occlusion_evaluation_app.py
├── verify_gui_setup.py
├── requirements-gui.txt
└── tests/smoke scripts
```

## Core model

`model_cnn_lstm.py` defines:

- `SUPPORTED_BACKBONES`: `vgg16`, `vgg19`, `resnet50`, `resnet152`, `efficientnet_b3`;
- `CNNLSTM`: backbone creation, feature-map-to-sequence conversion, forward pass, optional debug tensors, and 128-D feature extraction.

The forward data flow is:

```text
x
→ cnn_backbone
→ cnn_features [B,C,H,W]
→ permute/reshape [B,H×W,C]
→ LSTM
→ final timestep [B,256]
→ BatchNorm → Dropout → Dense → ReLU
→ deep_features [B,128]
→ classifier logits [B,4]
```

`model_resnet152_lstm.py` is the earlier ResNet152-specific implementation. It is historically important but should not be mixed with standardized artifacts unless paths and state dictionaries match.

## Data pipeline

`src/data_pipeline_torch.py` is the intended authority for dataset paths, class mapping, saved train/validation indices, training augmentation, and deterministic evaluation transform. It was referenced by all training, feature extraction, and GUI inference paths. This file was listed in the manifest but not included in the 25 attached files, so exact transform parameters must be verified from the actual repository.

## Training flow

`train_backbone.py`:

1. parses backbone/epochs/batch size/learning rate/patience/workers/resume;
2. fixes seed 42;
3. loads datasets using `create_datasets()`;
4. builds `CNNLSTM`;
5. freezes VGG16/VGG19 convolutional blocks;
6. trains with CrossEntropyLoss and RMSprop;
7. tracks validation performance, early stopping, and best checkpoint;
8. writes `training_results.json`.

The training loader shuffles with a seeded generator and uses `drop_last=True`; validation does not shuffle.

## Standardized evaluation flow

`evaluate_backbone.py` performs the full post-training pipeline for one backbone:

1. load best checkpoint;
2. extract deterministic 128-D features for train/validation/test;
3. save feature arrays;
4. fit StandardScaler on train only;
5. fit PCA(64) on scaled train only;
6. transform validation/test;
7. fit a linear SVM (`C=1`);
8. compute validation/test metrics, reports, and confusion matrices;
9. save scaler, PCA, SVM, arrays, and `final_results.json`.

This script is the main historical experiment producer. It must be rerun after clean splitting.

## Legacy first-baseline flow

`train_resnet152_lstm.py`, `extract_all_features.py`, `prepare_pca_features.py`, `train_svm.py`, and `evaluate_final_test.py` created root-level ResNet152 artifacts. The early SVM script used `probability=True`; current standardized source uses `False`, but historical logs require per-artifact verification. The GUI intentionally reads only `experiments/<backbone>/`.

## Diagnostic/smoke scripts

- `test_resnet152_features.py`: verifies convolutional output and spatial sequence shape.
- `test_lstm_section.py`: verifies LSTM mechanics.
- `test_post_lstm_layers.py`: verifies BN/dropout/dense path.
- `test_complete_model.py`: verifies complete forward modes.
- `test_training_step.py`: verifies loss, gradients, and optimizer step.
- `test_real_mri_batch.py`: verifies actual dataset batches and class mapping.
- `test_best_checkpoint_features.py`: verifies best checkpoint and deterministic feature extraction.
- `test_all_backbones.py`: instantiates/tests all supported backbones.
- `diagnose_vgg16_collapse.py`: diagnostic created after VGG instability.

These are mostly executable smoke scripts rather than isolated pytest tests. They may perform heavy imports/work at module level.

## GUI modules and flow

`model_registry.py` defines backbone order/display names and maps each backbone to checkpoint, SVM, scaler, PCA, final results, and training results.

`inference_engine.py`/`model_factory.py` load one model at a time, validate artifact/class compatibility, apply `eval_transform`, generate neural logits and 128-D features, apply the saved scaler/PCA/SVM, and return final and comparison outputs.

`results_loader.py` loads JSON results, constructs comparison tables, and formats confusion matrices.

`gradcam.py` hooks selected CNN layers and backpropagates a neural-head target. Its output cannot be claimed to directly explain the SVM.

`occlusion.py` masks input patches and reruns the complete final-SVM pipeline. It is slower and coarse.

`app.py` provides prediction, performance, comparison, and methodology tabs. It limits uploads to 25 MB and loads only one backbone in memory per session.

`gradcam_evaluation_app.py` and `occlusion_evaluation_app.py` support reproducible manual evaluation and CSV export.

## Dependencies

The GUI requirements snapshot lists Streamlit, Altair, pandas, Pillow, NumPy, joblib, scikit-learn, torch, and torchvision. The full training environment must be frozen from the real environment; the requirements file does not prove exact versions.

## Artifacts and compatibility

Every checkpoint must match the model class, backbone, class order, and preprocessing. Each scaler/PCA/SVM triplet belongs to the 128-D features from one specific checkpoint. Never mix artifacts across backbones or between legacy root-level and standardized directories.

## Conversation → decision → implementation → result examples

| Conversation/issue | Decision | Implementation | Result |
|---|---|---|---|
| “What is the LSTM timestep?” | Use spatial positions | `cnn_to_sequence()` | `[B,H×W,C]` input |
| TensorFlow PTX failure | Move to PyTorch | PyTorch/torchvision codebase | GPU training proceeded |
| VGG16 collapsed | Freeze VGG CNN | `requires_grad=False` | VGG16 test 94.97% historically |
| Grad-CAM cannot explain sklearn SVM | Separate explanation targets | Neural Grad-CAM + SVM occlusion | Honest dual explanation |
| Need final scientific check | Audit duplicates | exact/pixel/perceptual audit | material leakage found; lock stopped |

## Continuation checklist

Before executing anything, recover the complete repository, confirm `src/data_pipeline_torch.py` and all `gui/` modules, archive the contaminated-split outputs, implement group-aware split generation, add an automated zero-cross-group audit, and version the clean rerun separately.
