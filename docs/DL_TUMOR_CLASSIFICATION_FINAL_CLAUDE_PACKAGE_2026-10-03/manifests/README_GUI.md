# Brain Tumor MRI Classification GUI

This package adds a local Streamlit interface to the completed five-backbone
comparison project. It preserves the validated inference path:

`MRI image → evaluation transform → CNN → LSTM → 128-D dense features → StandardScaler → PCA → linear SVM`

The final displayed class is the SVM prediction. The neural classifier head is
shown separately as a comparison.

## Included screens

- **Predict MRI:** upload one image, select any of the five trained backbones,
  run inference, compare the SVM and neural-head outputs, and inspect class scores.
- **Model performance:** validation/test metrics, confusion matrices, per-class
  test metrics, and training curves for the selected backbone.
- **Compare models:** one standardized table and chart populated directly from
  each experiment's `final_results.json`.
- **Methodology:** inference path and artifact-readiness checks.
- **Grad-CAM explainability:** model-input view, contribution heatmap, adjustable
  MRI overlay, classifier-agreement notice, and PNG downloads.

## Install into the existing project

Copy these items into the root of `~/brain_tumor_project`:

```text
app.py
gui/
requirements-gui.txt
verify_gui_setup.py
README_GUI.md
```

The result should look like:

```text
brain_tumor_project/
├── app.py
├── gui/
├── model_cnn_lstm.py
├── src/data_pipeline_torch.py
├── experiments/
├── requirements-gui.txt
└── verify_gui_setup.py
```

Activate the existing environment and install only the GUI dependencies:

```bash
cd ~/brain_tumor_project
source venv/bin/activate
python -m pip install -r requirements-gui.txt
python verify_gui_setup.py
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Scientific safeguards

- Preprocessing imports `eval_transform` directly from
  `src/data_pipeline_torch.py`, so uploaded images use the same deterministic
  evaluation transform as saved feature extraction.
- Checkpoint class mapping and SVM class order are validated before prediction.
- StandardScaler and PCA are loaded from the chosen experiment; nothing is
  refitted from the uploaded image.
- The SVM was trained with `probability=False`. The GUI therefore labels its
  chart as a relative score, not a probability.
- Grad-CAM targets the selected CNN-LSTM neural-head class. The GUI explicitly
  distinguishes this explanation from the external scikit-learn SVM decision.
- Heatmaps are described as coarse contribution maps, not tumor-segmentation
  boundaries.
- Only one backbone is retained in memory per GUI session to limit VRAM usage.
- The GUI reads standardized files only from `experiments/<backbone>/`; it does
  not use the older root-level ResNet152 artifacts.

## Current input scope

PNG, JPG/JPEG, BMP, and TIFF images are supported. DICOM is intentionally not
included in this first version because the completed Point 1 dataset and
pipeline operate on ordinary image files. The application is a research and
educational prototype, not a medical device.
