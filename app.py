"""Streamlit GUI for the validated brain-tumor CNN-LSTM-SVM experiments."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from gui.gradcam import GradCAMResult, colorize_heatmap, overlay_heatmap, png_bytes
from gui.inference_engine import ModelManager, PredictionResult
from gui.occlusion import OcclusionResult, overlay_occlusion
from gui.model_registry import BACKBONE_ORDER, DISPLAY_NAMES, all_artifacts, get_artifacts
from gui.results_loader import (
    CLASS_DISPLAY,
    CLASS_NAMES,
    comparison_table,
    confusion_frame,
    load_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)


def _model_manager() -> ModelManager:
    manager = st.session_state.get("model_manager")
    if manager is None:
        manager = ModelManager(PROJECT_ROOT, device_preference="auto")
        st.session_state.model_manager = manager
    return manager


def _open_image(raw: bytes) -> Image.Image:
    if len(raw) > 25 * 1024 * 1024:
        raise ValueError("The uploaded image exceeds the 25 MB limit.")
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error


def _score_chart(scores: dict[str, float], title: str, percent: bool = True):
    frame = pd.DataFrame(
        {
            "Class": list(scores.keys()),
            "Score": [value * 100 if percent else value for value in scores.values()],
        }
    )
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            x=alt.X("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Class:N", sort="-x", title=None),
            color=alt.Color("Class:N", legend=None),
            tooltip=["Class:N", alt.Tooltip("Score:Q", format=".2f")],
        )
        .properties(title=title, height=230)
    )


def _render_prediction(result: PredictionResult) -> None:
    st.subheader("Prediction result")
    primary, secondary, agreement, runtime = st.columns(4)
    primary.metric("Final SVM prediction", result.svm_class_name)
    secondary.metric("Neural-head prediction", result.nn_class_name)
    agreement.metric("Classifier agreement", "Yes" if result.models_agree else "No")
    runtime.metric("Inference time", f"{result.elapsed_seconds:.2f} s")

    if result.models_agree:
        st.success(f"Both classifier heads predict **{result.svm_class_name}**.")
    else:
        st.warning(
            "The SVM and neural head disagree. The research pipeline's final output is "
            f"**{result.svm_class_name}** from the SVM."
        )

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            _score_chart(result.svm_relative_scores, "SVM relative class scores"),
            use_container_width=True,
        )
        st.caption(
            "Relative scores are a softmax view of the SVM decision values. They are not "
            "calibrated probabilities because the saved SVM uses probability=False."
        )
    with right:
        st.altair_chart(
            _score_chart(result.nn_scores, "Neural-head softmax scores"),
            use_container_width=True,
        )
        st.caption("Softmax scores express model confidence but are not clinically calibrated.")

    with st.expander("Inference details"):
        st.write(
            {
                "backbone": DISPLAY_NAMES[result.backbone],
                "device": result.device,
                "original_image_size": f"{result.image_size[0]} × {result.image_size[1]}",
                "deep_feature_dimension": result.deep_feature_dimension,
                "pca_feature_dimension": result.pca_feature_dimension,
                "pipeline": "CNN → LSTM → 128-D dense features → StandardScaler → PCA → linear SVM",
            }
        )


def _render_explanation(explanation: GradCAMResult, prediction: PredictionResult) -> None:
    st.divider()
    st.subheader("Explainability — Grad-CAM")
    st.write(
        f"Grad-CAM highlights the image regions that contributed most strongly to the "
        f"CNN–LSTM neural prediction of **{explanation.target_class_name}**."
    )
    if prediction.models_agree:
        st.success(
            f"The neural head and final SVM agree on **{prediction.svm_class_name}**. "
            "The heatmap explains the neural component of this agreed classification."
        )
    else:
        st.warning(
            f"The neural head predicts **{prediction.nn_class_name}**, while the final SVM "
            f"predicts **{prediction.svm_class_name}**. This heatmap explains only the "
            "neural-head prediction."
        )

    opacity = st.slider(
        "Heatmap overlay strength",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
    )
    layer_heatmaps = getattr(explanation, "layer_heatmaps", {})
    if {
        "cnn_backbone.layer3[-1]",
        "cnn_backbone.layer4[-1]",
    }.issubset(layer_heatmaps):
        _render_resnet_gradcam_comparison(explanation, opacity)
    else:
        heatmap_rgb = colorize_heatmap(explanation.heatmap)
        overlay_rgb = overlay_heatmap(explanation, opacity)
        original_col, heatmap_col, overlay_col = st.columns(3)
        with original_col:
            st.image(explanation.model_input_rgb, caption="Model input")
        with heatmap_col:
            st.image(heatmap_rgb, caption="Grad-CAM contribution heatmap")
        with overlay_col:
            st.image(overlay_rgb, caption=f"MRI overlay · strength {opacity:.2f}")

        download_left, download_right, details = st.columns([1, 1, 2])
        with download_left:
            st.download_button(
                "Download heatmap",
                data=png_bytes(heatmap_rgb),
                file_name=f"{explanation.backbone}_{explanation.target_class_key}_gradcam.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_right:
            st.download_button(
                "Download overlay",
                data=png_bytes(overlay_rgb),
                file_name=f"{explanation.backbone}_{explanation.target_class_key}_overlay.png",
                mime="image/png",
                use_container_width=True,
            )
        with details:
            st.caption(
                f"Target layer: `{explanation.target_layer_name}` · "
                f"Grad-CAM time: {explanation.elapsed_seconds:.2f} s"
            )

    st.caption(
        "Blue/dark regions had lower contribution; yellow/red regions had stronger "
        "contribution. This is a coarse explanation map, not a tumor-segmentation boundary."
    )


def _render_svm_occlusion(
    image: Image.Image,
    backbone: str,
    prediction: PredictionResult,
    record: dict,
) -> None:
    st.divider()
    st.subheader("Explainability — final-SVM occlusion sensitivity")
    st.write(
        "This test covers small regions of the transformed MRI and reruns the complete "
        "CNN–LSTM–StandardScaler–PCA–SVM pipeline. Yellow/red regions are locations whose "
        f"removal most reduced the final SVM decision score for **{prediction.svm_class_name}**."
    )

    result: OcclusionResult | None = record.get("occlusion")
    error_text = record.get("occlusion_error")
    if result is None:
        if st.button(
            "Generate final-SVM occlusion map",
            key=f"occlusion_{record['fingerprint']}_{backbone}",
            use_container_width=True,
        ):
            try:
                with st.spinner("Running the full prediction pipeline across occluded regions…"):
                    result = _model_manager().explain_svm_occlusion(
                        image,
                        backbone,
                        prediction.svm_class_key,
                        patch_size=32,
                        stride=16,
                        batch_size=8,
                    )
                record["occlusion"] = result
                record["occlusion_error"] = None
            except Exception as error:
                error_text = str(error)
                record["occlusion_error"] = error_text

    if error_text and result is None:
        st.warning(f"Occlusion sensitivity could not be generated: {error_text}")
        return
    if result is None:
        st.caption("This optional analysis is slower than Grad-CAM because it reruns inference many times.")
        return

    opacity = st.slider(
        "Occlusion overlay strength",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        key=f"occlusion_opacity_{record['fingerprint']}_{backbone}",
    )
    heatmap_rgb = colorize_heatmap(result.heatmap)
    overlay_rgb = overlay_occlusion(result, opacity)
    original_col, heatmap_col, overlay_col = st.columns(3)
    original_col.image(result.model_input_rgb, caption="Model input")
    heatmap_col.image(heatmap_rgb, caption="Positive final-SVM support")
    overlay_col.image(overlay_rgb, caption=f"SVM occlusion overlay · strength {opacity:.2f}")

    detail_columns = st.columns(4)
    detail_columns[0].metric("Target class", result.target_class_name)
    detail_columns[1].metric("Occlusion windows", result.occlusion_count)
    detail_columns[2].metric(
        "Prediction flips",
        f"{result.prediction_flip_fraction * 100:.1f}%",
    )
    detail_columns[3].metric("Runtime", f"{result.elapsed_seconds:.2f} s")
    st.caption(
        f"Patch {result.patch_size}×{result.patch_size} · stride {result.stride} · "
        "masked pixels replaced by the ImageNet channel means. The map explains the final "
        "SVM score, but remains a coarse sensitivity analysis—not tumor segmentation."
    )

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download SVM occlusion heatmap",
        data=png_bytes(heatmap_rgb),
        file_name=f"{backbone}_{result.target_class_key}_svm_occlusion.png",
        mime="image/png",
        use_container_width=True,
    )
    download_right.download_button(
        "Download SVM occlusion overlay",
        data=png_bytes(overlay_rgb),
        file_name=f"{backbone}_{result.target_class_key}_svm_occlusion_overlay.png",
        mime="image/png",
        use_container_width=True,
    )


def _render_resnet_gradcam_comparison(
    explanation: GradCAMResult,
    opacity: float,
) -> None:
    layer3 = explanation.layer_heatmaps["cnn_backbone.layer3[-1]"]
    layer4 = explanation.layer_heatmaps["cnn_backbone.layer4[-1]"]
    maps = [
        ("Layer 3", "14×14 spatial map", layer3),
        ("Layer 4", "7×7 semantic map", layer4),
        ("Fused", "50% Layer 3 + 50% Layer 4", explanation.heatmap),
    ]

    st.markdown("**Overlay comparison**")
    original_col, layer3_col, layer4_col, fused_col = st.columns(4)
    with original_col:
        st.image(explanation.model_input_rgb, caption="Model input")
    for column, (label, description, heatmap) in zip(
        (layer3_col, layer4_col, fused_col), maps
    ):
        with column:
            st.image(
                overlay_heatmap(explanation, opacity, heatmap),
                caption=f"{label} overlay · {description}",
            )

    with st.expander("Compare heatmaps without MRI overlay"):
        for column, (label, description, heatmap) in zip(st.columns(3), maps):
            with column:
                st.image(colorize_heatmap(heatmap), caption=f"{label} · {description}")

    download_columns = st.columns(3)
    for column, (label, _description, heatmap) in zip(download_columns, maps):
        slug = label.lower().replace(" ", "_")
        with column:
            st.download_button(
                f"Download {label} overlay",
                data=png_bytes(overlay_heatmap(explanation, opacity, heatmap)),
                file_name=(
                    f"{explanation.backbone}_{explanation.target_class_key}_{slug}_overlay.png"
                ),
                mime="image/png",
                use_container_width=True,
            )

    st.caption(
        f"Target layers: `{explanation.target_layer_name}` · "
        f"Grad-CAM time: {explanation.elapsed_seconds:.2f} s"
    )


def _render_metrics(backbone: str) -> None:
    artifacts = get_artifacts(PROJECT_ROOT, backbone)
    result = load_json(artifacts.final_results)
    training = load_json(artifacts.training_results)
    if result is None:
        st.info(f"No readable results file found at `{artifacts.final_results}`.")
        return

    neural = result.get("neural_network", {})
    validation = result.get("validation", {})
    test = result.get("test", {})
    st.subheader(f"{artifacts.display_name} performance")

    cols = st.columns(5)
    cols[0].metric("Best epoch", neural.get("best_epoch", "—"))
    cols[1].metric("NN val accuracy", _format_percent(neural.get("val_accuracy"), True))
    cols[2].metric("SVM val accuracy", _format_percent(validation.get("accuracy")))
    cols[3].metric("Test accuracy", _format_percent(test.get("accuracy")))
    cols[4].metric("Test macro F1", _format_percent(test.get("macro_f1")))

    val_tab, test_tab, class_tab, curve_tab = st.tabs(
        ["Validation confusion matrix", "Test confusion matrix", "Per-class test report", "Training curves"]
    )
    with val_tab:
        _render_confusion(result, "validation")
    with test_tab:
        _render_confusion(result, "test")
    with class_tab:
        report = test.get("classification_report", {})
        rows = []
        for key in CLASS_NAMES:
            values = report.get(key, {})
            rows.append(
                {
                    "Class": CLASS_DISPLAY[key],
                    "Precision (%)": _to_percent(values.get("precision")),
                    "Recall (%)": _to_percent(values.get("recall")),
                    "F1-score (%)": _to_percent(values.get("f1-score")),
                    "Support": values.get("support"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with curve_tab:
        _render_training_curves(training)


def _render_confusion(result: dict, split: str) -> None:
    frame = confusion_frame(result, split)
    if frame is None:
        st.info("Confusion-matrix data are unavailable.")
        return
    frame.index = [CLASS_DISPLAY[name] for name in CLASS_NAMES]
    frame.columns = [CLASS_DISPLAY[name] for name in CLASS_NAMES]
    frame.index.name = "Actual class"
    frame.columns.name = "Predicted class"
    st.dataframe(frame, use_container_width=True)


def _render_training_curves(training: dict | None) -> None:
    history = (training or {}).get("history", {})
    if not history:
        st.info("Training-history data are unavailable.")
        return
    keys = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
    lengths = [len(history.get(key, [])) for key in keys]
    epoch_count = min(lengths, default=0)
    if epoch_count == 0:
        st.info("Training-history arrays are empty.")
        return

    history_frame = pd.DataFrame(
        {
            "Epoch": range(1, epoch_count + 1),
            "Training loss": history["train_loss"][:epoch_count],
            "Validation loss": history["val_loss"][:epoch_count],
            "Training accuracy (%)": history["train_accuracy"][:epoch_count],
            "Validation accuracy (%)": history["val_accuracy"][:epoch_count],
        }
    )
    accuracy = history_frame[["Epoch", "Training accuracy (%)", "Validation accuracy (%)"]].melt(
        id_vars="Epoch", var_name="Series", value_name="Value"
    )
    loss = history_frame[["Epoch", "Training loss", "Validation loss"]].melt(
        id_vars="Epoch", var_name="Series", value_name="Value"
    )
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            alt.Chart(accuracy)
            .mark_line(point=True)
            .encode(x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1)), y="Value:Q", color="Series:N")
            .properties(title="Accuracy", height=300),
            use_container_width=True,
        )
    with right:
        st.altair_chart(
            alt.Chart(loss)
            .mark_line(point=True)
            .encode(x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1)), y="Value:Q", color="Series:N")
            .properties(title="Loss", height=300),
            use_container_width=True,
        )


def _to_percent(value):
    return round(float(value) * 100, 2) if value is not None else None


def _format_percent(value, already_percent: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value) if already_percent else float(value) * 100
    return f"{number:.2f}%"


st.title("Brain Tumor MRI Classification")
st.caption("Five-backbone CNN–LSTM–PCA–SVM research prototype")
st.warning(
    "Research and educational use only. This application is not a medical device and must "
    "not be used to diagnose, exclude, or treat disease."
)

with st.sidebar:
    st.header("Model settings")
    backbone = st.selectbox(
        "CNN backbone",
        BACKBONE_ORDER,
        index=3,
        format_func=lambda name: DISPLAY_NAMES[name],
    )
    artifacts = get_artifacts(PROJECT_ROOT, backbone)
    if artifacts.prediction_ready():
        st.success("Prediction artifacts ready")
    else:
        st.error("Prediction artifacts missing")
        for path in artifacts.missing_prediction_files():
            st.caption(str(path.relative_to(PROJECT_ROOT)))
    st.caption("Only one backbone is kept in memory at a time.")
    if st.button("Unload active model", use_container_width=True):
        _model_manager().unload()
        st.session_state.pop("prediction_record", None)
        st.success("Active model unloaded.")

predict_tab, performance_tab, comparison_tab, methodology_tab = st.tabs(
    ["Predict MRI", "Model performance", "Compare models", "Methodology"]
)

with predict_tab:
    st.subheader("Upload an MRI image")
    uploaded = st.file_uploader(
        "Supported formats: PNG, JPG/JPEG, BMP, TIFF",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is None:
        st.info("Upload one MRI image to begin.")
    else:
        raw = uploaded.getvalue()
        fingerprint = hashlib.sha256(raw).hexdigest()
        try:
            image = _open_image(raw)
            image_col, action_col = st.columns([1.25, 1])
            with image_col:
                st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height}")
            with action_col:
                st.write(f"**Selected model:** {DISPLAY_NAMES[backbone]}")
                st.write("**Final classifier:** Linear SVM")
                st.write("**Input preprocessing:** Project evaluation transform")
                analyze = st.button(
                    "Analyze MRI",
                    type="primary",
                    disabled=not artifacts.prediction_ready(),
                    use_container_width=True,
                )
                if analyze:
                    with st.spinner(f"Loading {DISPLAY_NAMES[backbone]} and running inference…"):
                        manager = _model_manager()
                        prediction = manager.predict(image, backbone)
                        explanation = None
                        explanation_error = None
                        try:
                            explanation = manager.explain(
                                image,
                                backbone,
                                prediction.nn_class_key,
                            )
                        except Exception as error:
                            explanation_error = str(error)
                    st.session_state.prediction_record = {
                        "fingerprint": fingerprint,
                        "backbone": backbone,
                        "result": prediction,
                        "explanation": explanation,
                        "explanation_error": explanation_error,
                        "occlusion": None,
                        "occlusion_error": None,
                    }

            record = st.session_state.get("prediction_record")
            if (
                record
                and record["fingerprint"] == fingerprint
                and record["backbone"] == backbone
            ):
                _render_prediction(record["result"])
                if record.get("explanation") is not None:
                    _render_explanation(record["explanation"], record["result"])
                else:
                    st.divider()
                    st.subheader("Explainability — Grad-CAM")
                    st.warning(
                        "The classification completed, but Grad-CAM could not be generated: "
                        f"{record.get('explanation_error', 'unknown error')}"
                    )
                _render_svm_occlusion(image, backbone, record["result"], record)
        except Exception as error:
            st.error(str(error))
            with st.expander("Technical details"):
                st.exception(error)

with performance_tab:
    _render_metrics(backbone)

with comparison_tab:
    st.subheader("Standardized five-backbone comparison")
    comparison = comparison_table(PROJECT_ROOT)
    if comparison.empty:
        st.info("No readable final_results.json files were found.")
    else:
        st.dataframe(comparison.drop(columns=["Backbone"]), hide_index=True, use_container_width=True)
        chart_data = comparison[["Model", "Test accuracy (%)", "Test macro F1 (%)"]].melt(
            id_vars="Model", var_name="Metric", value_name="Percent"
        )
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", sort=list(comparison["Model"]), title=None),
                y=alt.Y("Percent:Q", title="Score (%)", scale=alt.Scale(zero=False)),
                color="Metric:N",
                xOffset="Metric:N",
                tooltip=["Model:N", "Metric:N", alt.Tooltip("Percent:Q", format=".2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

with methodology_tab:
    st.subheader("Validated inference path")
    st.markdown(
        """
1. Convert the uploaded image to RGB and apply the project's deterministic evaluation transform.
2. Extract a spatial feature map with the selected pretrained-and-fine-tuned CNN backbone.
3. Reshape spatial positions into a sequence and process them with the trained LSTM.
4. Produce the trained 128-dimensional dense representation.
5. Apply the saved training-only StandardScaler and PCA transformations.
6. Classify the PCA representation with the saved linear SVM.
7. Generate Grad-CAM from the selected CNN–LSTM neural-head class and overlay it on the
   transformed model input.
8. Optionally run occlusion sensitivity through the complete pipeline to explain the final
   SVM decision score directly.

The GUI reads only the standardized artifacts under `experiments/<backbone>/`. It does not use
the older root-level ResNet152 artifacts. Grad-CAM explains the differentiable neural head; it
does not claim to segment the tumor or directly explain the external scikit-learn SVM.
Occlusion sensitivity complements Grad-CAM by perturbing the input and measuring the change
in the saved final SVM decision value.
"""
    )
    status_rows = []
    for item in all_artifacts(PROJECT_ROOT):
        status_rows.append(
            {
                "Model": item.display_name,
                "Prediction bundle": "Ready" if item.prediction_ready() else "Missing files",
                "Evaluation results": "Ready" if item.final_results.is_file() else "Missing",
                "Training history": "Ready" if item.training_results.is_file() else "Missing",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
