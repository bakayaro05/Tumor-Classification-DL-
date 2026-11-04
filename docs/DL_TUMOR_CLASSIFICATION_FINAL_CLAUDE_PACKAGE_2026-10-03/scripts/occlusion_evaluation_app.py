"""Frozen comparison of neural-head Grad-CAM and final-SVM occlusion sensitivity."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from gui.gradcam import colorize_heatmap, overlay_heatmap
from gui.inference_engine import ModelManager
from gui.occlusion import overlay_occlusion
from gui.results_loader import CLASS_DISPLAY


PROJECT_ROOT = Path(__file__).resolve().parent
BACKBONE = "resnet152"
PATCH_SIZE = 32
STRIDE = 16
BATCH_SIZE = 8
OVERLAY_OPACITY = 0.55

# Chosen before viewing occlusion results: representative good/partial/poor Grad-CAM
# cases, all tumor classes, two no-tumor controls, and the misclassified pituitary case.
FIXED_SAMPLE_NUMBERS = (1, 2, 5, 6, 7, 8, 11, 15, 16, 18, 19)
GRADCAM_CSV_NAME = "gradcam_evaluation_resnet152_seed42.csv"
GRADCAM_CSV_CANDIDATES = (
    PROJECT_ROOT / "gradcam_evaluation" / GRADCAM_CSV_NAME,
    PROJECT_ROOT / GRADCAM_CSV_NAME,
)
RESULT_DIR = PROJECT_ROOT / "occlusion_evaluation"
RESULT_CSV = RESULT_DIR / "occlusion_comparison_resnet152_seed42.csv"

LOCALIZATION_RATINGS = ("Select…", "Good", "Partial", "Poor", "Not applicable")
COMPARISONS = (
    "Select…",
    "Both lesion-focused",
    "Occlusion more anatomically relevant",
    "Grad-CAM more anatomically relevant",
    "Both predominantly off-lesion",
    "Inconclusive",
    "Not applicable / no tumor",
)
CSV_FIELDS = (
    "protocol_version",
    "sample_number",
    "image_id",
    "image_path",
    "true_class",
    "svm_prediction",
    "svm_correct",
    "occlusion_target_class",
    "gradcam_layer3_rating_frozen",
    "gradcam_layer4_rating_frozen",
    "gradcam_fused_rating_frozen",
    "gradcam_best_map_frozen",
    "occlusion_rating",
    "comparison",
    "observation",
    "patch_size",
    "stride",
    "occlusion_count",
    "baseline_decision_score",
    "maximum_score_drop",
    "prediction_flip_fraction",
    "evaluated_at_utc",
)


st.set_page_config(
    page_title="Grad-CAM vs SVM Occlusion",
    page_icon="🔬",
    layout="wide",
)


def _frozen_gradcam_csv() -> Path:
    for candidate in GRADCAM_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(path) for path in GRADCAM_CSV_CANDIDATES)
    raise FileNotFoundError(
        "The completed Grad-CAM CSV is required and will only be read, never modified. "
        f"Expected {expected}."
    )


def _test_directory() -> Path:
    try:
        from src.data_pipeline_torch import TEST_DIR
    except ImportError as error:
        raise RuntimeError(
            "Could not import src.data_pipeline_torch.TEST_DIR. Place this file in the "
            "brain_tumor_project root."
        ) from error
    path = Path(TEST_DIR).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Official test directory was not found: {path}")
    return path


@st.cache_data(show_spinner=False)
def _comparison_samples(csv_path_text: str, test_dir_text: str) -> list[dict[str, str]]:
    csv_path = Path(csv_path_text)
    test_dir = Path(test_dir_text)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_number = {int(row["sample_number"]): row for row in rows}
    missing = [number for number in FIXED_SAMPLE_NUMBERS if number not in by_number]
    if missing:
        raise RuntimeError(f"Frozen Grad-CAM CSV is missing sample numbers: {missing}")

    selected = []
    for number in FIXED_SAMPLE_NUMBERS:
        row = dict(by_number[number])
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = test_dir / image_path
        row["resolved_image_path"] = str(image_path.resolve())
        selected.append(row)
    return selected


def _model_manager() -> ModelManager:
    manager = st.session_state.get("occlusion_eval_model_manager")
    if manager is None:
        manager = ModelManager(PROJECT_ROOT, device_preference="auto")
        st.session_state.occlusion_eval_model_manager = manager
    return manager


def _open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read test image: {path}") from error


def _analyze(sample: dict[str, str]):
    image = _open_image(Path(sample["resolved_image_path"]))
    manager = _model_manager()
    prediction = manager.predict(image, BACKBONE)
    gradcam = manager.explain(image, BACKBONE, prediction.nn_class_key)
    occlusion = manager.explain_svm_occlusion(
        image,
        BACKBONE,
        prediction.svm_class_key,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )
    return prediction, gradcam, occlusion


def _load_records() -> dict[str, dict[str, str]]:
    if not RESULT_CSV.is_file():
        return {}
    with RESULT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def _save_records(records: dict[str, dict[str, str]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda row: int(row["sample_number"]))
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def _records_csv_bytes(records: dict[str, dict[str, str]]) -> bytes:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(sorted(records.values(), key=lambda row: int(row["sample_number"])))
    return buffer.getvalue().encode("utf-8")


def _set_index(index: int, total: int) -> None:
    st.session_state.occlusion_eval_index = max(0, min(index, total - 1))
    st.session_state.pop("occlusion_eval_analysis", None)


def _option_index(options: tuple[str, ...], value: str | None, default: str) -> int:
    selected = value if value in options else default
    return options.index(selected)


def _display_comparison(gradcam, occlusion) -> None:
    layer3 = gradcam.layer_heatmaps["cnn_backbone.layer3[-1]"]
    layer4 = gradcam.layer_heatmaps["cnn_backbone.layer4[-1]"]
    columns = st.columns(5)
    columns[0].image(gradcam.model_input_rgb, caption="Model input")
    columns[1].image(
        overlay_heatmap(gradcam, OVERLAY_OPACITY, layer3),
        caption="Grad-CAM Layer 3 · neural head",
    )
    columns[2].image(
        overlay_heatmap(gradcam, OVERLAY_OPACITY, layer4),
        caption="Grad-CAM Layer 4 · neural head",
    )
    columns[3].image(
        overlay_heatmap(gradcam, OVERLAY_OPACITY, gradcam.heatmap),
        caption="Grad-CAM fused · neural head",
    )
    columns[4].image(
        overlay_occlusion(occlusion, OVERLAY_OPACITY),
        caption="Occlusion · final SVM",
    )

    with st.expander("Heatmaps without MRI overlay"):
        heatmap_columns = st.columns(4)
        for column, label, heatmap in zip(
            heatmap_columns,
            ("Layer 3", "Layer 4", "Fused", "Final SVM occlusion"),
            (layer3, layer4, gradcam.heatmap, occlusion.heatmap),
            strict=True,
        ):
            column.image(colorize_heatmap(heatmap), caption=label)


def _render_summary(records: dict[str, dict[str, str]], total: int) -> None:
    st.divider()
    st.subheader("Occlusion comparison progress")
    st.progress(len(records) / total, text=f"{len(records)} of {total} fixed cases rated")
    if not records:
        return
    frame = pd.DataFrame(records.values()).sort_values("sample_number")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Download occlusion comparison CSV",
        data=_records_csv_bytes(records),
        file_name=RESULT_CSV.name,
        mime="text/csv",
        use_container_width=True,
    )


st.title("Frozen Grad-CAM vs final-SVM occlusion evaluation")
st.caption(
    "ResNet152 · 11 cases selected before occlusion review · patch 32×32 · stride 16 · "
    "fixed overlay opacity 0.55"
)
st.info(
    "Grad-CAM explains the neural head. Occlusion reruns the complete saved pipeline and "
    "measures the final SVM decision-score decrease. Rate anatomical localization, not "
    "classification confidence; neither map is a tumor boundary."
)

try:
    gradcam_csv = _frozen_gradcam_csv()
    test_dir = _test_directory()
    samples = _comparison_samples(str(gradcam_csv), str(test_dir))
except Exception as error:
    st.error(str(error))
    st.stop()

if "occlusion_eval_index" not in st.session_state:
    st.session_state.occlusion_eval_index = 0
if "occlusion_eval_records" not in st.session_state:
    st.session_state.occlusion_eval_records = _load_records()

index = int(st.session_state.occlusion_eval_index)
records: dict[str, dict[str, str]] = st.session_state.occlusion_eval_records
sample = samples[index]
existing = records.get(sample["image_id"], {})

with st.sidebar:
    st.header("Fixed comparison set")
    selected = st.selectbox(
        "Current case",
        range(len(samples)),
        index=index,
        format_func=lambda item: (
            f"{item + 1:02d} · original sample {samples[item]['sample_number']} · "
            f"{samples[item]['image_id']}"
        ),
    )
    if selected != index:
        _set_index(selected, len(samples))
        st.rerun()
    st.metric("Completed", f"{len(records)} / {len(samples)}")
    st.caption(f"Frozen source: `{gradcam_csv.relative_to(PROJECT_ROOT)}`")

header_left, header_middle, header_right = st.columns([2, 1, 1])
header_left.subheader(
    f"Comparison {index + 1} of {len(samples)} — {sample['image_id']}"
)
header_middle.metric("True class", CLASS_DISPLAY[sample["true_class"]])
header_right.metric("Status", "Rated" if existing else "Pending")

image = _open_image(Path(sample["resolved_image_path"]))
preview_col, action_col = st.columns([1, 2])
preview_col.image(image, caption=Path(sample["resolved_image_path"]).name)
with action_col:
    st.write(
        "**Frozen Grad-CAM ratings:** "
        f"Layer 3 = {sample['layer3_rating']}; "
        f"Layer 4 = {sample['layer4_rating']}; "
        f"Fused = {sample['fused_rating']}; "
        f"best = {sample['best_map']}"
    )
    analyze = st.button("Run Grad-CAM and final-SVM occlusion", type="primary", use_container_width=True)

if analyze:
    try:
        with st.spinner("Running prediction, Grad-CAM, and full-pipeline occlusion…"):
            prediction, gradcam, occlusion = _analyze(sample)
        st.session_state.occlusion_eval_analysis = {
            "image_id": sample["image_id"],
            "prediction": prediction,
            "gradcam": gradcam,
            "occlusion": occlusion,
        }
    except Exception as error:
        st.error(str(error))
        with st.expander("Technical details"):
            st.exception(error)

analysis = st.session_state.get("occlusion_eval_analysis")
if analysis and analysis.get("image_id") == sample["image_id"]:
    prediction = analysis["prediction"]
    gradcam = analysis["gradcam"]
    occlusion = analysis["occlusion"]

    metrics = st.columns(5)
    metrics[0].metric("True class", CLASS_DISPLAY[sample["true_class"]])
    metrics[1].metric("Final SVM", prediction.svm_class_name)
    metrics[2].metric("Neural head", prediction.nn_class_name)
    metrics[3].metric("Occlusion target", occlusion.target_class_name)
    metrics[4].metric("Prediction flips", f"{occlusion.prediction_flip_fraction * 100:.1f}%")
    _display_comparison(gradcam, occlusion)

    default_rating = "Not applicable" if sample["true_class"] == "notumor" else "Select…"
    default_comparison = (
        "Not applicable / no tumor" if sample["true_class"] == "notumor" else "Select…"
    )
    with st.form(f"occlusion_rating_{sample['image_id']}"):
        occlusion_rating = st.selectbox(
            "Final-SVM occlusion localization",
            LOCALIZATION_RATINGS,
            index=_option_index(
                LOCALIZATION_RATINGS,
                existing.get("occlusion_rating"),
                default_rating,
            ),
        )
        comparison = st.selectbox(
            "Grad-CAM versus occlusion conclusion",
            COMPARISONS,
            index=_option_index(COMPARISONS, existing.get("comparison"), default_comparison),
        )
        observation = st.text_area(
            "Brief observation",
            value=existing.get("observation", ""),
            placeholder=(
                "Example: Grad-CAM is peripheral, while masking the visible mass causes the "
                "largest reduction in the final SVM score."
            ),
        )
        submitted = st.form_submit_button("Save comparison", type="primary", use_container_width=True)

    if submitted:
        if "Select…" in {occlusion_rating, comparison}:
            st.error("Complete both ratings before saving.")
        elif not observation.strip():
            st.error("Add a brief observation before saving.")
        else:
            records[sample["image_id"]] = {
                "protocol_version": "1",
                "sample_number": sample["sample_number"],
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "true_class": sample["true_class"],
                "svm_prediction": prediction.svm_class_key,
                "svm_correct": str(prediction.svm_class_key == sample["true_class"]),
                "occlusion_target_class": occlusion.target_class_key,
                "gradcam_layer3_rating_frozen": sample["layer3_rating"],
                "gradcam_layer4_rating_frozen": sample["layer4_rating"],
                "gradcam_fused_rating_frozen": sample["fused_rating"],
                "gradcam_best_map_frozen": sample["best_map"],
                "occlusion_rating": occlusion_rating,
                "comparison": comparison,
                "observation": observation.strip(),
                "patch_size": str(occlusion.patch_size),
                "stride": str(occlusion.stride),
                "occlusion_count": str(occlusion.occlusion_count),
                "baseline_decision_score": f"{occlusion.baseline_decision_score:.8f}",
                "maximum_score_drop": f"{occlusion.maximum_score_drop:.8f}",
                "prediction_flip_fraction": f"{occlusion.prediction_flip_fraction:.8f}",
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _save_records(records)
            st.success("Comparison saved separately; the frozen Grad-CAM CSV was not modified.")

navigation_left, navigation_right = st.columns(2)
if navigation_left.button("← Previous case", disabled=index == 0, use_container_width=True):
    _set_index(index - 1, len(samples))
    st.rerun()
if navigation_right.button(
    "Next case →", disabled=index == len(samples) - 1, use_container_width=True
):
    _set_index(index + 1, len(samples))
    st.rerun()

_render_summary(records, len(samples))
