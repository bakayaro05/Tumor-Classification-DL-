"""Reproducible qualitative evaluation of the frozen multi-layer Grad-CAM."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from gui.gradcam import colorize_heatmap, overlay_heatmap
from gui.inference_engine import ModelManager, PredictionResult
from gui.results_loader import CLASS_DISPLAY, CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent
BACKBONE = "resnet152"
SAMPLE_SEED = 42
SAMPLES_PER_CLASS = 5
OVERLAY_OPACITY = 0.55
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RATINGS = ("Select…", "Good", "Partial", "Poor", "Not applicable")
BEST_MAPS = ("Select…", "Layer 3", "Layer 4", "Fused", "None / not applicable")
RESULT_DIR = PROJECT_ROOT / "gradcam_evaluation"
FIGURE_DIR = RESULT_DIR / "figures"
RESULT_CSV = RESULT_DIR / "gradcam_evaluation_resnet152_seed42.csv"
CSV_FIELDS = (
    "protocol_version",
    "backbone",
    "sample_seed",
    "samples_per_class",
    "overlay_opacity",
    "sample_number",
    "image_id",
    "image_path",
    "true_class",
    "svm_prediction",
    "neural_prediction",
    "gradcam_target_class",
    "svm_correct",
    "neural_correct",
    "classifiers_agree",
    "layer3_rating",
    "layer4_rating",
    "fused_rating",
    "best_map",
    "observation",
    "evaluated_at_utc",
)


st.set_page_config(
    page_title="Grad-CAM Evaluation",
    page_icon="🔍",
    layout="wide",
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


def _class_directory(test_dir: Path, class_name: str) -> Path:
    direct = test_dir / class_name
    if direct.is_dir():
        return direct
    matches = [item for item in test_dir.iterdir() if item.is_dir() and item.name.lower() == class_name]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find the {class_name!r} class folder under {test_dir}.")


@st.cache_data(show_spinner=False)
def _evaluation_samples(test_dir_text: str) -> list[dict[str, str]]:
    test_dir = Path(test_dir_text)
    samples: list[dict[str, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = _class_directory(test_dir, class_name)
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Class {class_name!r} contains only {len(candidates)} readable images; "
                f"{SAMPLES_PER_CLASS} are required."
            )
        rng = random.Random(f"gradcam-evaluation-{SAMPLE_SEED}-{class_name}")
        selected = sorted(rng.sample(candidates, SAMPLES_PER_CLASS))
        for image_path in selected:
            samples.append(
                {
                    "image_id": f"{class_name}/{image_path.name}",
                    "image_path": str(image_path),
                    "true_class": class_name,
                }
            )
    return samples


def _model_manager() -> ModelManager:
    manager = st.session_state.get("gradcam_eval_model_manager")
    if manager is None:
        manager = ModelManager(PROJECT_ROOT, device_preference="auto")
        st.session_state.gradcam_eval_model_manager = manager
    return manager


def _open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read test image: {path}") from error


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


def _set_sample(index: int, total: int) -> None:
    st.session_state.gradcam_eval_index = max(0, min(index, total - 1))
    st.session_state.pop("gradcam_eval_analysis", None)


def _analyze(sample: dict[str, str]) -> tuple[PredictionResult, object]:
    image = _open_image(Path(sample["image_path"]))
    manager = _model_manager()
    prediction = manager.predict(image, BACKBONE)
    explanation = manager.explain(image, BACKBONE, prediction.nn_class_key)
    return prediction, explanation


def _display_maps(explanation: object) -> None:
    layer3 = explanation.layer_heatmaps["cnn_backbone.layer3[-1]"]
    layer4 = explanation.layer_heatmaps["cnn_backbone.layer4[-1]"]
    maps = (
        ("Layer 3", "14×14 spatial map", layer3),
        ("Layer 4", "7×7 semantic map", layer4),
        ("Fused", "50% Layer 3 + 50% Layer 4", explanation.heatmap),
    )
    original_col, layer3_col, layer4_col, fused_col = st.columns(4)
    with original_col:
        st.image(explanation.model_input_rgb, caption="Model input")
    for column, (label, description, heatmap) in zip(
        (layer3_col, layer4_col, fused_col), maps
    ):
        with column:
            st.image(
                overlay_heatmap(explanation, OVERLAY_OPACITY, heatmap),
                caption=f"{label} overlay · {description}",
            )
    with st.expander("Heatmaps without MRI overlay"):
        for column, (label, description, heatmap) in zip(st.columns(3), maps):
            with column:
                st.image(colorize_heatmap(heatmap), caption=f"{label} · {description}")


def _save_figures(sample: dict[str, str], explanation: object) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{int(st.session_state.gradcam_eval_index) + 1:02d}_{sample['true_class']}_{Path(sample['image_path']).stem}"
    Image.fromarray(explanation.model_input_rgb).save(FIGURE_DIR / f"{stem}_input.png")
    maps = {
        "layer3": explanation.layer_heatmaps["cnn_backbone.layer3[-1]"],
        "layer4": explanation.layer_heatmaps["cnn_backbone.layer4[-1]"],
        "fused": explanation.heatmap,
    }
    for name, heatmap in maps.items():
        Image.fromarray(overlay_heatmap(explanation, OVERLAY_OPACITY, heatmap)).save(
            FIGURE_DIR / f"{stem}_{name}_overlay.png"
        )


def _rating_index(options: tuple[str, ...], value: str | None, default: str) -> int:
    selected = value if value in options else default
    return options.index(selected)


def _render_summary(records: dict[str, dict[str, str]], total: int) -> None:
    st.divider()
    st.subheader("Evaluation progress and summary")
    complete = len(records)
    st.progress(complete / total, text=f"{complete} of {total} images rated")
    if not records:
        return
    frame = pd.DataFrame(records.values()).sort_values("sample_number")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if complete == total:
        st.success("All 20 images have been evaluated. The CSV is ready for report analysis.")
        summary = []
        for map_column, map_name in (
            ("layer3_rating", "Layer 3"),
            ("layer4_rating", "Layer 4"),
            ("fused_rating", "Fused"),
        ):
            counts = frame[map_column].value_counts()
            summary.append(
                {
                    "Map": map_name,
                    "Good": int(counts.get("Good", 0)),
                    "Partial": int(counts.get("Partial", 0)),
                    "Poor": int(counts.get("Poor", 0)),
                    "Not applicable": int(counts.get("Not applicable", 0)),
                }
            )
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.download_button(
        "Download evaluation CSV",
        data=_records_csv_bytes(records),
        file_name=RESULT_CSV.name,
        mime="text/csv",
        use_container_width=True,
    )


st.title("Frozen Grad-CAM qualitative evaluation")
st.caption(
    "ResNet152 · official test set · fixed seed 42 · five images per class · "
    "fixed overlay opacity 0.55"
)
st.info(
    "Rate localization, not classification confidence. Good = strongest activation "
    "substantially overlaps the apparent lesion; Partial = some overlap with considerable "
    "irrelevant activation; Poor = strongest activation is away from the apparent lesion. "
    "Use Not applicable for no-tumor localization."
)

try:
    test_dir = _test_directory()
    samples = _evaluation_samples(str(test_dir))
except Exception as error:
    st.error(str(error))
    st.stop()

if "gradcam_eval_index" not in st.session_state:
    st.session_state.gradcam_eval_index = 0
if "gradcam_eval_records" not in st.session_state:
    st.session_state.gradcam_eval_records = _load_records()

index = int(st.session_state.gradcam_eval_index)
records: dict[str, dict[str, str]] = st.session_state.gradcam_eval_records

with st.sidebar:
    st.header("Evaluation set")
    selected_index = st.selectbox(
        "Current image",
        range(len(samples)),
        index=index,
        format_func=lambda item: (
            f"{item + 1:02d} · {CLASS_DISPLAY[samples[item]['true_class']]} · "
            f"{Path(samples[item]['image_path']).name}"
        ),
    )
    if selected_index != index:
        _set_sample(selected_index, len(samples))
        st.rerun()
    st.metric("Completed", f"{len(records)} / {len(samples)}")
    st.caption(f"Results file: `{RESULT_CSV.relative_to(PROJECT_ROOT)}`")

sample = samples[index]
existing = records.get(sample["image_id"], {})
header_left, header_middle, header_right = st.columns([2, 1, 1])
header_left.subheader(f"Sample {index + 1} of {len(samples)} — {sample['image_id']}")
header_middle.metric("Ground-truth class", CLASS_DISPLAY[sample["true_class"]])
header_right.metric("Status", "Rated" if existing else "Pending")

image = _open_image(Path(sample["image_path"]))
preview_col, action_col = st.columns([1, 2])
with preview_col:
    st.image(image, caption=Path(sample["image_path"]).name)
with action_col:
    st.write(f"**Official test path:** `{Path(sample['image_path']).relative_to(test_dir)}`")
    analyze = st.button("Analyze this evaluation image", type="primary", use_container_width=True)

if analyze:
    with st.spinner("Running prediction and Layer 3/Layer 4 Grad-CAM…"):
        prediction, explanation = _analyze(sample)
        st.session_state.gradcam_eval_analysis = {
            "image_id": sample["image_id"],
            "prediction": prediction,
            "explanation": explanation,
        }

analysis = st.session_state.get("gradcam_eval_analysis")
if analysis and analysis.get("image_id") == sample["image_id"]:
    prediction = analysis["prediction"]
    explanation = analysis["explanation"]
    result_columns = st.columns(4)
    result_columns[0].metric("True class", CLASS_DISPLAY[sample["true_class"]])
    result_columns[1].metric("Final SVM", prediction.svm_class_name)
    result_columns[2].metric("Neural head", prediction.nn_class_name)
    result_columns[3].metric("Heads agree", "Yes" if prediction.models_agree else "No")
    if prediction.svm_class_key == sample["true_class"]:
        st.success("The final SVM prediction is correct for this selected test image.")
    else:
        st.warning("The final SVM prediction is incorrect. Rate what the heatmap actually shows.")

    _display_maps(explanation)

    default_rating = "Not applicable" if sample["true_class"] == "notumor" else "Select…"
    with st.form(f"gradcam_rating_{sample['image_id']}"):
        rating_columns = st.columns(3)
        layer3_rating = rating_columns[0].selectbox(
            "Layer 3 localization",
            RATINGS,
            index=_rating_index(RATINGS, existing.get("layer3_rating"), default_rating),
        )
        layer4_rating = rating_columns[1].selectbox(
            "Layer 4 localization",
            RATINGS,
            index=_rating_index(RATINGS, existing.get("layer4_rating"), default_rating),
        )
        fused_rating = rating_columns[2].selectbox(
            "Fused localization",
            RATINGS,
            index=_rating_index(RATINGS, existing.get("fused_rating"), default_rating),
        )
        best_map = st.selectbox(
            "Most anatomically relevant map",
            BEST_MAPS,
            index=_rating_index(
                BEST_MAPS,
                existing.get("best_map"),
                "None / not applicable" if sample["true_class"] == "notumor" else "Select…",
            ),
        )
        observation = st.text_area(
            "Brief observation",
            value=existing.get("observation", ""),
            placeholder="Example: Layer 4 overlaps the visible lateral mass; Layer 3 extends into the face.",
        )
        submitted = st.form_submit_button("Save rating", type="primary", use_container_width=True)

    if submitted:
        if "Select…" in {layer3_rating, layer4_rating, fused_rating, best_map}:
            st.error("Complete all ratings before saving this sample.")
        elif not observation.strip():
            st.error("Add a brief observation before saving this sample.")
        else:
            records[sample["image_id"]] = {
                "protocol_version": "1",
                "backbone": BACKBONE,
                "sample_seed": str(SAMPLE_SEED),
                "samples_per_class": str(SAMPLES_PER_CLASS),
                "overlay_opacity": f"{OVERLAY_OPACITY:.2f}",
                "sample_number": str(index + 1),
                "image_id": sample["image_id"],
                "image_path": str(Path(sample["image_path"]).relative_to(test_dir)),
                "true_class": sample["true_class"],
                "svm_prediction": prediction.svm_class_key,
                "neural_prediction": prediction.nn_class_key,
                "gradcam_target_class": prediction.nn_class_key,
                "svm_correct": str(prediction.svm_class_key == sample["true_class"]),
                "neural_correct": str(prediction.nn_class_key == sample["true_class"]),
                "classifiers_agree": str(prediction.models_agree),
                "layer3_rating": layer3_rating,
                "layer4_rating": layer4_rating,
                "fused_rating": fused_rating,
                "best_map": best_map,
                "observation": observation.strip(),
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _save_records(records)
            _save_figures(sample, explanation)
            st.success("Rating and comparison figures saved.")

navigation_left, navigation_right = st.columns(2)
if navigation_left.button("← Previous image", disabled=index == 0, use_container_width=True):
    _set_sample(index - 1, len(samples))
    st.rerun()
if navigation_right.button(
    "Next image →", disabled=index == len(samples) - 1, use_container_width=True
):
    _set_sample(index + 1, len(samples))
    st.rerun()

_render_summary(records, len(samples))
