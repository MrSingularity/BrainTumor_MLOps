"""NeuroScan — Brain Tumor Detection Frontend.

Clinical radiology UI with dark medical theme, structured diagnostic
reports, and side-by-side checkpoint comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_project.models.factory import load_checkpoint
from mlops_project.api.metrics import metrics

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle_3m"
MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "norm_stats.json"
SAMPLE_LIMIT = 6
IMAGE_SIZE = 256

# ── Clinical CSS ──────────────────────────────────────────────────────────────
CLINICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0c0f !important;
    color: #c8d0d8 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1500px !important; }

[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2936 !important;
}
[data-testid="stSidebar"] * { color: #8a9bb0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4fc3f7 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
[data-testid="stSlider"] > div > div > div { background: #1e2936 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #4fc3f7 !important; border-color: #4fc3f7 !important;
}
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2936;
    border-radius: 2px;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #4fc3f7 !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #e8f0f8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
}
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid #2a3a4a !important;
    color: #8a9bb0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 2px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: #0d2137 !important;
    border-color: #4fc3f7 !important;
    color: #4fc3f7 !important;
}
[data-testid="stButton"] button:hover {
    border-color: #4fc3f7 !important; color: #4fc3f7 !important;
}
[data-testid="stRadio"] label { color: #8a9bb0 !important; font-size: 0.8rem !important; }
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #2a3a4a !important;
    border-radius: 2px !important;
}
[data-testid="stProgressBar"] > div { background: #1e2936 !important; border-radius: 1px !important; }
[data-testid="stProgressBar"] > div > div { background: #4fc3f7 !important; border-radius: 1px !important; }
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #1e2936 !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    color: #4fc3f7 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stImage"] img { border: 1px solid #1e2936 !important; border-radius: 2px; }
[data-testid="stAlert"] { background: #0d1117 !important; border-radius: 2px !important; border-width: 1px !important; }
hr { border-color: #1e2936 !important; }

table { border-collapse: collapse; width: 100%; }
th {
    background: #0d1117; color: #4fc3f7;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 0.5rem 0.75rem; border: 1px solid #1e2936;
}
td {
    color: #c8d0d8; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    padding: 0.45rem 0.75rem; border: 1px solid #1e2936;
}
tr:nth-child(even) td { background: #0d1117; }

.clinical-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #4fc3f7; border-bottom: 1px solid #1e2936;
    padding-bottom: 0.4rem; margin-bottom: 0.8rem; margin-top: 1.2rem;
}
.badge-tumor {
    display: inline-block; background: #1a0a0a;
    border: 1px solid #ef4444; color: #ef4444;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.15em; padding: 0.35rem 0.8rem;
    text-transform: uppercase; border-radius: 2px;
}
.badge-clear {
    display: inline-block; background: #0a1a10;
    border: 1px solid #22c55e; color: #22c55e;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.15em; padding: 0.35rem 0.8rem;
    text-transform: uppercase; border-radius: 2px;
}
.badge-conflict {
    display: inline-block; background: #1a1200;
    border: 1px solid #f59e0b; color: #f59e0b;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.15em; padding: 0.35rem 0.8rem;
    text-transform: uppercase; border-radius: 2px;
}
.study-info { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4a6070; line-height: 1.9; }
.study-info span { color: #8a9bb0; }
.scan-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    color: #4fc3f7; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem;
}
.report-box {
    background: #0d1117; border: 1px solid #1e2936; border-radius: 2px;
    padding: 1.25rem 1.5rem; margin-top: 0.5rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; line-height: 1.9;
}
.report-label { color: #4fc3f7; font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; }
.report-finding { color: #e8f0f8; font-size: 0.78rem; line-height: 1.7; margin-top: 0.5rem; }
.report-impression-tumor { color: #ef4444; font-weight: 500; }
.report-impression-clear { color: #22c55e; font-weight: 500; }
.vs-divider {
    text-align: center; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: #2a3a4a; letter-spacing: 0.2em;
    padding: 0.5rem 0; border-left: 1px solid #1e2936;
}
.disclaimer {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #2a3a4a;
    letter-spacing: 0.08em; text-transform: uppercase; text-align: center;
    padding: 0.5rem; border-top: 1px solid #1e2936; margin-top: 1rem;
}
</style>
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    risk_score: float
    model_name: str
    latency_ms: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_sample_images(limit: int = SAMPLE_LIMIT) -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    samples: list[Path] = []
    for patient_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        candidate = next(
            (p for p in sorted(patient_dir.glob("*.tif")) if not p.name.endswith("_mask.tif")),
            None,
        )
        if candidate:
            samples.append(candidate)
        if len(samples) >= limit:
            break
    return samples


def available_checkpoints() -> list[Path]:
    if not MODELS_ROOT.exists():
        return []
    return sorted(p for p in MODELS_ROOT.glob("*.pt") if p.is_file())


def load_image(path: Path) -> Image.Image:
    return Image.open(path)


def load_bytes_from_path(path: Path) -> bytes:
    return path.read_bytes()


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.tif")


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


@st.cache_data
def load_normalization_stats(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            "Normalization statistics not found. "
            "Run `python -m mlops_project.data.prepare` to generate them."
        )
    payload = json.loads(path.read_text())
    mean = np.array(payload["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.clip(np.array(payload["std"], dtype=np.float32).reshape(3, 1, 1), 1e-6, None)
    return mean, std


@st.cache_resource
def load_model_bundle(checkpoint_path_str: str):
    model, ckpt = load_checkpoint(checkpoint_path_str, device="cpu", eval_mode=True)
    return model, ckpt


def preprocess_for_model(image: Image.Image, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.array(rgb, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    normalized = (chw - mean) / std
    try:
        return torch.from_numpy(normalized).float().unsqueeze(0)
    except RuntimeError:
        return torch.tensor(normalized.tolist(), dtype=torch.float32).unsqueeze(0)


def run_local_predictor(image: Image.Image, checkpoint_path: Path, threshold: float) -> PredictionResult:
    start = time.perf_counter()
    mean, std = load_normalization_stats(str(PROCESSED_STATS_PATH))
    model, ckpt = load_model_bundle(str(checkpoint_path))
    x = preprocess_for_model(image, mean, std)
    with torch.no_grad():
        score = float(torch.sigmoid(model(x)).item())
    latency_ms = (time.perf_counter() - start) * 1000.0
    label = "tumor" if score >= threshold else "no_tumor"
    confidence = score if label == "tumor" else 1.0 - score
    model_name = str(ckpt.get("model_name", checkpoint_path.stem))

    metrics.log_prediction(
        label=label,
        confidence=confidence,
        risk_score=score,
        model_name=model_name,
        latency_ms=latency_ms,
        image_hash=hashlib.sha256(image.tobytes()).hexdigest()[:16],
        checkpoint_name=checkpoint_path.name,
        threshold=threshold,
    )

    return PredictionResult(
        label=label,
        confidence=confidence,
        risk_score=score,
        model_name=model_name,
        latency_ms=latency_ms,
    )


# ── UI helpers ────────────────────────────────────────────────────────────────

def render_badge(label: str) -> None:
    if label == "tumor":
        st.markdown("<div class='badge-tumor'>⚠ TUMOR DETECTED</div><br>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-clear'>✓ NO TUMOR DETECTED</div><br>", unsafe_allow_html=True)


def render_metrics(result: PredictionResult) -> None:
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Classification", result.label.replace("_", " ").upper())
    with m2: st.metric("Confidence", f"{result.confidence * 100:.1f}%")
    with m3: st.metric("Risk Score", f"{result.risk_score:.3f}")
    with m4: st.metric("Latency", f"{result.latency_ms:.0f} ms")
    st.progress(result.risk_score, text=f"Risk score: {result.risk_score:.3f} / 1.000")


def render_diagnostic_report(
    result: PredictionResult,
    file_bytes: bytes,
    display_name: str,
    threshold: float,
    now: datetime,
) -> None:
    study_id = hashlib.sha256(file_bytes).hexdigest()[:12].upper()

    st.markdown(
        f"""
        <div class='report-box'>
            <div style='border-bottom:1px solid #1e2936; margin-bottom:0.8rem; padding-bottom:0.4rem;'>
                <span class='report-label'>NeuroScan Diagnostic Report</span>
            </div>
            <table>
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Study ID</td><td>{study_id}</td></tr>
                <tr><td>Filename</td><td>{display_name}</td></tr>
                <tr><td>Analysis Date</td><td>{now.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td>Model</td><td>{result.model_name}</td></tr>
                <tr><td>Modality</td><td>MRI / T1</td></tr>
                <tr><td>Threshold</td><td>{threshold:.2f}</td></tr>
                <tr><td>Risk Score</td><td>{result.risk_score:.4f}</td></tr>
                <tr><td>Confidence</td><td>{result.confidence * 100:.2f}%</td></tr>
                <tr><td>Latency</td><td>{result.latency_ms:.1f} ms</td></tr>
            </table>
            <div style='margin-top:0.8rem; color:#2a3a4a; font-size:0.6rem;'>
                ⚠ NOT FOR CLINICAL USE · Research and educational purposes only ·
                Results are not clinically validated and must not influence medical decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_report(
    r1: PredictionResult,
    r2: PredictionResult,
    file_bytes: bytes,
    display_name: str,
    threshold: float,
    now: datetime,
) -> None:
    agree = r1.label == r2.label
    score_delta = abs(r1.risk_score - r2.risk_score)

    if agree:
        consensus_badge = (
            "<div class='badge-tumor'>⚠ CONSENSUS: TUMOR</div>"
            if r1.label == "tumor"
            else "<div class='badge-clear'>✓ CONSENSUS: NO TUMOR</div>"
        )
    else:
        consensus_badge = "<div class='badge-conflict'>⚡ CONFLICT: MODELS DISAGREE</div>"

    st.markdown(f"{consensus_badge}<br>", unsafe_allow_html=True)

    study_id = hashlib.sha256(file_bytes).hexdigest()[:12].upper()
    agreement_text = (
        f"Both models agree on the classification. Score delta is {score_delta:.4f}"
        + (", indicating high consistency." if score_delta < 0.1 else ", but risk scores differ — review with caution.")
        if agree else
        f"Models produce conflicting classifications. {r1.model_name} scores {r1.risk_score:.3f} "
        f"vs {r2.model_name} at {r2.risk_score:.3f}. Consider ensemble or human review."
    )

    st.markdown(
        f"""
        <div class='report-box'>
            <div style='border-bottom:1px solid #1e2936; margin-bottom:0.8rem; padding-bottom:0.4rem;'>
                <span class='report-label'>Comparison Report · {study_id}</span>
                &nbsp;&nbsp;<span style='color:#4a6070; font-size:0.62rem;'>{now.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <table>
                <tr><th>Parameter</th><th>{r1.model_name}</th><th>{r2.model_name}</th><th>Delta</th></tr>
                <tr>
                    <td>Classification</td>
                    <td style='color:{"#ef4444" if r1.label=="tumor" else "#22c55e"}'>{r1.label.upper()}</td>
                    <td style='color:{"#ef4444" if r2.label=="tumor" else "#22c55e"}'>{r2.label.upper()}</td>
                    <td>{"— agree" if agree else "⚡ differ"}</td>
                </tr>
                <tr><td>Risk Score</td><td>{r1.risk_score:.4f}</td><td>{r2.risk_score:.4f}</td><td>{r2.risk_score - r1.risk_score:+.4f}</td></tr>
                <tr><td>Confidence</td><td>{r1.confidence*100:.1f}%</td><td>{r2.confidence*100:.1f}%</td><td>{(r2.confidence - r1.confidence)*100:+.1f}%</td></tr>
                <tr><td>Latency</td><td>{r1.latency_ms:.0f} ms</td><td>{r2.latency_ms:.0f} ms</td><td>{r2.latency_ms - r1.latency_ms:+.0f} ms</td></tr>
                <tr><td>Threshold</td><td colspan='3'>{threshold:.2f} (shared)</td></tr>
            </table>
            <div style='margin-top:1rem;'>
                <span class='report-label'>Agreement Analysis</span>
                <div class='report-finding'>{agreement_text}</div>
            </div>
            <div style='margin-top:0.6rem; color:#2a3a4a; font-size:0.6rem;'>
                ⚠ NOT FOR CLINICAL USE · Research and educational purposes only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NeuroScan · Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    now = datetime.now()
    checkpoints = available_checkpoints()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧠 NeuroScan")
        st.markdown(
            "<div class='study-info'>Brain Tumor Detection System<br>MLOps Pipeline · v0.1.0</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("<div class='clinical-header'>Mode</div>", unsafe_allow_html=True)
        mode = st.radio(
            "Analysis mode",
            ["Single model", "Compare checkpoints"],
            horizontal=False,
            label_visibility="collapsed",
        )

        st.markdown("<div class='clinical-header'>Parameters</div>", unsafe_allow_html=True)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50)

        st.markdown("<div class='clinical-header'>Checkpoint(s)</div>", unsafe_allow_html=True)
        if not checkpoints:
            st.error("No model checkpoints found in models/")
            ckpt_a = ckpt_b = None
        else:
            names = [p.name for p in checkpoints]
            if mode == "Single model":
                sel_a = st.selectbox("Checkpoint", names, index=0)
                ckpt_a = MODELS_ROOT / sel_a
                ckpt_b = None
            else:
                sel_a = st.selectbox("Model A", names, index=0)
                sel_b = st.selectbox("Model B", names, index=min(1, len(names) - 1))
                ckpt_a = MODELS_ROOT / sel_a
                ckpt_b = MODELS_ROOT / sel_b

        st.divider()
        st.markdown(
            "<div class='study-info'>"
            "① Select mode &amp; checkpoint(s)<br>"
            "② Upload or pick a scan<br>"
            "③ Run analysis<br>"
            "④ Read diagnostic report"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='disclaimer'>⚠ Not for clinical use</div>",
            unsafe_allow_html=True,
        )

    # ── Page header ───────────────────────────────────────────────────────────
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown("## Brain Tumor Detection · MRI Analysis")
        st.markdown(
            f"<div class='study-info'>"
            f"Date: <span>{now.strftime('%Y-%m-%d')}</span> &nbsp;|&nbsp; "
            f"Time: <span>{now.strftime('%H:%M:%S')}</span> &nbsp;|&nbsp; "
            f"Modality: <span>MRI / T1</span> &nbsp;|&nbsp; "
            f"Mode: <span>{'Comparison' if mode == 'Compare checkpoints' else 'Single inference'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.info("Results are **not clinically validated**.")
    st.divider()

    # ── Image input ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.0, 1.3], gap="large")

    with left_col:
        st.markdown("<div class='clinical-header'>Image Input</div>", unsafe_allow_html=True)
        input_mode = st.radio(
            "Source", ["Upload image", "Dataset examples"],
            horizontal=True, label_visibility="collapsed",
        )

        file_bytes: bytes | None = None
        display_name = ""
        source_image: Image.Image | None = None
        preview_image: Image.Image | None = None
        mask_image: Image.Image | None = None
        selected_example: Path | None = None

        if input_mode == "Upload image":
            uploaded_file = st.file_uploader(
                "Upload MRI scan", type=["png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=False,
            )
            if uploaded_file is None:
                st.caption("Upload a TIFF, PNG, or JPEG MRI scan to begin analysis.")
            else:
                file_bytes = uploaded_file.getvalue()
                display_name = uploaded_file.name
                source_image = Image.open(uploaded_file).convert("RGB")
                preview_image = to_grayscale(source_image)

        else:
            sample_images = find_sample_images()
            if not sample_images:
                st.error("No sample images found in data/raw/kaggle_3m.")
            else:
                cols = st.columns(3)
                for idx, sp in enumerate(sample_images):
                    with cols[idx % 3]:
                        st.image(to_grayscale(load_image(sp)), caption=sp.parent.name, use_container_width=True)
                        is_sel = st.session_state.selected_sample == str(sp)
                        if st.button(
                            f"{'▶ ' if is_sel else ''}Select",
                            key=f"s_{sp.as_posix()}",
                            type="primary" if is_sel else "secondary",
                        ):
                            st.session_state.selected_sample = str(sp)
                            st.rerun()

                if st.session_state.selected_sample:
                    selected_example = Path(st.session_state.selected_sample)
                    file_bytes = load_bytes_from_path(selected_example)
                    display_name = selected_example.name
                    source_image = load_image(selected_example).convert("RGB")
                    preview_image = to_grayscale(source_image)
                    pm = mask_path_for(selected_example)
                    if pm.exists():
                        mask_image = load_image(pm).convert("RGB")

        if file_bytes is None or source_image is None:
            return

        st.markdown("<div class='clinical-header'>Study Information</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='study-info'>"
            f"Filename: <span>{display_name}</span><br>"
            f"Size: <span>{len(file_bytes):,} bytes</span><br>"
            f"SHA-256: <span>{hashlib.sha256(file_bytes).hexdigest()[:20]}…</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='clinical-header'>MRI Scan</div>", unsafe_allow_html=True)
        st.markdown("<div class='scan-label'>Axial · T1 · Grayscale</div>", unsafe_allow_html=True)
        st.image(preview_image, use_container_width=True)

        if mask_image is not None:
            st.markdown("<div class='clinical-header'>Ground Truth Mask</div>", unsafe_allow_html=True)
            st.markdown("<div class='scan-label'>Tumor region annotation</div>", unsafe_allow_html=True)
            st.image(mask_image, use_container_width=True)

        predict_clicked = (input_mode == "Upload image") and st.button(
            "▶  Run Analysis", type="primary", use_container_width=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    with right_col:
        should_predict = predict_clicked or (input_mode == "Dataset examples" and file_bytes is not None)

        if not should_predict:
            st.markdown(
                "<div class='study-info' style='padding:3rem 0; text-align:center;'>"
                "Awaiting image input…<br><br>"
                "Select a scan on the left to generate the diagnostic report."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if ckpt_a is None:
            st.error("No checkpoint selected. Add a .pt file to the models/ directory.")
            return

        # ── Single model ──────────────────────────────────────────────────────
        if mode == "Single model":
            st.markdown("<div class='clinical-header'>Diagnostic Report</div>", unsafe_allow_html=True)
            try:
                result = run_local_predictor(source_image, ckpt_a, float(threshold))
            except FileNotFoundError as e:
                st.error(str(e)); return
            except Exception as e:
                st.error(f"Analysis failed: {e}"); return

            render_badge(result.label)
            render_metrics(result)
            render_diagnostic_report(result, file_bytes, display_name, float(threshold), now)

            with st.expander("⚙ Analysis Details"):
                st.markdown(
                    f"**Checkpoint:** `{ckpt_a.name}`  \n"
                    f"**Architecture:** `{result.model_name}`  \n"
                    f"**Input resolution:** {IMAGE_SIZE}×{IMAGE_SIZE} px  \n"
                    f"**Normalization:** per-channel mean/std from training set  \n"
                    f"**Inference device:** CPU  \n"
                    f"**Decision threshold:** {float(threshold):.2f}  \n"
                    f"**Prediction logged:** yes"
                )

        # ── Comparison mode ───────────────────────────────────────────────────
        else:
            if ckpt_b is None:
                st.error("Select a second checkpoint for comparison."); return

            st.markdown("<div class='clinical-header'>Checkpoint Comparison</div>", unsafe_allow_html=True)
            try:
                r1 = run_local_predictor(source_image, ckpt_a, float(threshold))
                r2 = run_local_predictor(source_image, ckpt_b, float(threshold))
            except FileNotFoundError as e:
                st.error(str(e)); return
            except Exception as e:
                st.error(f"Analysis failed: {e}"); return

            col_a, col_vs, col_b = st.columns([1, 0.12, 1])

            with col_a:
                st.markdown(f"<div class='scan-label'>Model A · {r1.model_name}</div>", unsafe_allow_html=True)
                render_badge(r1.label)
                st.metric("Classification", r1.label.replace("_", " ").upper())
                st.metric("Confidence", f"{r1.confidence * 100:.1f}%")
                st.metric("Risk Score", f"{r1.risk_score:.3f}")
                st.metric("Latency", f"{r1.latency_ms:.0f} ms")
                st.progress(r1.risk_score, text=f"{r1.risk_score:.3f}")

            with col_vs:
                st.markdown("<div class='vs-divider' style='margin-top:3rem;'>VS</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown(f"<div class='scan-label'>Model B · {r2.model_name}</div>", unsafe_allow_html=True)
                render_badge(r2.label)
                st.metric("Classification", r2.label.replace("_", " ").upper())
                st.metric("Confidence", f"{r2.confidence * 100:.1f}%")
                st.metric("Risk Score", f"{r2.risk_score:.3f}")
                st.metric("Latency", f"{r2.latency_ms:.0f} ms")
                st.progress(r2.risk_score, text=f"{r2.risk_score:.3f}")

            st.markdown("<div class='clinical-header'>Comparison Report</div>", unsafe_allow_html=True)
            render_comparison_report(r1, r2, file_bytes, display_name, float(threshold), now)

            with st.expander("⚙ Analysis Details"):
                st.markdown(
                    f"**Model A:** `{ckpt_a.name}` ({r1.model_name})  \n"
                    f"**Model B:** `{ckpt_b.name}` ({r2.model_name})  \n"
                    f"**Shared threshold:** {float(threshold):.2f}  \n"
                    f"**Input resolution:** {IMAGE_SIZE}×{IMAGE_SIZE} px  \n"
                    f"**Normalization:** per-channel mean/std from training set  \n"
                    f"**Inference device:** CPU  \n"
                    f"**Both predictions logged:** yes"
                )


if __name__ == "__main__":
    main()