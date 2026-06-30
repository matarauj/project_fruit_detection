"""
Fruit Harvest MVP — Streamlit application.

Run locally:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline.pipeline import FruitPipeline, PipelineConfig
from src.utils import load_config


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Harvest MVP",
    page_icon="🍎",
    layout="wide",
)


# ── Sidebar — configuration ────────────────────────────────────────────────────
st.sidebar.header("Configuration")

cfg = load_config()

conf_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=cfg["detection"]["confidence_threshold"],
    step=0.05
)
tracker_threshold = st.sidebar.slider(
    "Tracker high-confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=cfg["tracking"].get("track_high_thresh", 0.25),
    step=0.05
)
avg_weight_g = st.sidebar.number_input(
    "Average apple weight (g)",
    min_value=50,
    max_value=500,
    value=int(cfg["yield"]["avg_apple_weight_g"]),
    step=10
)
line_position = st.sidebar.slider(
    "Counting line position (Method 1)",
    min_value=0.1,
    max_value=0.9,
    value=cfg["counting"]["method1"]["line_position"],
    step=0.05
)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🍎 Fruit Harvest MVP")
st.caption("Apple detection, tracking, and yield estimation from orchard video.")

uploaded_file = st.file_uploader(
    "Upload orchard video (≤ 50 MB)",
    type=["mp4", "mov", "avi"],
    help="Recorded on a mobile phone walking through the orchard."
)

if uploaded_file is None:
    st.info("Upload a video to begin processing.")
    st.stop()


# ── Processing ────────────────────────────────────────────────────────────────
run_button = st.button("▶ Run analysis", type="primary")

if not run_button:
    st.stop()

# Save upload to a temp file that persists for the duration of the run.
with tempfile.NamedTemporaryFile(
    suffix=Path(uploaded_file.name).suffix, delete=False
) as tmp_in:
    tmp_in.write(uploaded_file.read())
    input_path = Path(tmp_in.name)

# Prepare a temp file for the annotated output video.
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
    output_path = Path(tmp_out.name)

# Build pipeline config from YAML defaults, then override with sidebar values.
pipeline_cfg                  = PipelineConfig.from_yaml()
pipeline_cfg.conf             = conf_threshold
pipeline_cfg.avg_apple_weight_g = float(avg_weight_g)
pipeline_cfg.line_position    = line_position
pipeline_cfg.track_high_thresh   = tracker_threshold

pipeline = FruitPipeline(pipeline_cfg)

try:
    with st.spinner("Running detection and tracking — this may take a moment…"):
        result = pipeline.run(input_path, output_path=output_path)
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()


# ── Results layout ────────────────────────────────────────────────────────────
st.subheader("Results")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Frames processed",        result.total_frames)
col2.metric("Count (Method 1 — line)", result.count_m1)
col3.metric("Count (Method 2 — ID)",   result.count_m2)
col4.metric(
    "Estimated yield (kg)",
    f"{result.estimated_kg_m2:.2f}",
    help="Based on Method 2 count x average apple weight."
)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Processed video")
    st.video(str(output_path))

with right:
    st.subheader("Detection confidence distribution")
    if result.confidence_scores:
        fig = px.histogram(
            result.confidence_scores,
            nbins=20,
            labels={"value": "Confidence", "count": "Detections"},
            color_discrete_sequence=["#e04a2f"]
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections recorded.")

st.divider()

st.subheader("Per-frame statistics")

if result.per_frame_stats:
    df = pd.DataFrame([
        {
            "frame":         fs.frame_idx,
            "detections":    fs.detections_in_frame,
            "active_tracks": fs.active_tracks,
            "count_m1":      fs.count_m1,
            "count_m2":      fs.count_m2
        }
        for fs in result.per_frame_stats
    ])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )
else:
    st.info("No per-frame data available.")