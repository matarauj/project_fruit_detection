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
    min_value=0.05, max_value=0.95,
    value=cfg["detection"]["confidence_threshold"],
    step=0.05,
)
tracker_threshold = st.sidebar.slider(
    "Tracker high-confidence threshold",
    min_value=0.05, max_value=0.95,
    value=cfg["tracking"].get("track_high_thresh", 0.25),
    step=0.05,
)
avg_weight_g = st.sidebar.number_input(
    "Average apple weight (g)",
    min_value=50, max_value=500,
    value=int(cfg["yield"]["avg_apple_weight_g"]),
    step=10,
)
line_position = st.sidebar.slider(
    "Counting line position (Method 1)",
    min_value=0.1, max_value=0.9,
    value=cfg["counting"]["method1"]["line_position"],
    step=0.05,
)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🍎 Fruit Harvest MVP")
st.caption("Apple detection, tracking, and yield estimation from orchard video.")

uploaded_file = st.file_uploader(
    "Upload orchard video (≤ 50 MB)",
    type=["mp4", "mov", "avi"],
    help="Recorded on a mobile phone walking through the orchard.",
)

if uploaded_file is None:
    st.info("Upload a video to begin processing.")
    st.stop()

# ── Processing ────────────────────────────────────────────────────────────────
run_button = st.button("▶ Run analysis", type="primary")

if not run_button:
    st.stop()

# Save upload to a temp file
with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = Path(tmp.name)

# TODO: Initialise pipeline with sidebar values (Phase 5)
# pipeline_cfg = PipelineConfig.from_yaml()
# pipeline_cfg.conf = conf_threshold
# pipeline_cfg.avg_apple_weight_g = avg_weight_g
# pipeline_cfg.line_position = line_position
# pipeline = FruitPipeline(pipeline_cfg)

# ── Results layout ────────────────────────────────────────────────────────────
st.subheader("Results")

col1, col2, col3, col4 = st.columns(4)
# Placeholder metric cards — replace with real values from pipeline result
col1.metric("Frames processed", "—")
col2.metric("Count (Method 1 — line)", "—")
col3.metric("Count (Method 2 — ID)", "—")
col4.metric("Estimated yield (kg)", "—")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Processed video")
    # TODO: st.video(output_video_path) after pipeline runs

with right:
    st.subheader("Detection confidence distribution")
    # TODO: Plot confidence histogram from pipeline result
    # fig = px.histogram(confidence_scores, nbins=20, labels={"value": "Confidence"})
    # st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Per-frame statistics")
# TODO: Display per_frame_stats as a DataFrame
# df = pd.DataFrame([...])
# st.dataframe(df, use_container_width=True)

# CSV download
# csv = df.to_csv(index=False)
# st.download_button("⬇ Download CSV", data=csv, file_name="results.csv", mime="text/csv")
