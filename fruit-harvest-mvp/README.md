# Fruit Harvest MVP

Apple detection, tracking, and yield estimation from orchard video.

## Overview

A Streamlit web application that accepts a short orchard video and returns:
- The number of apples detected (two independent counting methods)
- An estimated harvest weight in kilograms
- A processed video with bounding boxes and track trails
- A CSV export of per-frame statistics

## Tech stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLO11 (Ultralytics) |
| Tracking  | ByteTrack (via Ultralytics) |
| Counting  | Line-crossing (M1) + Global ID (M2) |
| App UI    | Streamlit |
| Deployment| Hugging Face Spaces (Docker) |

## Project structure

```
fruit-harvest-mvp/
├── configs/             # settings.yaml, bytetrack.yaml
├── data/                # Videos, frames, annotations, splits
│   └── splits/dataset.yaml
├── models/              # Weights (checkpoints, exported, metrics)
├── notebooks/           # Training & tracking experiments
├── scripts/             # Data preparation utilities
│   ├── extract_frames.py
│   └── create_splits.py
├── src/                 # Core library
│   ├── detection/       # YOLO inference
│   ├── tracking/        # ByteTrack wrapper
│   ├── counting/        # Methods 1 & 2, yield estimation
│   ├── pipeline/        # End-to-end orchestration
│   ├── evaluation/      # Metrics
│   └── utils/           # Config loading, shared helpers
├── streamlit_app/       # UI
├── tests/
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fruit-harvest-mvp.git
cd fruit-harvest-mvp

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install PyTorch (match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## Data preparation

```bash
# Extract every 15th frame from all videos in data/raw/
python scripts/extract_frames.py

# After annotating with Label Studio, create the dataset splits
python scripts/create_splits.py
```

## Training

Open `notebooks/01_training_experiments.ipynb` in Google Colab for initial
experiments, then run the full training session on Kaggle (GPU P100).

## Running the app

```bash
streamlit run streamlit_app/app.py
```

## Target metrics

| Metric       | Target  |
|--------------|---------|
| Precision    | ≥ 0.80  |
| Recall       | ≥ 0.80  |
| F1           | ≥ 0.80  |
| mAP50        | ≥ 0.75  |
| mAP50-95     | ≥ 0.40  |
| IDF1         | ≥ 0.75  |
| Counting MAE | ≤ 1.5   |
| Counting SD  | ≤ 1.5   |
