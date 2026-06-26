# ── Fruit Harvest MVP — Hugging Face Spaces Dockerfile ────────────────────────
#
# Hugging Face Spaces expects the app to listen on port 7860.
# Model weights are loaded from the HF model repo at startup
# (or baked into the image if size permits).
#
# Build locally:
#   docker build -t fruit-harvest-mvp .
#   docker run -p 7860:7860 fruit-harvest-mvp

FROM python:3.12-slim

WORKDIR /app

# System dependencies required by OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (app-only, no dev tools)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY streamlit_app/ ./streamlit_app/

# TODO: Copy or download model weights before deployment.
# Option A — bake weights into image:
#   COPY models/exported/best.pt ./models/exported/best.pt
#
# Option B — download from HF model repo at runtime (recommended for large files):
#   Add a startup script that calls huggingface_hub.hf_hub_download()

# HF Spaces runs as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
