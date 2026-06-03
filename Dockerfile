# syntax=docker/dockerfile:1
#
# Fire / Smoke / Person YOLOv8 detector — Streamlit demo image.
#
# Build:  docker build -t fire-smoke-detector .
# Run:    docker run --rm -p 8501:8501 -v "$(pwd)/models:/app/models:ro" fire-smoke-detector
# (or just `docker compose up` — see docker-compose.yml)

FROM python:3.11-slim AS runtime

# ---------------------------------------------------------------------------
# System libraries
#   libgl1 / libglib2.0-0 : required by opencv-python (cv2) at import time
#   curl                  : used by the container HEALTHCHECK
# NOTE: no system ffmpeg — the opencv-python wheel bundles its own FFmpeg libs
# (and ignores the system ones), so apt ffmpeg would add ~470 MB for nothing.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep ML tool caches inside /app so the non-root user can always write them.
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# Install CPU-only PyTorch FIRST from the dedicated index so the (multi-GB) CUDA
# wheels are never pulled. torch then already satisfies the >=2.0 pin in
# requirements.txt, so the second install won't replace it.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code. Model weights (models/) and results (outputs/) are provided
# as volumes at run time — see docker-compose.yml — so they're NOT copied here.
COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py ./

# Run as a non-root user; give it a writable /app (caches, mounted dirs).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache /app/models /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit exposes a lightweight liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
