# Human or AI? — web app image.
#
# Default build = FULL detector (classical + DINOv2). Needs ~2 GB RAM to run, so
# use it on a host with enough memory (e.g. Hugging Face Spaces free CPU = 16 GB).
#
# Lean build = classical-only fallback (no torch/timm), fits a 512 MB free host
# (Render/Koyeb) that allows a custom domain. The app auto-detects the missing
# torch and serves the 85-feature classical model:
#     docker build --build-arg INCLUDE_TORCH=0 -t humanorai .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# OpenCV runtime libs (opencv-python-headless still needs these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Always-needed Python deps (pinned scientific stack for clean model unpickle).
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# Full model: CPU-only PyTorch (the cpu index avoids the multi-GB CUDA build).
ARG INCLUDE_TORCH=1
RUN if [ "$INCLUDE_TORCH" = "1" ]; then \
        pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu && \
        pip install timm==1.0.27 ; \
    fi

# App code + the two v2 model files (everything else excluded via .dockerignore).
COPY . /app

# Bake the DINOv2 weights into the image so the first scan is fast (skipped for
# the lean build; non-fatal if the download is unavailable at build time).
RUN if [ "$INCLUDE_TORCH" = "1" ]; then \
        python -c "from src.embedding_extractor import DinoEmbedder; DinoEmbedder()" || true ; \
    fi

# Run unprivileged (Hugging Face Spaces requires uid 1000) with a writable cache.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects 7860; Render/Fly inject $PORT.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
