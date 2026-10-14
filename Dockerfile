# NVIDIA CUDA-enabled PyTorch runtime. Koyeb must deploy this on a GPU instance.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    MODEL_PATH=/data/models/VoxCPM2 \
    HF_HOME=/data/huggingface \
    MODELSCOPE_CACHE=/data/modelscope \
    DEVICE=cuda \
    OPTIMIZE=0 \
    LOAD_DENOISER=0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

COPY app.py download_model.py test_api.py README_ID.md .env.example /app/
RUN mkdir -p /data/models /data/huggingface /data/modelscope \
    && chmod -R a+rwX /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
