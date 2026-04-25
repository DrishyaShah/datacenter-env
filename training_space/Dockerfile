FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Training dependencies (Unsloth + project deps)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
        torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        stable-baselines3 gymnasium numpy fastapi uvicorn pydantic \
        huggingface_hub trl accelerate bitsandbytes xformers

WORKDIR /app
COPY . /app

# Health endpoint so HF doesn't kill the Space during training
# Port 7860 returns 200 OK while training runs in background
ENV PORT=7860

CMD ["python", "training_space/run_training.py"]
