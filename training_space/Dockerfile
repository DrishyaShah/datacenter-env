FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python3 and python3-pip are both for Python 3.10 on Ubuntu 22.04 — no version mismatch.
RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-dev python3-pip git &&     rm -rf /var/lib/apt/lists/* &&     ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip + setuptools from PyPI (unsloth needs modern setuptools)
RUN pip3 install --upgrade pip setuptools wheel

# Unsloth owns the torch version
RUN pip3 install unsloth

# All remaining deps — numpy is explicit, no version pins that can conflict with torch
RUN pip3 install     numpy     "torchvision>=0.20"     "torchaudio>=2.0"     stable-baselines3     gymnasium     huggingface_hub     trl     accelerate     bitsandbytes     xformers

WORKDIR /app
COPY . /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PORT=7860

CMD ["python3", "training_space/run_training.py"]
