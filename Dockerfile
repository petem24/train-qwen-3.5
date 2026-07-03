FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATASET_DIR=/workspace/dataset \
    OUTPUT_DIR=/workspace/output \
    DATASET_FORMAT=coco \
    MODEL_ID=Qwen/Qwen3.5-0.8B \
    TRUST_REMOTE_CODE=true \
    ATTN_IMPLEMENTATION=sdpa \
    TORCH_DTYPE=bfloat16 \
    BITS=4 \
    LORA=true \
    LORA_R=16 \
    LORA_ALPHA=32 \
    EPOCHS=3 \
    BATCH_SIZE=1 \
    GRAD_ACCUM_STEPS=8 \
    LR=2e-4 \
    GRADIENT_CHECKPOINTING=true \
    MASK_PROMPT=true \
    TENSORBOARD=true \
    WANDB=false

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    python3 \
    python3-dev \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install \
        "transformers @ git+https://github.com/huggingface/transformers.git@main" \
        accelerate \
        bitsandbytes \
        datasets \
        peft \
        pillow \
        qwen-vl-utils \
        roboflow \
        tensorboard \
        wandb

COPY train_qwen35_vlm.py /opt/qwen35/train_qwen35_vlm.py

CMD ["python3", "/opt/qwen35/train_qwen35_vlm.py"]
