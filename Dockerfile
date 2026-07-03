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
    EPOCHS=100 \
    BATCH_SIZE=8 \
    EVAL_BATCH_SIZE=8 \
    GRAD_ACCUM_STEPS=2 \
    LR=1e-4 \
    NUM_WORKERS=2 \
    GRADIENT_CHECKPOINTING=true \
    MASK_PROMPT=true \
    TENSORBOARD=true \
    WANDB=false \
    EVAL=true \
    EVAL_STRATEGY=epoch \
    SAVE_STRATEGY=epoch \
    LOAD_BEST_MODEL_AT_END=true \
    METRIC_FOR_BEST_MODEL=eval_loss \
    GREATER_IS_BETTER=false \
    EARLY_STOPPING=true \
    EARLY_STOPPING_PATIENCE=10 \
    EARLY_STOPPING_THRESHOLD=0.0 \
    TRAINING_ARGS_JSON='{"dataloader_pin_memory":true,"dataloader_persistent_workers":true,"dataloader_prefetch_factor":4}'

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
