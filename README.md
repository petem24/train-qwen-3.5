# Qwen3.5 Roboflow OCR Training Template

This template fine-tunes Qwen3.5 as a vision-language OCR model from a Roboflow dataset. It downloads the dataset at runtime from Roboflow, converts COCO annotations into image-to-answer chat samples, then trains with LoRA.

The Roboflow project URL is already set in `.env.example`:

```text
https://app.roboflow.com/elite-portal/basketball-jersey-numbers-ocr-4hrs4/1
```

## Build

```bash
docker build -t petemaher/qwen35-jersey-ocr:1.0.0 .
```

## Run Locally

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env.qwen35.example \
  -e ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY" \
  -v "$PWD/output-qwen35:/workspace/output" \
  petemaher/qwen35-jersey-ocr:1.0.0
```

Do not bake `ROBOFLOW_API_KEY` into the image. Pass it at runtime.

## RunPod Template

Use these values for a RunPod template:

```text
Container image: petemaher/qwen35-jersey-ocr:1.0.0
Container disk:  80 GB
Volume disk:     150 GB or larger
Volume mount:    /workspace
Docker command:  python3 /opt/qwen35/train_qwen35_vlm.py
```

Environment:

```bash
ROBOFLOW_API_KEY=...
ROBOFLOW_DATASET_URL=https://app.roboflow.com/elite-portal/basketball-jersey-numbers-ocr-4hrs4/1
DATASET_FORMAT=coco
MODEL_ID=Qwen/Qwen3.5-0.8B
BITS=4
LORA=true
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LR=2e-4
OUTPUT_DIR=/workspace/output
DATASET_DIR=/workspace/dataset
```

The default model is the smallest Qwen3.5 checkpoint so the template can start on a single GPU. For larger GPUs, change `MODEL_ID` to a larger Qwen3.5 model and adjust `BITS`, `BATCH_SIZE`, and `GRAD_ACCUM_STEPS`.

## How Labels Become OCR Answers

The script expects Roboflow COCO exports such as:

```text
/workspace/dataset/train/_annotations.coco.json
/workspace/dataset/valid/_annotations.coco.json
```

For each image, it sorts annotations from left to right and combines their category names:

```text
["2", "3"] -> "23"
["23"] -> "23"
["home", "23"] -> "home 23"
```

Converted records are written to:

```text
/workspace/output/qwen35_sft/train.jsonl
/workspace/output/qwen35_sft/valid.jsonl
```

Final LoRA adapter/model files are written to:

```text
/workspace/output/final
```

## Notes

Qwen3.5 training is intentionally configured with `ATTN_IMPLEMENTATION=sdpa` by default. The Qwen VL fine-tuning community notes currently recommend disabling Flash Attention 2 for Qwen3.5 stability.

Roboflow download uses the Python SDK with:

```python
version.download(model_format="coco", location="/workspace/dataset")
```

You can skip downloading and train from an already-mounted dataset by setting:

```bash
SKIP_DATASET_DOWNLOAD=true
DATASET_DIR=/workspace/dataset
```
