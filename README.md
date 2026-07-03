# Qwen3.5 Roboflow OCR Training Template

This template fine-tunes Qwen3.5 as a vision-language OCR model from a Roboflow dataset. It downloads the dataset at runtime from Roboflow, converts text-image-pair JSONL records into image-to-answer chat samples, then trains with LoRA.

Set your Roboflow dataset at runtime with either `ROBOFLOW_DATASET_URL` or the explicit workspace/project/version variables:

```text
ROBOFLOW_DATASET_URL=<your-roboflow-dataset-url>
```

## Build

```bash
docker build -t <your-image-name>:1.0.0 .
```

## Run Locally

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env.example \
  -e ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY" \
  -v "$PWD/output-qwen35:/workspace/output" \
  <your-image-name>:1.0.0
```

Do not bake `ROBOFLOW_API_KEY` into the image. Pass it at runtime.

## RunPod Template

Use these values for a RunPod template:

```text
Container image: <your-image-name>:1.0.0
Container disk:  80 GB
Volume disk:     150 GB or larger
Volume mount:    /workspace
Docker command:  python3 /opt/qwen35/train_qwen35_vlm.py
```

Environment:

```bash
ROBOFLOW_API_KEY=...
ROBOFLOW_DATASET_URL=<your-roboflow-dataset-url>
DATASET_FORMAT=jsonl
MODEL_ID=Qwen/Qwen3.5-0.8B
BITS=4
LORA=true
EPOCHS=100
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LR=1e-4
NUM_WORKERS=2
OUTPUT_DIR=/workspace/output
DATASET_DIR=/workspace/dataset
OCR_PROMPT=Read the text in this image. Answer with only the text.
EVAL=true
EVAL_STRATEGY=epoch
SAVE_STRATEGY=epoch
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL=eval_loss
GREATER_IS_BETTER=false
EARLY_STOPPING=true
EARLY_STOPPING_PATIENCE=10
TRAINING_ARGS_JSON={"dataloader_pin_memory":true,"dataloader_persistent_workers":true,"dataloader_prefetch_factor":4}
```

The default model is the smallest Qwen3.5 checkpoint, with a larger-batch preset to keep the GPU busier. Start with `BATCH_SIZE=8` and `GRAD_ACCUM_STEPS=2`; if VRAM is still underused, try `BATCH_SIZE=16` and `GRAD_ACCUM_STEPS=1`. If you hit out-of-memory, lower `BATCH_SIZE` first. For larger GPUs, you can also change `MODEL_ID` to a larger Qwen3.5 model.

With the defaults above, training runs for at most 100 epochs, evaluates and saves once per epoch, keeps the best checkpoint by `eval_loss`, and stops early if validation loss does not improve for 10 validation epochs. Early stopping requires a validation split such as `valid.jsonl` or a `valid/` directory with JSONL records.

## Dataset Format

For Roboflow `text-image-pairs` projects, use:

```bash
DATASET_FORMAT=jsonl
```

Roboflow may also allow `openai` for the same project type; the loader accepts both `jsonl` and `openai` style records. COCO is only valid for object-detection projects, not text-image-pair projects.

The JSONL loader accepts common shapes such as:

```json
{"image": "train/example.jpg", "text": "23"}
{"image": "train/example.jpg", "question": "Read the number.", "answer": "23"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "Read the text."}, {"type": "image_url", "image_url": {"url": "train/example.jpg"}}]}, {"role": "assistant", "content": "23"}]}
```

If your labels include a dataset-specific prefix that should not appear in the answer, set:

```bash
LABEL_PREFIXES_TO_STRIP=prefix-,other_prefix_
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
version.download(model_format="jsonl", location="/workspace/dataset")
```

You can skip downloading and train from an already-mounted dataset by setting:

```bash
SKIP_DATASET_DOWNLOAD=true
DATASET_DIR=/workspace/dataset
```
